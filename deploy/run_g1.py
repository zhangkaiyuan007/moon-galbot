"""G1 语音抓取主入口：语音说物体 → VLM 定位 → ACT 执行 → 语音反馈，支持在线修正。

双速：VLMWorker 低频刷新目标/框中心(线程)，PolicyRuntime 高频控制环读最新中心画标记。
语音设目标触发一轮抓取；"往上抓一点/抓紧点"实时修正并记忆。

两种模式：
- **无语音 bring-up（推荐先用）**：--target-label 直接指定目标，跑一轮抓取，不需要 mic 服务。
- **语音模式**：不给 --target-label，改用 --mic-addr 消费 on-robot mic 服务。

用法：
    # 先 dry-run 无语音，把抓取调稳
    uv run python deploy/run_g1.py --checkpoint <ckpt> --model <LA-3B> \\
        --target-label "cola bottle" --bin-label "basket"
    # 调好后 --execute 真动
    uv run python deploy/run_g1.py --checkpoint <ckpt> --model <LA-3B> \\
        --target-label "cola bottle" --execute --go-home
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# 减少显存碎片（错误信息里 PyTorch 自己的建议）。必须在 import torch 前设置。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    ARM_GROUP, GRIP_VELOCITY_MPS, GRIPPER_FULL_OPEN_M, GRIPPER_NAME,
    HOME_HEAD, HOME_LEG, HOME_RIGHT_ARM,
)
from correction import CorrectionMemory  # noqa: E402
from policy_runtime import ACTPolicyWrapper, PolicyRuntime  # noqa: E402
from shared_state import SharedState  # noqa: E402
from vlm_worker import VLMWorker  # noqa: E402
from voice import VoiceMicClient  # noqa: E402

DEFAULT_LA_REPO = Path(__file__).resolve().parents[1].parent / "eagle" / "Embodied"


def confirm_or_exit(msg: str) -> None:
    while True:
        k = input(f"{msg} (y/n): ").strip().lower()
        if k == "y":
            return
        if k == "n":
            raise SystemExit("aborted by operator")


def go_home(robot, gm) -> None:
    from galbot_sdk.g1 import ControlStatus

    for group, target in [("leg", HOME_LEG), ("head", HOME_HEAD), (ARM_GROUP, HOME_RIGHT_ARM)]:
        st = robot.set_joint_positions(target, joint_groups=[group], joint_names=[],
                                       is_blocking=True, speed_rad_s=0.15, timeout_s=40.0)
        if st != ControlStatus.SUCCESS:
            raise SystemExit(f"go-home failed on {group}: {st}")
    robot.set_gripper_command(GRIPPER_NAME, GRIPPER_FULL_OPEN_M, GRIP_VELOCITY_MPS, 30, True)


def wait_for_center(shared: SharedState, timeout: float = 20.0) -> bool:
    """新目标后等 VLM 至少产出一次目标中心，避免对着旧/空标记就动。
    首轮要 target+bin 两次生成(int4 每次~2.6s)加首次 warmup，故给 20s 余量。"""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if shared.get_centers()[0] is not None:
            return True
        time.sleep(0.05)
    return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="lerobot ACT checkpoint 目录")
    p.add_argument("--model", default="nvidia/LocateAnything-3B", help="locate-anything 权重路径")
    p.add_argument("--la-repo", type=Path, default=DEFAULT_LA_REPO)
    p.add_argument("--memory", default="corrections.json", help="修正记忆 JSON")
    p.add_argument("--target-label", default=None,
                   help="无语音 bring-up：直接指定目标检测词，跑一轮抓取（给了就不启用语音）")
    p.add_argument("--bin-label", default="basket", help="框的检测词")
    p.add_argument("--mic-addr", default="tcp://127.0.0.1:6000",
                   help="语音模式：galbot-mic-service 的 ZMQ 地址 tcp://<机器人IP>:6000")
    p.add_argument("--execute", action="store_true", help="真动机器人（默认 dry-run）")
    p.add_argument("--go-home", action="store_true", help="先回采集姿态")
    p.add_argument("--vlm-full", action="store_true",
                   help="VLM 用全精度 bf16(~6GB)；默认 4-bit 量化(~2GB) 以适配 8GB 显存")
    p.add_argument("--slow", type=float, default=1.0)
    p.add_argument("--max-chunks", type=int, default=90)
    args = p.parse_args()

    import galbot_sdk.g1 as gm
    from galbot_sdk.g1 import GalbotRobot, SensorType

    if args.execute:
        print("⚠️  EXECUTE 模式：确认急停可及、工作区清空、手放急停上。")
        confirm_or_exit("确认安全条件")

    robot = None
    try:
        robot = GalbotRobot()
        if not robot.init({SensorType.HEAD_LEFT_CAMERA, SensorType.RIGHT_ARM_CAMERA}):
            raise SystemExit("GalbotRobot init failed")
        time.sleep(3)
        # 打印当前实际姿态：把机器人摆到你采集数据时的起手姿态，启动一次即可抄进 config 的 HOME_*
        for g in ("leg", "head", ARM_GROUP):
            q = [round(v, 4) for v in robot.get_joint_positions([g], [])]
            print(f"[pose] {g} = {q}")
        if args.go_home and args.execute:
            confirm_or_exit(f"回采集姿态？arm {HOME_RIGHT_ARM}")
            go_home(robot, gm)

        shared = SharedState()
        mem = CorrectionMemory(args.memory)
        policy = ACTPolicyWrapper(args.checkpoint)
        vlm = VLMWorker(robot, shared, args.model, args.la_repo, load_in_4bit=not args.vlm_full)
        vlm.start()
        runtime = PolicyRuntime(robot, gm, policy, shared, mem, target_obj="", slow=args.slow)
        voice = None

        if args.target_label:
            # —— 无语音 bring-up：命令行直接给目标，跑一轮抓取 ——
            shared.set_labels(args.target_label, args.bin_label)
            runtime.target_obj = args.target_label
            print(f"[no-voice] 目标={args.target_label} 框={args.bin_label}，等待 VLM 定位…")
            if wait_for_center(shared):
                runtime.run_episode(max_chunks=args.max_chunks, execute=args.execute)
                print("[no-voice] episode 结束")
            else:
                print("[no-voice] VLM 未定位到目标，检查检测词/相机")
        else:
            # —— 语音模式：消费 on-robot mic 服务 ——
            from tts import make_tts
            voice = VoiceMicClient(shared, mem, args.mic_addr, tts=make_tts(robot))
            voice.start()
            voice.say("准备好了，请说要拿什么")
            last_cmd = 0
            while robot.is_running():
                cmd = shared.command_id()
                if cmd != last_cmd:
                    last_cmd = cmd
                    if not wait_for_center(shared):
                        voice.say("没找到目标，请换个说法")
                        continue
                    runtime.target_obj = voice.current_obj
                    runtime.run_episode(max_chunks=args.max_chunks, execute=args.execute)
                    voice.say("完成")
                time.sleep(0.2)
    finally:
        if robot is not None:
            try:
                if voice is not None:
                    voice.stop()
                vlm.stop()
            except Exception:
                pass
            robot.request_shutdown()
            robot.wait_for_shutdown()
            robot.destroy()


if __name__ == "__main__":
    main()
