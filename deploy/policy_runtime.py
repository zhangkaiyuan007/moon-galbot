"""高频 ACT 控制环：抓头相机→按 SharedState 最新中心(+修正偏移)画标记→resize→
本地 ACT 推理出 action chunk→SDK 执行；夹爪按滞回开合，effort 受修正记忆调节。

执行/安全/夹爪逻辑改自已验证的 pi0.5 bridge。与 pi0.5 的差别：策略是本地 lerobot ACT、
头相机喂进策略前画上 locate-anything 的目标/框标记、夹爪 effort 走修正记忆。

⚠️ 需真机 + 训练好的 ACT checkpoint 才能完整运行，无法离线端到端测；纯逻辑
(chunk_is_safe / 标记与 obs 组装) 已就地自测。控制频率/pipeline 在真机 bring-up 时按实测调。
ponytail: 先用最简单的"抓→推理→执行"非流水线循环；若 15Hz 达不到再加 bridge 那套 pipeline。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))  # markers
from config import (  # noqa: E402
    ACTION_KEY, ARM_GROUP, CONTROL_HZ, GRIP_CLOSE_BELOW, GRIP_EFFORT,
    GRIP_OPEN_ABOVE, GRIP_VELOCITY_MPS, GRIPPER_FULL_OPEN_M, GRIPPER_NAME,
    HEAD_KEY, HEAD_SIZE_WH, HORIZON, MAX_STEP_RAD, MAX_TRANSIT_RAD, STATE_KEY,
    TRANSIT_SPEED_RAD_S, WRIST_KEY,
)
from markers import draw_markers  # noqa: E402  (tools/markers.py，与训练同一份)


def decode_rgb(compressed: dict) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(compressed["data"], np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("camera frame decode failed")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# —— 2D 光流跟踪：填补 VLM(~0.4Hz)之间的空档，让标记按控制环频率跟手 ——
_LK = dict(winSize=(21, 21), maxLevel=3,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def _feats_near(gray: np.ndarray, center, r: int = 40) -> np.ndarray:
    """在 center 周围取角点作为 LK 跟踪集，多点中位数抗遮挡(手挡住部分点仍存活)。"""
    x, y = int(center[0]), int(center[1])
    mask = np.zeros(gray.shape, np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=40, qualityLevel=0.01, minDistance=3, mask=mask)
    return pts if pts is not None else np.array([[[float(x), float(y)]]], np.float32)


def chunk_is_safe(current: np.ndarray, chunk: np.ndarray) -> tuple[bool, str]:
    """拒绝过大的重规划跳变和 chunk 内步跳变（护栏，来自 bridge）。"""
    transit = float(np.abs(chunk[0, :7] - current).max())
    if transit > MAX_TRANSIT_RAD:
        return False, f"re-plan transition {transit:.3f} rad > {MAX_TRANSIT_RAD}"
    if len(chunk) > 1:
        step = float(np.abs(np.diff(chunk[:, :7], axis=0)).max())
        if step > MAX_STEP_RAD:
            return False, f"per-step delta {step:.3f} rad > {MAX_STEP_RAD}"
    return True, ""


def build_trajectory(gm, current: np.ndarray, chunk: np.ndarray, dt: float):
    """右臂 Trajectory：当前位姿 → 过渡 → chunk 各步（来自 bridge）。"""
    transit = float(np.abs(chunk[0, :7] - current).max())
    transit_time = max(dt, transit / TRANSIT_SPEED_RAD_S)
    traj = gm.Trajectory()
    traj.joint_groups = [ARM_GROUP]
    traj.joint_names = []
    points, t = [], 0.05
    for i, q in enumerate(np.vstack([current, chunk[:, :7]])):
        pt = gm.TrajectoryPoint()
        pt.time_from_start_second = t
        t += transit_time if i == 0 else dt
        pt.joint_command_vec = [_jc(gm, p) for p in q]
        points.append(pt)
    traj.points = points
    return traj


def _jc(gm, pos: float):
    c = gm.JointCommand()
    c.position = float(pos)
    return c


class ACTPolicyWrapper:
    """加载 lerobot ACT checkpoint，按 obs 出 action chunk (T,8)。"""

    def __init__(self, checkpoint: str | Path, device: str = "cuda"):
        import torch
        from lerobot.common.policies.act.modeling_act import ACTPolicy

        self.torch = torch
        self.device = device
        self.policy = ACTPolicy.from_pretrained(checkpoint).to(device).eval()
        self.horizon = getattr(self.policy.config, "n_action_steps", HORIZON)

    def _to_batch(self, obs: dict) -> dict:
        t = self.torch
        b = {}
        for k in (HEAD_KEY, WRIST_KEY):
            img = obs[k]  # HWC uint8 RGB
            b[k] = (t.from_numpy(img).permute(2, 0, 1).float() / 255.0).unsqueeze(0).to(self.device)
        b[STATE_KEY] = t.from_numpy(obs[STATE_KEY]).float().unsqueeze(0).to(self.device)
        return b

    def infer_chunk(self, obs: dict) -> np.ndarray:
        """镜像 ACTPolicy.select_action 的前向（不走单步队列），直接取整段 chunk。"""
        t = self.torch
        with t.no_grad():
            batch = self.policy.normalize_inputs(self._to_batch(obs))
            if self.policy.config.image_features:
                batch = dict(batch)
                batch["observation.images"] = [batch[k] for k in self.policy.config.image_features]
            actions = self.policy.model(batch)[0][:, : self.horizon]
            actions = self.policy.unnormalize_outputs({ACTION_KEY: actions})[ACTION_KEY]
        return actions[0].cpu().numpy()  # (T, 8)


class PolicyRuntime:
    def __init__(self, robot, gm, policy: ACTPolicyWrapper, shared, correction,
                 target_obj: str, slow: float = 1.0, vis_dir=None, record_dir=None):
        self.robot = robot
        self.gm = gm
        self.policy = policy
        self.shared = shared
        self.correction = correction
        self.target_obj = target_obj
        self.dt = (1.0 / CONTROL_HZ) * slow
        self.grip_closed = False
        self.did_grasp = False
        self.still_chunks = 0
        self.vis_dir = vis_dir  # 非 None 时每帧存带标记的头相机图，看 VLM 识别结果
        self._vis_i = 0
        self.record_dir = record_dir  # 非 None 时每帧存原始(无标记)头相机帧，供离线验证 2D tracker
        self._rec_i = 0
        self._prev_gray = None       # LK 跟踪状态
        self._track_pts = None
        self._track_center: tuple | None = None
        self._last_gen = -1

    def _grab(self, sensor, tries: int = 15) -> dict:
        """相机偶发丢帧(返回无 'data')时重试等下一帧，避免 execute 中途崩。"""
        for _ in range(tries):
            c = self.robot.get_rgb_data(sensor)
            if c and "data" in c:
                return c
            time.sleep(0.02)
        raise RuntimeError(f"{sensor} 连续 {tries} 次无帧数据")

    def _track(self, head_rgb: np.ndarray, vlm_xy, gen: int):
        """LK 光流每帧跟随 target；VLM 出新结果(gen 变)时用其重置跟踪点纠偏，防漂移/跟到手上。"""
        gray = cv2.cvtColor(head_rgb, cv2.COLOR_RGB2GRAY)
        if gen != self._last_gen and vlm_xy is not None:  # VLM 纠偏帧：重置到语义真值
            self._track_pts = _feats_near(gray, vlm_xy)
            self._track_center = vlm_xy
            self._last_gen = gen
        elif self._track_pts is not None and self._prev_gray is not None:  # 空档帧：光流跟随
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, self._track_pts, None, **_LK)
            good = nxt[st.ravel() == 1].reshape(-1, 2) if st is not None else np.empty((0, 2))
            if len(good) >= 3:  # 存活点够 → 中位数作中心；不够就保持上次(等下次 VLM 拉回)
                self._track_center = (float(np.median(good[:, 0])), float(np.median(good[:, 1])))
                self._track_pts = good.reshape(-1, 1, 2)
        self._prev_gray = gray
        return self._track_center

    def capture(self) -> tuple[dict, np.ndarray]:
        from galbot_sdk.g1 import SensorType

        head = decode_rgb(self._grab(SensorType.HEAD_LEFT_CAMERA))
        if self.record_dir is not None:  # 原始帧(画标记前)，控制环频率的连续帧，给 tracker 验证
            cv2.imwrite(f"{self.record_dir}/frame_{self._rec_i:04d}.png",
                        cv2.cvtColor(head, cv2.COLOR_RGB2BGR))
            self._rec_i += 1
        target_xy, bin_xy, gen = self.shared.get_centers()
        target_xy = self._track(head, target_xy, gen)  # LK 高频跟随 + VLM 低频纠偏
        target_xy = self.correction.apply_offset(self.target_obj, target_xy)  # 修正偏移
        head = draw_markers(head, target_xy, bin_xy)                          # 原分辨率画点
        if self.vis_dir is not None:  # 存原分辨率带标记图，比对 VLM 识别位置是否和检测时一致
            cv2.imwrite(f"{self.vis_dir}/vis_{self._vis_i:04d}.png",
                        cv2.cvtColor(head, cv2.COLOR_RGB2BGR))
            self._vis_i += 1
        head = cv2.resize(head, HEAD_SIZE_WH, interpolation=cv2.INTER_AREA)
        wrist = decode_rgb(self._grab(SensorType.RIGHT_ARM_CAMERA))
        arm_q = np.array(self.robot.get_joint_positions([ARM_GROUP], []), dtype=np.float32)
        grip = float(np.clip(
            self.robot.get_gripper_state(GRIPPER_NAME).width / GRIPPER_FULL_OPEN_M, 0.0, 1.0))
        obs = {
            HEAD_KEY: head,
            WRIST_KEY: wrist,
            STATE_KEY: np.concatenate([arm_q, [grip]]).astype(np.float32),
        }
        return obs, arm_q

    def set_gripper(self, close: bool) -> None:
        width = 0.0 if close else GRIPPER_FULL_OPEN_M
        effort = self.correction.grip_effort(self.target_obj, GRIP_EFFORT)
        self.robot.set_gripper_command(GRIPPER_NAME, width, GRIP_VELOCITY_MPS, effort, True)
        self.grip_closed = close
        if close:
            self.did_grasp = True

    def _grip_event(self, chunk: np.ndarray) -> tuple[int, str] | None:
        prof = chunk[:, 7]
        if not self.grip_closed:
            idx = np.nonzero(prof < GRIP_CLOSE_BELOW)[0]
            return (int(idx[0]), "close") if len(idx) else None
        idx = np.nonzero(prof > GRIP_OPEN_ABOVE)[0]
        return (int(idx[0]), "open") if len(idx) else None

    def _done(self, chunk: np.ndarray, arm_q: np.ndarray) -> bool:
        if self.did_grasp and not self.grip_closed:
            self.still_chunks = self.still_chunks + 1 if np.abs(chunk[:, :7] - arm_q).max() < 0.05 else 0
            return self.still_chunks >= 4
        return False

    def run_episode(self, max_chunks: int = 90, execute: bool = False) -> None:
        from galbot_sdk.g1 import ControlStatus

        self.grip_closed = (
            self.robot.get_gripper_state(GRIPPER_NAME).width / GRIPPER_FULL_OPEN_M < 0.5)
        for i in range(max_chunks):
            obs, arm_q = self.capture()
            chunk = self.policy.infer_chunk(obs)
            print(f"[chunk {i}] action[0] {np.round(chunk[0], 3)} grip "
                  f"{chunk[0,7]:.2f}→{chunk[-1,7]:.2f}")
            ok, reason = chunk_is_safe(arm_q, chunk)
            if not ok:
                print(f"❌ unsafe: {reason}"); break
            if self._done(chunk, arm_q):
                print("✅ settled after grasp — episode complete"); break
            if not execute:
                continue
            event = self._grip_event(chunk)
            if event is not None:
                cut, action = event
                if cut > 0:
                    if self.robot.execute_joint_trajectory(
                            build_trajectory(self.gm, arm_q, chunk[:cut], self.dt), True) != ControlStatus.SUCCESS:
                        print("❌ traj failed"); break
                self.set_gripper(close=(action == "close"))
                print(f"  → gripper {action.upper()} at step {cut}")
                continue
            if self.robot.execute_joint_trajectory(
                    build_trajectory(self.gm, arm_q, chunk, self.dt), True) != ControlStatus.SUCCESS:
                print("❌ traj failed"); break


def _selfcheck() -> None:
    import numpy as np
    ok, _ = chunk_is_safe(np.zeros(7), np.zeros((HORIZON, 8)))
    assert ok
    bad, reason = chunk_is_safe(np.zeros(7), np.full((HORIZON, 8), 1.0))  # 首步跳变 1rad
    assert not bad and "transition" in reason, reason
    print("policy_runtime selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
