"""低频 VLM 刷新环：持续用 locate-anything 检测头相机里的目标/框中心，写进 SharedState。

在独立线程跑，频率 VLM_REFRESH_HZ（默认 1.5Hz）。控制环高频读最新中心去画标记。
物体被挪动 → 下次刷新 → 中心跳到新位置 → 抗挪动闭环。

检测语义与训练一致：对 target/bin 各 ground_single、取最大框中心（见 detect_markers.py）。
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import VLM_REFRESH_HZ  # noqa: E402

XY = tuple[float, float] | None

# 8GB 卡上 vision attention 是 O(patch²)。1280x960(6188 patch) 会 OOM，960x720(3468) 峰值 4.73GiB。
# 喂 VLM 前缩到这个上界；检测坐标再按比例还原回原图，供 policy_runtime 在原分辨率画标记。
VLM_INPUT_MAX_WH = (960, 720)


def _rescale_center(c: XY, src_wh: tuple[int, int], dst_wh: tuple[int, int]) -> XY:
    """把在 src_wh 图上检出的中心，按比例映射回 dst_wh(原图) 坐标系。"""
    if c is None:
        return None
    return (c[0] * dst_wh[0] / src_wh[0], c[1] * dst_wh[1] / src_wh[1])


def _largest_box_center(boxes: list[dict]) -> XY:
    """最大框中心；与 tools/detect_markers.py 一致，保证训练/部署同语义。"""
    if not boxes:
        return None
    b = max(boxes, key=lambda d: (d["x2"] - d["x1"]) * (d["y2"] - d["y1"]))
    return ((b["x1"] + b["x2"]) / 2.0, (b["y1"] + b["y2"]) / 2.0)


class VLMWorker:
    def __init__(self, robot, shared, model_path: str, la_repo: str | Path,
                 refresh_hz: float = VLM_REFRESH_HZ, load_in_4bit: bool = False):
        self.robot = robot
        self.shared = shared
        self.period = 1.0 / refresh_hz
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        sys.path.insert(0, str(la_repo))
        from locateanything_worker import LocateAnythingWorker
        self.worker = LocateAnythingWorker(model_path, load_in_4bit=load_in_4bit)
        self.parse_boxes = LocateAnythingWorker.parse_boxes
        self._bin_xy: XY = None          # bin 只首帧检测一次（篮子不动），之后每轮只检 target
        self._bin_label_seen: str | None = None

    def _grab_head(self) -> Image.Image:
        from galbot_sdk.g1 import SensorType

        comp = self.robot.get_rgb_data(SensorType.HEAD_LEFT_CAMERA)
        bgr = cv2.imdecode(np.frombuffer(comp["data"], np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("head camera decode failed")
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def _detect(self, img: Image.Image, label: str) -> XY:
        # 大图缩到显存安全上界再喂 VLM；坐标在缩图尺度解析，末尾还原回原图。
        if img.width > VLM_INPUT_MAX_WH[0] or img.height > VLM_INPUT_MAX_WH[1]:
            small = img.resize(VLM_INPUT_MAX_WH, Image.BILINEAR)
        else:
            small = img
        ans = self.worker.ground_single(small, label)["answer"]
        boxes = self.parse_boxes(ans, small.width, small.height)
        if not boxes:  # 诊断：框不到时看模型原始输出，判断是没检测到还是解析问题
            print(f"[vlm] '{label}' @{small.width}x{small.height} 未框到; raw={ans[:200]!r}")
        c = _largest_box_center(boxes)
        return _rescale_center(c, (small.width, small.height), (img.width, img.height))

    def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            target_label, bin_label = self.shared.get_labels()
            if target_label is not None:
                try:
                    img = self._grab_head()
                    t = self._detect(img, target_label)
                    if bin_label != self._bin_label_seen:  # 换了框标签 → 重新首帧检测
                        self._bin_xy = None
                        self._bin_label_seen = bin_label
                    if bin_label and self._bin_xy is None:  # bin 只检一次，之后复用
                        self._bin_xy = self._detect(img, bin_label)
                    self.shared.set_centers(t, self._bin_xy)
                    print(f"[vlm] target={t} bin={self._bin_xy}")  # 诊断：移动瓶子看 target 坐标变不变
                except Exception as e:  # 检测偶发失败不该弄崩控制环
                    print(f"[vlm] detect error: {e}")
            self._stop.wait(max(0.0, self.period - (time.monotonic() - t0)))

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            # 单次生成可达 2.6s，join 要够长让线程从推理里退出，否则 destroy 时
            # 线程还在用 CUDA 会 "terminate called" abort。
            self._thread.join(timeout=8.0)


def _selfcheck() -> None:
    # 缩图上 (480,360) 中心 → 原图 1280x960 应还原到 (640,480)
    c = _rescale_center((480, 360), (960, 720), (1280, 960))
    assert c == (640.0, 480.0), c
    assert _rescale_center(None, (960, 720), (1280, 960)) is None
    print("vlm_worker selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
