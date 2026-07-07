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


def _largest_box_center(boxes: list[dict]) -> XY:
    """最大框中心；与 tools/detect_markers.py 一致，保证训练/部署同语义。"""
    if not boxes:
        return None
    b = max(boxes, key=lambda d: (d["x2"] - d["x1"]) * (d["y2"] - d["y1"]))
    return ((b["x1"] + b["x2"]) / 2.0, (b["y1"] + b["y2"]) / 2.0)


class VLMWorker:
    def __init__(self, robot, shared, model_path: str, la_repo: str | Path,
                 refresh_hz: float = VLM_REFRESH_HZ):
        self.robot = robot
        self.shared = shared
        self.period = 1.0 / refresh_hz
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        sys.path.insert(0, str(la_repo))
        from locateanything_worker import LocateAnythingWorker
        self.worker = LocateAnythingWorker(model_path)
        self.parse_boxes = LocateAnythingWorker.parse_boxes

    def _grab_head(self) -> Image.Image:
        from galbot_sdk.g1 import SensorType

        comp = self.robot.get_rgb_data(SensorType.HEAD_LEFT_CAMERA)
        bgr = cv2.imdecode(np.frombuffer(comp["data"], np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("head camera decode failed")
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def _detect(self, img: Image.Image, label: str) -> XY:
        ans = self.worker.ground_single(img, label)["answer"]
        return _largest_box_center(self.parse_boxes(ans, img.width, img.height))

    def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            target_label, bin_label = self.shared.get_labels()
            if target_label is not None:
                try:
                    img = self._grab_head()
                    t = self._detect(img, target_label)
                    b = self._detect(img, bin_label) if bin_label else None
                    self.shared.set_centers(t, b)
                except Exception as e:  # 检测偶发失败不该弄崩控制环
                    print(f"[vlm] detect error: {e}")
            self._stop.wait(max(0.0, self.period - (time.monotonic() - t0)))

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
