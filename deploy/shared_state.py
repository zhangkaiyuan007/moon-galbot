"""双速架构的共享状态：低频 VLM 线程写，高频控制环读最新值。

VLM 刷新环(vlm_worker)持续把最新的目标/框中心写进来；控制环(policy_runtime)每步
读最新中心去画标记。语音(voice)设置要检测的物体词。全部加锁，简单可靠。

坐标是**头相机原始分辨率(1280x960)**下的 (x, y)，与训练标记一致；None = 尚无有效检测。
"""

from __future__ import annotations

import threading

XY = tuple[float, float] | None


class SharedState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._target_label: str | None = None
        self._bin_label: str | None = None
        self._target_xy: XY = None
        self._bin_xy: XY = None
        self._gen = 0      # 每次 VLM 写入自增，控制环判断是否刷新过
        self._cmd_id = 0   # 每次语音设新目标自增，主循环判断是否来了新指令

    def set_labels(self, target_label: str, bin_label: str) -> None:
        """语音选定目标后调用；同时清空旧中心，避免用上一目标的坐标。"""
        with self._lock:
            self._target_label = target_label
            self._bin_label = bin_label
            self._target_xy = None
            self._bin_xy = None
            self._cmd_id += 1

    def command_id(self) -> int:
        """主循环轮询：值变化即有新语音目标（同物体重复也算）。"""
        with self._lock:
            return self._cmd_id

    def get_labels(self) -> tuple[str | None, str | None]:
        with self._lock:
            return self._target_label, self._bin_label

    def set_centers(self, target_xy: XY, bin_xy: XY) -> None:
        """VLM 线程每次检测后写入最新中心。"""
        with self._lock:
            self._target_xy = target_xy
            self._bin_xy = bin_xy
            self._gen += 1

    def get_centers(self) -> tuple[XY, XY, int]:
        """控制环读最新中心；返回 (target_xy, bin_xy, generation)。"""
        with self._lock:
            return self._target_xy, self._bin_xy, self._gen


def _selfcheck() -> None:
    s = SharedState()
    assert s.get_labels() == (None, None)
    assert s.get_centers() == (None, None, 0)
    assert s.command_id() == 0
    s.set_labels("cola bottle", "basket")
    assert s.command_id() == 1
    s.set_labels("cola bottle", "basket")  # 同物体重复也应自增
    assert s.command_id() == 2
    assert s.get_labels() == ("cola bottle", "basket")
    assert s.get_centers()[:2] == (None, None), "换标签应清空旧中心"
    s.set_centers((500.0, 480.0), (900.0, 500.0))
    tx, bx, gen = s.get_centers()
    assert tx == (500.0, 480.0) and bx == (900.0, 500.0) and gen == 1
    s.set_labels("apple", "basket")
    assert s.get_centers()[:2] == (None, None)
    print("shared_state selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
