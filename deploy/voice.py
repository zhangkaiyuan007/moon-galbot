"""语音接口

服务在机器人 HPU 上跑 VAD + KWS，把命中的物料/指令标签通过 ZMQ 6000 发出
（消息 `{"type":"kwd_cmd","data":"<标签>"}`）。我们只当消费者：订阅、按标签路由到
SharedState(物体)/CorrectionMemory(修正)。**不需要 Whisper/ASR**——KWS 直接给标签。

标签来自服务的 keywords.txt（`拼音 @标签`）。这里把标签映射到含义：
- 物体标签 → locate-anything 的检测短语（喂给 VLM）。
- 修正标签 → (字段, 方向)。
在服务的 keywords.txt 里加对应词条即可扩充。
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

# KWS 物料标签 → locate-anything 检测短语。按比赛物料在此 + 服务 keywords.txt 同步扩充。
OBJECT_LABELS: dict[str, str] = {
    "cola": "cola bottle",
    "green_bottle": "green bottle",
}
# KWS 标签 → 中文播报名（仅 TTS 用；检测短语仍是英文喂 VLM）
DISPLAY_NAMES: dict[str, str] = {"cola": "可乐", "green_bottle": "雪碧"}
# KWS 修正标签 → (字段, 方向)。图像坐标：上=y 减小，左=x 减小。
CORRECTION_LABELS: dict[str, tuple[str, int]] = {
    "corr_up": ("dy", -1), "corr_down": ("dy", +1),
    "corr_left": ("dx", -1), "corr_right": ("dx", +1),
    "corr_tight": ("grip", +1), "corr_loose": ("grip", -1),
}
BIN_LABEL = "basket"


def route_label(label: str, object_labels=OBJECT_LABELS, correction_labels=CORRECTION_LABELS):
    """把 KWS 标签路由成动作。
    返回 ("target", 检测短语) | ("correction", (字段,方向)) | ("unknown", None)"""
    if label in object_labels:
        return ("target", object_labels[label])
    if label in correction_labels:
        return ("correction", correction_labels[label])
    return ("unknown", None)


class VoiceMicClient:
    def __init__(self, shared, correction, sub_addr: str,
                 object_labels: dict[str, str] = OBJECT_LABELS,
                 correction_labels: dict[str, tuple[str, int]] = CORRECTION_LABELS,
                 bin_label: str = BIN_LABEL, tts: Callable[[str], None] | None = None):
        self.shared = shared
        self.correction = correction
        self.sub_addr = sub_addr           # 例：tcp://<机器人IP>:6000
        self.object_labels = object_labels
        self.correction_labels = correction_labels
        self.bin_label = bin_label
        self.tts = tts
        self.current_obj: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def say(self, text: str) -> None:
        print(f"[robot] {text}")
        if self.tts is not None:
            self.tts(text)

    def _handle_wakeup(self, data: dict | None = None) -> None:
        self.say("我在，请说要拿什么")

    def _handle_label(self, label: str) -> None:
        kind, payload = route_label(label, self.object_labels, self.correction_labels)
        if kind == "target":
            self.current_obj = label
            self.shared.set_labels(payload, self.bin_label)
            self.say(f"好的，去拿{DISPLAY_NAMES.get(label, label)}")
        elif kind == "correction":
            if self.current_obj is None:
                self.say("还没有目标，请先说要拿什么")
                return
            self.correction.apply_hits(self.current_obj, [payload])
            self.say("好的，记住了")

    def _loop(self) -> None:
        import zmq

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.connect(self.sub_addr)
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.setsockopt(zmq.RCVTIMEO, 500)  # ms，便于响应 stop
        while not self._stop.is_set():
            try:
                raw = sock.recv_string()
            except zmq.Again:
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            msg_type = msg.get("type")
            if msg_type == "waken_up":
                print(f"[kws] waken_up")
                data = msg.get("data")
                self._handle_wakeup(data if isinstance(data, dict) else None)
            elif msg.get("type") == "kwd_cmd":
                data = msg.get("data")
                if isinstance(data, str):
                    print(f"[kws] {data}")
                    self._handle_label(data)
        sock.close(0)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)


def _selfcheck() -> None:
    assert route_label("cola") == ("target", "cola bottle")
    assert route_label("corr_up") == ("correction", ("dy", -1))
    assert route_label("nonsense") == ("unknown", None)

    # 路由 + 修正记忆联动（不连 ZMQ）
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from correction import CorrectionMemory
    from shared_state import SharedState
    import tempfile

    shared, mem = SharedState(), CorrectionMemory(Path(tempfile.mkdtemp()) / "m.json")
    c = VoiceMicClient(shared, mem, "tcp://127.0.0.1:6000")
    c._handle_label("cola")
    assert shared.get_labels() == ("cola bottle", "basket") and c.current_obj == "cola"
    c._handle_label("corr_up")
    assert mem.apply_offset("cola", (0.0, 0.0)) == (0.0, -40.0)  # 往上 = y 减 40
    print("voice selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
