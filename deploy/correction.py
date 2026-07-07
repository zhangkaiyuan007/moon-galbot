"""在线修正三件套（策略外的轻量层）：

1. 空间类("往上/往左一点") → 给目标标记加像素偏移，策略自然跟过去。
2. 力度类("抓紧一点")     → 调夹爪 effort。
3. 记忆                    → 物体名→修正量，存 JSON；下次说同一物体自动预加载。

语音修正文本用关键词解析成 (字段, 方向)。步长是校准量——操作者看效果微调。
ponytail: 图像"上"≈世界"上"只在头相机近似成立，magnitude 是旋钮不是常数。
"""

from __future__ import annotations

import json
from pathlib import Path

XY = tuple[float, float] | None

STEP_PX = 40        # 每次"一点"的像素偏移（头相机原分辨率 1280x960 下）
GRIP_STEP = 15      # 每次"紧/松一点"的 effort 增量

# 关键词 → (字段, 方向)。图像坐标：上=y 减小，左=x 减小。
CORRECTION_KEYWORDS: dict[str, tuple[str, int]] = {
    "往上": ("dy", -1), "上面": ("dy", -1), "高": ("dy", -1),
    "往下": ("dy", +1), "下面": ("dy", +1), "低": ("dy", +1),
    "往左": ("dx", -1), "左边": ("dx", -1),
    "往右": ("dx", +1), "右边": ("dx", +1),
    "抓紧": ("grip", +1), "紧": ("grip", +1), "用力": ("grip", +1),
    "松": ("grip", -1), "轻": ("grip", -1),
}


def parse_correction(text: str) -> list[tuple[str, int]]:
    """从语音文本里提取修正意图，返回 [(字段, 方向), ...]。无匹配则空列表。

    多关键词按最长优先去重（"往上" 命中就不再让 "上面" 之类重复触发同字段）。
    """
    hits: dict[str, int] = {}
    for kw in sorted(CORRECTION_KEYWORDS, key=len, reverse=True):
        if kw in text:
            field, sign = CORRECTION_KEYWORDS[kw]
            hits.setdefault(field, sign)  # 同字段只取第一个命中
    return list(hits.items())


class CorrectionMemory:
    """物体名 → {dx, dy, grip} 的累积修正，可存取 JSON。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._mem: dict[str, dict[str, float]] = {}
        if self.path and self.path.exists():
            self._mem = json.loads(self.path.read_text())

    def _entry(self, obj: str) -> dict[str, float]:
        return self._mem.setdefault(obj, {"dx": 0.0, "dy": 0.0, "grip": 0.0})

    def apply_hits(self, obj: str, hits: list[tuple[str, int]]) -> list[tuple[str, int]]:
        """把 (字段, 方向) 修正项累加到该物体记忆并存盘；返回命中项。

        KWS 标签路径直接给 hits；自由文本路径经 apply_voice → parse_correction。
        """
        e = self._entry(obj)
        for field, sign in hits:
            step = GRIP_STEP if field == "grip" else STEP_PX
            e[field] += sign * step
        if hits:
            self.save()
        return hits

    def apply_voice(self, obj: str, text: str) -> list[tuple[str, int]]:
        """解析自由文本并累加（可选的 ASR 兜底路径）。"""
        return self.apply_hits(obj, parse_correction(text))

    def apply_offset(self, obj: str, xy: XY) -> XY:
        """把该物体记忆里的 (dx,dy) 加到目标标记中心；xy 为 None 则不变。"""
        if xy is None:
            return None
        e = self._mem.get(obj)
        if not e:
            return xy
        return (xy[0] + e["dx"], xy[1] + e["dy"])

    def grip_effort(self, obj: str, base_effort: float) -> float:
        """按记忆调整夹爪 effort，夹到 [1, 100]。"""
        e = self._mem.get(obj)
        adj = base_effort + (e["grip"] if e else 0.0)
        return float(max(1.0, min(100.0, adj)))

    def save(self) -> None:
        if self.path:
            self.path.write_text(json.dumps(self._mem, ensure_ascii=False, indent=2))


def _selfcheck() -> None:
    import tempfile

    assert parse_correction("往上抓一点") == [("dy", -1)]
    assert dict(parse_correction("抓紧一点，往右挪")) == {"grip": 1, "dx": 1}
    assert parse_correction("放这里") == []

    tmp = Path(tempfile.mkdtemp()) / "mem.json"
    m = CorrectionMemory(tmp)
    m.apply_voice("apple", "往上抓一点")     # dy -= 40
    m.apply_voice("apple", "再抓紧一点")     # grip += 15
    assert m.apply_offset("apple", (100.0, 200.0)) == (100.0, 160.0)
    assert m.grip_effort("apple", 50.0) == 65.0
    assert m.apply_offset("pear", (10.0, 10.0)) == (10.0, 10.0)  # 无记忆不变

    m2 = CorrectionMemory(tmp)  # 重新加载，验证持久化
    assert m2.apply_offset("apple", (0.0, 0.0)) == (0.0, -40.0)
    assert m2.grip_effort("apple", 50.0) == 65.0
    print("correction selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
