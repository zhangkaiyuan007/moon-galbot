"""检查阶段 A 的检测效果：把 detections 的中心画回真实头相机帧，抽几帧存成 PNG，
并打印漏检率。绿点=目标物、蓝点=框（与阶段 B 写进数据集的完全一致）。

用法：
    uv run python tools/preview_markers.py \\
        --data-dir /home1/jiajunjie/1105_1696 \\
        --detections-dir detections --out-dir preview --n 12
默认预览按名字排序的第一条 episode；--episode 指定某条。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from galbot_mcap import decimate_indices, read_episode  # noqa: E402
from markers import draw_markers, zoh_fill  # noqa: E402

SOURCE_FPS = 30.0
XY = tuple[float, float] | None


def _centers(df: pd.DataFrame, frame_ts: np.ndarray) -> tuple[list[XY], list[XY], int, int]:
    df = df.set_index("head_ts")
    tg, bn = [], []
    for ts in frame_ts:
        if ts in df.index:
            r = df.loc[ts]
            tg.append(None if np.isnan(r["target_x"]) else (float(r["target_x"]), float(r["target_y"])))
            bn.append(None if np.isnan(r["bin_x"]) else (float(r["bin_x"]), float(r["bin_y"])))
        else:
            tg.append(None); bn.append(None)
    miss_t = sum(c is None for c in tg)
    miss_b = sum(c is None for c in bn)
    return zoh_fill(tg), zoh_fill(bn), miss_t, miss_b


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--detections-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("preview"))
    p.add_argument("--episode", default=None, help="episode 名（不含后缀）；默认第一条")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--n", type=int, default=12, help="抽几帧预览")
    args = p.parse_args()

    eps = sorted(args.data_dir.glob("*.SYNC.mcap"))
    if args.episode:
        eps = [e for e in eps if e.name.replace(".SYNC.mcap", "") == args.episode]
        if not eps:
            raise FileNotFoundError(f"no episode named {args.episode}")
    ep = eps[0]
    name = ep.name.replace(".SYNC.mcap", "")
    det = args.detections_dir / f"{name}.parquet"
    if not det.exists():
        raise FileNotFoundError(f"missing detections: {det}")

    streams = read_episode(ep)
    sel = decimate_indices(len(streams.head_ts), SOURCE_FPS, args.fps)
    frame_ts = streams.head_ts[sel]
    tg, bn, miss_t, miss_b = _centers(pd.read_parquet(det), frame_ts)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pick = np.linspace(0, len(sel) - 1, min(args.n, len(sel)), dtype=int)
    for k in pick:
        rgb = cv2.cvtColor(
            cv2.imdecode(np.frombuffer(streams.head_jpeg[sel[k]], np.uint8), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        )
        marked = draw_markers(rgb, tg[k], bn[k])
        cv2.imwrite(str(args.out_dir / f"{name}_f{k:04d}.png"),
                    cv2.cvtColor(marked, cv2.COLOR_RGB2BGR))

    n = len(sel)
    print(f"{name}: {n} frames")
    print(f"  目标漏检 {miss_t}/{n} ({miss_t/n:.0%})，框漏检 {miss_b}/{n} ({miss_b/n:.0%})  [ZOH 前]")
    print(f"  已写 {len(pick)} 张预览 -> {args.out_dir}/  （绿=目标, 蓝=框）")
    print("  肉眼确认：绿点是否稳定压在目标物上、蓝点在框上；抓取遮挡段应 ZOH 不乱跳。")


if __name__ == "__main__":
    main()
