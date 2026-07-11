"""离线验证 2D 跟踪能否在'手移动瓶子'(含遮挡)下跟住目标。不需要机器人。
思路 = 部署要用的混合方案：VLM 首帧给点 → LK 光流每帧跟随 → 每 N 帧 VLM 语义纠偏。
输入是 run_g1 --record-frames 录下的连续帧。输出带点的可视化帧，肉眼看跟不跟得住。

用法：
    uv run python deploy/verify_tracker.py --frames deploy/frames --out deploy/track \\
        --label "green bottle" --vlm-every 8
    绿点=LK 跟踪, 红点=VLM 纠偏那帧的真值。看红绿点是否咬合、遮挡时绿点漂多远。
"""
import argparse
import glob
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/home/galbot/eagle/Embodied")
import cv2
import numpy as np
from PIL import Image
from locateanything_worker import LocateAnythingWorker

MODEL = "/home/galbot/LocateAnything-3B"
VLM_INPUT = (960, 720)  # 与 vlm_worker 一致：大图缩到这再喂 VLM，坐标还原回原图
LK = dict(winSize=(21, 21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def vlm_center(worker, path, label):
    """VLM 检测 → 最大框中心(原图坐标)；测不到返回 None。"""
    img = Image.open(path).convert("RGB")
    small = img.resize(VLM_INPUT, Image.BILINEAR) if img.width > VLM_INPUT[0] else img
    ans = worker.ground_single(small, label)["answer"]
    boxes = LocateAnythingWorker.parse_boxes(ans, small.width, small.height)
    if not boxes:
        return None
    b = max(boxes, key=lambda d: (d["x2"]-d["x1"])*(d["y2"]-d["y1"]))
    cx, cy = (b["x1"]+b["x2"])/2, (b["y1"]+b["y2"])/2
    return (cx*img.width/small.width, cy*img.height/small.height)


def feats_near(gray, center, r=40):
    """在 center 周围取角点作为 LK 跟踪集，比单点鲁棒(遮挡时靠多数存活点)。"""
    x, y = int(center[0]), int(center[1])
    mask = np.zeros(gray.shape, np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=40, qualityLevel=0.01, minDistance=3, mask=mask)
    return pts if pts is not None else np.array([[[float(x), float(y)]]], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", default="deploy/track")
    ap.add_argument("--label", default="green bottle")
    ap.add_argument("--vlm-every", type=int, default=8, help="每几帧用 VLM 纠偏一次(模拟低频 VLM)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    frames = sorted(glob.glob(f"{args.frames}/frame_*.png")) or sorted(glob.glob(f"{args.frames}/*.png"))
    assert frames, f"{args.frames} 下没有帧"
    worker = LocateAnythingWorker(MODEL, load_in_4bit=True)

    c0 = vlm_center(worker, frames[0], args.label)
    assert c0, "首帧 VLM 没检测到目标，换 --label 或换起始帧"
    prev = cv2.cvtColor(cv2.imread(frames[0]), cv2.COLOR_BGR2GRAY)
    pts = feats_near(prev, c0)
    center = c0
    for i, f in enumerate(frames):
        cur_bgr = cv2.imread(f)
        cur = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)
        vlm_pt = None
        if i > 0:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, **LK)
            good = nxt[st == 1] if st is not None else np.empty((0, 2))
            if len(good) >= 3:
                center = tuple(np.median(good, axis=0))  # 存活点中位数=抗遮挡的中心
                pts = good.reshape(-1, 1, 2)
        if args.vlm_every and i % args.vlm_every == 0 and i > 0:  # 低频 VLM 纠偏
            vlm_pt = vlm_center(worker, f, args.label)
            if vlm_pt:
                center = vlm_pt
                pts = feats_near(cur, vlm_pt)  # 用 VLM 真值重置跟踪集
        cv2.circle(cur_bgr, (int(center[0]), int(center[1])), 8, (0, 255, 0), 2)  # 绿=跟踪
        if vlm_pt:
            cv2.circle(cur_bgr, (int(vlm_pt[0]), int(vlm_pt[1])), 5, (0, 0, 255), -1)  # 红=VLM真值
        cv2.imwrite(f"{args.out}/track_{i:04d}.png", cur_bgr)
        drift = f"drift={np.hypot(center[0]-vlm_pt[0], center[1]-vlm_pt[1]):.0f}px" if vlm_pt else ""
        print(f"[{i:03d}] center=({center[0]:.0f},{center[1]:.0f}) live={len(pts)} {drift}")
        prev = cur
    print(f"\n完成，看 {args.out}/track_*.png：绿点跟不跟瓶子、遮挡时漂多远、红点(VLM)一到能否拉回")


if __name__ == "__main__":
    main()
