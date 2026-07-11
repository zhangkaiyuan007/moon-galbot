"""在训练机上跑：统计 galbot_g1_marked 数据集里【绿色目标标记点】的空间分布。
分布集中 → 坐实"目标位置多样性不足"，策略靠死记轨迹就能拟合、无视头相机标记。

标记是 markers.py 画的纯绿实心圆 TARGET_COLOR=(0,255,0)；这里从每帧头相机图找绿色质心。
注意：本脚本在部署机没数据、未验证，dataset 图像 key/格式可能要按你的 lerobot 版本微调。

用法(训练机)：
    python tools/analyze_target_dist.py --root /home1/jiajunjie/lerobot_data/galbot_g1_marked \\
        --repo-id galbot_g1_marked --sample 2000
"""
import argparse
import numpy as np


def green_centroid(img_hwc_uint8):
    """找纯绿标记像素质心(x, y)；没有绿点返回 None。target=绿(0,255,0)，bin=蓝需排除。"""
    r, g, b = img_hwc_uint8[..., 0], img_hwc_uint8[..., 1], img_hwc_uint8[..., 2]
    mask = (g > 180) & (r < 90) & (b < 90)
    if mask.sum() < 5:
        return None
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def to_uint8_hwc(img):
    """lerobot 图可能是 torch CHW float[0,1] 或 numpy；统一成 HWC uint8。"""
    a = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if a.ndim == 3 and a.shape[0] in (1, 3):   # CHW → HWC
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:                      # float[0,1] → uint8
        a = (a * 255).clip(0, 255).astype(np.uint8)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo-id", default="galbot_g1_marked")
    ap.add_argument("--sample", type=int, default=2000, help="抽样帧数，避免全量太慢")
    ap.add_argument("--key", default="observation.images.head")
    ap.add_argument("--out", default="tools/target_dist.png")
    args = ap.parse_args()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(args.repo_id, root=args.root)
    n = len(ds)
    idxs = np.linspace(0, n - 1, min(args.sample, n)).astype(int)

    pts = []
    for i in idxs:
        c = green_centroid(to_uint8_hwc(ds[int(i)][args.key]))
        if c:
            pts.append(c)
    pts = np.array(pts)
    assert len(pts), "没找到任何绿色标记，检查 --key 或标记颜色阈值"

    x, y = pts[:, 0], pts[:, 1]
    print(f"帧数={n} 抽样={len(idxs)} 检出标记={len(pts)}")
    print(f"x: min={x.min():.0f} max={x.max():.0f} 范围={x.max()-x.min():.0f} std={x.std():.0f}")
    print(f"y: min={y.min():.0f} max={y.max():.0f} 范围={y.max()-y.min():.0f} std={y.std():.0f}")
    print("判读：范围/std 越小=目标越集中=越坐实'多样性不足'。")
    print("参考：头相机 1280x960，若 x/y 范围都不到画面 1/4(~300/240px)，就是明显集中。")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.hexbin(x, y, gridsize=40, cmap="hot")
        plt.gca().invert_yaxis()
        plt.xlim(0, 1280); plt.ylim(960, 0)
        plt.title(f"target marker distribution (n={len(pts)})")
        plt.colorbar(label="count")
        plt.savefig(args.out, dpi=120)
        print(f"热图已存 {args.out}")
    except Exception as e:
        print(f"(跳过画图: {e})")


if __name__ == "__main__":
    main()
