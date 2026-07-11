"""纯点图训练 wrapper：在训练加载帧时，把已标注头相机图的背景抹黑、只留目标/框点。
不重跑 VLM、不物理改 dataset——monkey-patch LeRobotDataset 取帧，在线检测已画的绿/蓝点
质心并黑底重画。逼策略只能从点读位置(对抗 causal confusion)。部署端 markers.py 已同款黑底。

用法(训练机，参数同 train_act.sh)：
    python training/train_act_pointonly.py \
      --dataset.repo_id=galbot_g1_marked \
      --dataset.root=$HF_LEROBOT_HOME/galbot_g1_marked \
      --dataset.video_backend=pyav --policy.type=act \
      --policy.n_action_steps=15 --policy.chunk_size=50 \
      --output_dir=outputs/act_pointonly --batch_size=8 --steps=100000 \
      --save_freq=10000 --log_freq=200 --wandb.enable=false

注意：仅头相机(observation.images.head)变黑；阈值/半径按你 dataset 图尺寸可能要调，
先拿 1 个 batch 存出来肉眼确认点位对、再全量训。
"""
import runpy
import sys

import cv2
import numpy as np
import torch

HEAD_KEY = "observation.images.head"


def _centroid(hsv_ok: np.ndarray):
    if hsv_ok.sum() < 5:
        return None
    ys, xs = np.nonzero(hsv_ok)
    return int(round(xs.mean())), int(round(ys.mean()))


def _point_only(img_chw: torch.Tensor) -> torch.Tensor:
    """(C,H,W) float[0,1] RGB 已标注图 → 黑底+重画的目标绿点/框蓝点。"""
    a = (img_chw.numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    green = (g > 150) & (r < 110) & (b < 110)   # 阈值放宽以吃掉 h264 压缩伪影
    blue = (b > 150) & (r < 110) & (g < 110)
    h, w = a.shape[:2]
    rad = max(4, round(w / 91))                  # 与原图 1280→r14 同比例
    out = np.zeros_like(a)
    for mask, color in ((green, (0, 255, 0)), (blue, (0, 0, 255))):
        c = _centroid(mask)
        if c is not None:
            cv2.circle(out, c, rad, color, thickness=-1)
    return torch.from_numpy(out.transpose(2, 0, 1).astype(np.float32) / 255.0)


def _patch_dataset():
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    orig = LeRobotDataset.__getitem__

    def patched(self, idx):
        item = orig(self, idx)
        if HEAD_KEY in item and torch.is_tensor(item[HEAD_KEY]) and item[HEAD_KEY].ndim == 3:
            item[HEAD_KEY] = _point_only(item[HEAD_KEY])
        return item

    LeRobotDataset.__getitem__ = patched
    print("[pointonly] LeRobotDataset.__getitem__ patched：头相机背景已在线抹黑")


def _demo():
    """自检：黑底重画后，只应有绿/蓝两点，背景全黑。"""
    img = np.zeros((96, 128, 3), np.uint8)
    cv2.circle(img, (40, 30), 4, (0, 255, 0), -1)   # 假目标绿点
    cv2.circle(img, (90, 70), 4, (0, 0, 255), -1)   # 假框蓝点
    t = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255.0)
    out = (_point_only(t).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    assert out[:, :, 1].max() > 200 and out[:, :, 2].max() > 200, "绿/蓝点应保留"
    # 抠掉两点邻域后应全黑(背景无残留)
    m = np.ones(out.shape[:2], bool)
    cv2.circle(m.view(np.uint8), (40, 30), 8, 0, -1); cv2.circle(m.view(np.uint8), (90, 70), 8, 0, -1)
    assert out[m].sum() == 0, "背景应全黑"
    print("pointonly selfcheck OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _demo()
    else:
        _patch_dataset()
        sys.argv[0] = "lerobot.scripts.train"
        runpy.run_module("lerobot.scripts.train", run_name="__main__")
