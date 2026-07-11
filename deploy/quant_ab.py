"""离线量化/分辨率精度对照：同图不同配置，比框中心偏差。回答"int4/降分辨率影响多大"。
- bf16 vs int4 @ 640x480：纯量化损失（bf16=训练精度基准）
- int4 @ 640 vs 960：分辨率漂移趋势
都在 8G 卡上能装下。不需要机器人。测完可删。"""
import os, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/home/galbot/eagle/Embodied")
import torch
from PIL import Image
from locateanything_worker import LocateAnythingWorker

MODEL = "/home/galbot/LocateAnything-3B"
# 换成图里真实存在的物体标签；这几张论文图用 "person" 能测到
IMGS = [
    ("/home/galbot/eagle/Embodied/assets/images/qualitative_examples.jpg", "person"),
]
PB = LocateAnythingWorker.parse_boxes


def center_norm(worker, path, label, wh):
    """返回归一化中心 (x/W, y/H)，跨分辨率可比；测不到返回 None。"""
    img = Image.open(path).convert("RGB").resize(wh, Image.BICUBIC)
    ans = worker.ground_single(img, label)["answer"]
    boxes = PB(ans, img.width, img.height)
    if not boxes:
        return None
    b = max(boxes, key=lambda d: (d["x2"]-d["x1"])*(d["y2"]-d["y1"]))
    return ((b["x1"]+b["x2"])/2/img.width, (b["y1"]+b["y2"])/2/img.height)


def load(**kw):
    t = time.monotonic()
    w = LocateAnythingWorker(MODEL, **kw)
    print(f"  loaded in {time.monotonic()-t:.1f}s")
    return w


def dist_px(a, b, W=1280, H=960):
    """两个归一化中心在 1280x960 下相差多少像素。"""
    if a is None or b is None:
        return None
    return ((a[0]-b[0])*W, (a[1]-b[1])*H)


print("=== int8 (接近全精度基准，8bit 损失通常<1%) ===")
w = load(load_in_8bit=True)
int8_640 = {p: center_norm(w, p, l, (640, 480)) for p, l in IMGS}
int8_960 = {p: center_norm(w, p, l, (960, 720)) for p, l in IMGS}
del w; torch.cuda.empty_cache()

print("=== int4 ===")
w = load(load_in_4bit=True)
int4_640 = {p: center_norm(w, p, l, (640, 480)) for p, l in IMGS}
int4_960 = {p: center_norm(w, p, l, (960, 720)) for p, l in IMGS}

print("\n=== 结果（中心以 1280x960 像素计的偏差）===")
for p, l in IMGS:
    name = os.path.basename(p)
    print(f"[{name}] label={l!r}")
    print(f"  int8@640 = {int8_640[p]}   int8@960 = {int8_960[p]}")
    print(f"  int4@640 = {int4_640[p]}   int4@960 = {int4_960[p]}")
    print(f"  量化损失  int8→int4 @960 : {dist_px(int8_960[p], int4_960[p])} px  ← 部署分辨率下 4bit 影响")
    print(f"  量化损失  int8→int4 @640 : {dist_px(int8_640[p], int4_640[p])} px")
    print(f"  分辨率漂移 int8 640→960  : {dist_px(int8_640[p], int8_960[p])} px  ← 纯分辨率影响")
