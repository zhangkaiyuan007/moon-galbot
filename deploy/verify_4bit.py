"""离线验证修复后的真机路径：4-bit 加载 + VLMWorker._detect(1280x960 原图)。
_detect 会自动缩到 960x720 喂 VLM，再把中心还原回 1280x960。确认不 OOM + 坐标合理。
不需要机器人（_detect 不碰 robot）。验证完可删。"""
import os, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(__file__))  # import vlm_worker
import torch
from PIL import Image
from vlm_worker import VLMWorker

MODEL = "/home/galbot/LocateAnything-3B"
REPO = "/home/galbot/eagle/Embodied"
IMG = "/home/galbot/eagle/Embodied/assets/images/qualitative_examples.jpg"

t0 = time.monotonic()
vlm = VLMWorker(robot=None, shared=None, model_path=MODEL, la_repo=REPO, load_in_4bit=True)
torch.cuda.synchronize()
print(f"[load] {time.monotonic()-t0:.1f}s  weights={torch.cuda.memory_allocated()/2**30:.2f}GiB")

img = Image.open(IMG).convert("RGB").resize((1280, 960), Image.BICUBIC)  # 模拟真机头相机
torch.cuda.reset_peak_memory_stats()
t1 = time.monotonic()
c = vlm._detect(img, "person")  # 内部缩 960x720 检测，坐标还原回 1280x960
torch.cuda.synchronize()
print(f"[detect] {time.monotonic()-t1:.1f}s  peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB")
print(f"[center] {c}  (应落在 0..1280 × 0..960 内)")
assert c is None or (0 <= c[0] <= 1280 and 0 <= c[1] <= 960), "坐标越界！"
print("OK: 真机 1280x960 输入不再 OOM，坐标已还原到原图坐标系")
