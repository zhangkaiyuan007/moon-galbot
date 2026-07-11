<div align="center">

# ReLocateACT

**语音驱动的桌面抓取** —— 用「VLM 检测 + 视觉标记」把感知和控制解耦，双速运行、在线修正，
在单卡 8 GB 消费级 GPU 上跑通整条闭环。

`Galbot G1` · `LeRobot-ACT` · `NVIDIA LocateAnything-3B`

</div>

---

## 这是什么

在 Galbot **G1** 上「说一句话 → 机器人把目标物抓进框」。初衷是**不训练一个大的
VLA**，而是把任务拆成两半：

- **感知**交给开放词表检测器 **LocateAnything-3B**：定位目标物和框；
- **控制**一个轻量 **ACT** 策略：把中心点画成一个**彩色标记**叠在头相机画面上喂给它，它输出动作。

两者之间的接口就是「画在图上的那个点」。好处是：换检测器、换要抓的东西都不用重训控制策略；
检测器负责开放词表泛化，策略保持小而快。

流程：**说要拿什么 → VLM 定位 → 画标记 → ACT 抓取 → 放进框 → 语音反馈**，中途还能
「往上抓一点 / 抓紧点」实时修正，也能在你**用手挪动物体**时跟着重新定位。

---

## 亮点

| | 做法 | 为什么这么做 |
|---|------|------|
| **解耦** | 检测(语义) + ACT(控制),接口是画在图上的标记点 | 不用训大 VLA;检测器管开放词表,策略保持廉价 |
| **训练/部署同一份画标记代码** ([`tools/markers.py`](tools/markers.py)) | - | 一份代码两边共用,从根上杜绝不一致 |
| **双速运行** | 低频 VLM 线程写中心点进 `SharedState`,高频控制环每步读最新值 | 检测器慢(3B)、控制要快,拆开互不拖累 |
| **光流跟踪** | LK 光流在控制环频率跟目标,VLM 每次刷新就重置一下 | 填补 VLM 两次刷新之间的空档,让标记**跟着移动的物体走** → 抗挪动抓取 |
| **8 GB 单卡部署** | 检测器 4-bit 量化 + 输入降分辨率 | 一张笔记本级显卡就能跑 |
| **语音闭环 + 在线修正** | 抓取中口头修正标记偏移/夹爪力度并存盘 | 人可以随时插话纠正,修正按物体记忆持久化 |

---

## 架构

```
                     ┌─────────────────────────────────────────────┐
   语音  ──────────▶ │  机器人本地麦克风: VAD + KWS (ZMQ)  →  目标标签 │
                     └───────────────┬─────────────────────────────┘
                                     │ 设目标
                                     ▼
 头相机 ───┐    ┌───────────────────────────────┐  中心(x,y)   ┌──────────────┐
           ├───▶│ VLMWorker   LocateAnything-3B  │─────────────▶│ SharedState  │
           │    │  低频 ~0.4 Hz, 4-bit 量化       │   线程安全   │ 目标 / 框     │
           │    └───────────────────────────────┘              └──────┬───────┘
           │                                                          │ 最新中心
           │    ┌─────────────────────────────────────────────────────▼──────┐
           └───▶│ PolicyRuntime   高频控制环                                   │
 腕相机 ──────▶│  LK 光流跟踪 → 画标记 → ACT 推理 →                            │
 关节  ───────▶│  安全护栏 → SDK 执行 / 夹爪滞回 →                             │
                │  在线修正 (CorrectionMemory)                                 │
                └─────────────────────────────────────────────────────────────┘
```

喂给 ACT 的观测:`head`、`wrist`、`state`(7 臂关节 + 1 夹爪)。
输出:8 维动作 chunk(7 关节 + 夹爪),按右臂轨迹执行,带单步和重规划的安全限幅
(见 [`deploy/config.py`](deploy/config.py))。

---

## 数据管线

```
*.SYNC.mcap ──(A)──▶ detections/*.parquet ──(B)──▶ LeRobot 数据集(标记已画入) ──▶ ACT
             detect_markers          convert_mcap_to_lerobot        train_act.sh
```

- **(A) 检测** — [`tools/detect_markers.py`](tools/detect_markers.py):LocateAnything-3B 逐帧对
  target/bin 检测,取最大框中心 → parquet。
- **(B) 转换** — [`tools/convert_mcap_to_lerobot.py`](tools/convert_mcap_to_lerobot.py):时间对齐的
  mcap + 检测结果 → LeRobot 数据集(h264),用共用的 [`tools/markers.py`](tools/markers.py) 把标记画进
  `head` 帧(绿=目标、蓝=框;漏检用零阶保持,复现部署时低频刷新的行为)。
- **预览** — [`tools/preview_markers.py`](tools/preview_markers.py):打印漏检率、把中心叠回真实帧肉眼检查。

---

## 安装

**前提**:装了 `uv`;有 NVIDIA 驱动(`nvidia-smi` 能输出);LocateAnything 仓库
`eagle/Embodied` 放在本仓库的**上一级**

```bash
cd moon-galbot
uv sync
```

---

## 快速开始

下面路径以训练服务器 `/home1/jiajunjie/...` 为例,按你的机器改;所有命令在仓库根目录跑。

### 1. 采数据（遥操作）
录成 Galbot `*.SYNC.mcap`(一集一条)。数据配方(关键,见 [`PLAN.md`](PLAN.md)):每集随机摆物体和框
位置;约 1/4~1/3 的集在接近途中**人为挪动物体**再重新抓;桌上多物体、只抓目标;目标轮换。
> 管线只用 `*.SYNC.mcap`;`*.FIN.mcap` 是未对齐的原始数据,用不上。

### 2. 阶段 A · 检测
```bash
uv run python tools/detect_markers.py \
  --data-dir /home1/jiajunjie/1105_1696 --out-dir detections \
  --target-label "cola bottle" --bin-label "basket" \
  --model /home1/jiajunjie/LocateAnything-3B --max-episodes 1
```
先跑 1 条,没问题再去掉 `--max-episodes` 全量。

### 3. 阶段 A · 检查
```bash
uv run python tools/preview_markers.py \
  --data-dir /home1/jiajunjie/1105_1696 --detections-dir detections --out-dir preview --n 12
```
确认**绿点压在目标、蓝点在框**;框漏检率应≈0%。

### 4. 阶段 B · 转换
```bash
export HF_LEROBOT_HOME=/home1/jiajunjie/lerobot_data
uv run python tools/convert_mcap_to_lerobot.py \
  --data-dir /home1/jiajunjie/1105_1696 --detections-dir detections \
  --output-root $HF_LEROBOT_HOME --repo-id galbot_g1_marked --fps 15 --overwrite
```

### 5. 训练 ACT
```bash
export HF_LEROBOT_HOME=/home1/jiajunjie/lerobot_data
bash training/train_act.sh          # lerobot 原生 ACT;显存不够改脚本里 --batch_size
```
checkpoint 在 `outputs/act_galbot_g1_marked/checkpoints/`。

### 6. 真机部署
先 dry-run(读观测、定位、画标记、推理、
打印动作,机器人不动):
```bash
uv run python deploy/run_g1.py \
  --checkpoint outputs/act_galbot_g1_marked/checkpoints/<step>/pretrained_model \
  --model /home/galbot/LocateAnything-3B \
  --target-label "cola bottle" --bin-label "basket" --save-vis deploy/vis
```
真动:
```bash
uv run python deploy/run_g1.py \
  --checkpoint .../pretrained_model --model /home/galbot/LocateAnything-3B \
  --target-label "cola bottle" --execute --go-home
```

**加语音**(不给 `--target-label`,改用 `--mic-addr`):
```bash
uv run python deploy/run_g1.py --checkpoint .../pretrained_model \
  --model /home/galbot/LocateAnything-3B \
  --mic-addr tcp://<机器人IP>:6000 --execute --go-home
```
语音走机器人本地的 mic 服务(VAD+KWS over ZMQ);TTS 是 edge-tts + ffmpeg,带磁盘缓存
(`deploy/tts.py --prebake` 可预烘焙固定话术、离线可用)。

常用开关:`--save-vis DIR`(存带标记的头相机帧)、`--vlm-full`(检测器用 bf16,需要大显存)、
`--slow`、`--max-chunks`。

---

## 8 GB 显存怎么塞下

3B 检测器和 ACT 策略要共存在一张消费卡上。三个措施,各治一种**不同的**显存报错:

| 现象 | 真正原因 | 解法 |
|---|---|---|
| 加载时 OOM(~6.6 GiB 权重) | 3B 检测器 bf16 ≈ 6 GB | **4-bit NF4 量化**(`--vlm-full` 可关) → ~2 GB |
| 注意力里 OOM(单个算子要 ~6.5 GiB) | vision attention 是 O(patch²),原生输入太大 | **降 VLM 输入分辨率** 到安全预算,坐标再还原回原图 |

之后峰值约 3–4 GB。复现脚本:[`deploy/verify_4bit.py`](deploy/verify_4bit.py)、
[`deploy/verify_tracker.py`](deploy/verify_tracker.py)。

---

## 目录

```
tools/                       数据处理:mcap → 带标记的 LeRobot 数据集
  galbot_mcap.py               SYNC mcap 读取 + 时间对齐 + 8 维 state/action
  markers.py                   cv2 画标记 + 零阶保持(训练/部署共用)
  detect_markers.py            阶段A:LocateAnything 逐帧检测 → parquet
  convert_mcap_to_lerobot.py   阶段B:mcap + 检测 → LeRobot 数据集(h264)
  preview_markers.py           阶段A 可视化检查 + 漏检率
  analyze_target_dist.py       诊断:数据集里目标位置的分布
training/
  train_act.sh                 lerobot 原生 ACT 训练
  retrain_pointonly.md         重训笔记(state 加噪配方,见「目前的局限」)
deploy/                        真机运行时(双速)
  config.py                    部署契约常量(关节组/夹爪/home 姿态/安全阈值)
  shared_state.py              线程安全:VLM 中心 + 标签 + 命令计数
  vlm_worker.py                低频 LocateAnything 刷新线程 → shared_state
  policy_runtime.py            高频 ACT 控制环:抓图→跟踪→画标记→推理→执行
  correction.py                在线修正(标记平移/夹爪 effort/JSON 记忆)
  voice.py                     消费机器人本地 mic 服务(ZMQ):标签→物体/修正
  tts.py                       edge-tts + ffmpeg 语音播报,带缓存/预烘焙
  run_g1.py                    主入口
```

---

## 后续

- [ ] 用 `analyze_target_dist.py` 确认目标位置覆盖;集中就补采不同摆位的 demo。
- [ ] **坐标塞进 state** 的目标表示(训练+部署,含归一化)。
- [ ] 更强的视觉编码器,或换个紧凑 VLA 主干,救"读标记"这条路。
- [ ] 抗挪动成功率的量化 benchmark。

---

## 致谢

- **NVIDIA LocateAnything-3B** —— 开放词表 2D/3D 定位(检测器)
- **LeRobot**(Hugging Face)—— 数据集格式与 ACT 训练
- **ACT** —— Action Chunking with Transformers(ALOHA)
- **Galbot G1 SDK** —— 机器人硬件 I/O
