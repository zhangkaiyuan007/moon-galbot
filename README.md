## 0. 目录

```
tools/        数据处理（mcap → 带标记的 lerobot 数据集）
  galbot_mcap.py            读 SYNC mcap + 时间对齐 + 8维 state/action（库）
  markers.py               纯 cv2 画点 + ZOH（训练/部署共用，保证一致）
  detect_markers.py        阶段A：locate-anything 逐帧检测 → detections/*.parquet
  convert_mcap_to_lerobot.py 阶段B：mcap + detections → lerobot 数据集（h264）
  preview_markers.py       检查阶段A：把中心画回真实帧 + 打印漏检率
training/
  train_act.sh             lerobot 原生 ACT 训练启动
deploy/       真机部署（双速运行时）
  config.py                部署契约常量（关节组/夹爪/home 姿态/安全阈值）
  shared_state.py          线程安全：VLM 中心 + 目标标签 + 命令计数
  vlm_worker.py            低频 locate-anything 刷新环（线程）→ shared_state
  policy_runtime.py        高频 ACT 控制环：抓图+画标记→推理→SDK 执行
  correction.py            在线修正三件套（标记平移/夹爪 effort/记忆 JSON）
  voice.py                 消费 on-robot mic 服务（ZMQ）：KWS 标签→物体/修正路由
  run_g1.py                主入口，串起以上全部
```

---

## 1. 环境配置

前提：装了 `uv`；有 NVIDIA 驱动（`nvidia-smi` 能出）；`eagle/Embodied`(locate-anything)
放在 moon-galbot 的**上一级**（`X/moon-galbot` 和 `X/eagle/Embodied`）。

```bash
cd moon-galbot
uv sync
```

---

## 2. 端到端流程

路径以服务器为例（`/home1/jiajunjie/...`），按实际改。所有命令在 moon-galbot 根目录跑。

### 步骤 1 · 采数据（比赛现场，遥操作）
录成 Galbot 的 `*.SYNC.mcap`（每条一集）。数据配方（关键，见 PLAN）：
- 每条随机摆物体+框位置；约 1/4~1/3 集在接近途中**人为挪动物体**并重新接近完成；
- 桌上多物体、只抓目标；目标物轮换。
> 管线只用 `*.SYNC.mcap`；`*.FIN.mcap` 是原始未对齐数据、用不上。

### 步骤 2 · 阶段A 检测
```bash
uv run python tools/detect_markers.py \
  --data-dir /home1/jiajunjie/1105_1696 --out-dir detections \
  --target-label "cola bottle" --bin-label "basket" \
  --model /home1/jiajunjie/LocateAnything-3B --max-episodes 1
```
先 1 条，没问题再去掉 `--max-episodes` 全量。多物体数据每条传各自 `--target-label`。`--la-repo` 默认自动推导（上一级 eagle/Embodied），不对再手动指。

### 步骤 2.5 · 检查检测效果
```bash
uv run python tools/preview_markers.py \
  --data-dir /home1/jiajunjie/1105_1696 --detections-dir detections --out-dir preview --n 12
```
看打印的漏检率（框应 ~0%，目标遮挡漏几帧正常）；翻 `preview/` 的图确认**绿点压在目标、
蓝点在框**。

### 步骤 3 · 阶段B 转换
```bash
export HF_LEROBOT_HOME=/home1/jiajunjie/lerobot_data
uv run python tools/convert_mcap_to_lerobot.py \
  --data-dir /home1/jiajunjie/1105_1696 --detections-dir detections \
  --output-root $HF_LEROBOT_HOME --repo-id galbot_g1_marked --fps 15 --overwrite
```
默认 h264 编码。产物是 lerobot 数据集，特征键 `observation.images.head/wrist`、
`observation.state`、`action`（8 维）。

### 步骤 4 · 训练 ACT
```bash
export HF_LEROBOT_HOME=/home1/jiajunjie/lerobot_data
bash training/train_act.sh              # 内部用 video_backend=pyav
```
产物 checkpoint 在 `outputs/act_galbot_g1_marked/checkpoints/`。显存不够改脚本里 `--batch_size`。

### 步骤 5 · 真机部署

**先无语音把抓取调稳** —— 命令行直接给目标，不需要 mic 服务：
```bash
# dry-run：读观测、VLM 定位、画标记、推理、打印动作，机器人不动
uv run python deploy/run_g1.py \
  --checkpoint outputs/act_galbot_g1_marked/checkpoints/<step>/pretrained_model \
  --model /home1/jiajunjie/LocateAnything-3B \
  --target-label "cola bottle" --bin-label "basket"
# 调好后真动
uv run python deploy/run_g1.py \
  --checkpoint outputs/act_galbot_g1_marked/checkpoints/<step>/pretrained_model \
  --model /home1/jiajunjie/LocateAnything-3B \
  --target-label "cola bottle" --bin-label "basket" \
  --execute --go-home
```
这一步就能验证整条抓取闭环：VLM 定位 → 标记 → ACT 抓取 → 放框 → 抗挪动。联调重点：
go-home 姿态、标记与训练一致、动作/安全合理、夹爪 effort、控制频率。

**抓稳之后再加语音** —— 不给 `--target-label`，改用 `--mic-addr`：
```bash
uv run python deploy/run_g1.py \
  --checkpoint outputs/act_galbot_g1_marked/checkpoints/<step>/pretrained_model \
  --model /home1/jiajunjie/LocateAnything-3B \
  --mic-addr tcp://<机器人IP>:6000 --execute --go-home
```
流程：说"帮我拿可乐" → 抓取 → "完成"；抓取中"往上抓一点/抓紧点"实时修正并记进
`corrections.json`。语音走同事的 on-robot mic 服务（`/home/galbot/galbot-mic-service`），不用 Whisper：
- 在机器人上部署并启动 `galbot_mic`（见该项目 `使用流程.md`），它做 VAD+KWS，命中标签经
  ZMQ 6000 发出。`run_g1.py` 用 `--mic-addr` 订阅。
- 在服务的 `keywords.txt` 里配物料词（`拼音 @cola`）和修正词（`拼音 @corr_up` 等）；
  `deploy/voice.py` 顶部 `OBJECT_LABELS`/`CORRECTION_LABELS` 把标签映射到 VLM 检测短语和
  修正量，两处标签要对上。

---

## 3. 传输到新机器（rsync）
```bash
# 项目（排除 .venv/产物；靠 uv sync 重建环境）
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='detections' \
  --exclude='lerobot_data' --exclude='outputs' <本机>/moon-galbot/  用户@服务器:/路径/moon-galbot/
# locate-anything 仓库（保持在 moon-galbot 上一级）
rsync -avz <本机>/eagle/Embodied/  用户@服务器:/路径/eagle/Embodied/
# 模型权重（从 HF 缓存 snapshot 用 -L 解引用成扁平目录）
snap=~/.cache/huggingface/hub/models--nvidia--LocateAnything-3B/snapshots/*/
rsync -avzL $snap  用户@服务器:/路径/LocateAnything-3B/
# 数据（只需 SYNC）
rsync -av -P --include='*.SYNC.mcap' --exclude='*' <本机>/1105_1696/  用户@服务器:/路径/1105_1696/
```