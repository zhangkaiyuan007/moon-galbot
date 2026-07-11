# 纯点图 + state 加噪 重训指南

## 为什么要重训（问题诊断）

ACT 策略**无视头相机上的 VLM 标记**，靠 `observation.state` 的本体惯性走一条固定轨迹到固定区域（模仿学习经典的 **causal confusion / state 泄漏**）。

证据：
- 换任意 step 的 checkpoint 行为都一样 → 训练层面问题，非过拟合、非部署 bug。
- 两位置对比：目标物在头相机里位移 **124px**，手臂 `action` 轨迹几乎重合（`chunk 0` 完全一样）。
- 腕相机学到了近距离夹取时机，头相机（全局定位）完全没被利用。

结论：只能抓固定位置的物体。要让手臂响应目标位置，必须在训练时**逼策略用视觉**。

## 方案：三管齐下

| 改动 | 文件 | 状态 | 作用 |
|---|---|---|---|
| 纯点图（头相机只剩黑底+点） | `tools/markers.py` + `training/train_act_pointonly.py` | ✅ 已改 | 消灭背景纹理，逼策略从点读位置 |
| state 加噪 | lerobot `modeling_act.py`（+4 行） | ⏳ 训练机手动加 | 断掉本体惯性抄近路 |
| LK 光流 tracker（部署侧） | `deploy/policy_runtime.py` | ✅ 已就绪 | 部署时标记按控制环频率跟手 |

纯点图 + state 加噪**夹逼**：前者让视觉任务变简单，后者让 state 不可靠，两者缺一效果打折。

## 训练机执行步骤

### 1. 同步项目
把 `moon-galbot` 同步到训练机，确认带上：
- `tools/markers.py`（`MASK_BACKGROUND = True`）
- `training/train_act_pointonly.py`

### 2. 给 lerobot 加 state 加噪（4 行）
编辑训练机的 `.../site-packages/lerobot/common/policies/act/modeling_act.py`，
在 `ACTPolicy.forward` 里 `batch = self.normalize_inputs(batch)` **之后**加：

```python
        batch = self.normalize_inputs(batch)
        # state 加噪：削弱本体状态可靠性，逼策略用视觉(对抗 causal confusion)。std 可调 0.3~0.5
        if self.training and "observation.state" in batch:
            batch = dict(batch)
            batch["observation.state"] = batch["observation.state"] + \
                torch.randn_like(batch["observation.state"]) * 0.4
```
（只在 `forward`=训练路径生效，`select_action`=推理不受影响。）

### 3. 训练前验证在线抹黑（必做，2 分钟）
真实 dataset 图经 h264 压缩，绿/蓝点的**颜色阈值和半径可能要调**。在 `train_act_pointonly.py`
的 `patched` 里临时加一行 dump：

```python
if idx < 5:
    cv2.imwrite(f"/tmp/pt_{idx}.png",
                (item[HEAD_KEY].numpy().transpose(1,2,0)*255).astype("uint8")[...,::-1])
```
跑起来后看 `/tmp/pt_*.png`：**背景全黑、绿蓝两点在正确位置** = 对。
- 全黑无点 → 放宽 `_point_only` 的颜色阈值
- 点太小/太大 → 调 `rad`

确认无误后删掉这行 dump。

### 4. 训练
用 `train_act_pointonly.py` 替代 `python -m lerobot.scripts.train`，参数照搬 `train_act.sh`：

```bash
export HF_LEROBOT_HOME=/home1/jiajunjie/lerobot_data
python training/train_act_pointonly.py \
  --dataset.repo_id=galbot_g1_marked \
  --dataset.root="$HF_LEROBOT_HOME/galbot_g1_marked" \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.n_action_steps=15 \
  --policy.chunk_size=50 \
  --output_dir=outputs/act_pointonly \
  --batch_size=8 --steps=100000 --save_freq=10000 --log_freq=200 \
  --wandb.enable=false
```
不用重跑阶段 B、不改 dataset——`train_act_pointonly.py` 在加载时在线抹黑。

### 5. 部署新 checkpoint
`--checkpoint outputs/act_pointonly/checkpoints/<step>/pretrained_model`，其余命令不变。

## 关键一致性约束

- **`MASK_BACKGROUND` 训练和部署必须一致**。训练用了纯点图，部署 `markers.py` 也必须 `True`（已默认）。
- **旧 checkpoint 不能配纯点图部署**：旧 checkpoint 是"有背景图"训过的，纯点图部署会输入不匹配。想用旧 checkpoint 测就把 `MASK_BACKGROUND=False`。
- **`std=0.4` 是起点**：太大伤夹取（夹爪宽度也在 state 里），太小不够力，看效果调。

## 验证是否成功

新 checkpoint 出来后，dry-run 做**两位置对比**：目标物放明显不同的左/右两处各跑一次，
看 `[chunk 0~7]` 手臂 `action`：
- **随目标位置明显变化** → 成功，策略开始用点定位。
- **仍几乎不变** → 加大 state 噪声 std、或检查抹黑图点位是否正确、或目标位置多样性仍不足
  （跑 `tools/analyze_target_dist.py` 看数据分布）。
