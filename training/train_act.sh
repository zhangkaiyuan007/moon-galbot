#!/usr/bin/env bash
# ACT 训练启动（lerobot 原生）。相机键/维度从数据集自动推导，归一化统计自动计算，
# 无需单独跑 norm-stats。
#
# 用法：
#   export HF_LEROBOT_HOME=/home1/jiajunjie/lerobot_data
#   bash training/train_act.sh
# 从 moon-galbot 根目录跑（用项目 venv）。GPU 显存不够就调小 --batch_size。
set -euo pipefail

REPO_ID="${REPO_ID:-galbot_g1_marked}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:?先 export HF_LEROBOT_HOME}"
OUT="${OUT:-outputs/act_${REPO_ID}}"

uv run python -m lerobot.scripts.train \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${HF_LEROBOT_HOME}/${REPO_ID}" \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.n_action_steps=15 \
  --policy.chunk_size=50 \
  --output_dir="${OUT}" \
  --batch_size=8 \
  --steps=100000 \
  --save_freq=10000 \
  --log_freq=200 \
  --wandb.enable=false

# - video_backend=pyav：数据集是 h264，pyav 能解；torchcodec 在本机 FFmpeg 上加载失败。
# - n_action_steps=15：~1s@15fps 执行一段再重规划+刷新标记，抗挪动反应性够；默认 100 太长。
# - chunk_size=50：ACT 预测 50 步、执行前 15 步。
# 训练产物在 $OUT/checkpoints/；部署用 --checkpoint 指向某个 checkpoint 目录。
