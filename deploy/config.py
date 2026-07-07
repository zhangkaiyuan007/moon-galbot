"""部署契约常量（右臂桌面抓取，G1）。

这些值必须与训练数据一致，否则策略看到的是分布外输入。多数从已验证的
pi0.5 bridge(`RLinf/toolkits/galbot_g1/g1_pi05_bridge.py`)提炼——同一台机器、同一任务、
同样的采集姿态，ACT 部署沿用。

ponytail: 硬件相关的数(home 姿态、夹爪行程、安全阈值)是校准量，不是魔法值——
真机 bring-up 时按实际微调。
"""

from __future__ import annotations

# --- 控制/时序 ---
CONTROL_HZ = 15.0          # 与训练 fps 一致
HORIZON = 10               # ACT action chunk 步数
HEAD_SIZE_WH = (640, 480)  # 头相机喂给策略前 resize 到此

# --- 关节组 / 夹爪（SDK 名称）---
ARM_GROUP = "right_arm"
GRIPPER_NAME = "right_gripper"
GRIPPER_FULL_OPEN_M = 0.12  # 夹爪全开宽度(m)；开度[0,1] = width/0.12

# --- 数据集特征键（与 convert_mcap_to_lerobot.py 一致）---
HEAD_KEY = "observation.images.head"
WRIST_KEY = "observation.images.wrist"
STATE_KEY = "observation.state"
ACTION_KEY = "action"

# --- 采集姿态（头相机视角由腿/头决定，必须匹配训练）---
HOME_LEG = [0.5505, 1.7068, 1.1645, 0.0002, -0.0039]
HOME_HEAD = [0.0, 0.0]
HOME_RIGHT_ARM = [-2.007, 1.312, 0.613, 1.726, 0.272, 0.730, -0.030]

# --- 安全护栏 ---
MAX_STEP_RAD = 0.20        # chunk 内相邻步最大关节跳变
MAX_TRANSIT_RAD = 0.40     # 当前位姿→新 chunk 首步的最大跳变
TRANSIT_SPEED_RAD_S = 0.4

# --- 夹爪滞回（训练里夹爪指令是二值 0/100，部署按开/合驱动）---
GRIP_CLOSE_BELOW = 0.45
GRIP_OPEN_ABOVE = 0.65
GRIP_VELOCITY_MPS = 0.2
GRIP_EFFORT = 50

# --- VLM 刷新 ---
VLM_REFRESH_HZ = 1.5       # locate-anything 低频刷新；抗挪动的响应下限
