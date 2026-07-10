# test_voice_wakeup.py —— 只测语音：唤醒/关键词 → 路由 → 机器人喇叭 TTS
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from galbot_sdk.g1 import GalbotRobot

from voice import VoiceMicClient
from tts import make_tts
from shared_state import SharedState
from correction import CorrectionMemory

shared = SharedState()
mem = CorrectionMemory(None)

robot = GalbotRobot()
if not robot.init():  # 只需音频，不订阅相机
    raise SystemExit("GalbotRobot init failed")
robot.set_volume(100)  # 系统全局音量 0-100

try:
    voice = VoiceMicClient(
        shared,
        mem,
        sub_addr="tcp://192.168.1.88:6000",
        tts=make_tts(robot),
    )
    voice.start()
    voice.say("voice client started, waiting wakeup")

    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    try:
        voice.stop()
    except NameError:
        pass
    robot.request_shutdown()
    robot.wait_for_shutdown()
    robot.destroy()
