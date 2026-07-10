"""机器人 TTS：文字 → edge-tts(mp3) → ffmpeg 转 16k/16bit/mono PCM → 机器人喇叭。

SDK 无内置 TTS，`write_audio_stream_output` 只吃 16k 16bit mono 裸 PCM，按小块喂。
edge-tts 在线合成、输出 24k mp3，用系统 ffmpeg 转码。
依赖：`pip install edge-tts` + 系统 `ffmpeg`。接线见 run_g1.py：
    voice = VoiceMicClient(shared, mem, addr, tts=make_tts(robot))
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import time
from typing import Callable

VOICE = "zh-CN-XiaoyiNeural"
CHUNK = 2560  # 字节；16k*16bit*mono 下 = 80ms/块，与 audio_example 一致


def _synth_mp3(text: str, voice: str) -> bytes:
    """edge-tts 合成 → mp3 字节（阻塞）。"""
    import edge_tts

    async def run() -> bytes:
        buf = bytearray()
        async for c in edge_tts.Communicate(text, voice).stream():
            if c["type"] == "audio":
                buf += c["data"]
        return bytes(buf)

    return asyncio.run(run())


def _mp3_to_pcm(mp3: bytes) -> bytes:
    """ffmpeg：mp3 → 16k 16bit mono 裸 PCM。"""
    p = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", "pipe:0",
         "-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1"],
        input=mp3, stdout=subprocess.PIPE, check=True,
    )
    return p.stdout


def _play(robot, pcm: bytes, stream_id: str) -> None:
    for i in range(0, len(pcm), CHUNK):
        robot.write_audio_stream_output(pcm[i:i + CHUNK], stream_id)
        time.sleep(0.05)  # ponytail: 固定节流，跟 audio_example 一致；机器人侧无背压回执


def make_tts(robot, stream_id: str = "robot_tts", voice: str = VOICE) -> Callable[[str], None]:
    """返回 say(text)：合成失败只打日志不抛，别让一句播报崩了主流程。"""
    assert shutil.which("ffmpeg"), "需要系统 ffmpeg（apt install ffmpeg）"

    def say(text: str) -> None:
        try:
            pcm = _mp3_to_pcm(_synth_mp3(text, voice))
        except Exception as e:  # 网络/合成/转码任一失败
            print(f"[tts] 合成失败，跳过播报: {e}")
            return
        _play(robot, pcm, stream_id)

    return say


def _selfcheck() -> None:
    # 分块逻辑：不连网、不连机器人，用假 synth + 假 robot 验证喂进去的字节完整、每块 ≤ CHUNK。
    class FakeRobot:
        def __init__(self):
            self.chunks: list[bytes] = []

        def write_audio_stream_output(self, chunk: bytes, sid: str) -> bool:
            self.chunks.append(chunk)
            return True

    pcm = bytes(range(256)) * 40  # 10240 字节 = 4 整块 + 余
    robot = FakeRobot()
    _play(robot, pcm, "t")
    assert b"".join(robot.chunks) == pcm, "重组后应与原 PCM 完全一致"
    assert all(len(c) <= CHUNK for c in robot.chunks), "每块不超过 CHUNK"
    assert len(robot.chunks) == (len(pcm) + CHUNK - 1) // CHUNK
    print("tts selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
