"""
Microphone capture, playback, and VAD. Uses sounddevice (PortAudio) which is
lightweight and works well on Windows without extra native deps beyond the
bundled PortAudio binary that the `sounddevice` wheel ships.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import webrtcvad

from app.providers.base import VADProvider

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)


class WebRTCVAD(VADProvider):
    def __init__(self, aggressiveness: int = 2):
        # aggressiveness 0-3, higher = more aggressive filtering of non-speech
        self._vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, frame_pcm16: bytes, sample_rate: int = SAMPLE_RATE) -> bool:
        try:
            return self._vad.is_speech(frame_pcm16, sample_rate)
        except Exception:
            # webrtcvad requires exact frame sizes (10/20/30ms) -- mismatched frames
            # shouldn't crash the pipeline, just be treated as non-speech.
            return False


def list_input_devices() -> list[dict]:
    devices = sd.query_devices()
    return [
        {"index": i, "name": d["name"], "channels": d["max_input_channels"]}
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]


class MicStream:
    """
    Push-to-talk / continuous capture. Call start(), then read frames from
    `.frames` (a Queue of int16 PCM byte chunks) until stop().
    """

    def __init__(self, device_index: Optional[int] = None, sample_rate: int = SAMPLE_RATE):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.frames: "queue.Queue[bytes]" = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self.level_callback: Optional[Callable[[float], None]] = None

    def _callback(self, indata, frames, time_info, status):
        pcm16 = (indata[:, 0] * 32767).astype(np.int16).tobytes()
        self.frames.put(pcm16)
        if self.level_callback:
            rms = float(np.sqrt(np.mean(np.square(indata[:, 0])))) if len(indata) else 0.0
            self.level_callback(rms)

    def start(self) -> None:
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SAMPLES,
            device=self.device_index,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> bytes:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        chunks = []
        while not self.frames.empty():
            chunks.append(self.frames.get_nowait())
        return b"".join(chunks)


class Player:
    """Plays PCM16 audio and supports immediate stop for barge-in interruption."""

    def __init__(self):
        self._stop_flag = threading.Event()

    def play(self, pcm16: bytes, sample_rate: int) -> None:
        self._stop_flag.clear()
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        sd.play(audio, samplerate=sample_rate)
        # Poll so we can react to stop() promptly instead of blocking on sd.wait().
        while sd.get_stream().active if sd.get_stream() else False:
            if self._stop_flag.is_set():
                sd.stop()
                return
            sd.sleep(20)

    def stop(self) -> None:
        self._stop_flag.set()
        sd.stop()
