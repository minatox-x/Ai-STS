from __future__ import annotations

import io
import wave
from pathlib import Path

from app.providers.base import ASRProvider, ModelLoadError, Transcript


class FasterWhisperASR(ASRProvider):
    def __init__(self, model_id: str, model_dir: Path, compute_type: str = "int8", threads: int | None = None):
        self.model_id = model_id
        self.model_dir = model_dir
        self.compute_type = compute_type
        self.threads = threads
        self._model = None
        self._loaded = False

    def load(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ModelLoadError(
                "faster-whisper is not installed.",
                suggestion="pip install faster-whisper",
            ) from e

        if not self.model_dir.exists():
            raise ModelLoadError(
                f"ASR model files not found at {self.model_dir}.",
                suggestion="Download this model from Settings -> Models -> Speech Recognition.",
            )

        try:
            self._model = WhisperModel(
                str(self.model_dir),
                device="cpu",
                compute_type=self.compute_type,
                cpu_threads=self.threads or 0,
            )
            self._loaded = True
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load ASR model: {e}",
                suggestion="The model files may be corrupted. Try re-downloading from the Model Manager.",
            ) from e

    def unload(self) -> None:
        self._model = None
        self._loaded = False

    def transcribe(self, audio_pcm16: bytes, sample_rate: int) -> Transcript:
        if not self._loaded:
            raise ModelLoadError("ASR model not loaded. Call load() first.")

        # faster-whisper accepts a file path or numpy array; write a small in-memory WAV
        # rather than requiring the caller to know internal format details.
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_pcm16)
        buf.seek(0)

        segments, info = self._model.transcribe(
            buf,
            language=None,  # auto-detect; caller can force via a config layer if desired
            vad_filter=True,
            beam_size=5,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return Transcript(text=text, language=info.language, is_final=True)
