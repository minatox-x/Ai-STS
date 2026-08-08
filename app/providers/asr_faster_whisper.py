from __future__ import annotations

import io
import wave
from pathlib import Path

from app.providers.base import ASRProvider, ModelLoadError, Transcript


class FasterWhisperASR(ASRProvider):
    def __init__(
        self,
        model_id: str = "small",
        model_dir: Path | str | None = None,
        compute_type: str = "int8",
        threads: int | None = None,
    ):
        self.model_id = model_id
        self.model_dir = Path(model_dir) if model_dir else None
        self.compute_type = compute_type
        self.threads = threads
        self._model = None
        self._loaded = False

    def _resolve_model_path(self) -> str:
        """
        Determines whether self.model_dir contains an existing HuggingFace snapshot,
        a flat local folder, or if we should rely on standard model_id lookup.
        """
        if not self.model_dir or not self.model_dir.exists():
            return self.model_id

        # 1. Direct path check (user placed model files directly in self.model_dir)
        if (self.model_dir / "model.bin").exists():
            return str(self.model_dir)

        # 2. HuggingFace Hub Cache check (models--org--repo/snapshots/hash)
        snapshots = list(self.model_dir.glob("models--*--*/snapshots/*"))
        if snapshots:
            # Pick the latest modified snapshot directory
            latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
            if (latest_snapshot / "model.bin").exists():
                return str(latest_snapshot)

        # 3. Fall back to standard model ID
        return self.model_id

    def load(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ModelLoadError(
                "faster-whisper is not installed.",
                suggestion="pip install faster-whisper",
            ) from e

        model_path_or_id = self._resolve_model_path()

        # If it resolved to a pure ID and custom dir exists, check for emptiness
        if model_path_or_id == self.model_id and self.model_dir and self.model_dir.exists():
            contents = list(self.model_dir.glob("*"))
            if not contents:
                raise ModelLoadError(
                    f"ASR model files not found at {self.model_dir}.",
                    suggestion="Download this model from Settings -> Models -> Speech Recognition.",
                )

        try:
            # Passing download_root ensures faster-whisper downloads to or uses self.model_dir
            kwargs = {
                "device": "cpu",
                "compute_type": self.compute_type,
                "cpu_threads": self.threads or 0,
            }
            if self.model_dir:
                kwargs["download_root"] = str(self.model_dir)

            self._model = WhisperModel(model_path_or_id, **kwargs)
            self._loaded = True
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load ASR model: {e}",
                suggestion="The model files may be missing or corrupted. Try re-downloading from the Model Manager.",
            ) from e

    def unload(self) -> None:
        self._model = None
        self._loaded = False

    def transcribe(self, audio_pcm16: bytes, sample_rate: int) -> Transcript:
        if not self._loaded or self._model is None:
            raise ModelLoadError("ASR model not loaded. Call load() first.")

        # Write raw PCM16 into an in-memory WAV container for faster-whisper
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_pcm16)
        buf.seek(0)

        segments, info = self._model.transcribe(
            buf,
            language=None,  # Auto-detect language
            vad_filter=True,
            beam_size=5,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return Transcript(text=text, language=info.language, is_final=True)
