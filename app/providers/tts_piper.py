from __future__ import annotations

from pathlib import Path

from app.providers.base import ModelLoadError, TTSProvider, TTSSettings


class PiperTTS(TTSProvider):
    """
    Default CPU-friendly TTS. Piper voices are per-language .onnx + .onnx.json pairs,
    so `model_dir` is expected to contain one subfolder per language/voice
    (e.g. model_dir/hi/voice.onnx, model_dir/en/voice.onnx).
    """

    def __init__(self, model_id: str, model_dir: Path):
        self.model_id = model_id
        self.model_dir = model_dir
        self._voices: dict[str, "object"] = {}
        self._loaded = False

    def load(self) -> None:
        try:
            from piper import PiperVoice
        except ImportError as e:
            raise ModelLoadError(
                "piper-tts is not installed.",
                suggestion="pip install piper-tts",
            ) from e

        if not self.model_dir.exists():
            raise ModelLoadError(
                f"TTS voice files not found at {self.model_dir}.",
                suggestion="Download from Settings -> Models -> Voice.",
            )

        found_any = False
        for onnx_file in self.model_dir.rglob("*.onnx"):
            lang = onnx_file.parent.name  # expects e.g. .../hi/xx.onnx, .../en/xx.onnx
            try:
                self._voices[lang] = PiperVoice.load(str(onnx_file))
                found_any = True
            except Exception:
                continue

        if not found_any:
            raise ModelLoadError(
                f"No usable Piper voice files found under {self.model_dir}.",
                suggestion="Re-download the voice pack from the Model Manager.",
            )
        self._loaded = True

    def unload(self) -> None:
        self._voices = {}
        self._loaded = False

    def _pick_voice(self, language_hint: str | None):
        if language_hint and language_hint in self._voices:
            return self._voices[language_hint]
        # fall back to whatever's loaded, preferring English if present
        return self._voices.get("en") or next(iter(self._voices.values()))

    def synthesize(self, text: str, settings: TTSSettings) -> tuple[bytes, int]:
        if not self._loaded:
            raise ModelLoadError("TTS not loaded. Call load() first.")
        voice = self._pick_voice(settings.language_hint)
        audio_chunks = []
        sample_rate = voice.config.sample_rate
        for audio_bytes in voice.synthesize_stream_raw(text, length_scale=1.0 / max(settings.speed, 0.1)):
            audio_chunks.append(audio_bytes)
        return b"".join(audio_chunks), sample_rate

    def synthesize_stream(self, text: str, settings: TTSSettings):
        if not self._loaded:
            raise ModelLoadError("TTS not loaded. Call load() first.")
        voice = self._pick_voice(settings.language_hint)
        sample_rate = voice.config.sample_rate
        for audio_bytes in voice.synthesize_stream_raw(text, length_scale=1.0 / max(settings.speed, 0.1)):
            yield audio_bytes, sample_rate
