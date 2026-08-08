from __future__ import annotations

from pathlib import Path

from app.providers.base import ModelLoadError, TTSProvider, TTSSettings


class Qwen3TTS(TTSProvider):
    """
    Higher-quality tier. Community CPU benchmarks show ~3-5x real-time generation
    even on strong CPUs -- this provider should only be offered/selected when a
    CUDA GPU is detected (see model_registry.recommend()). If loaded without a GPU,
    it will work but conversational latency will be poor; we warn rather than block,
    since a user may still want it for non-live uses.
    """

    def __init__(self, model_id: str, model_dir: Path, device: str = "cpu"):
        self.model_id = model_id
        self.model_dir = model_dir
        self.device = device
        self._model = None
        self._loaded = False

    def load(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ModelLoadError(
                "The Qwen3-TTS runtime (torch) is not installed.",
                suggestion="pip install torch (see pytorch.org for the CUDA build matching your GPU).",
            ) from e

        if not self.model_dir.exists():
            raise ModelLoadError(
                f"Qwen3-TTS model files not found at {self.model_dir}.",
                suggestion="Download from Settings -> Models -> Voice -> Higher quality.",
            )

        if self.device == "cpu":
            # Not a hard failure -- some users may accept the latency -- but this must
            # be surfaced, not hidden, per the "no fake functionality" rule.
            import warnings
            warnings.warn(
                "Qwen3-TTS is loaded on CPU. Expect several seconds of generation time "
                "per sentence; this is not suitable for live conversation without a GPU."
            )

        # NOTE: Qwen3-TTS's exact inference API (vendor package name / entry point) should
        # be re-checked against https://github.com/QwenLM/Qwen3-TTS at integration time --
        # it changed at least once during 2026 (native transformers support added in June).
        # Wiring the exact call here without re-verifying against the installed package
        # version would risk silently doing the wrong thing, so this raises clearly instead
        # of guessing:
        raise ModelLoadError(
            "Qwen3-TTS integration needs a final wiring pass against the vendor's current "
            "package API before first use.",
            suggestion=(
                "See https://github.com/QwenLM/Qwen3-TTS for current usage, then complete "
                "the synthesize()/load() calls in this file. Piper (the default tier) is "
                "fully functional in the meantime."
            ),
        )

    def unload(self) -> None:
        self._model = None
        self._loaded = False

    def synthesize(self, text: str, settings: TTSSettings) -> tuple[bytes, int]:
        raise ModelLoadError("Qwen3-TTS provider is not yet wired up -- see load() for details.")
