from __future__ import annotations

import io
import wave
from pathlib import Path

from app.providers.base import ASRProvider, ModelLoadError, Transcript


class Qwen3ASR(ASRProvider):
    """
    Higher-quality tier. Official usage is torch/transformers-based; runs on CPU but
    is noticeably slower there (no dedicated CPU-optimized runtime was found as of the
    verification date in models.json). Only recommend this tier when a CUDA GPU with
    enough VRAM is detected -- see model_registry.recommend().
    """

    def __init__(self, model_id: str, model_dir: Path, device: str = "cpu"):
        self.model_id = model_id
        self.model_dir = model_dir
        self.device = device
        self._model = None
        self._processor = None
        self._loaded = False

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForMultimodalLM
        except ImportError as e:
            raise ModelLoadError(
                "The Qwen3-ASR runtime (torch + transformers) is not installed.",
                suggestion=(
                    "This is a large optional dependency. Install with: "
                    "pip install torch transformers --extra-index-url "
                    "https://download.pytorch.org/whl/cu121 (or the CPU wheel index if no GPU)."
                ),
            ) from e

        if not self.model_dir.exists():
            raise ModelLoadError(
                f"Qwen3-ASR model files not found at {self.model_dir}.",
                suggestion="Download from Settings -> Models -> Speech Recognition -> Higher quality.",
            )

        try:
            self._processor = AutoProcessor.from_pretrained(str(self.model_dir))
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            self._model = AutoModelForMultimodalLM.from_pretrained(str(self.model_dir), dtype=dtype).to(self.device).eval()
            self._torch = torch
            self._loaded = True
        except Exception as e:
            raise ModelLoadError(f"Failed to load Qwen3-ASR: {e}") from e

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False

    def transcribe(self, audio_pcm16: bytes, sample_rate: int) -> Transcript:
        if not self._loaded:
            raise ModelLoadError("Qwen3-ASR not loaded. Call load() first.")

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_pcm16)
        buf.seek(0)

        inputs = self._processor.apply_transcription_request(audio=[buf]).to(self.device)
        with self._torch.inference_mode():
            output_ids = self._model.generate(**inputs, max_new_tokens=256, do_sample=False)
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        text = self._processor.decode(generated, return_format="transcription_only")[0]
        return Transcript(text=text.strip(), is_final=True)
