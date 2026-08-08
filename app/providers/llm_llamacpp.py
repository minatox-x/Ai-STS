from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from app.providers.base import ChatMessage, GenerationSettings, LLMProvider, ModelLoadError


class LlamaCppLLM(LLMProvider):
    def __init__(self, model_id: str, gguf_path: Path):
        self.model_id = model_id
        self.gguf_path = gguf_path
        self._llm = None
        self._loaded = False
        self._context_size = 4096

    def load(self, settings: GenerationSettings | None = None) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ModelLoadError(
                "llama-cpp-python is not installed.",
                suggestion="pip install llama-cpp-python",
            ) from e

        if not self.gguf_path.exists():
            raise ModelLoadError(
                f"LLM model file not found at {self.gguf_path}.",
                suggestion="Download this model from Settings -> Models -> Local AI.",
            )

        settings = settings or GenerationSettings()
        self._context_size = settings.context_size
        try:
            self._llm = Llama(
                model_path=str(self.gguf_path),
                n_ctx=settings.context_size,
                n_threads=settings.threads or (os.cpu_count() or 4),
                n_gpu_layers=settings.gpu_layers,
                seed=settings.seed if settings.seed is not None else -1,
                verbose=False,
            )
            self._loaded = True
        except Exception as e:
            msg = str(e)
            suggestion = "The model file may be corrupted or incompatible with this llama-cpp-python version."
            if "memory" in msg.lower() or "alloc" in msg.lower():
                suggestion = "Not enough RAM to load this model. Try a smaller model or lower context size."
            raise ModelLoadError(f"Failed to load local LLM: {e}", suggestion=suggestion) from e

    def unload(self) -> None:
        self._llm = None
        self._loaded = False

    def _to_llamacpp_messages(self, messages: list[ChatMessage]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def generate(self, messages: list[ChatMessage], settings: GenerationSettings) -> str:
        if not self._loaded:
            raise ModelLoadError("LLM not loaded. Call load() first.")
        result = self._llm.create_chat_completion(
            messages=self._to_llamacpp_messages(messages),
            temperature=settings.temperature,
            top_p=settings.top_p,
            top_k=settings.top_k,
            repeat_penalty=settings.repeat_penalty,
            max_tokens=settings.max_output_tokens,
        )
        return result["choices"][0]["message"]["content"].strip()

    def generate_stream(self, messages: list[ChatMessage], settings: GenerationSettings) -> Iterator[str]:
        """
        Yields text as it's generated so the pipeline can start TTS on the first
        completed sentence rather than waiting for the whole reply (spec section 27).
        """
        if not self._loaded:
            raise ModelLoadError("LLM not loaded. Call load() first.")
        stream = self._llm.create_chat_completion(
            messages=self._to_llamacpp_messages(messages),
            temperature=settings.temperature,
            top_p=settings.top_p,
            top_k=settings.top_k,
            repeat_penalty=settings.repeat_penalty,
            max_tokens=settings.max_output_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk["choices"][0]["delta"].get("content")
            if delta:
                yield delta
