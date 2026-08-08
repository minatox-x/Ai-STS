"""
Abstract interfaces every model provider implements.

Adding a new model = add a models.json entry + implement one of these classes.
Nothing else in the app should need to change (see model_registry.py's factory).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional


@dataclass
class Transcript:
    text: str
    language: Optional[str] = None
    is_final: bool = True
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class GenerationSettings:
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_output_tokens: int = 200
    seed: Optional[int] = None
    context_size: int = 4096
    threads: Optional[int] = None
    gpu_layers: int = 0


@dataclass
class TTSSettings:
    voice: Optional[str] = None
    speed: float = 1.0
    pitch: float = 0.0
    volume: float = 1.0
    language_hint: Optional[str] = None


class ASRProvider(ABC):
    """Speech -> text."""

    model_id: str

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory. Raise ModelLoadError on failure."""

    @abstractmethod
    def unload(self) -> None:
        """Free memory."""

    @abstractmethod
    def transcribe(self, audio_pcm16: bytes, sample_rate: int) -> Transcript:
        """One-shot transcription of a complete utterance."""

    def transcribe_stream(self, audio_chunks: Iterator[bytes], sample_rate: int) -> Iterator[Transcript]:
        """Optional: streaming transcription. Default falls back to buffering + one-shot."""
        buf = b"".join(audio_chunks)
        yield self.transcribe(buf, sample_rate)

    @property
    def is_loaded(self) -> bool:
        return getattr(self, "_loaded", False)


class LLMProvider(ABC):
    """Chat messages -> response text."""

    model_id: str

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def generate(self, messages: list[ChatMessage], settings: GenerationSettings) -> str:
        """Full (non-streaming) generation."""

    def generate_stream(self, messages: list[ChatMessage], settings: GenerationSettings) -> Iterator[str]:
        """Optional: token/sentence streaming. Default falls back to one-shot generate()."""
        yield self.generate(messages, settings)

    @property
    def is_loaded(self) -> bool:
        return getattr(self, "_loaded", False)


class TTSProvider(ABC):
    """Text -> speech audio (PCM16)."""

    model_id: str

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def synthesize(self, text: str, settings: TTSSettings) -> tuple[bytes, int]:
        """Returns (pcm16_bytes, sample_rate)."""

    def synthesize_stream(self, text: str, settings: TTSSettings) -> Iterator[tuple[bytes, int]]:
        """Optional: streaming synthesis. Default falls back to one-shot synthesize()."""
        yield self.synthesize(text, settings)

    @property
    def is_loaded(self) -> bool:
        return getattr(self, "_loaded", False)


class VADProvider(ABC):
    @abstractmethod
    def is_speech(self, frame_pcm16: bytes, sample_rate: int) -> bool: ...


class ModelLoadError(RuntimeError):
    """Raised when a model fails to load (missing file, insufficient memory, corrupt weights)."""

    def __init__(self, message: str, *, suggestion: Optional[str] = None):
        super().__init__(message)
        self.suggestion = suggestion
