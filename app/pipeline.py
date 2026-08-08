"""
Orchestrates: mic -> VAD -> ASR -> memory retrieval -> router -> local LLM / Gemini
-> TTS -> playback. Each stage is a swappable provider (app/providers/base.py).

This is Phase 1: the full loop works end-to-end for a single push-to-talk turn.
Sentence-level LLM->TTS streaming and true barge-in are structured with clear
TODOs (see spec sections 22 and 27) but need testing against a real mic/speaker,
which this sandbox doesn't have.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from app import gemini_usage
from app.config import AppConfig
from app.memory import MemoryStore
from app.model_registry import ModelRegistry
from app.providers.base import ChatMessage, GenerationSettings, ModelLoadError, TTSSettings
from app.router import RouteDecision, route


@dataclass
class TurnResult:
    user_text: str
    assistant_text: str
    routed_to: str
    latency_ms: dict


class ProviderFactory:
    """Instantiates the right provider class for a given model registry entry."""

    def __init__(self, config: AppConfig, registry: ModelRegistry):
        self.config = config
        self.registry = registry
        self.models_dir = Path(config.models_dir)

    def build_asr(self, model_id: str):
        entry = self.registry.get(model_id)
        if entry is None:
            raise ModelLoadError(f"Unknown ASR model id: {model_id}")
        model_dir = self.models_dir / "ASR" / model_id
        if entry.raw["runtime"] == "faster_whisper":
            from app.providers.asr_faster_whisper import FasterWhisperASR
            return FasterWhisperASR(model_id, model_dir, threads=self.config.threads)
        if entry.raw["runtime"] == "transformers_qwen3_asr":
            from app.providers.asr_qwen3 import Qwen3ASR
            device = "cuda" if self.config.gpu_layers > 0 else "cpu"
            return Qwen3ASR(model_id, model_dir, device=device)
        raise ModelLoadError(f"No provider implemented for ASR runtime: {entry.raw['runtime']}")

    def build_llm(self, model_id: str):
        entry = self.registry.get(model_id)
        if entry is None:
            raise ModelLoadError(f"Unknown LLM model id: {model_id}")
        model_dir = self.models_dir / "LLM" / model_id
        gguf_files = list(model_dir.glob("*.gguf")) if model_dir.exists() else []
        gguf_path = gguf_files[0] if gguf_files else model_dir / f"{model_id}.gguf"
        from app.providers.llm_llamacpp import LlamaCppLLM
        return LlamaCppLLM(model_id, gguf_path)

    def build_tts(self, model_id: str):
        entry = self.registry.get(model_id)
        if entry is None:
            raise ModelLoadError(f"Unknown TTS model id: {model_id}")
        model_dir = self.models_dir / "TTS" / model_id
        if entry.raw["runtime"] == "piper":
            from app.providers.tts_piper import PiperTTS
            return PiperTTS(model_id, model_dir)
        if entry.raw["runtime"] == "qwen3_tts":
            from app.providers.tts_qwen3 import Qwen3TTS
            device = "cuda" if self.config.gpu_layers > 0 else "cpu"
            return Qwen3TTS(model_id, model_dir, device=device)
        raise ModelLoadError(f"No provider implemented for TTS runtime: {entry.raw['runtime']}")


_SENTENCE_END = re.compile(r"(?<=[.!?\u0964])\s+")  # includes Hindi danda \u0964


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_END.split(text) if p.strip()]
    return parts


class ConversationPipeline:
    def __init__(self, config: AppConfig, registry: Optional[ModelRegistry] = None):
        self.config = config
        self.registry = registry or ModelRegistry()
        self.factory = ProviderFactory(config, self.registry)
        self.memory = MemoryStore()
        self.conversation_id = str(uuid.uuid4())

        self.asr = None
        self.llm = None
        self.tts = None
        self.gemini = None

    # --- lifecycle ---
    def load_local_models(self, on_status: Optional[Callable[[str], None]] = None) -> None:
        status = on_status or (lambda s: None)

        status("Loading speech recognition model...")
        self.asr = self.factory.build_asr(self.config.asr_model_id)
        self.asr.load()

        status("Loading local language model...")
        self.llm = self.factory.build_llm(self.config.llm_model_id)
        gen_settings = GenerationSettings(
            context_size=self.config.context_size,
            threads=self.config.threads,
            gpu_layers=self.config.gpu_layers,
        )
        self.llm.load(gen_settings)

        status("Loading voice model...")
        self.tts = self.factory.build_tts(self.config.tts_model_id)
        self.tts.load()

        status("Ready.")

    def maybe_load_gemini(self):
        if not self.config.gemini_enabled or self.config.offline_mode:
            self.gemini = None
            return
        from app.providers.gemini_provider import GeminiProvider
        self.gemini = GeminiProvider(self.config.gemini_model)
        self.gemini.load()

    def unload_all(self) -> None:
        for provider in (self.asr, self.llm, self.tts, self.gemini):
            if provider is not None:
                try:
                    provider.unload()
                except Exception:
                    pass

    # --- one full turn from raw audio to spoken response ---
    def run_turn_from_audio(self, audio_pcm16: bytes, sample_rate: int) -> TurnResult:
        transcript = self.asr.transcribe(audio_pcm16, sample_rate)
        return self.run_turn_from_text(transcript.text)

    def run_turn_from_text(self, user_text: str) -> TurnResult:
        if not user_text.strip():
            return TurnResult(user_text="", assistant_text="", routed_to="none", latency_ms={})

        self.memory.add_message(self.conversation_id, "user", user_text)

        gemini_enabled = bool(self.gemini) and not self.config.offline_mode
        decision = route(user_text, gemini_enabled=gemini_enabled)

        assistant_text, routed_to = self._answer(user_text, decision)

        self.memory.add_message(self.conversation_id, "assistant", assistant_text)
        return TurnResult(user_text=user_text, assistant_text=assistant_text, routed_to=routed_to, latency_ms={})

    def _build_messages(self, user_text: str) -> list[ChatMessage]:
        system_prompt = self.config.build_system_prompt()
        summary = self.memory.get_latest_summary(self.conversation_id)
        if summary:
            system_prompt += f"\n\nConversation summary so far:\n{summary}"

        recent = self.memory.get_recent_messages(self.conversation_id, limit=12)
        messages = [ChatMessage("system", system_prompt)]
        for m in recent[:-1]:  # exclude the message we just added, added separately below
            role = "assistant" if m["role"] == "assistant" else "user"
            messages.append(ChatMessage(role, m["content"]))
        messages.append(ChatMessage("user", user_text))
        return messages

    def _answer(self, user_text: str, decision) -> tuple[str, str]:
        gen_settings = GenerationSettings(
            max_output_tokens=self._max_tokens_for_response_length(),
            context_size=self.config.context_size,
        )

        if decision.decision == RouteDecision.LOCAL or self.gemini is None:
            messages = self._build_messages(user_text)
            return self.llm.generate(messages, gen_settings), "local"

        ok, reason = gemini_usage.within_budget(self.config.gemini_daily_limit, self.config.gemini_monthly_limit)
        if not ok:
            # Budget exhausted: fall back to local rather than blocking the user.
            messages = self._build_messages(user_text)
            return self.llm.generate(messages, gen_settings), f"local (gemini budget: {reason})"

        # Minimal context only -- never the full raw history (spec section 17).
        summary = self.memory.get_latest_summary(self.conversation_id) or ""
        gemini_settings = GenerationSettings(max_output_tokens=self.config.gemini_max_output_tokens)
        messages = [
            ChatMessage("system", self.config.build_system_prompt()),
        ]
        if summary:
            messages.append(ChatMessage("system", f"Conversation summary: {summary}"))
        messages.append(ChatMessage("user", user_text))

        try:
            reply = self.gemini.generate(messages, gemini_settings)
            approx_in = sum(len(m.content) for m in messages) // 4
            approx_out = len(reply) // 4
            gemini_usage.record_request(approx_in, approx_out)
            return reply, decision.decision.value
        except ModelLoadError:
            messages = self._build_messages(user_text)
            return self.llm.generate(messages, gen_settings), "local (gemini failed)"

    def _max_tokens_for_response_length(self) -> int:
        return {
            "very_concise": 40,
            "concise": 90,
            "natural": 200,
            "detailed": 500,
            "custom": self.config.gemini_max_output_tokens,
        }.get(self.config.response_length, 90)

    # --- speech output ---
    def speak(self, text: str, player, stop_check: Optional[Callable[[], bool]] = None) -> None:
        """
        Splits into sentences and synthesizes/plays incrementally so the user
        starts hearing the reply without waiting for the whole thing (spec section 27).
        `stop_check` lets the caller interrupt (barge-in) between sentences; interrupting
        mid-sentence requires a lower-level stop on the audio stream, see audio_io.Player.stop().
        """
        tts_settings = TTSSettings(language_hint=None if self.config.language_mode == "auto" else self.config.language_mode)
        for sentence in split_sentences(text) or [text]:
            if stop_check and stop_check():
                return
            pcm16, sample_rate = self.tts.synthesize(sentence, tts_settings)
            player.play(pcm16, sample_rate)
