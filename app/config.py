"""
Settings persistence. Plain JSON (portable, human-editable, easy to export/import
as a "Friend" config per the spec) written atomically so a crash mid-save can't
corrupt it -- write to a temp file, then os.replace() over the real one.
"""
from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


def app_data_dir() -> Path:
    if platform.system() == "Windows":
        base = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
    else:
        base = str(Path.home() / ".local" / "share")
    d = Path(base) / "Friend"
    d.mkdir(parents=True, exist_ok=True)
    return d


CONFIG_PATH = app_data_dir() / "config.json"


@dataclass
class PersonalityConfig:
    description: str = (
        "A warm, casual friend who talks naturally in whatever mix of Hindi and English "
        "the user uses, keeps answers short unless asked for detail, and remembers what "
        "the user tells them."
    )
    friendliness: int = 70
    humor: int = 50
    formality: int = 20
    energy: int = 60
    emotional_expression: int = 60
    talkativeness: int = 40
    patience: int = 80
    curiosity: int = 60
    sarcasm: int = 20
    seriousness: int = 30
    advanced_system_prompt: Optional[str] = None  # overrides everything above when set


@dataclass
class AppConfig:
    # Models
    asr_model_id: str = "faster-whisper-small"
    llm_model_id: str = "qwen3-0.6b-q4"
    tts_model_id: str = "piper-hi-en-medium"
    models_dir: str = str(app_data_dir() / "Models")

    # Performance
    resource_protection_pct: int = 60
    speed_mode: str = "balanced"  # fast | balanced | quality | custom
    memory_strategy: str = "balanced"  # aggressive_unload | balanced | keep_loaded
    context_size: int = 4096
    threads: Optional[int] = None
    gpu_layers: int = 0

  
    # Language / conversation
    language_mode: str = "auto"  # auto | en | hi | hinglish
    response_length: str = "concise"  # very_concise | concise | natural | detailed | custom

    # Voice
    push_to_talk: bool = True
    continuous_listening: bool = False
    allow_interruption: bool = True
    vad_sensitivity: float = 0.5
    silence_timeout_ms: int = 900

    # Memory
    memory_enabled: bool = True
    conversation_history_enabled: bool = True

    # Gemini
    gemini_enabled: bool = False
    gemini_model: str = "gemini-2.5-flash"
    gemini_max_output_tokens: int = 200
    gemini_daily_limit: int = 10
    gemini_monthly_limit: int = 100
    gemini_use_for_current_info: bool = True
    gemini_use_for_difficult: bool = True
    gemini_use_for_web: bool = True
    gemini_use_as_fallback: bool = True

    # Privacy
    offline_mode: bool = False

    # Personality
    personality: PersonalityConfig = field(default_factory=PersonalityConfig)

    def build_system_prompt(self) -> str:
        """Turns the personality config into a system prompt. Advanced override wins."""
        p = self.personality
        if p.advanced_system_prompt:
            return p.advanced_system_prompt

        traits = []
        if p.humor > 60:
            traits.append("You joke around sometimes.")
        if p.sarcasm > 60:
            traits.append("You use light sarcasm occasionally.")
        if p.formality < 30:
            traits.append("You speak casually, not formally.")
        if p.curiosity > 60:
            traits.append("You ask the user follow-up questions when it feels natural.")
        if p.energy > 70:
            traits.append("Your tone is upbeat and energetic.")
        elif p.energy < 30:
            traits.append("Your tone is calm and low-key.")

        length_map = {
            "very_concise": "Keep replies to a single short sentence unless asked for more.",
            "concise": "Keep replies to 1-3 short sentences unless asked for more.",
            "natural": "Reply at whatever length feels natural for the question.",
            "detailed": "Feel free to give thorough, detailed answers.",
            "custom": "",
        }
        lang_map = {
            "auto": "Mirror the user's language -- if they write in Hindi, English, or a mix, reply the same way. Don't force translation.",
            "en": "Always reply in English.",
            "hi": "Always reply in Hindi.",
            "hinglish": "Reply in natural Hinglish (mixed Hindi/English).",
        }

        return (
            f"{p.description}\n\n"
            f"{lang_map.get(self.language_mode, lang_map['auto'])}\n"
            f"{length_map.get(self.response_length, length_map['concise'])}\n"
            + (" ".join(traits))
        ).strip()


def load_config() -> AppConfig:
    if not CONFIG_PATH.exists():
        cfg = AppConfig()
        save_config(cfg)
        return cfg
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        personality_data = data.pop("personality", {})
        cfg = AppConfig(**{k: v for k, v in data.items() if k in AppConfig.__dataclass_fields__})
        cfg.personality = PersonalityConfig(**{k: v for k, v in personality_data.items() if k in PersonalityConfig.__dataclass_fields__})
        return cfg
    except Exception:
        # Corrupt config: don't crash the app, fall back to defaults and keep the
        # broken file around for diagnostics instead of silently overwriting it.
        backup = CONFIG_PATH.with_suffix(".json.broken")
        try:
            CONFIG_PATH.replace(backup)
        except Exception:
            pass
        cfg = AppConfig()
        save_config(cfg)
        return cfg


def save_config(cfg: AppConfig) -> None:
    tmp_path = CONFIG_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, CONFIG_PATH)  # atomic on both Windows and POSIX
