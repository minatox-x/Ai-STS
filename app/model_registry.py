"""
Loads app/models.json and implements the "Best fit / Safer / Higher quality"
recommendation engine described in the spec (section 4). Never maximizes model
size -- reserves a resource-protection margin for the OS and the other two
pipeline stages (a machine running ASR+LLM+TTS needs room for all three).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.hardware import HardwareProfile

CATALOG_PATH = Path(__file__).parent / "models.json"

ModelType = Literal["asr", "llm", "tts", "vad"]


@dataclass
class ModelEntry:
    raw: dict

    def __getattr__(self, item):
        try:
            return self.raw[item]
        except KeyError as e:
            raise AttributeError(item) from e

    @property
    def id(self) -> str:
        return self.raw["id"]

    @property
    def ram_recommended_gb(self) -> float:
        return self.raw.get("ram_recommended_gb", self.raw.get("ram_minimum_gb", 0))

    @property
    def needs_gpu_for_practical_use(self) -> bool:
        return bool(self.raw.get("gpu_recommended")) and not self.raw.get("gpu_required")


class ModelRegistry:
    def __init__(self, catalog_path: Path = CATALOG_PATH):
        with open(catalog_path, "r", encoding="utf-8") as f:
            self._catalog = json.load(f)

    def list(self, model_type: ModelType) -> list[ModelEntry]:
        return [ModelEntry(m) for m in self._catalog.get(model_type, [])]

    def get(self, model_id: str) -> ModelEntry | None:
        for model_type in ("asr", "llm", "tts", "vad"):
            for m in self._catalog.get(model_type, []):
                if m["id"] == model_id:
                    return ModelEntry(m)
        return None

    def recommend(
        self,
        model_type: ModelType,
        hw: HardwareProfile,
        resource_protection_pct: int = 60,
    ) -> dict[str, ModelEntry | None]:
        """
        Returns {"safer": ModelEntry, "recommended": ModelEntry, "higher_quality": ModelEntry}.
        Any of these can be None if nothing in the catalog fits at all (e.g. disk full).

        `resource_protection_pct` is "max % of available RAM the whole AI stack may use" --
        we further split that three ways across ASR+LLM+TTS+VAD headroom, since all four may
        be resident at once depending on the memory strategy (see config.py MemoryStrategy).
        """
        budget_gb = hw.ram_available_gb * (resource_protection_pct / 100.0) / 3.0
        candidates = [m for m in self.list(model_type) if m.raw.get("download_size_gb", 0) <= hw.disk_free_gb]

        def fits(m: ModelEntry) -> bool:
            if m.needs_gpu_for_practical_use and not (hw.gpu.vendor == "nvidia" and (hw.gpu.vram_gb or 0) >= (m.raw.get("vram_gb") or 0)):
                return False
            return m.ram_recommended_gb <= budget_gb

        safer = min(
            (m for m in candidates if m.raw.get("tier") == "safer"),
            key=lambda m: m.ram_recommended_gb,
            default=None,
        )
        fitting = [m for m in candidates if fits(m)]
        recommended = max(fitting, key=lambda m: m.raw.get("quality_rating", 0), default=safer)
        higher_quality = max(
            (m for m in candidates if m.raw.get("tier") == "higher_quality"),
            key=lambda m: m.raw.get("quality_rating", 0),
            default=None,
        )

        return {"safer": safer, "recommended": recommended, "higher_quality": higher_quality}

    def recommend_all(self, hw: HardwareProfile, resource_protection_pct: int = 60) -> dict:
        return {
            model_type: self.recommend(model_type, hw, resource_protection_pct)
            for model_type in ("asr", "llm", "tts")
        }
