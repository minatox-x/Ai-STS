"""
Local-first request router. Classifies each message so the app defaults to the
local LLM and only reaches for Gemini when genuinely needed (current info,
web-dependent, or the user explicitly asked for it) -- this is what keeps
Gemini cost near zero per the spec.

This is a fast keyword/heuristic classifier, not an LLM call, so routing itself
costs nothing. It's intentionally simple and overridable -- see `override`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class RouteDecision(str, Enum):
    LOCAL = "local"
    GEMINI_CURRENT_INFO = "gemini_current_info"
    GEMINI_WEB = "gemini_web"
    GEMINI_COMPLEX = "gemini_complex"


_CURRENT_INFO_PATTERNS = [
    r"\btoday'?s?\b", r"\bcurrent\b", r"\blatest\b", r"\bright now\b", r"\bweather\b",
    r"\bnews\b", r"\bstock price\b", r"\bscore\b", r"\bwho is the (current|new) \b",
    r"\baaj\b", r"\babhi\b", r"\bmausam\b",
]
_WEB_PATTERNS = [r"\bsearch for\b", r"\blook up\b", r"\bfind (a|the) (link|website|article)\b"]
_COMPLEX_HINTS = [r"\banalyz", r"\bin depth\b", r"\bresearch\b", r"\bcompare .* (vs|versus)\b"]


@dataclass
class RouteResult:
    decision: RouteDecision
    reason: str


def route(
    message: str,
    *,
    gemini_enabled: bool,
    local_confidence_threshold: float = 0.35,
    local_llm_confidence: float | None = None,
) -> RouteResult:
    """
    `local_llm_confidence`, if provided (0-1), lets the pipeline route to Gemini when
    the local model itself signals low confidence (e.g. via a short self-rated
    preamble) -- optional, defaults to always trusting local for anything that
    doesn't match a current-info/web pattern.
    """
    text = message.lower()

    if not gemini_enabled:
        return RouteResult(RouteDecision.LOCAL, "Gemini disabled -- always local.")

    for pat in _CURRENT_INFO_PATTERNS:
        if re.search(pat, text):
            return RouteResult(RouteDecision.GEMINI_CURRENT_INFO, f"Matched current-information pattern: {pat}")

    for pat in _WEB_PATTERNS:
        if re.search(pat, text):
            return RouteResult(RouteDecision.GEMINI_WEB, f"Matched web-required pattern: {pat}")

    for pat in _COMPLEX_HINTS:
        if re.search(pat, text):
            return RouteResult(RouteDecision.GEMINI_COMPLEX, f"Matched complex-reasoning hint: {pat}")

    if local_llm_confidence is not None and local_llm_confidence < local_confidence_threshold:
        return RouteResult(RouteDecision.GEMINI_COMPLEX, "Local model reported low confidence.")

    return RouteResult(RouteDecision.LOCAL, "No cloud-requiring pattern matched -- handled locally.")
