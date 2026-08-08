"""
Tracks Gemini request counts locally so the app can enforce the daily/monthly
budget caps from Settings and show the usage dashboard, without claiming exact
$ cost (the app has no reliable live pricing feed -- showing a fabricated cost
number would violate the "don't pretend" rule).
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from app.config import app_data_dir

USAGE_PATH = app_data_dir() / "gemini_usage.json"


def _load() -> dict:
    if not USAGE_PATH.exists():
        return {"requests": []}
    try:
        with open(USAGE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"requests": []}


def _save(data: dict) -> None:
    tmp = USAGE_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(USAGE_PATH)


def record_request(estimated_input_tokens: int, estimated_output_tokens: int) -> None:
    data = _load()
    data["requests"].append({
        "ts": time.time(),
        "input_tokens": estimated_input_tokens,
        "output_tokens": estimated_output_tokens,
    })
    _save(data)


def _count_since(cutoff_ts: float) -> int:
    data = _load()
    return sum(1 for r in data["requests"] if r["ts"] >= cutoff_ts)


def requests_today() -> int:
    start_of_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    return _count_since(start_of_day)


def requests_this_month() -> int:
    now = datetime.now()
    start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()
    return _count_since(start_of_month)


def within_budget(daily_limit: int, monthly_limit: int) -> tuple[bool, str]:
    today = requests_today()
    month = requests_this_month()
    if today >= daily_limit:
        return False, f"Daily Gemini limit reached ({today}/{daily_limit})."
    if month >= monthly_limit:
        return False, f"Monthly Gemini limit reached ({month}/{monthly_limit})."
    return True, ""


def clear_statistics() -> None:
    _save({"requests": []})


def summary() -> dict:
    return {
        "requests_today": requests_today(),
        "requests_this_month": requests_this_month(),
    }
