# Friend — Local-First AI Voice Companion (Phase 1)

A privacy-first, Hindi/English/Hinglish voice assistant that runs primarily on your own PC.
Local models are the default. Gemini is optional and only used when you explicitly allow it.

> **Status: Phase 1 (functional MVP).** This repo implements the core end-to-end pipeline
> (hardware detection → model registry → local ASR → local LLM → local TTS → playback),
> the provider abstraction layer, and Windows packaging. Phases 2–5 from the original spec
> (streaming, interruption, memory/summarization UI, tray, full settings surface) are designed
> for in the architecture but not all wired into the UI yet — see [ROADMAP.md](ROADMAP.md).

---

## Why the model choices differ from "just use Qwen3 for everything"

Qwen3-ASR and Qwen3-TTS are real, Apache-2.0, high quality models — but they were benchmarked
here as GPU-oriented. Community CPU benchmarks for Qwen3-TTS show ~3–5x real-time generation
even on a strong desktop CPU, which is unusable for "graceful degradation on weak hardware."

So the default ("Safer"/"Recommended") tier uses the standard lightweight offline-voice stack:

| Role | Default (CPU-friendly)              | Higher quality (GPU recommended) |
|------|--------------------------------------|-----------------------------------|
| ASR  | faster-whisper (int8, CTranslate2)   | Qwen3-ASR 0.6B / 1.7B             |
| LLM  | Qwen3 GGUF via llama.cpp             | Larger Qwen3 GGUF / more layers on GPU |
| TTS  | Piper (ONNX)                         | Qwen3-TTS 0.6B / 1.7B             |

This maps directly onto the app's own "Best fit / Safer / Higher quality" recommendation engine —
you can switch tiers per component in Settings → Models at any time. Nothing is hard-coded; see
`app/models.json`.

**Verify before you rely on it:** model URLs, VRAM numbers, and CPU speed all drift. The registry
in `app/models.json` records what was verified and when — check `verified_on` per entry and
re-verify before a release if it's been more than a few months.

---

## Architecture

```
app/
  main.py                 Entry point
  config.py                Settings persistence (JSON, atomic writes)
  hardware.py               CPU/RAM/GPU/VRAM/disk detection (psutil + GPUtil/WMI)
  model_registry.py         Model catalog + Best fit/Safer/Higher-quality recommender
  models.json                The model catalog (edit this to add models — no code changes needed)
  downloader.py             Threaded downloader: pause/resume/checksum/retry
  router.py                 Local-first request classifier (LOCAL vs GEMINI)
  memory.py                 Short-term + long-term memory, local SQLite
  pipeline.py               Orchestrates mic -> VAD -> ASR -> router -> LLM -> TTS -> speaker
  audio_io.py                Mic capture / playback / VAD
  providers/
    base.py                  ASRProvider / LLMProvider / TTSProvider / VADProvider interfaces
    asr_faster_whisper.py, asr_qwen3.py
    llm_llamacpp.py
    tts_piper.py, tts_qwen3.py
    gemini_provider.py
  ui/
    main_window.py           PySide6 chat + mic UI
    setup_wizard.py           First-run hardware scan + model picker
```

Every provider implements the same interface (`app/providers/base.py`), so adding a new model
means adding a `models.json` entry + one provider class — the rest of the app doesn't change.

---

## Requirements

- Windows 10/11 (primary target), Python 3.11 for development
- ~500 MB free for the app itself; models are downloaded separately, sizes shown before download

## Development setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m app.main
```

## Building the Windows .exe

**Option A — locally on Windows:**
```bash
pip install pyinstaller
pyinstaller build.spec
# → dist/Friend/Friend.exe
```

**Option B — GitHub Actions (recommended):**
Push a tag like `v0.1.0`. `.github/workflows/build-windows.yml` runs on `windows-latest`,
installs dependencies, runs PyInstaller, zips the result, and attaches it to a GitHub Release
automatically. No local Windows machine needed.

```bash
git tag v0.1.0
git push origin v0.1.0
```

The installer/zip does **not** contain AI models — those download on first run via the Model
Manager, exactly per the "keep the installer lightweight" requirement.

## Adding a new model

Add an entry to `app/models.json` with real, verified download info (never invented URLs —
see section on verification above), then implement/extend the matching provider class in
`app/providers/`. No other file needs to change.

## Adding a new provider (e.g. a different LLM runtime)

Subclass the relevant interface in `app/providers/base.py` and register it in
`app/pipeline.py`'s provider factory.

## Privacy

Audio, transcripts, memory, and conversation history stay on disk locally
(`%APPDATA%\Friend\`) unless you explicitly enable Gemini. When Gemini is used, only the
minimum context (current message + summary + relevant memory) is sent — never full history.
See `app/router.py` and `app/providers/gemini_provider.py`.

## Known limitations of this Phase 1 drop

- UI covers core chat/voice loop + a first-run wizard; the full settings surface (44 sections
  in the original spec) is not all built yet — `config.py` already has fields for most of it.
- Streaming LLM→TTS sentence-chunking and barge-in interruption are stubbed with clear `TODO`s
  in `pipeline.py` — the hooks exist, the real-time audio interrupt logic needs testing on real
  hardware with a mic, which isn't possible in this sandbox.
- Tray mode, hot-swap-without-restart, and the diagnostics viewer are not yet implemented.
- I could not test on actual Windows hardware, a real microphone, or a GPU in this environment —
  treat this as a solid, real foundation that needs a pass on your machine, not a finished product.
