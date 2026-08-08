# Roadmap

Phase 1 (this drop) implements the architecture and a real end-to-end local voice loop.
Remaining phases, in the order they build on each other:

## Phase 2
- [ ] Wire `router.py` fully into the UI (currently used by `pipeline.py` but not exposed as
      per-message "Ask locally / Ask Gemini / Auto" override in the UI)
- [ ] Gemini provider: budget limits (daily/monthly request caps), usage dashboard
- [ ] Long-term memory UI (view/edit/delete individual memories)
- [ ] Context auto-summarization triggered at N% of context window

## Phase 3
- [ ] Real barge-in: needs a VAD running concurrently with TTS playback, tested on real hardware
- [ ] Sentence-level streaming from LLM into TTS (skeleton exists in `pipeline.py`)
- [ ] Developer/debug overlay (latency breakdown, tokens/sec, RAM/CPU/GPU live)
- [ ] Resource monitor + "switch to lightweight model" prompt under memory pressure

## Phase 4
- [ ] Friend personality: natural-language description → structured prompt (currently a
      simpler template-based version exists in `config.py::build_system_prompt`)
- [ ] Import/export friend config + memory + settings as JSON
- [ ] System tray mode
- [ ] Full settings UI for all sections listed in the spec (currently only Setup Wizard +
      a basic Models/Audio panel exist)

## Phase 5
- [ ] Installer polish (icon, code signing if you have a cert)
- [ ] Crash recovery / atomic-write audit pass
- [ ] Full automated test suite per the spec's testing section (only a starter set exists in `tests/`)
- [ ] Portable mode (no install, run from a folder)

None of these require architecture changes — the provider/router/memory/config abstractions
in Phase 1 were designed to absorb all of the above without rewrites.
