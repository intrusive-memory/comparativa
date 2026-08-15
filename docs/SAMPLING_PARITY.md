---
type: reference
state: current
updated: 2026-08-15
---

# SAMPLING_PARITY.md — Swift `GenerationSettings` → Python `comparativa`

**Purpose.** Condition A (Swift/Produciesta) and condition C (Python/qwen3) are
supposed to measure the *port*, not the tuning. If the two stacks sample
differently, every quality difference is confounded and the comparison is worth
nothing (EXECUTION_PLAN.md Risk §8.2). This document records, value by value,
what the Swift stack does and what `comparativa` does about it.

**Source of truth (read-only).**
`~/Projects/package-collection/pkg/SwiftVoxAlta/Sources/SwiftVoxAlta/GenerationSettings.swift`
— read at mission time on 2026-08-15. `GenerationSettings.default` (line 118) is
`GenerationSettings()`, so the `init` defaults on **lines 97–113** are the values
production actually runs with. `VoiceLockManager.swift:255` logs exactly these
four sampling values, and `:296–329` passes them to the model, confirming that
nothing overrides them on the way down.

The machine-readable copy of this table is
`comparativa.generation.sampling.SWIFT_PARITY`; `tests/test_engines.py`
re-reads the Swift file and asserts that every value below is still on the line
it cites, and that this document mentions each one.

---

## 1. Qwen3 sampling parameters (matched)

| Parameter (Python) | Swift property | Value | Declared at | Default at | Notes |
|---|---|---|---|---|---|
| `temperature` | `temperature` | `0.7` | `GenerationSettings.swift:39` | `GenerationSettings.swift:98` | Swift doc comment calls 0.6–0.7 "balanced stability and naturalness". |
| `top_p` | `topP` | `0.9` | `GenerationSettings.swift:47` | `GenerationSettings.swift:99` | Nucleus threshold; Swift doc comment calls 0.9 "balanced". |
| `repetition_penalty` | `repetitionPenalty` | `1.3` | `GenerationSettings.swift:55` | `GenerationSettings.swift:100` | Top of the Swift "light penalty (recommended)" band (1.1–1.3). |
| `max_tokens` | `maxTokens` | `16384` | `GenerationSettings.swift:61` | `GenerationSettings.swift:101` | 12 Hz token rate; the Swift doc comment reads ≈22 minutes of audio. |

### `top_k` — the knob that only exists on the Python side

`GenerationSettings` has **no** top-k property, and neither does the Swift
generation entry point:
`mlx-audio-swift/Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTSVoiceClonePrompt.swift:230–238`
takes `temperature`, `topP`, `repetitionPenalty`, `maxTokens` and nothing else.

Python `mlx-audio` 0.4.8 defaults `top_k=50`
(`mlx_audio/tts/models/qwen3_tts/qwen3_tts.py:1158`), which would truncate the
candidate set in a way the Swift stack never does — a silent divergence in the
one comparison the mission cares about. `comparativa` therefore passes
**`top_k = 0`**, which the library treats as "disabled"
(`qwen3_tts.py:859` and `:934` both gate on `top_k > 0`).

### Chunking parameters (matched, applied in `comparativa.generation.engines`)

These are not sampling knobs, but they change what text reaches the model and
are part of the same settings object.

| Parameter | Swift property | Value | Declared at | Default at | Python mirror |
|---|---|---|---|---|---|
| auto-chunking on | `enableAutoChunking` | `true` | `GenerationSettings.swift:77` | `GenerationSettings.swift:102` | `EngineSpec.chunks_long_lines` |
| chunk target | `chunkTargetDuration` | `12.0` s | `GenerationSettings.swift:89` | `GenerationSettings.swift:103` | `parsing.textprep.CHUNK_TARGET_SECONDS` |
| inter-chunk pause | `chunkPauseDuration` | `0.25` s | `GenerationSettings.swift:95` | `GenerationSettings.swift:104` | `parsing.textprep.CHUNK_PAUSE_SECONDS` |

Chunk *boundaries* can still differ: Swift splits with Foundation's ICU sentence
segmenter and Python with a regex approximation (divergence D-5 in
`docs/TEXT_PREP.md`).

### Language token (matched)

SwiftVoxAlta resolves any English locale to `TTSLanguage.english`, whose
`modelName` is `"english"` (`TTSLanguage.swift:25`, `:41–47`), and threads it
into every generation call (`VoiceLockManager.swift:299`, `:325`). The granville
corpus is English, so `comparativa` passes `lang_code="english"`
(`generation.engines.QWEN3_LANGUAGE`) rather than mlx-audio's `"auto"`, which
emits no language token at all.

---

## 2. What is *not* matched, and why

| Topic | Swift (condition A) | Python (condition C) | Why |
|---|---|---|---|
| Voice source | ICL voice clone from a `.vox` reference (`Qwen3TTSVoiceClonePrompt.generateWithClonePrompt`) | CustomVoice **preset speaker** (`generate_custom_voice`) | RD-2: round 1 is defaults-only. This is the A-vs-C confound the report must state; the clean port pair this round is E vs F. |
| Checkpoint | `Qwen3-TTS-12Hz-1.7B-**Base**-bf16` (the clone path needs the base model) | `…-1.7B-**CustomVoice**-bf16` (presets are CustomVoice-only) | Follows from the line above, not an independent choice. |
| ICL penalty bump | n/a | `qwen3_tts.py` raises `repetition_penalty` to ≥1.5 in ICL mode | Not reached: the preset path never enters ICL mode, so 1.3 is what runs. |
| Text preparation | see `docs/TEXT_PREP.md` | see `docs/TEXT_PREP.md` | Divergences D-1…D-7 are registered there. |

---

## 3. Non-parity engines (library defaults, recorded verbatim)

Chatterbox and Soprano have no Swift counterpart in this mission, so there is
nothing to match; `comparativa` runs mlx-audio 0.4.8's own defaults and records
them so the report can state exactly what produced the audio.

| Engine | temperature | top_p | top_k | repetition_penalty | max tokens | extras | Source |
|---|---|---|---|---|---|---|---|
| `chatterbox` | `0.8` | `1.0` | — | `1.2` | `1000` | `min_p=0.05`, `cfg_weight=0.5`, `exaggeration=0.1` | `mlx_audio/tts/models/chatterbox/chatterbox.py:759-782` |
| `chatterbox-turbo` | `0.8` | `0.95` | `1000` | `1.2` | `800` | `min_p=0.0` | `mlx_audio/tts/models/chatterbox_turbo/chatterbox_turbo.py:780-797` |
| `soprano` | `0.3` | `0.95` | — | — | `512` | — | `mlx_audio/tts/models/soprano/soprano.py:362-371` |

Soprano exposes no top-k and no repetition penalty at all; its `voice` argument
is accepted and discarded (`soprano.py:387`).

---

## 4. Engine capability flags

`comparativa.generation.ENGINE_SPECS[key].capabilities()`:

| Engine | Checkpoint | Preset voices | Instruct | Seeding | Long lines chunked by |
|---|---|---|---|---|---|
| `qwen3-1.7b` | `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16` | yes (9 speakers, 7 assignable) | yes | yes | comparativa (12.0 s) |
| `qwen3-0.6b` | `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16` | yes (9 speakers, 7 assignable) | yes | yes | comparativa (12.0 s) |
| `chatterbox` | `mlx-community/chatterbox-fp16` | no (1 built-in voice) | no | yes | comparativa (12.0 s) |
| `chatterbox-turbo` | `mlx-community/chatterbox-turbo-fp16` | no (1 built-in voice) | no | yes | the model (sentence `split_pattern`) |
| `soprano` | `mlx-community/Soprano-80M-bf16` | no (1 built-in voice) | no | yes | the model (`_preprocess_text`) |

### Seeding (FR-10)

All five engines sample through MLX's **global** RNG, so
`mlx.core.random.seed(n)` immediately before the call pins the sample stream:

- qwen3 — `categorical_sampling` (`qwen3_tts.py:868`, `:944`)
- soprano — `mlx_lm.sample_utils.make_sampler` (`soprano.py:335`)
- chatterbox / turbo — `make_sampler` in `chatterbox/t3/t3.py:363`, plus
  `mx.random.normal` in the flow-matching decoder
  (`chatterbox/s3gen/flow_matching.py:47`)

No engine exposes a per-call generator or an explicit RNG key, so seeding is
process-global by construction: it is reproducible within a run, and reproducible
across runs only if the same calls happen in the same order. Every line record
carries the seed it used and how it was applied; if a future mlx-audio release
breaks the global-RNG assumption, set `EngineSpec.seeding = False` and the record
reads `"seeding unavailable"` instead.

`tests/smoke/test_engine_smoke.py::test_seeding_is_reproducible_on_the_smallest_engine`
verifies the claim empirically on `qwen3-0.6b` (same seed → bit-identical
waveform).

---

## 5. Truncation detection (Risk §8.3)

Autoregressive TTS fails silently in both directions, so every generated span is
duration-checked (`comparativa.generation.truncation`):

- expected = `max(words / 2.8, chars × 0.055, 0.30 s)` — the character rate is
  SwiftVoxAlta's own `estimatedSecondsPerChar` (`VoiceLockManager`), mirrored in
  `parsing.textprep.SECONDS_PER_CHAR`
- **truncated** when the audio is under `0.5 ×` expected (or effectively silent)
- **overrun** when it is over `4.0 ×` expected
- a flagged span is regenerated **once**, from a different point in the sample
  stream (`seed + 1_000_003`); the attempt closer to the expected duration is
  kept and the line record carries `truncation_retry: true` plus the rejected
  attempt's duration

The thresholds are deliberately loose. This is a detector for gross failure, not
a prosody judge; a WER round-trip (Sortie 9, optional) is the tighter check.

---

## 6. Reproducing

```bash
# Unit suite (fast, no checkpoints)
uv run pytest

# Engine smoke: one fixed line per engine, local checkpoints only
HF_HUB_OFFLINE=1 uv run pytest tests/smoke --smoke -s

# Or the script directly
HF_HUB_OFFLINE=1 uv run python -m comparativa.generation.smoke -o out/smoke
```

`HF_HUB_OFFLINE=1` is what makes "no checkpoint downloads" verifiable rather
than merely intended: with it set, `huggingface_hub` resolves from the local
cache or raises.

---

## 7. Verification status — 2026-08-15 (Sortie 5)

Smoke line: *"Granville is a small town with a long memory, and tonight it
remembers everything."* (14 words, expected ≈5.0 s), seed `20260815`,
`HF_HUB_OFFLINE=1` throughout.

| Engine | Audio | Expected | Ratio | Load | RTF | Verdict |
|---|---|---|---|---|---|---|
| `qwen3-1.7b` | 6.56 s | 5.0 s | 1.31 | 1.6 s | 0.65 | pass |
| `qwen3-0.6b` | 8.96 s | 5.0 s | 1.79 | 0.4 s | 0.48 | pass |
| `chatterbox` | 4.00 s | 5.0 s | 0.80 | 0.9 s | 0.81 | pass |
| `chatterbox-turbo` | 4.52 s | 5.0 s | 0.90 | 1.3 s | 0.45 | pass (optional engine) |
| `soprano` | — | — | — | — | — | **blocked** |

No truncation, no overrun, no retry on any passing engine.

**Soprano is blocked on a missing checkpoint, not on code.** The local HF cache
entry for `mlx-community/Soprano-80M-bf16` holds only `config.json`, `README.md`
and `.gitattributes` (16 KB total); `model.safetensors` (217,333,883 bytes),
`model.safetensors.index.json`, `tokenizer.json` and `tokenizer_config.json` are
absent, so the loader raises:

```
EngineError: could not load engine 'soprano' from checkpoint
'mlx-community/Soprano-80M-bf16': No weight files (safetensors or npz) found in
~/.cache/huggingface/hub/models--mlx-community--Soprano-80M-bf16/snapshots/b7da048eff3dfd556409a44b78b9c61a9dd4ccfa
```

The expected file list is in
`~/Projects/package-collection/pkg/acervo-manifests/models/soprano-tts-80m/manifest.json`.
Once the snapshot is complete, `HF_HUB_OFFLINE=1 uv run pytest tests/smoke
--smoke` closes both the smoke criterion and the RD-1 residual check
(`test_soprano_loads_the_rd1_bf16_conversion`), which asserts that the v1 decoder
config is selected for a non-`soprano-1.1` path — the specific compatibility
worry RD-1 raised.

**Seeding is confirmed empirically**, not merely assumed: `qwen3-0.6b` generated
a bit-identical waveform twice from seed `4242`
(`test_seeding_is_reproducible_on_the_smallest_engine`), and the two full smoke
runs above reproduced identical durations for every engine.
