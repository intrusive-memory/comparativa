---
type: doc
---

# VOICE_CLONING.md — round 2: `.vox`-cloned voices across every engine

Round 1 (RD-2) was defaults-only: every Python condition used its engine's
built-in voices, and the report had to flag A-vs-C as confounded by voice
design. Round 2 removes that confound. Every clone-capable engine now speaks
in the **production voices** — the same `.vox` bundles the Swift stack
(Produciesta / SwiftVoxAlta) renders with — giving the matrix a clean
Swift-vs-Python Qwen3 pair (**A vs B**) and a cross-model clone probe
(**B vs G**).

Everything here was implemented against read-only inspection of the Swift
sources; nothing in the Swift stack was modified.

## 1. What a `.vox` actually contains

A `.vox` is a **zip archive** written by SwiftVoxAlta's `VoxExporter`
(`SwiftVoxAlta/Sources/SwiftVoxAlta/VoxExporter.swift`), produced for a cast by
`echada cast`. Layout (vox_version 0.4.0):

```
manifest.json                                  # voice name/description, provenance
embeddings/qwen3-tts/1.7b/sample-audio.wav     # engine-generated reference clip (24 kHz mono)
embeddings/qwen3-tts/1.7b/clone-prompt.bin     # serialized VoiceClonePrompt
embeddings/qwen3-tts/0.6b/sample-audio.wav     # (older exports may lack the 0.6b pair)
embeddings/qwen3-tts/0.6b/clone-prompt.bin
```

`clone-prompt.bin` is `VoiceClonePrompt.serialize()` output
(`mlx-audio-swift/Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTSVoiceClonePrompt.swift`):

```
[4 bytes little-endian metadata length]
[JSON metadata: refText, language, hasEmbedding, refCodesSize, speakerDataSize]
[refCodes safetensors]                # encoded reference audio codes [1, 16, T]
[speaker-embedding safetensors]       # x-vector, when hasEmbedding
```

The load-bearing discovery: the JSON header carries **`refText` — the
transcript of `sample-audio.wav`**. Python mlx-audio's Qwen3 ICL cloning
requires `ref_audio` *and* `ref_text`, and the bundle supplies both. The
safetensors payloads are Swift-side caches; Python re-encodes the reference
audio itself, so only the header is parsed
(`comparativa/voices/vox.py::parse_clone_prompt_header`).

## 2. Per-engine clone capability (mlx-audio 0.4.8)

| engine | checkpoint | mechanism | needs ref text | fallback without `.vox` |
|---|---|---|---|---|
| `qwen3-1.7b-clone` | `Qwen3-TTS-12Hz-1.7B-Base-bf16` | ICL (`ref_audio` + `ref_text`, `qwen3_tts.py::_generate_icl`) | yes | **none** — Base has no preset speakers; planning fails loudly |
| `qwen3-0.6b-clone` | `Qwen3-TTS-12Hz-0.6B-Base-bf16` | same | yes | none (same) |
| `chatterbox` | `chatterbox-fp16` | `prepare_conditionals(ref, sr)` → `generate(conds=…)` | no | built-in `conds.safetensors` voice |
| `chatterbox-turbo` | `chatterbox-turbo-fp16` | `prepare_conditionals` stores `model._conds` | no | built-in voice |
| `soprano` | `Soprano-80M-bf16` | **cannot clone** — one baked-in voice | — | its single voice, recorded as `mode: default` |

Prepared conditioning is cached **per clone name** in the engine layer
(`Engine._conds_cache` / `_clone_audio_cache`), so an episode pays the
reference-encoding cost once per character, not once per line — including
interleaved characters on chatterbox-turbo, whose cached `Conditionals` are
restored onto the model between lines.

The qwen3 clone engines mirror SwiftVoxAlta's own clone path (Base checkpoint,
ICL prompt from the same reference), which is what makes A-vs-B a port
comparison rather than a voice-design comparison.

### Cache status caveat

`Qwen3-TTS-12Hz-1.7B-Base-bf16` is in the local HF cache (4.2 GB, verified
2026-08-19). **`0.6B-Base-bf16` is refs-only** — `qwen3-0.6b-clone` is fully
wired but fails cleanly offline until
`hf download mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16` is run.

## 3. Sampling: the one recorded divergence

Clone sampling starts from the same Swift-parity values as round 1
(`docs/SAMPLING_PARITY.md`), with one exception: mlx-audio's ICL path floors
the repetition penalty at 1.5 (`icl_rep_penalty = max(repetition_penalty, 1.5)`,
to prevent codec degeneration after the long reference prefill), while Swift's
default is 1.3. `QWEN3_ICL_SAMPLING` records **1.5** — the value that actually
runs — with a `source` string explaining why, so every manifest states the
truth rather than the intention. This divergence must be mentioned wherever
A-vs-B is scored.

## 4. `presets-cloned.yaml` (schema_version 2)

```sh
uv run comparativa voices corpus/frozen --mode cloned --write   # → presets-cloned.yaml
```

Per character × engine, an entry is one of:

- `{mode: clone, vox, vox_sha256, model_size, member, ref_seconds, ref_sample_rate, ref_text, language}`
- `{mode: default, voice: default}` (soprano; chatterbox family without a `.vox`)
- `null` — unresolved (qwen3 clone engines without a `.vox`); the command exits
  1 and `generate` refuses to plan that character.

The document is **deterministic** (no timestamps; byte-identical regeneration)
and records the cast sha, so staleness shows up as a diff. `generate` verifies
each bundle's sha at plan time and marks the provenance `stale: true` on
mismatch. Schema-1 `presets.yaml` continues to work everywhere; resolution is
schema-aware (`comparativa/voices/cloned.py::resolve_voice_entry`).

Manifest additions: `voices.mode` (`defaults` / `cloned`) and
`voices.clones.<CHARACTER>` carrying the full clone provenance per character.

## 5. Corpus facts worth knowing (frozen snapshot)

- All 25 cast members have a readable `.vox` with a 1.7b entry and a non-empty
  `refText`; condition B covers the whole cast.
- `KEVIN`'s reference is **4.96 s**, just under chatterbox-turbo's documented
  “should be > 5 seconds” guidance. It clones anyway; the generated document
  carries a listen-for-drift note.
- `narrator.vox` (lowercase on disk, found case-insensitively) is an older
  export with **no 0.6b pair**; `qwen3-0.6b-clone` substitutes the 1.7b
  reference and notes the substitution (`size_substituted: true`).

## 6. Running round 2

```sh
# one-off, any script, any engine
HF_HUB_OFFLINE=1 uv run comparativa generate <episode.fountain> \
    --engine qwen3-1.7b-clone        # auto-selects presets-cloned.yaml
HF_HUB_OFFLINE=1 uv run comparativa generate <episode.fountain> \
    --engine chatterbox --presets presets-cloned.yaml   # cloned chatterbox (G)

# the matrix (B joined the default set)
uv run comparativa bench --conditions B,G
```

Smoke coverage: `tests/smoke/test_clone_smoke.py` (one cloned line per clone
engine, plus a conditionals-cache reuse check). Unit coverage:
`tests/test_vox.py`, `tests/test_cloned_voices.py` — the synthetic fixtures
reproduce the exact Swift serialization framing.
