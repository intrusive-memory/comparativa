---
type: project
state: draft
updated: 2026-08-15
---

# Comparativa — Requirements

**Status: decisions locked 2026-08-15 — awaiting final read-through, then
mission-supervisor breakdown**

A Python reference pipeline that parses Fountain screenplays and generates scripted
episode audio with MLX speech models, built to answer one question with controlled
evidence: **why does Produciesta/SwiftVoxAlta output sound worse than the Python
mlx-audio reference implementation, and what would fix it?**

---

## 1. Background

- The production pipeline (Produciesta → SwiftVoxAlta → mlx-audio-swift) generates
  podcast episode audio from Fountain scripts using Qwen3-TTS
  (`mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` / `0.6B`) with per-character `.vox`
  voice identities.
- Ad-hoc listening tests (2026-08-15) found Python `mlx-audio` output — same Qwen3-TTS
  1.7B checkpoint family, preset voices — **significantly better** than current
  Produciesta output.
- Two confounded suspects:
  1. **Voice design**: `.vox` voices are *synthesized from text prompts* (echada
     `cast`, provenance `method: synthesized`) rather than cloned from real
     recordings. Prompt-designed voices may simply be worse raw material.
  2. **Swift adaptation**: mlx-audio-swift's Qwen3-TTS port (sampling, quantization,
     codec decode, chunking, clone-prompt handling) may degrade output vs the Python
     reference.
- Key enabler for isolation: every `.vox` is a zip containing per-size
  `sample-audio.wav` and `clone-prompt.bin`. The Python pipeline can clone from the
  **identical reference audio** the Swift stack uses, on the **identical checkpoint**.

## 2. Goals

1. Build a Python CLI (`comparativa`) that: parses a Fountain episode → extracts the
   ordered (character, parenthetical, dialogue) sequence → maps characters to voices
   from a granville-style `CAST.md` + `voices/*.vox` → generates per-line audio via
   Python mlx-audio → assembles a full episode with silence gaps → emits audio + a
   per-line timing/provenance manifest (JSON).
2. Run a controlled comparison against the Swift pipeline on the granville corpus
   across **performance**, **audio quality**, and **ease of use**.
3. Produce a written verdict (`REPORT.md`) that says which suspect (voice design,
   Swift port, model ceiling) accounts for the quality gap, with audio evidence
   file-linked per condition.

## 3. Non-goals

- Not a production replacement for Produciesta; throwaway-quality code is acceptable
  where it doesn't bias the comparison.
- No Swift porting work in this project (that's the follow-on mission if the verdict
  implicates the port).
- No changes to the granville project, `.vox` format, or SwiftVoxAlta.
- No model training/finetuning; no CDN shipping.

## 4. Corpus

- Source: `~/Projects/podcasts/granville`
  - Scripts: `episodes/*.fountain` (audio-drama style: heavy NARRATOR, parentheticals,
    `[[notes]]`, centered titles, `(CONT'D)` suffixes).
  - Cast: `CAST.md` (YAML frontmatter, `cast:` list with `character`, `voicePrompt`,
    `voices.voxalta` paths).
  - Voices: `voices/*.vox` (zip: `manifest.json`, per-size `sample-audio.wav`,
    `clone-prompt.bin`).
  - Existing Swift-generated baseline audio: `audio/*.m4a` (+ `.vtt`).
- Iteration corpus: `episode_1_01_cold_open.fountain` + one bumper
  (`episode_1_01a_…`). Full-season sweep only in the final benchmark run.
- The corpus is read-only input; comparativa never writes into the granville tree.

## 5. Functional requirements

### 5.1 Fountain parsing
- FR-1: Parse Fountain via **jouvence** (fallback: `fountain-tools`) into an ordered
  element stream: scene headings, action, character cues, parentheticals, dialogue,
  notes, centered text.
- FR-2: Normalize character cues: strip `(CONT'D)` / `(V.O.)` / `(O.S.)` extensions;
  map cue → CAST.md character; NARRATOR is a first-class speaking character.
- FR-3: Non-spoken elements (action lines, `[[notes]]`, scene headings, `SHOT PROMPT`
  panels) are excluded from speech but preserved in the manifest for timing context.
- FR-4: Text preparation for TTS must be **documented and, where feasible, matched to
  Produciesta's semantics** (parenthetical handling, em-dashes, ellipses, ALL-CAPS
  words) so the two stacks speak the same strings. Divergences are listed in the
  report.

### 5.2 Voice resolution
- FR-5: Read `CAST.md` frontmatter; resolve each character to its `.vox`; extract
  `sample-audio.wav` (per model size) as the cloning reference. `clone-prompt.bin`
  is out of scope for Python (Swift-internal format) unless trivially decodable.
- FR-6: Support three voice modes per run: `vox-clone` (ref_audio = .vox
  sample-audio), `preset` (model built-in voices, mapped per character), and
  `ref-clone` (arbitrary external reference wav per character, for the
  "better raw material" probe).

### 5.3 Generation
- FR-7: Engines, all via pinned `mlx-audio` (Python, uv-managed, Python 3.12):
  - `qwen3-1.7b` — `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` (parity engine)
  - `qwen3-0.6b` — `…-0.6B-Base-bf16` (size-degradation probe)
  - `chatterbox` — `mlx-community/chatterbox-fp16` (model-ceiling probe)
  - `soprano` — `mlx-community/Soprano-1.1-80M-bf16` (lightweight probe; also ported
    in mlx-audio-swift, enabling a second Swift-vs-Python cross-check)
  - `chatterbox-turbo` — optional, speed probe
- FR-8: Per-line generation with fixed, recorded sampling params (temperature,
  top_p/top_k, repetition penalty, max_tokens); params logged in the manifest.
  Truncation must be detected (duration sanity check vs word count) and retried once.
- FR-9: Episode assembly: concatenate line audio with configurable inter-line and
  inter-scene gaps; loudness-normalize per line (simple RMS/LUFS target); output
  `wav` + `m4a`; write `manifest.json` with per-line char/text/engine/voice/params/
  duration/offset.
- FR-10: Deterministic re-runs where the engine allows seeding; otherwise record that
  seeding is unavailable.

### 5.4 Benchmarking & comparison
- FR-11: Conditions matrix (per episode) — all conditions run in round 1, listened
  side-by-each:
  | Cond | Stack | Engine/ckpt | Voices |
  |---|---|---|---|
  | A | Swift (Produciesta, regenerated fresh) | Qwen3 1.7B bf16 | .vox (current) |
  | B | Python comparativa | same checkpoint | vox-clone (same refs) |
  | C | Python comparativa | same checkpoint | preset voices |
  | D | Python comparativa | Chatterbox fp16 | cloned from same refs |
  | E | Python comparativa | Soprano 1.1 80M bf16 | preset (cloning only if supported) |
  | F (stretch) | Swift (mlx-audio-swift Soprano) | Soprano | preset |
  - A vs B isolates the **Swift port** (same model, same voice material).
  - B vs C isolates **voice design** (same stack, different voice material).
  - B/C vs D/E isolates the **model ceiling**.
  - E vs F (if F is feasible via existing Swift tooling without new build work) is a
    second, independent Swift-vs-Python port check on a different architecture.
- FR-12: Performance metrics per condition: wall-clock per episode, real-time factor,
  model load time, peak RSS (via `/usr/bin/time -l` or psutil). Swift side measured
  by timing the existing `produciesta` CLI export on the same machine, same session.
- FR-13: Quality protocol: human blind A/B — the tool emits a randomized,
  filename-blinded listening set + scoring sheet (naturalness, prosody, artifacts,
  voice consistency across lines, character distinctness); scores entered manually;
  tool tabulates. Optional objective proxies (STT WER round-trip, e.g. UTMOS) only
  if they run locally without new infrastructure.
- FR-14: Ease-of-use is a short written comparison in the report: setup steps, LOC of
  the integration layer, API friction notes for both stacks.
- FR-15: `REPORT.md` deliverable: results tables, verdict per hypothesis
  (voice-design / swift-port / model-ceiling), and a recommended next mission.

## 6. CLI shape (indicative)

```
comparativa parse   <episode.fountain>                 # dump element stream JSON
comparativa voices  <project-dir>                      # resolve cast → refs, audition each voice
comparativa generate <episode> --engine qwen3-1.7b --voices vox-clone -o out/
comparativa bench   <project-dir> --episodes ep1 --conditions A,B,C,D
comparativa listen  <bench-dir>                        # build blinded A/B set + score sheet
comparativa report  <bench-dir>                        # tabulate scores + metrics → REPORT.md
```

## 7. Constraints & environment

- Apple Silicon only; models from local HF cache (`~/.cache/huggingface`); no
  re-downloads of already-cached checkpoints.
- Python 3.12 via `uv`; project managed with `pyproject.toml` + `uv.lock`;
  `mlx-audio` version pinned (record the exact version in the report).
- Audio artifacts and model files are **never committed**; `.gitignore` covers
  `out/`, `bench/`, `*.wav`, `*.m4a`.
- Swift-side runs use existing signed binaries (`produciesta`) — no rebuilds; build
  tooling rules (no `swift build`) don't apply here since this repo is Python.

## 8. Risks / candor

- **Preset-voice halo**: today's "Python sounds better" impression compared preset
  voices to designed voices. Condition B may reveal the Python stack sounds equally
  rough with .vox material — that's a *useful* outcome (verdict: voice design).
- **Sampling-param mismatch**: if Swift and Python defaults differ, A-vs-B isn't a
  pure port comparison; params must be extracted from SwiftVoxAlta's
  `GenerationSettings.swift` and mirrored.
- **Chatterbox truncation**: observed 11.3s output for a ~20s paragraph in the first
  smoke test; long-line chunking needs explicit handling before D is trustworthy.
- **mlx-audio STT is broken** in the installed release (whisper processor bug) —
  WER-based objective scoring may need `mlx-whisper` directly or be dropped.

## 9. Decisions (resolved 2026-08-15)

1. **Project name**: `comparativa` — confirmed.
2. **Corpus scope**: one episode + one bumper (`episode_1_01_cold_open` +
   `episode_1_01a_bumper_donnie_and_arnie_1`); full season out of scope for round 1.
3. **Engine scope**: Chatterbox **and** Soprano both included in round 1 — all
   conditions generated and listened side-by-each in a single benchmark pass.
4. **Swift baseline**: regenerate condition A fresh with current Produciesta
   binaries; the 2026-08-09 `audio/*.m4a` files are reference-only.
