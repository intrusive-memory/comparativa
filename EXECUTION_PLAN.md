---
type: execution-plan
---

# EXECUTION_PLAN.md — Comparativa

## Terminology

> **Mission** — A definable, testable scope of work. Defines scope, acceptance criteria, and dependency structure.

> **Sortie** — An atomic, testable unit of work executed by a single autonomous AI agent in one dispatch. One aircraft, one mission, one return.

> **Work Unit** — A grouping of sorties (package, component, phase).

## Mission Scope

Build an **all-Python** reference pipeline (`comparativa`) that parses Fountain screenplays and generates episode audio with MLX speech models, then run a controlled comparison against the current Swift stack (Produciesta/SwiftVoxAlta) on the granville corpus. The Swift stack is measured, never modified: condition A is regenerated with the existing signed `produciesta` binary; stretch condition F uses mlx-audio-swift's Soprano only if it runs with existing tooling.

Source requirements: `REQUIREMENTS.md` (decisions locked 2026-08-15), amended by user decisions of 2026-08-15 recorded in § Resolved Decisions below — most notably **round 1 is defaults-only**: every Python condition uses each engine's built-in/default voices; voice cloning (`.vox`-clone, ref-clone) is deferred to a follow-on mission.

**Grounded facts (verified at breakdown time)**:
- Corpus: `~/Projects/podcasts/granville` — `episodes/episode_1_01_cold_open.fountain`, `episodes/episode_1_01a_bumper_donnie_and_arnie_1.fountain`, `CAST.md`, `voices/*.vox` all present. Read-only input.
- `uv 0.12.1` installed at `/opt/homebrew/bin/uv`.
- Python `mlx-audio` **0.4.8** is the version in the local uv cache (used for the smoke test); it includes models for qwen3-tts, chatterbox, chatterbox_turbo, and soprano.
- HF cache has: `Qwen3-TTS-12Hz-1.7B-Base-bf16`, `1.7B-CustomVoice-bf16`, `0.6B-Base-bf16`, `0.6B-CustomVoice-bf16`, `chatterbox-fp16`, `chatterbox-turbo-fp16`, `Soprano-80M-bf16`.
- Swift baseline binary: `~/.local/bin/produciesta` (signed, no rebuilds).
- Sampling-parity source: `~/Projects/package-collection/pkg/SwiftVoxAlta/Sources/SwiftVoxAlta/GenerationSettings.swift` (read-only).
- mlx-audio-swift Soprano port pins `mlx-community/Soprano-80M-bf16` (`~/Projects/package-collection/pkg/mlx-audio-swift/Sources/MLXAudioCore/AudioModelManager.swift:51`).
- Python mlx-audio Soprano (`mlx_audio/tts/models/soprano/soprano.py`): loader accepts any HF repo id; non-"soprano-1.1" paths select the correct v1 decoder config; the `voice` parameter is unused — **Soprano has exactly one built-in voice, no cloning, no presets** (every character shares it in condition E/F; acceptable for a port/quality probe).

## Conditions Matrix (round 1, defaults-only)

| Cond | Stack | Engine/ckpt | Voices |
|------|-------|-------------|--------|
| A | Swift (Produciesta, regenerated fresh) | Qwen3 1.7B bf16 | `.vox` (current production voices, as-is) |
| C | Python comparativa | qwen3-1.7b | built-in preset voices, auto-assigned per character |
| D | Python comparativa | chatterbox fp16 | engine default voice |
| E | Python comparativa | Soprano-80M-bf16 | single built-in voice |
| F (stretch) | Swift (mlx-audio-swift Soprano) | Soprano-80M-bf16 | single built-in voice |

- **E vs F** is the clean Swift-vs-Python **port** comparison (same checkpoint, same default voice, different stack).
- **A vs C** compares the production stack to the Python stack but **confounds** the Swift port with voice design (`.vox` vs presets) — the report must state this explicitly.
- **C vs D/E** probes the **model ceiling**.
- Condition B (qwen3 cloned from `.vox` refs) is **deferred** to the follow-on custom-voices mission; it remains the future clean Qwen3 port test.
- qwen3-0.6b stays available as an engine (size-degradation probe, used in integration tests) but is not a listened round-1 condition.

## Work Units

| Work Unit | Directory | Sorties | Layer | Dependencies |
|-----------|-----------|---------|-------|--------------|
| Foundation | `.` (repo root) | 1 | 1 | none |
| Parsing | `src/comparativa/parsing/` | 2 | 2 | Foundation |
| Voices | `src/comparativa/voices/` | 1 | 2 | Foundation |
| Generation | `src/comparativa/generation/` | 2 | 3 | Parsing, Voices |
| Benchmark | `src/comparativa/bench/` | 2 | 4 | Generation |
| Evaluation | `src/comparativa/eval/` | 3 | 5 | Benchmark |

Layer 2 work units (Parsing, Voices) are independent of each other and may run in parallel.

## Parallelism Structure

**Critical Path**: Sortie 1 → Sortie 2 → Sortie 3 → Sortie 5 → Sortie 6 → Sortie 7 → Sortie 8 → Sortie 10 → Sortie 11 (length: 9 sorties)

**Parallel Execution Groups**:
- **Group 1** (sequential): Sortie 1 — **SUPERVISING AGENT ONLY** (creates the uv environment: `uv sync`, lockfile commit)
- **Group 2** (can run in parallel, after Sortie 1):
  - Parsing: Sorties 2 → 3 (Agent 1)
  - Voices: Sortie 4 (Agent 2)
- **Group 3** (sequential, after Group 2): Sortie 5 → Sortie 6 (Agent 1)
- **Group 4** (can run in parallel, after Sortie 6):
  - Benchmark: Sorties 7 → 8 (Agent 1)
  - Evaluation tooling: Sortie 9 (Agent 2)
- **Group 5** (sequential, after Group 4): Sortie 10, then Sortie 11 (deferred on human scores)

**Agent Constraints**:
- **Supervising agent**: Owns every step that mutates the shared environment or lockfile — `uv sync`, `uv add`, any `pyproject.toml` dependency change (the Python equivalent of a build step). Sortie 1 is supervising-agent-only for this reason; if any later sortie discovers a missing dependency, it must report back rather than run `uv add` itself.
- **Sub-agents (up to 4; this plan needs at most 2 concurrent)**: May run `uv run pytest` and `uv run comparativa ...` for verification (read-only use of the env), but perform **no environment mutations and no Swift builds** (Sortie 8 invokes only the existing signed binary).
- Maximum concurrency: 2 agents (Groups 2 and 4). Layer barriers are hard: Group N+1 never starts before every sortie in Group N is verified COMPLETED.

---

## Work Unit: Foundation

### Sortie 1: Project scaffold

**Priority**: 33.5 — blocks all 10 downstream sorties (dep depth 10); establishes the uv env, CLI skeleton, and test harness every other sortie reuses.

**Entry criteria**:
- [ ] First sortie — no prerequisites

**Tasks**:
1. Create `pyproject.toml`: Python 3.12, uv-managed, deps pinned to exact versions: `mlx-audio==0.4.8` (the version validated by the smoke test; record it — it goes in the report per §7 of requirements), `jouvence`, `psutil`, `pyloudnorm`, `soundfile`/`numpy` as needed, `pytest` (dev).
2. Run `uv sync`; commit `uv.lock`.
3. Package skeleton `src/comparativa/` with a `comparativa` console-script entry point exposing six stub subcommands — `parse`, `voices`, `generate`, `bench`, `listen`, `report` — each printing help and exiting 0.
4. `.gitignore` covering `out/`, `bench/`, `*.wav`, `*.m4a`, `.venv/`, `__pycache__/`.
5. Minimal pytest (`tests/test_cli.py`) asserting the CLI entry point loads and lists all six subcommands.

**Exit criteria**:
- [ ] `uv run comparativa --help` exits 0 and lists all six subcommands
- [ ] `uv.lock` exists and is committed
- [ ] `uv run pytest` exits 0
- [ ] `.gitignore` contains `out/`, `bench/`, `*.wav`, `*.m4a`

---

## Work Unit: Parsing

### Sortie 2: Fountain element stream + `parse` command

**Priority**: 29 — blocks 8 sorties; the element stream is the foundational data structure for all text prep and generation; moderate risk (new parser library against real corpus).

**Entry criteria**:
- [ ] Sortie 1 exit criteria met (CLI skeleton, uv env)

**Tasks**:
1. Implement a jouvence-based parser wrapper producing an ordered element stream — scene headings, action, character cues, parentheticals, dialogue, `[[notes]]`, centered text — with source-line references (FR-1). Fall back to `fountain-tools` only if jouvence fails on the corpus files; record which parser is active.
2. Implement `comparativa parse <episode.fountain>` emitting the element stream as JSON (stdout or `-o` file).
3. pytest tests against both corpus episodes (`episode_1_01_cold_open.fountain`, `episode_1_01a_bumper_donnie_and_arnie_1.fountain`): assert nonzero dialogue count, element ordering, and known first/last dialogue lines.

**Exit criteria**:
- [ ] `uv run comparativa parse ~/Projects/podcasts/granville/episodes/episode_1_01_cold_open.fountain` exits 0 and emits JSON containing >0 dialogue elements
- [ ] `uv run pytest tests/test_parse.py` exits 0
- [ ] No file writes inside the granville tree

### Sortie 3: Cue normalization + TTS text preparation

**Priority**: 26 — blocks 7 sorties; text-prep parity is a stated comparison-validity risk (§8.2-adjacent); requires read-only archaeology in two Swift codebases.

**Entry criteria**:
- [ ] Sortie 2 exit criteria met (element stream available)

**Tasks**:
1. Cue normalization (FR-2): strip `(CONT'D)` / `(V.O.)` / `(O.S.)` extensions; map cue → CAST.md character name; NARRATOR is a first-class speaking character.
2. Speech classification (FR-3): action lines, `[[notes]]`, scene headings, `SHOT PROMPT` panels marked `spoken: false` — excluded from speech, preserved in the stream for timing context.
3. Text preparation module (FR-4): read Produciesta/SwiftVoxAlta text-prep semantics from `~/Projects/apps/Produciesta` and `~/Projects/package-collection/pkg/SwiftVoxAlta` sources (read-only) — parenthetical handling, em-dashes, ellipses, ALL-CAPS words — implement matched transforms; write `docs/TEXT_PREP.md` documenting each matched behavior and every divergence.
4. Tests: every character cue in both corpus episodes resolves to a CAST.md character (unresolved list is empty).

**Exit criteria**:
- [ ] `uv run pytest tests/test_textprep.py` exits 0
- [ ] `docs/TEXT_PREP.md` exists and contains a divergence table (may be empty)
- [ ] `uv run comparativa parse` on each of the two corpus episodes reports an empty `unresolved_cues` list in its JSON output

---

## Work Unit: Voices

### Sortie 4: CAST.md roster + default-voice assignment and `voices` command

**Priority**: 25 — blocks 7 sorties; `presets.yaml` is a committed artifact consumed by every generation condition; runs in parallel with Sorties 2–3 (layer 2).

**Entry criteria**:
- [ ] Sortie 1 exit criteria met (CLI skeleton, uv env)

**Tasks**:
1. Parse `CAST.md` YAML frontmatter `cast:` list (`character`, `voicePrompt`, `voices.voxalta`) into a character roster (FR-5, reduced: `.vox` extraction is out of scope this round — defaults-only decision; `voices.voxalta` paths are recorded in the roster for the future cloning mission but never opened).
2. Enumerate each engine's available built-in/default voices from mlx-audio 0.4.8: qwen3 preset speakers (CustomVoice checkpoints), chatterbox default conditioning, soprano single voice; record per-engine voice lists in code.
3. Default-voice assignment (Resolved Decision RD-2): deterministically auto-assign each character an available, appropriate built-in voice per engine using gender/age keywords from `voicePrompt`; engines with a single voice (soprano, chatterbox default) assign it to all characters; write the full assignment to a committed `presets.yaml`.
4. `comparativa voices <project-dir>`: print a table of character → per-engine assigned voice; nonzero exit on unresolvable entries.
5. Tests: all granville cast members appear in `presets.yaml` with an assignment for every engine.

**Exit criteria**:
- [ ] `uv run comparativa voices ~/Projects/podcasts/granville` exits 0 with every cast member assigned a voice for every engine
- [ ] `presets.yaml` exists, is committed, and covers the full cast × engine matrix
- [ ] `uv run pytest tests/test_voices.py` exits 0
- [ ] No writes inside the granville tree

---

## Work Unit: Generation

### Sortie 5: Engine layer + sampling parity

**Priority**: 24 — blocks 6 sorties; highest-risk sortie (four MLX engines, external checkpoint loading, RD-1 residual check); the unified engine interface is reused by generate, bench, and audition.

**Entry criteria**:
- [ ] Sortie 3 and Sortie 4 exit criteria met (prepared text + voice assignments)

**Tasks**:
1. Extract Swift sampling defaults (temperature, top_p/top_k, repetition penalty, max_tokens) from `~/Projects/package-collection/pkg/SwiftVoxAlta/Sources/SwiftVoxAlta/GenerationSettings.swift` (read-only); write `docs/SAMPLING_PARITY.md` and use them as the qwen3 engine defaults (Risk §8.2).
2. Unified engine interface over `mlx-audio==0.4.8` (FR-7): `qwen3-1.7b`, `qwen3-0.6b`, `chatterbox`, `soprano` (checkpoint `mlx-community/Soprano-80M-bf16` per RD-1), optional `chatterbox-turbo`; per-engine capability flags (preset voices available, seeding supported). Confirm in the smoke test that Python mlx-audio loads the mlx-community bf16 Soprano conversion cleanly (RD-1 residual check).
3. Per-line generation with fixed, recorded sampling params; seed where the engine allows, otherwise record "seeding unavailable" (FR-8, FR-10).
4. Truncation detection (FR-8, Risk §8.3): duration sanity check vs word count; retry once and flag in the line record; explicit long-line chunking for chatterbox.
5. Smoke tests: generate one fixed line on each of the four engines using only locally cached checkpoints; assert sane duration and logged params.

**Exit criteria**:
- [ ] Smoke script generates one line on each of `qwen3-1.7b`, `qwen3-0.6b`, `chatterbox`, `soprano` with no network downloads of new checkpoints
- [ ] All smoke outputs pass the duration sanity check
- [ ] `docs/SAMPLING_PARITY.md` exists listing each parameter name and value with its `GenerationSettings.swift` source line

### Sortie 6: Episode assembly + `generate` command

**Priority**: 20 — blocks 5 sorties; assembly + manifest schema is reused by bench and eval; unlocks both remaining layers (Sortie 9 becomes dispatchable in parallel with 7–8).

**Entry criteria**:
- [ ] Sortie 5 exit criteria met (engines generate single lines)

**Tasks**:
1. Assembly (FR-9): concatenate line audio with configurable inter-line and inter-scene gaps; per-line loudness normalization to **−16 LUFS integrated via `pyloudnorm`** (RD-3).
2. Outputs: episode `.wav` + `.m4a` (via `afconvert`); `manifest.json` with per-line character, text, engine, assigned voice, sampling params, seed, duration, offset, truncation-retry flag.
3. `comparativa generate <episode> --engine <e> -o out/` end-to-end (voices always resolved from `presets.yaml`; no cloning flags this round).
4. `comparativa voices --audition`: generate one fixed sentence per character × engine assignment (uses the engine layer).
5. Integration test: full cold-open episode via `qwen3-0.6b`; assert episode duration ≈ Σ(line durations) + gaps (±0.1 s/line tolerance) and manifest line count == spoken-line count from parse.

**Exit criteria**:
- [ ] `uv run comparativa generate` on the cold open exits 0 producing `.wav`, `.m4a`, and `manifest.json` in `out/`
- [ ] Manifest line count equals the spoken-line count reported by `parse`
- [ ] `uv run pytest tests/test_generate.py` exits 0

---

## Work Unit: Benchmark

### Sortie 7: `bench` runner + performance metrics (conditions C–E)

**Priority**: 13 — blocks 3 sorties; fixes the `metrics.json` schema that Sortie 8's Swift-side entries must match, so it must precede 8.

**Entry criteria**:
- [ ] Sortie 6 exit criteria met (end-to-end generate works)

**Tasks**:
1. Encode the round-1 conditions matrix (see § Conditions Matrix) as config: C (qwen3-1.7b presets), D (chatterbox default), E (soprano default); optional chatterbox-turbo speed probe.
2. Performance capture per condition run (FR-12): wall-clock per episode, real-time factor, model load time, peak RSS via psutil; write `metrics.json` in a schema shared with Swift-side entries.
3. `comparativa bench <project-dir> --episodes <ids> --conditions C,D,E` orchestrating generate per condition into `bench/<cond>/<episode>/`.
4. `--dry-run` mode validating condition + episode resolution without generating; covered by a test.

**Exit criteria**:
- [ ] `bench --conditions C,E` on the bumper episode completes exit 0 with per-condition audio, manifest, and `metrics.json`
- [ ] `uv run pytest tests/test_bench.py` (dry-run coverage) exits 0

### Sortie 8: Swift baselines — condition A (and stretch F)

**Priority**: 8.5 — blocks 2 sorties; moderate risk (external signed binary, condition-F runnability unknown) but low complexity; runs in parallel with Sortie 9.

**Entry criteria**:
- [ ] Sortie 7 exit criteria met (bench dir schema and metrics.json schema fixed)

**Tasks**:
1. Wrapper script invoking the existing signed `~/.local/bin/produciesta` to regenerate condition A audio fresh for both corpus episodes into `bench/A/`, timing wall-clock and peak RSS via `/usr/bin/time -l` (no rebuilds; Decision §9.4 — the 2026-08-09 `audio/*.m4a` are reference-only).
2. Record produciesta version and write `bench/A/metrics.json` in the shared schema.
3. Condition F: run mlx-audio-swift's Soprano via existing Swift tooling only — no build work; checkpoint is `mlx-community/Soprano-80M-bf16` on both stacks (RD-1, already verified as the Swift port's pinned repo). If no existing runnable tool wraps the Soprano port, write `bench/F/SKIPPED.md` stating the specific blocker.

**Exit criteria**:
- [ ] `bench/A/` contains freshly regenerated audio + `metrics.json` for both corpus episodes
- [ ] `bench/F/` contains either outputs + `metrics.json` or `SKIPPED.md` with a stated reason
- [ ] No Swift builds were performed: `grep -rE "xcodebuild|swift build|swift-build" <condition-A/F wrapper scripts>` returns no matches, and only pre-existing signed binaries were invoked

---

## Work Unit: Evaluation

### Sortie 9: `listen` + `report` tooling

**Priority**: 10 — blocks 2 sorties; pure-Python tooling on fixture data (low risk); its early entry gate (only Sortie 6) makes it the primary parallelism win — dispatch alongside Sorties 7–8.

**Entry criteria**:
- [ ] Sortie 6 exit criteria met (manifests exist to build fixtures from); may run in parallel with Sorties 7–8

**Tasks**:
1. `comparativa listen <bench-dir>` (FR-13): randomized, filename-blinded listening set (opaque-id copies), a key file stored separately, and a scoring-sheet CSV with columns for naturalness, prosody, artifacts, voice consistency across lines, character distinctness.
2. `comparativa report <bench-dir>`: tabulate scoring CSV + all `metrics.json` files into results tables; render `REPORT.md` skeleton with templated verdict sections; unblind via the key file.
3. Objective proxy (optional per FR-13): STT WER round-trip via `mlx-whisper` directly (installed mlx-audio STT is broken per Risk §8.4); include only if it runs locally with cached models, otherwise record "dropped" with reason in the report skeleton.
4. Tests: `listen` on a synthetic fixture bench dir yields blinded names recoverable via the key; `report` renders tables from fixture scores.

**Exit criteria**:
- [ ] `uv run pytest tests/test_eval.py` exits 0
- [ ] `listen` + `report` round-trip on fixture data produces a `REPORT.md` skeleton containing results tables

### Sortie 10: Round-1 benchmark execution

**Priority**: 5.5 — blocks only Sortie 11; execution-heavy (long wall-clock, few decisions); risk is runtime failures on the full matrix, not design.

**Entry criteria**:
- [ ] Sortie 7, Sortie 8, and Sortie 9 exit criteria met

**Tasks**:
1. Run the full round-1 matrix — A, C, D, E, plus F if Sortie 8 found it runnable — on both corpus episodes (Decision §9.2/§9.3: one episode + one bumper, all conditions in one pass).
2. Generate the blinded listening set + empty scoring sheet via `listen`.
3. Populate performance-metric tables in the report skeleton; draft the ease-of-use comparison (FR-14): setup steps, LOC of each integration layer, API friction notes for both stacks.

**Exit criteria**:
- [ ] `bench/` contains audio + manifest + metrics for every non-skipped condition on both episodes
- [ ] Blinded listening set and empty scoring sheet exist
- [ ] `uv run comparativa report bench/` exits 0 and the rendered `REPORT.md` skeleton contains a performance-table row for every non-skipped condition (A, C, D, E, and F unless `bench/F/SKIPPED.md` exists) on both episodes

### Sortie 11: REPORT.md verdict (deferred — human scores required)

**Priority**: 1.5 — terminal sortie; gated on an external condition (human listening scores) — hold in PENDING without FATAL escalation while waiting.

**Entry criteria**:
- [ ] Sortie 10 exit criteria met
- [ ] User has completed the scoring sheet (external condition — do NOT escalate to FATAL while waiting)

**Tasks**:
1. Tabulate human scores via `report`.
2. Write the verdict per hypothesis (FR-15) using the defaults-only evidence: **swift-port** (E vs F — the clean pair; A vs C only with the voice-design confound stated), **model-ceiling** (C vs D/E), **voice-design** (partial evidence only this round — a definitive voice-design verdict requires the deferred condition B; say so explicitly). Each verdict cites file-linked audio evidence per condition.
3. Write the recommended next mission section — including whether the deferred custom-voices mission (condition B/D-cloned) is warranted by what round 1 showed.

**Exit criteria**:
- [ ] `REPORT.md` exists with a verdict (or explicit "insufficient evidence this round" finding) for all three hypotheses
- [ ] Every verdict cites file-linked audio evidence per condition
- [ ] All results tables are populated from the scores CSV and metrics files

---

## Resolved Decisions

Decisions made by the user on 2026-08-15 resolving breakdown's open questions:

### RD-1: Soprano checkpoint = `mlx-community/Soprano-80M-bf16` (resolves OQ-1)
Research-backed: the mlx-audio-swift Soprano port hardcodes this repo id (`AudioModelManager.swift:51`); Python mlx-audio 0.4.8's Soprano loader accepts arbitrary repo ids and selects the correct v1 decoder config for non-"soprano-1.1" paths; the checkpoint is already in the local HF cache. The requirements' `Soprano-1.1-80M-bf16` reference is treated as an erratum — 1.1 would need a new download and would break E-vs-F parity. Residual check in Sortie 5: confirm the bf16 conversion loads cleanly in Python.

### RD-2: Defaults-only voices in round 1 (resolves OQ-2, supersedes FR-6/FR-11 in part)
Every Python condition uses each engine's built-in/default voices, auto-assigned per character where the engine offers a choice (committed `presets.yaml`). Voice cloning (`vox-clone`, `ref-clone`, conditions B and cloned-D) is deferred to a follow-on custom-voices mission. Consequence, acknowledged: E-vs-F is round 1's only clean port comparison; A-vs-C confounds port with voice design and the report must state it.

### RD-3: Loudness target = −16 LUFS integrated via `pyloudnorm` (resolves OQ-3)
Applied per line in Python assembly; the Swift condition-A audio is measured with the same meter so the report can state any loudness delta between stacks.

## Open Questions

<!-- Consumed by Pass 1 of refine (`refine-blockers`). -->

_No blocking open questions remain — all three breakdown questions were resolved by the user on 2026-08-15 (see Resolved Decisions RD-1..RD-3)._

---

## Summary

| Metric | Value |
|--------|-------|
| Work units | 6 |
| Total sorties | 11 |
| Open questions | 0 (3 resolved → RD-1..RD-3) |
| Dependency structure | 5 layers; Parsing ∥ Voices in layer 2; Sortie 9 ∥ Sorties 7–8 |
| Critical path | 9 sorties (1→2→3→5→6→7→8→10→11) |
| Parallelism | 1 supervising agent + up to 2 concurrent sub-agents (Groups 2 and 4) |
| Refined | 2026-08-15 — all 5 refinement passes complete; plan ready to execute |
