---
type: supervisor-state
---

# SUPERVISOR_STATE.md — OPERATION BATTLING BARDS

## Terminology

> **Mission** — A definable, testable scope of work; the whole campaign.
> **Sortie** — An atomic, testable unit of work executed by a single autonomous AI agent in one dispatch.
> **Work Unit** — A grouping of sorties (package, component, phase).

## Mission Metadata

- Operation name: OPERATION BATTLING BARDS
- Iteration: 1
- Mission branch: `mission/battling-bards/01`
- Starting point commit: `84edd15cba0b7ac5f925feee60ffba0b554c18fe`
- Mission started: 2026-08-15T22:24:48Z
- max_retries: 3
- Pre-build clean: skipped (non-Swift — pure Python/uv project, no Package.swift or *.xcodeproj)
- Dependency graph: untouched (no floor bumps, no Package.resolved deletion, no SPM cache clear)

## Plan Summary

- Work units: 6
- Total sorties: 11
- Dependency structure: layers (5 layers; Parsing ∥ Voices in layer 2; Sortie 9 ∥ Sorties 7–8 in layers 4/5)
- Dispatch mode: dynamic (no template in plan)
- Max concurrency: 2 sub-agents; layer barriers are hard
- Environment rule: only ONE agent may mutate the uv env/lockfile (`uv sync`, `uv add`, pyproject dep changes). Parallel-phase sorties must report missing deps back to the supervisor, never run `uv add` themselves.

## Work Units

| Name | Directory | Sorties | Dependencies |
|------|-----------|---------|--------------|
| Foundation | `.` | 1 (Sortie 1) | none |
| Parsing | `src/comparativa/parsing/` | 2 (Sorties 2–3) | Foundation |
| Voices | `src/comparativa/voices/` | 1 (Sortie 4) | Foundation |
| Generation | `src/comparativa/generation/` | 2 (Sorties 5–6) | Parsing, Voices |
| Benchmark | `src/comparativa/bench/` | 2 (Sorties 7–8) | Generation |
| Evaluation | `src/comparativa/eval/` | 3 (Sorties 9–11) | Benchmark (Sortie 9 gates only on Sortie 6; Sortie 11 deferred on human scores) |

## Work Unit States

### Foundation
- Work unit state: COMPLETED
- Current sortie: 1 of 1
- Sortie state: COMPLETED
- Sortie type: code
- Model: opus
- Complexity score: 12 (+ force-opus override: foundation pattern-setter, 10 dependents)
- Attempt: 1 of 3
- Last verified: 2026-08-15T22:28Z — supervisor re-ran exit criteria: commit `baa61af`; `comparativa --help` exit 0 (six subcommands); `pytest` 11 passed; `uv.lock` committed. Pinned: mlx-audio==0.4.8, jouvence==0.4.2, psutil==7.2.2, pyloudnorm==0.2.0, soundfile==0.14.0, numpy==2.5.2, pytest==9.1.1 (CPython 3.12.13).
- Notes: Downstream sorties should import `comparativa.cli.SUBCOMMANDS` and replace stub handlers rather than re-declaring the CLI.

### Parsing
- Work unit state: COMPLETED
- Current sortie: 3 of 3 (sorties 2–3)
- Sortie state: COMPLETED
- Sortie type: code
- Model: opus
- Complexity score: 18
- Attempt: 1 of 3
- Last verified: Sortie 3 COMPLETED 2026-08-15T23:18Z — supervisor re-ran: 127 tests pass, TEXT_PREP.md has 7-row divergence table, unresolved_cues empty on both episodes, commit `70a59cb`. Ground truth: 208/208 transcript-line parity with Produciesta VTTs under `produciesta-parity` (test-asserted).
- Notes: Sortie 2 = `d21b271` (jouvence, dual-dialogue rescue). Swift stack does NOT transform em-dashes/ellipses/ALL-CAPS (matching = no-op). 7 divergences in `textprep.DIVERGENCES`. Sortie 4's test_cli.py watch item resolved at Sortie 4 verification.

### Voices
- Work unit state: COMPLETED
- Current sortie: 4 (single sortie)
- Sortie state: COMPLETED
- Sortie type: code
- Model: opus
- Complexity score: 15
- Attempt: 1 of 3
- Last verified: 2026-08-15T22:52Z — supervisor re-ran: voices exit 0 (25 chars × 5 engines), presets.yaml committed, full suite 64 passed, commit `d3fd625`. test_cli.py stub-help contract preserved (Sortie 2's watch item resolved).
- Notes: qwen3 presets: 9 speakers, but eric/dylan excluded (dialect-forcing); only 2 English presets, ZERO English female presets → condition-C quality caveat MUST go in REPORT.md. presets.yaml 0.6b entries flagged `voices_verified_locally: false`.

### Generation
- Work unit state: COMPLETED
- Current sortie: 6 of 5–6
- Sortie state: COMPLETED
- Sortie type: code
- Model: opus
- Complexity score: 17 (Sortie 6); Sortie 5 was 21
- Attempt: 1 of 3
- Sortie 6 verified: COMPLETED 2026-08-16T00:30Z — supervisor re-ran: out/{wav,m4a,manifest.json} present, manifest schema 1 / 189 lines / policy recorded / full sampling provenance, test_generate 32 passed, full suite 194 passed, commit `8269778`. Cold-open integration on 0.6b: RTF 0.53, timeline delta 0.001 s.
- Sortie 5 verified: COMPLETED 2026-08-15T23:48Z — after supervisor prefetched Soprano-80M-bf16 (217,333,883-byte safetensors verified), full offline smoke passes all 4 engines (qwen3-1.7b RTF 0.63, 0.6b 0.47, chatterbox 0.79, soprano 0.18); RD-1 ANSWERED: bf16 Soprano loads cleanly in Python, v1 decoder config selected. Commit `5679165`. 162 fast tests pass.
- Notes: Sortie 6 dispatched. Engine API: `load_engine(key)` / `generate_line(LineRequest)` → `LineResult` (to_dict = manifest record). Sample rates differ (qwen3/chatterbox 24000, soprano 32000). Engine layer owns intra-line silence; assembly owns inter-line/inter-scene only. Seeding = mlx global RNG, verified bit-identical.

### Benchmark
- Work unit state: COMPLETED
- Current sortie: 8 of 7–8
- Sortie state: COMPLETED
- Sortie type: code
- Model: opus
- Complexity score: 15
- Attempt: 1 of 3
- Sortie 7: COMPLETED 2026-08-16T01:12Z — bench C,E bumper run verified (C RTF 1.346 @ 4.4 GiB / E RTF 0.167 @ 526 MiB, parity policy); frozen validated metrics schema; commits `d17a385` + deflake continuation `0775d5b` (RSS test now deterministic, 3× consecutive green verified by supervisor; attempt counter unchanged — PARTIAL path).
- Notes: Accepted boundary deviation in Sortie 7: `.gitignore` `bench/` anchored to `/bench/` (unanchored rule hid `src/comparativa/bench/` from git). Subprocess-per-cell isolates peak RSS. Sortie 8 orders: never write in granville (copy project to scratch if produciesta lacks an output flag), no Swift builds, metrics via make_entry/write_metrics.

### Evaluation
- Work unit state: RUNNING
- Current sortie: 11 of 9–11
- Sortie state: PENDING (deferred on external condition: human listening scores — never FATAL)
- Sortie type: code (verdict writing, gated on scored bench/listen/scoring_sheet.csv)
- Model: TBD at Sortie 11 dispatch
- Complexity score: scored at dispatch
- Attempt: 1 of 3
- Last verified: Sortie 10b COMPLETED (verified post-04:09Z drain-window run) — supervisor re-ran all exit criteria: A cold-open metrics fresh (21:09 local), all 6 C/D/E cells fresh (21:21–21:38 local), `comparativa report bench/` exit 0 with 8 populated perf rows (A/C/D/E × both episodes), pytest 243 passed / 8 skipped, commit `b950d96` (metrics.py schema fix only). Clean quiet-machine RTFs: A 2.590/2.488, C 0.606/0.594, D 0.816/0.738, E 0.131/0.110 (cold-open/bumper).
- Notes: Sortie 10 (10a+10b) fully complete. Earlier contaminated timings superseded — contention affected ALL conditions (C bumper RTF 1.346→0.594), not just A. Sortie 11 remains deferred on the human scoring the 8-clip blinded set; audio is final. Carry-forwards binding on Sortie 11: (1) condition F skipped ⇒ swift-port hypothesis "insufficient evidence this round" with A-vs-C confounds stated; (2) zero English female qwen3 presets ⇒ condition-C quality caveat in REPORT.md.

## Active Agents

| Work Unit | Sortie | Sortie State | Attempt | Model | Complexity Score | Task ID | Output File | Dispatched At |
|-----------|--------|--------------|---------|-------|------------------|---------|-------------|---------------|
| (none — 10b agent completed and verified; Sortie 11 deferred on human scores) | | | | | | | | |

_USER DIRECTIVE 2026-08-16: run non-timed work now under GPU contention; run performance measurements only once the machine is quiet. Sortie 10 split: 10a = audio matrix completion + blinded listening package + ease-of-use draft (contended OK — seeded generation is deterministic, audio identical); 10b = timed re-runs (A cold-open + all Python cells) + final perf tables. **10b gate status corrected at 04:27Z resume: machine NOT quiet** — `produciesta export episode_1_03` running at ~58% CPU from the user's render_queue.sh session; the earlier "drained 05:55Z" note carried an impossible future timestamp and is disregarded. 10b holds as deferred (never increments attempts, never FATAL) until supervisor observes render_queue.sh and produciesta gone from ps._

## Decisions Log

| Timestamp | Work Unit | Sortie | Decision | Rationale |
|-----------|-----------|--------|----------|-----------|
| 2026-08-16T03:10Z | Benchmark | 8 | Sortie 8 verified COMPLETED: bench/A both episodes (produciesta 1.0.0, ckpt Qwen3-1.7B-Base), bench/F/SKIPPED.md (no built mlx-audio-swift artifact + Soprano absent from SwiftAcervo container), no Swift builds (grep clean). Commit `b0e748d`. | Cold-open A timing contaminated by concurrent GPU load — flagged for re-run in Sortie 10. |
| 2026-08-16T03:10Z | — | — | ALARM: external Claude session (render_queue.sh, granville cwd) is mutating the corpus mid-mission — MICKEY added to CAST.md, reference bumper m4a overwritten, bumper FOUNTAIN rewritten (live sha no longer matches benchmarked sha). NOT interfered with — it is the user's own session. | Corpus can no longer be treated as frozen input. |
| 2026-08-16T03:20Z | — | — | CORPUS FROZEN: extracted mission-consistent corpus from granville git history into `corpus/frozen/` (episodes at exact benchmarked SHAs — cold open `c909b1d`, bumper `d7dd1ea`; CAST.md+PROJECT.md+voices from pre-MICKEY `bfd6df5`). SHA match verified against bench/A metrics; `voices corpus/frozen` exits 0 against committed presets.yaml. presets.yaml NOT regenerated (would shift condition-C assignments mid-mission). | All remaining mission work (tests, Sortie 10 matrix, A re-run) uses `corpus/frozen/`; live granville is abandoned as an input. Non-destructive: read-only `git show`/`git archive` against the user's repo. |
| 2026-08-16T03:20Z | Evaluation | 11 | CARRY-FORWARD: condition F skipped ⇒ round 1 has NO clean Swift-vs-Python port pair; swift-port hypothesis must be recorded as "insufficient evidence this round" (A-vs-C carries voice-design + lyric-splitting + NARRATOR-casting + loudness + AAC-decode confounds per docs/SWIFT_BASELINE.md §5). | From Sortie 8 recon; binding on Sorties 10–11. |
| 2026-08-16T04:27Z | Evaluation | 10a | RESUME RECONCILIATION: 10a verdict PARTIAL. Done: full A/C/D/E matrix in bench/, blind.py envelope-parsing fix + 2 tests (uncommitted, suite 243 green), docs/EASE_OF_USE.md draft. Missing: blinded listening set + scoring sheet, commit. Continuation dispatched (sonnet, attempt unchanged). | Ground truth (git + bench/ contents) over stale Active Agents table; PARTIAL path per execution.md §7. |
| 2026-08-16T04:27Z | Evaluation | 10b | GATE CORRECTION: state file claimed render queue drained at 05:55Z — a future timestamp relative to now (04:27Z). ps shows produciesta export (episode_1_03) at ~58% CPU + render_queue.sh alive. 10b stays deferred. | Observed state wins over recorded state (execution.md §1 Step 5). Timed runs on a contended GPU would repeat Sortie 7/8's contaminated-timing failure. |
| 2026-08-16T04:30Z | Evaluation | 10a | Model: sonnet for continuation | PARTIAL continuation minimum-model rule; remaining work is well-defined (run listen, verify report exit 0, commit) — no opus needed. |
| 2026-08-16T04:30Z | — | — | Supervisor committed EXECUTION_PLAN.md frontmatter (feature_name/starting_point_commit/mission_branch/iteration) left uncommitted since `start`. | Supervisor-owned metadata from the initialization sequence; keeping it out of the sortie's commit keeps sortie diffs clean. |
| 2026-08-16T04:32Z | Evaluation | 10a | Sortie 10a verified COMPLETED: commit `ec56055`; 8-clip blinded set + separate key.csv + empty scoring sheet in bench/listen/; report exit 0 w/ 8 skeleton rows; 243 tests green. bench/ + REPORT.md left untracked per Sortie 7/8 convention. | Supervisor re-ran all four exit criteria independently; key.csv fields prove the envelope fix works on real data. |
| 2026-08-16T04:33Z | Evaluation | 10b | 10b gate still closed (produciesta export 1_03 at ~48% CPU); launched supervisor watcher polling every 30s (re-arms every ~9 min). Human scoring of the blinded set may proceed in parallel — audio is final. | Deferred external condition; polling never increments attempts (execution.md §7). |
| 2026-08-16T04:41Z | Evaluation | 10b | FALSE-CLEAR caught: instantaneous pgrep at 04:40:32Z showed no produciesta, but the queue had merely moved between episodes — a new export (episode_1_02) appeared seconds later at 50% CPU. Watcher hardened: DRAINED now requires 3 consecutive quiet polls (90 s sustained). 10b NOT dispatched. | Inter-episode gaps in render_queue.sh make single-sample checks unreliable; dispatching timed runs on a false clear would repeat the Sortie 7/8 contaminated-timing failure. |
| post-drain resume | Evaluation | 10b | Sortie 10b verified COMPLETED: condition A cold-open re-rendered on quiet machine (47 min wall, zero contention — supervisor's independent watcher confirmed no render_queue throughout); all 6 C/D/E cells re-run with --force; report exit 0 with 8 populated rows; 243 tests green; commit `b950d96`. Agent also found+fixed a real bug: `discover_metrics` read the schema-1 envelope flat instead of `entries[]→performance`, leaving the perf table blank. Contaminated-run comparison shows contention had inflated ALL conditions (C bumper 1.346→0.594), vindicating the deferred-gate discipline. | Supervisor re-ran every exit criterion independently; commit diff confined to src/comparativa/eval/metrics.py. |
| resume (post-05:09Z) | Evaluation | 10b | GATE OPENED: sustained-drain check passed — 3 consecutive quiet polls over 90 s (no produciesta, no render_queue.sh). 10b dispatched. Model: sonnet — score 5 (boundary), upgraded from haiku because timing integrity is the sortie's whole purpose and the agent must abort on reappearing contention. Sortie 11 still deferred: scoring_sheet.csv unscored (all 8 rows empty). | Hardened 90 s rule from the 04:41Z false-clear; deferred waits never increment attempts. |
| 2026-08-16T05:09Z | Evaluation | 10b | QUEUE INTEL (read-only peek at user's render_queue.sh v3): 6 pieces queued (1_02, bumpers 2/3/4, 1_04, 1_05; 1_03 excluded — being rewritten). Currently on piece 1 of 6. Estimated hours of contention remain. 10b stays deferred; supervisor keeps cycling ~9-min sustained-drain watchers. | Deferred waits are free (no attempt increments); timing integrity is the whole point of 10b. |

_(earlier log continues below)_

| Timestamp | Work Unit | Sortie | Decision | Rationale |
|-----------|-----------|--------|----------|-----------|
| 2026-08-15T22:24Z | — | — | Created root commit `84edd15` from REQUIREMENTS.md + EXECUTION_PLAN.md | Repo had an unborn HEAD (zero commits); a starting_point_commit and mission branch require one. Also renamed default branch master → main per repo convention. |
| 2026-08-15T22:24Z | — | — | THE RITUAL: OPERATION BATTLING BARDS (haiku) | Validated: ALL CAPS, 2 words, non-literal. |
| 2026-08-15T22:24Z | — | — | Pre-build clean skipped | Non-Swift project (no Package.swift / *.xcodeproj). Silent skip per execution.md §1a. |
| 2026-08-15T22:25Z | Foundation | 1 | Sortie 1 dispatched to a single background agent rather than executed by the supervisor | Plan marks Sortie 1 "supervising agent only" to guarantee exclusive env/lockfile access; skill.md forbids the supervisor writing production/test code and wins operational conflicts. Dispatching ONE agent with nothing else running preserves the exclusivity the plan intends. Env-mutation ban for parallel-phase sorties stands. |
| 2026-08-15T22:25Z | Foundation | 1 | Model: opus | Score 12 (complexity 5, ambiguity 0, foundation 5, risk 2) → sonnet band, but force-opus override applies: establishes core architectural patterns with 10 dependents. |
| 2026-08-15T22:52Z | Voices | 4 | PLAN FACT CORRECTED: qwen3-0.6B checkpoints are NOT in the HF cache (refs only, no blobs) — plan's grounded fact was wrong | Sortie 4 verified by direct cache inspection. Affects Sortie 5 smoke (no-download criterion) and Sortie 6 integration test. |
| 2026-08-15T22:53Z | — | — | Supervisor pre-fetching `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16` in background (hf download) | Environment provisioning is supervisor-owned (like uv sync). Only CustomVoice needed (presets are CustomVoice-only; Base has empty spk_id). Preserves Sortie 5's "no downloads during smoke" criterion in spirit. Reversible (cache entry, deletable). |
| 2026-08-15T23:05Z | — | — | 0.6B-CustomVoice prefetch complete: 2.3 GB snapshot verified in HF cache (config, safetensors, speech_tokenizer present) | Sortie 5's four-engine smoke can now run with zero network downloads. |
| 2026-08-15T23:19Z | Parsing | 3 | RULING: benchmark conditions C/D/E run with `--speech-policy produciesta-parity`; policy recorded in each manifest.json and stated in REPORT.md. FR-3 remains CLI default. | Sortie 3 proved Produciesta narrates action/sluglines/title cards (208 lines vs FR-3's 189). Strict FR-3 would benchmark two different episodes, defeating A-vs-C validity. Both policies implemented + tested → reversible. |
| 2026-08-15T22:53Z | Voices | 4 | REPORT CARRY-FORWARD: condition C has zero English female qwen3 presets (25-char cast served by ryan/aiden + Asian-language presets) | Quality confound beyond the acknowledged A-vs-C voice-design confound; strengthens the case for deferred condition B. Must appear in REPORT.md (Sorties 9–11). |

## Overall Status

Sorties 1–10 complete and verified (10a + 10b). The ONLY remaining work is Sortie 11 (REPORT.md verdict), deferred on the human scoring `bench/listen/scoring_sheet.csv` (8 clips, all rows still empty). Clean quiet-machine performance tables are rendered in REPORT.md. When the scoring sheet is filled, run `/mission-supervisor resume` to dispatch Sortie 11 and enter the post-mission flow (test-cleanup → brief → clean).
