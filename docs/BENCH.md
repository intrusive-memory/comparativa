---
type: reference
state: current
updated: 2026-08-15
---

# BENCH.md — conditions matrix, the bench runner, and the frozen `metrics.json`

Deliverable of OPERATION BATTLING BARDS, Sortie 7 (FR-12).

`comparativa bench` runs the round-1 conditions matrix and writes one
performance record per (condition, episode). **The `metrics.json` schema in § 4
is frozen**: Sortie 8's Swift-side entries (conditions A and F) and Sortie 9's
report both read it, so nothing in it may be Python-only.

**Implementation**: `src/comparativa/bench/conditions.py` (the matrix as
config), `src/comparativa/bench/metrics.py` (the schema),
`src/comparativa/bench/perf.py` (peak RSS, tool versions, host),
`src/comparativa/bench/runner.py` (resolution, layout, subprocess execution),
`src/comparativa/bench/command.py` (CLI). **Tests**: `tests/test_bench.py`.

---

## 1. Conditions

| id | stack | engine / checkpoint | voices | runner |
|---|---|---|---|---|
| A | swift | Qwen3 1.7B bf16 (Produciesta records the exact repo) | `.vox` production voices | external — Sortie 8 |
| C | python | `qwen3-1.7b` → `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16` | built-in presets per character (`presets.yaml`) | `bench` |
| D | python | `chatterbox` → `mlx-community/chatterbox-fp16` | engine default | `bench` |
| E | python | `soprano` → `mlx-community/Soprano-80M-bf16` | single built-in | `bench` |
| F | swift | `mlx-community/Soprano-80M-bf16` (RD-1, same as E) | single built-in | external — Sortie 8 |
| T | python | `chatterbox-turbo` → `mlx-community/chatterbox-turbo-fp16` | engine default | `bench`, optional speed probe |

`--conditions` defaults to `C,D,E`; `all` selects `A,C,D,E` (the non-optional
set). Condition **B** (qwen3 cloned from `.vox`) is deferred to the follow-on
custom-voices mission (RD-2) and deliberately has no entry.

A and F are declared here but refused by the runner — they exist so both sorties
use the same condition ids, labels, and `metrics.json` fields. Asking for only
A/F exits 2 pointing at Sortie 8.

### Speech policy — supervisor ruling

Every bench condition runs with **`--speech-policy produciesta-parity`**, and it
is the `bench` default. The Swift baseline narrates action lines and sluglines,
so an A-vs-C comparison at `fr3` would compare two different episodes. The
policy is recorded in each run's `manifest.json` *and* in its metrics entry;
two runs at different policies must never be pooled.

## 2. Layout

One run per output directory (the manifest filename is fixed):

```
bench/<condition>/<episode>/<episode>.wav
                           /<episode>.m4a
                           /manifest.json
                           /metrics.json      # one entry
                           /generate.log      # the child's stdout+stderr
bench/summary.json                            # every entry from one invocation
```

`<episode>` is the `.fountain` stem — **the join key across stacks**. Sortie 8
must use the same string or the report cannot pair A with C.

A cell that already has `metrics.json` + `manifest.json` is skipped; `--force`
re-runs it.

## 3. Why each run is a subprocess

Peak RSS is the only FR-12 number generation does not already measure, and it is
only meaningful if a run does not inherit the previous condition's resident
checkpoint — an in-process condition E after condition C would report qwen3's
1.7 B footprint as Soprano's. One process per cell also means a model that dies
taking the interpreter with it costs one cell rather than the matrix, and the
child's stdout (including `chatterbox-turbo`'s tqdm bar) lands in
`generate.log` rather than the operator's terminal. `TQDM_DISABLE=1` and
`HF_HUB_DISABLE_PROGRESS_BARS=1` silence it at the source as well.

Children run with **`HF_HUB_OFFLINE=1`**: nothing in a timed run may block on,
or be timed against, a network fetch.

Wall-clock, RTF, and model-load time are read from the manifest's `totals`
(`wall_seconds`, `real_time_factor`, `load_seconds`) rather than re-derived, so
manifest and metrics cannot disagree. `performance.process_wall_seconds` is the
bench-level figure (interpreter start, imports, encode, writes included) and is
always the larger of the two.

Peak RSS comes from a `psutil` sampler thread reading the child's process tree
every 50 ms (`perf.PeakRSS`). Polling can only under-report a spike shorter than
the interval; `ru_maxrss` of `RUSAGE_CHILDREN` was rejected because it is a
high-water mark across *all* reaped children and cannot be attributed to one run.

## 4. `metrics.json` — schema version 1 (frozen)

Document envelope, identical wherever it is written:

```json
{ "schema_version": 1, "generated_by": "...", "written_at": "...", "entries": [ … ] }
```

`metrics.load_entries(bench_dir)` globs `**/metrics.json` and de-duplicates on
`(stack, condition, episode)`, so **Sortie 8 may write either
`bench/A/metrics.json` with both episodes' entries or one file per run** — both
layouts are read correctly and a run appearing in both collapses to one.

### 4.1 Entry fields

Keys marked ✔ are enforced by `metrics.validate_entry` (which `write_metrics`
calls on every entry it writes).

| field | ✔ | meaning |
|---|---|---|
| `condition` | ✔ | `A` / `C` / `D` / `E` / `F` / `T` |
| `stack` | ✔ | `python` or `swift` |
| `label` | | human-readable condition description |
| `engine` | ✔* | engine key, or the Swift tool's name (`produciesta`) |
| `checkpoint` | ✔* | HF repo id actually loaded |
| `voices` | | how voices were chosen (`presets`, `.vox`, …) |
| `episode` | ✔ | the `.fountain` stem — the cross-stack join key |
| `episode_path`, `episode_sha256` | | proof both stacks were given the same input |
| `speech_policy` | ✔ | `produciesta-parity` for every round-1 bench condition |
| `status` | ✔ | `ok` / `failed` / `skipped` |
| `error` | | failure text when `status == "failed"` |
| `performance.wall_seconds` | ✔ | episode wall-clock (manifest `totals.wall_seconds`) |
| `performance.real_time_factor` | ✔ | generate-seconds per audio-second; lower is faster |
| `performance.model_load_seconds` | ✔ | checkpoint load time |
| `performance.peak_rss_bytes` | ✔ | peak resident set of the run's process tree |
| `performance.generate_seconds`, `process_wall_seconds`, `start_rss_bytes`, `rss_samples` | | extras; may be omitted |
| `audio.audio_seconds` | ✔ | voiced audio, excluding gaps |
| `audio.duration_seconds`, `gap_seconds`, `sample_rate` | | per-engine sample rate is passed through, not normalized |
| `lines.line_count` | ✔ | records in the manifest = spoken lines in the script |
| `lines.placed_line_count`, `script_line_count`, `truncated_lines`, `truncation_retry_lines`, `overrun_lines` | | |
| `loudness.mean_output_lufs` | ✔ | mean measured output loudness |
| `loudness.mean_shortfall_db`, `max_shortfall_db` | ✔ | peak-guard shortfall vs target (`null` on a stack that does not normalize) |
| `loudness.meter`, `target_lufs`, `peak_limited_lines`, `unnormalized_lines` | | |
| `tool_versions` | ✔ | free-form `{name: version-or-null}` |
| `host` | | platform, machine, cpu count, memory of the measuring machine |
| `run` | | `started_at`, `finished_at`, `command`, `returncode`, `log` |
| `outputs` | | `{wav, m4a, manifest}` |
| `notes` | | free-form strings (e.g. `"peak RSS from /usr/bin/time -l"`) |

\* at least one of `engine` / `checkpoint` must be non-null.

Every `performance` / `audio` / `lines` / `loudness` key is **present** even when
`null`, so the report's table gets a cell rather than a hole.

### 4.2 Building an entry from Swift (Sortie 8)

```python
from comparativa.bench import make_entry, write_metrics

entry = make_entry(
    condition="A", stack="swift",
    engine="produciesta", checkpoint="<repo id produciesta loaded>",
    voices=".vox", episode="episode_1_01a_bumper_donnie_and_arnie_1",
    speech_policy="produciesta-parity",
    wall_seconds=..., real_time_factor=..., model_load_seconds=...,
    peak_rss_bytes=...,            # /usr/bin/time -l "maximum resident set size"
    audio_seconds=..., line_count=...,
    loudness={"meter": "pyloudnorm.Meter (ITU-R BS.1770-4, 400 ms block)",
              "mean_output_lufs": ...},   # measure the Swift audio with OUR meter
    tool_versions={"produciesta": "<--version>", "macos": "..."},
    notes=["peak RSS from /usr/bin/time -l"],
)
write_metrics("bench/A/metrics.json", [entry], generated_by="sortie-8 condition A")
```

`make_entry` validates before returning, so a schema break fails at write time
rather than at report time. If a Swift stack cannot report `model_load_seconds`
separately, pass `None` — the key stays, the value is null, and the report shows
the gap honestly.

## 5. Commands

```bash
# Plan every cell — parse, cues, voices — without loading a model
uv run comparativa bench ~/Projects/podcasts/granville \
    --episodes episode_1_01a_bumper_donnie_and_arnie_1 --conditions C,D,E --dry-run

# Run it for real (children force HF_HUB_OFFLINE=1 themselves)
uv run comparativa bench ~/Projects/podcasts/granville \
    --episodes episode_1_01a_bumper_donnie_and_arnie_1 --conditions C,E

# The whole matrix on both corpus episodes (Sortie 10)
uv run comparativa bench ~/Projects/podcasts/granville --episodes all --conditions C,D,E

# The conditions table
uv run comparativa bench --list-conditions

uv run pytest tests/test_bench.py
```

Episode tokens resolve as: a path, then an id (`.fountain` stem), then a
filename, then a unique substring of an id (`bumper_donnie_and_arnie_1`);
`all` selects the whole project. An ambiguous substring is refused rather than
guessed.

## 6. Verification status — 2026-08-15 (Sortie 7)

`bench --conditions C,E` on `episode_1_01a_bumper_donnie_and_arnie_1`
(15 lines at `produciesta-parity`), exit 0, macOS 27 / arm64, mlx-audio 0.4.8,
mlx 0.32.0, Python 3.12.13:

| cond | engine | lines | audio s | wall s | RTF | load s | peak RSS | mean LUFS | mean shortfall |
|---|---|---|---|---|---|---|---|---|---|
| C | qwen3-1.7b | 15 | 141.45 | 199.07 | 1.346 | 1.97 | 4.40 GiB | −16.86 | 0.85 dB (max 3.09) |
| E | soprano | 15 | 170.07 | 31.40 | 0.167 | 1.21 | 526 MiB | −17.41 | 1.41 dB (max 6.45) |

Soprano is ~8× faster than qwen3-1.7b in RTF and needs ~1/9 the memory, and
produces ~20 % more audio for the same 15 lines (slower delivery); it also hit
the truncation retry three times where qwen3 hit it zero times. All of that is
material for the report, none of it is a verdict — the listening scores decide.

`bench/` is gitignored, so these numbers live here and in the Sortie 7 report,
not in a committed artifact.
