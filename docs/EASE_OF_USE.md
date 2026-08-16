---
type: reference
state: draft
updated: 2026-08-15
---

# EASE_OF_USE.md — setup, integration surface, and API friction (FR-14)

Deliverable of OPERATION BATTLING BARDS, Sortie 10a. **Draft** — Sortie 11
finalizes this alongside the verdict once human listening scores exist. This
document does not carry performance numbers (those are dirty under Sortie 10a's
GPU contention; see `docs/BENCH.md` and `docs/SWIFT_BASELINE.md` for the
timing caveats). It answers a narrower, timing-independent question: **which
stack is cheaper to stand up and to integrate against**, going by what both
sides of this mission actually needed.

---

## 1. Setup steps

### Python (`comparativa`)

```sh
uv sync
uv run comparativa --help
```

Two commands, both idempotent, both driven by a committed `uv.lock`. Everything
downstream — parsing, voices, generation, bench, listen — runs through the same
`uv run comparativa ...` entry point once `uv sync` has resolved the
environment. Model checkpoints (`mlx-audio` 0.4.8: qwen3-1.7b/0.6b,
chatterbox, chatterbox-turbo, soprano) are pulled from the local HuggingFace
cache; every timed or benchmarked invocation forces `HF_HUB_OFFLINE=1` so a
run never silently blocks on, or gets credited/debited for, a network fetch.
First-run cost is whatever `uv sync` takes to resolve and install the pinned
dependency set (`mlx-audio==0.4.8`, `jouvence==0.4.2`, `psutil==7.2.2`,
`pyloudnorm==0.2.0`, `soundfile==0.14.0`, `numpy==2.5.2`) — no compiler, no
code signing, no separate model-store step beyond having the checkpoints in
the HF cache already (this mission's environment did).

### Swift (Produciesta / SwiftVoxAlta)

There is no "build" step in this mission — the hard constraint on Sortie 8
and this sortie is that the Swift stack is *measured, never modified*, so
setup means locating and trusting an already-signed, already-installed
artifact:

- a **signed binary**, `~/.local/bin/produciesta` (→
  `~/Projects/apps/Produciesta/bin/produciesta`), produced by a release
  pipeline this mission never runs;
- a **separate model store**: SwiftAcervo's App Group container
  (`~/Library/Group Containers/group.intrusive-memory.models/SharedModels`),
  populated ahead of time and invisible to `HF_HUB_OFFLINE`-style controls —
  there is no offline flag to pass, the store either has the checkpoint or it
  doesn't (`docs/CONDITION_F_RECON.md` hit exactly this: Soprano is in the
  Python HF cache but absent from the Swift store, and there is no CLI path
  that would even try to fetch it from anywhere else);
- a **project tree read via explicit flags**, not a working directory:
  `--project-md`, `--voices-dir`, `--cache-dir` all have to be pointed at the
  right places by the caller (`scripts/bench_condition_a.sh`), because
  Produciesta's own defaults would otherwise write into
  `~/Library/Application Support/Produciesta` or expect a project layout the
  benchmark harness does not assume.

Net: Python setup is two commands against a lockfile a contributor can run
unattended on a clean machine. Swift setup is "trust that a signed binary and
a populated model container already exist" — reproducible only by someone who
can run the release pipeline that produced them, which this mission
deliberately never touches (see `docs/CONDITION_F_RECON.md` § "Unblocking it"
for what that pipeline actually looks like: `make release`, `make
codesign-cli`, plus manually placing a checkpoint in the Acervo container).
That asymmetry is a *consequence of the mission's own hard constraint*, not
proof Swift setup is intrinsically harder — but it is the setup experience
this mission actually had.

---

## 2. Integration-layer LOC

Simple `wc -l` over each package under `src/comparativa/`, no vendored code,
no generated code:

| package | LOC | purpose |
|---|---:|---|
| `parsing/` | 1,807 | Fountain → element stream, cue normalization, text prep (matched to Swift semantics per `docs/TEXT_PREP.md`) |
| `generation/` | 3,103 | engine layer (4 MLX engines), sampling, truncation handling, assembly, encode |
| `bench/` | 1,788 | conditions matrix, metrics schema, perf capture, runner, CLI |
| `eval/` | 1,041 | blinding, metrics tabulation, scoring, report rendering |
| `voices/` | 907 | CAST.md roster, per-engine voice catalog/assignment |
| top-level (`cli.py`, `__init__.py`, `__main__.py`) | 132 | CLI dispatch |
| **`src/comparativa/` total** | **8,778** | the whole Python integration + generation + benchmark layer |
| `tests/` | 3,379 | pytest suite (241 tests green at Sortie 9 entry, +2 this sortie) |

That 8,778 is the **entire** Python side: it includes not just "talk to
mlx-audio" glue but the parser, text-prep parity layer, voice assignment,
benchmark harness, and evaluation tooling this mission needed to build from
scratch because none of it existed on the Python side before.

The Swift side has no equivalent scope to measure — Produciesta/SwiftVoxAlta
are a production app this mission does not own or modify — so the honest
comparison is narrower: **the integration surface this mission built on top
of each stack to drive it from a benchmark harness.**

| artifact | LOC | purpose |
|---|---:|---|
| `scripts/bench_condition_a.sh` | 157 | wrapper: resolve binary/paths, redirect `--cache-dir`, invoke `produciesta export`, capture `/usr/bin/time -l` |
| `scripts/bench_condition_a_metrics.py` | 356 | turn a run dir + `.vtt` sidecar into a schema-conforming `metrics.json` entry (afconvert decode, pyloudnorm measurement, tool-version capture) |
| **Swift-side wrapper total** | **513** | everything needed to drive the signed binary and normalize its output into the shared schema |

513 lines is not "how complex Swift TTS integration is" — it is how much
*wrapper* a black-box CLI needs before its output is comparable to a stack
this mission controls end-to-end: decode `.m4a` back to `.wav` because
Produciesta exports AAC only (§3.4 below), parse an `.vtt` sidecar because
there's no manifest, measure loudness with comparativa's own meter because
the Swift stack applies no LUFS target, and reconstruct `model_load_seconds`
as `null` because the CLI doesn't report it. Every one of those 513 lines is
compensating for a black box; none of comparativa's 8,778 lines are
compensating for *comparativa* being a black box to itself.

The honest read: Python's number is high because this mission built an
entire pipeline, not because MLX/mlx-audio integration is inherently
expensive — Sortie 5's engine layer is a fairly thin unified interface over
four already-existing model implementations. Swift's number is low because
the mission was forbidden from touching Swift internals, so it only shows
the outside-the-box cost of consuming a finished product.

---

## 3. API friction notes

Concrete frictions hit during this mission, one stack at a time, each traced
to a sortie doc or source file.

### 3.1 Python / mlx-audio

- **`top_k` exists only on the Python side.** `GenerationSettings.swift` has
  no top-k property at all; Python `mlx-audio` 0.4.8 defaults `top_k=50`,
  which would silently truncate the candidate set in a way Swift never does.
  `comparativa` has to explicitly pass `top_k=0` ("disabled") to match —
  an easy value to miss (`docs/SAMPLING_PARITY.md` § "the knob that only
  exists on the Python side").
- **Per-engine sample rate, not a pipeline-wide one.** qwen3 and chatterbox
  emit 24 kHz; Soprano emits 32 kHz. Assembly refuses to mix rates on one
  timeline (`AssemblyError`), which is correct but means every caller has to
  treat sample rate as a per-run fact, not a constant (`docs/ASSEMBLY.md` §4).
- **`chatterbox-turbo` draws a `tqdm` progress bar** straight to stdout
  inside a subprocess the benchmark harness needs quiet (its stdout is
  captured into `generate.log`, not the operator's terminal). Silencing it
  needs *two* env vars at the source, not one:
  `TQDM_DISABLE=1` **and** `HF_HUB_DISABLE_PROGRESS_BARS=1`
  (`src/comparativa/bench/runner.py`, `docs/BENCH.md` §3) — the second one is
  easy to miss if you only know about `tqdm`'s own disable flag.
- **`afconvert`'s AAC-LC bitrate ceiling is rate-dependent, and it fails
  silently-ish rather than clearly.** At 24 kHz mono (qwen3/chatterbox
  output), any `.m4a` bitrate above 64 kbps makes `afconvert` fail with
  `Couldn't set audio converter property ('!dat')` — an error message that
  gives no hint the fix is "lower the bitrate." `comparativa` targets
  64 kbps and falls back to `afconvert`'s own choice on a rejection rather
  than failing the run (`src/comparativa/generation/encode.py`,
  `docs/ASSEMBLY.md` §4).
- **The installed mlx-audio STT is broken**, so the optional objective proxy
  (STT round-trip WER) had to be sourced from `mlx-whisper` directly instead,
  with an explicit "dropped, reason recorded" fallback path when even that
  isn't available (`src/comparativa/eval/proxy.py`, EXECUTION_PLAN.md
  Risk §8.4).

### 3.2 Swift / Produciesta (consumed as a black box)

- **The `adapter-error` cache quirk.** A full-length episode's first attempt
  generated all 208 elements successfully and then died in compose with
  `error[adapter-error]: The operation could not be completed`, after
  45+ minutes of work. `ProduciestaModelStore` keeps generated audio in a
  `.store_SUPPORT/_EXTERNAL_DATA` sidecar next to a per-run SwiftData store;
  a failure to read those blobs back at compose time surfaces as exactly this
  opaque error, with no indication of *what* failed to read. The workaround
  found by trial is `CACHE_MODE=default` (Produciesta's own Application
  Support store) instead of a scratch `--cache-dir` under `/private/tmp` —
  one data point, not a diagnosis, but the operational rule the wrapper now
  follows (`docs/SWIFT_BASELINE.md` § "The compose-stage failure").
- **`.m4a`-only export forces an extra decode step for every listened
  clip.** Produciesta writes AAC, never WAV; the harness has to
  `afconvert`-decode it back to `.wav` before it can sit next to the native
  WAV output of every Python condition in the blinded listening set. That
  AAC round-trip is baked into condition A's listened signal and is *not*
  present in C/D/E — a confound the report has to state, not something
  `comparativa` can normalize away (`docs/SWIFT_BASELINE.md` §5.4).
- **No manifest, no per-line telemetry.** The only structured record of a
  Produciesta run is a `.vtt` sidecar (one cue per Element) plus
  `/usr/bin/time -l`'s process-level numbers. There is no equivalent of
  comparativa's `manifest.json` (per-line engine, checkpoint, sampling
  params, seed, truncation-retry flag). `model_load_seconds` in particular
  is unrecoverable — it stays `null` in every condition-A metrics entry
  because the CLI genuinely never reports it separately from total wall time
  (`docs/SWIFT_BASELINE.md` §3).
- **No CLI surface reaches models outside Qwen3-TTS.** `produciesta`, `diga`,
  and `echada` all resolve exclusively through `VoxAltaModelManager`'s
  Qwen3-TTS repos; the Soprano symbols are linked into the binary
  (`strings` finds `SopranoModel`, `SopranoDecoder`, `SopranoTTS`) but no
  flag on any signed tool reaches them, which is why condition F is skipped
  rather than run (`docs/CONDITION_F_RECON.md`).
- **No offline-mode equivalent.** Where the Python side has one env var
  (`HF_HUB_OFFLINE=1`) that guarantees a timed run never touches the network,
  the Swift model store either already has a checkpoint or it doesn't — there
  is no flag to fail loudly on a would-be fetch, so this mission's only
  option for the missing Soprano checkpoint was "the sortie is forbidden to
  even try," verified by inspection rather than by a runtime guarantee
  (`docs/CONDITION_F_RECON.md` § "The two blockers").
- **No loudness normalization at all.** Not strictly an API friction, but an
  integration-relevant asymmetry: Produciesta writes lines at whatever level
  the model produced (mean line loudness −19.7 to −21.1 LUFS across the two
  episodes), while every Python condition targets −16 LUFS integrated via
  `pyloudnorm`. `mean_shortfall_db`/`max_shortfall_db` are `null` for
  condition A, not `0` — the schema has to represent "this stack doesn't even
  have this concept," not just "this stack's value was zero"
  (`docs/SWIFT_BASELINE.md` §5.3, §3).

---

## 4. Summary for Sortie 11

| axis | Python (`comparativa`) | Swift (Produciesta) |
|---|---|---|
| setup | `uv sync` + `uv run` against a committed lockfile | trust a pre-existing signed binary + a pre-populated model container; no build path this mission is allowed to run |
| offline guarantee | one env var, enforced per run | none — the model store either has the checkpoint or the tool can't reach it |
| structured output | full `manifest.json` per line + schema-conforming `metrics.json` | `.vtt` sidecar only; no manifest; `model_load_seconds` permanently null |
| loudness control | `pyloudnorm` target, per line, with shortfall tracking | none; raw model output, ~3 dB quieter on average this mission's numbers |
| wrapper cost to benchmark | n/a — bench harness *is* the pipeline | 513 lines of glue (`scripts/bench_condition_a*`) to normalize a black box into the shared schema |
| what's forbidden | nothing — full read/write access to comparativa's own code | any build, any new-model install; measured strictly as a signed artifact |

This is a draft for Sortie 11 to finalize once the listening scores are in —
none of the above changes with clean timing numbers from Sortie 10b, since
every point here is about integration shape, not speed.
