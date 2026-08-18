---
type: project
---

# AGENTS.md — comparativa

Instructions for coding agents working in this repository. `CLAUDE.md` is a
symlink to this file.

## What this project is

An all-Python reference pipeline that parses Fountain screenplays and generates
episode audio with MLX speech models, so its output can be compared under
controlled conditions against the existing Swift stack (Produciesta /
SwiftVoxAlta). **The Swift stack is measured, never modified.** Do not edit
anything in the Swift projects from this repo, and do not "fix" Swift-side
behavior to make comparisons come out better — parity notes belong in `docs/`.

## Ground rules

- **Python 3.12 via `uv`.** Run everything through `uv run …`. Never pip-install
  into the system interpreter, and never touch `.venv/` by hand.
- **Pinned dependency:** `mlx-audio==0.4.8` is the version validated by the
  smoke test. Do not bump it casually; a bump invalidates the benchmark
  conditions and requires re-validating the smoke suite.
- **Apple Silicon required** for anything that loads MLX models. Pure parsing
  and unit tests run anywhere.
- **Offline generation:** checkpoints are cached locally; generation commands
  are run with `HF_HUB_OFFLINE=1`. Do not add code that fetches models at
  generation time.
- **Frozen corpus:** tests are pinned to the corpus snapshot described in
  `docs/CORPUS_PIN.md`. `corpus/` is extracted read-only from the granville
  project's git history and stays untracked. Never regenerate or edit it, and
  never commit it.
- **Never commit artifacts:** `out/`, `bench/`, `corpus/`, `*.wav`, `*.m4a`,
  and `default.profraw` are gitignored on purpose. Keep it that way.

## Commands

```sh
uv sync                      # set up the environment
uv run comparativa --help    # six subcommands: parse, voices, generate, bench, listen, report
```

All six subcommands are implemented; wiring lives in `src/comparativa/cli.py`
(`_WIRED` maps subcommand → module with `configure`/`handle` hooks, imported
lazily so `--help` stays fast).

## Tests

```sh
uv run pytest            # fast unit suite, no checkpoints loaded
uv run pytest --smoke    # + engine smoke and the full-episode integration test
```

- The default suite must stay fast and checkpoint-free. Anything that loads a
  model belongs behind `--smoke`.
- Tests are pinned to the frozen corpus snapshot — do not weaken that pinning.

## Layout

- `src/comparativa/` — the package: `parsing/`, `voices/`, `generation/`,
  `bench/`, `eval/`, with `cli.py` as the entry point.
- `scripts/` — the condition-A Swift-baseline wrapper and its metrics
  extractor.
- `presets.yaml` — voice/engine presets.
- `docs/` — design and parity documents (text prep, sampling parity, assembly,
  bench matrix, corpus pin, Swift baseline, condition-F recon, ease of use).
- `REQUIREMENTS.md` — locked requirements. Treat as read-only unless the user
  explicitly reopens them.
- `EXECUTION_PLAN.md`, `SUPERVISOR_STATE.md`, `REPORT.md` — mission-supervisor
  artifacts for the current mission.

## Conventions

- Keep the `metrics.json` schema frozen (see `docs/BENCH.md`); the report
  reader depends on it.
- Sampling parameters must stay traceable to `GenerationSettings.swift` — see
  `docs/SAMPLING_PARITY.md` before changing any generation defaults.
- Match existing code style: type-annotated, docstringed modules with lazy
  imports for heavyweight dependencies.
