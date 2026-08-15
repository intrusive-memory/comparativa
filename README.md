---
type: project
---

# comparativa

All-Python reference pipeline that parses Fountain screenplays and generates
episode audio with MLX speech models, so its output can be compared under
controlled conditions against the existing Swift stack (Produciesta /
SwiftVoxAlta). The Swift stack is measured, never modified.

## Requirements

- Python 3.12 (managed by [`uv`](https://docs.astral.sh/uv/))
- `mlx-audio==0.4.8` — the version validated by the smoke test
- Apple Silicon (MLX)

## Setup

```sh
uv sync
uv run comparativa --help
```

## Commands

| Command | Purpose |
|---------|---------|
| `parse` | Parse a Fountain screenplay into a JSON element stream |
| `voices` | Build the character roster and per-engine voice assignments |
| `generate` | Generate episode audio for one engine condition |
| `bench` | Run the benchmark matrix across conditions and episodes |
| `listen` | Build a blinded listening set and scoring sheet |
| `report` | Tabulate metrics and scores into a report |

All six subcommands are currently scaffolding stubs; they are implemented by
later sorties.

## Tests

```sh
uv run pytest
```

## Planning documents

- `REQUIREMENTS.md` — locked requirements
- `EXECUTION_PLAN.md` — mission plan and sortie breakdown
