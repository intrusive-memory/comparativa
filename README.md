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

| Command | Purpose | Status |
|---------|---------|--------|
| `parse` | Parse a Fountain screenplay into a JSON element stream | implemented |
| `voices` | Build the character roster and per-engine voice assignments (`--audition` generates a fixed sentence per character × engine) | implemented |
| `generate` | Generate episode audio for one engine condition | implemented |
| `bench` | Run the benchmark matrix across conditions and episodes | stub |
| `listen` | Build a blinded listening set and scoring sheet | stub |
| `report` | Tabulate metrics and scores into a report | stub |

```sh
# Generate an episode (offline; all checkpoints are cached locally)
HF_HUB_OFFLINE=1 uv run comparativa generate \
    ~/Projects/podcasts/granville/episodes/episode_1_01_cold_open.fountain \
    --engine qwen3-0.6b -o out/
```

## Tests

```sh
uv run pytest            # fast unit suite, no checkpoints loaded
uv run pytest --smoke    # + engine smoke and the full-episode integration test
```

## Planning documents

- `REQUIREMENTS.md` — locked requirements
- `EXECUTION_PLAN.md` — mission plan and sortie breakdown
- `docs/TEXT_PREP.md` — TTS text preparation and its Swift-stack parity
- `docs/SAMPLING_PARITY.md` — sampling parameters traced to `GenerationSettings.swift`
- `docs/ASSEMBLY.md` — episode assembly, loudness, and the `manifest.json` schema
