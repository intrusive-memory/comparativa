"""Engine smoke test: one fixed line, every engine, local checkpoints only.

Run it directly::

    HF_HUB_OFFLINE=1 uv run python -m comparativa.generation.smoke -o out/smoke

Each engine loads its cached checkpoint, generates :data:`SMOKE_TEXT` once, and
the result is checked for sane duration. The point is not audio quality: it is
proof that the engine layer drives all four families, that the recorded sampling
parameters survive the round trip, and (RD-1) that Python mlx-audio loads the
``mlx-community/Soprano-80M-bf16`` conversion the Swift port pins.

``tests/smoke/test_engine_smoke.py`` calls :func:`run_smoke`; it is skipped by
default because loading four checkpoints takes minutes (see ``tests/conftest.py``).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .engines import SMOKE_ENGINE_KEYS, EngineError, LineRequest, load_engine, spec

#: The one line every engine says. Two clauses, ordinary vocabulary, no numbers
#: or abbreviations: nothing here should stress a front-end.
SMOKE_TEXT: str = (
    "Granville is a small town with a long memory, and tonight it remembers everything."
)

#: Preset speaker used for the qwen3 engines (English male preset; see
#: presets.yaml). Ignored by the single-voice engines.
SMOKE_VOICE: str = "ryan"

#: Fixed seed so a smoke run is reproducible on engines that support seeding.
SMOKE_SEED: int = 20260815


@dataclass
class SmokeResult:
    """Outcome of one engine's smoke generation."""

    engine: str
    checkpoint: str
    ok: bool
    #: Populated on failure (missing checkpoint, generation error).
    error: str = ""
    record: dict[str, Any] | None = None
    wav_path: str | None = None
    load_seconds: float = 0.0

    @property
    def duration_seconds(self) -> float:
        return float((self.record or {}).get("duration_seconds", 0.0))

    @property
    def real_time_factor(self) -> float:
        return float((self.record or {}).get("real_time_factor", 0.0))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "engine": self.engine,
            "checkpoint": self.checkpoint,
            "ok": self.ok,
            "load_seconds": round(self.load_seconds, 3),
        }
        if self.error:
            d["error"] = self.error
        if self.record is not None:
            d["line"] = self.record
        if self.wav_path:
            d["wav"] = self.wav_path
        return d


def run_one(
    key: str,
    *,
    text: str = SMOKE_TEXT,
    out_dir: Path | None = None,
    seed: int | None = SMOKE_SEED,
) -> SmokeResult:
    """Load one engine, generate ``text`` once, and check the duration."""
    engine_spec = spec(key)
    try:
        engine = load_engine(key)
    except EngineError as exc:
        return SmokeResult(key, engine_spec.checkpoint, ok=False, error=str(exc))

    voice = SMOKE_VOICE if engine_spec.preset_voices else None
    try:
        result = engine.generate_line(
            LineRequest(text=text, voice=voice, seed=seed, label=f"smoke:{key}")
        )
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return SmokeResult(
            key,
            engine_spec.checkpoint,
            ok=False,
            error=f"generation failed: {exc}",
            load_seconds=engine.load_seconds,
        )

    wav_path: str | None = None
    if out_dir is not None:
        import soundfile as sf

        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"smoke-{key}.wav"
        sf.write(path, result.audio, result.sample_rate)
        wav_path = str(path)

    record = result.to_dict()
    ok = not result.truncated and not result.overrun and result.duration_seconds > 0
    error = ""
    if not ok:
        failed = [c.check.reason for c in result.chunks if not c.check.ok]
        error = "duration sanity check failed: " + "; ".join(failed or ["no audio"])

    return SmokeResult(
        engine=key,
        checkpoint=engine_spec.checkpoint,
        ok=ok,
        error=error,
        record=record,
        wav_path=wav_path,
        load_seconds=engine.load_seconds,
    )


def run_smoke(
    keys: Sequence[str] = SMOKE_ENGINE_KEYS,
    *,
    text: str = SMOKE_TEXT,
    out_dir: Path | None = None,
    seed: int | None = SMOKE_SEED,
) -> list[SmokeResult]:
    """Run :func:`run_one` for every engine in ``keys``, never raising."""
    return [run_one(k, text=text, out_dir=out_dir, seed=seed) for k in keys]


def format_table(results: Sequence[SmokeResult]) -> str:
    """A fixed-width summary of a smoke run."""
    header = f"{'engine':<17} {'ok':<4} {'load s':>8} {'audio s':>8} {'RTF':>7}  detail"
    rows = [header, "-" * len(header)]
    for r in results:
        detail = r.error
        if r.ok and r.record is not None:
            detail = (
                f"voice={r.record.get('voice')} "
                f"seed={r.record.get('seed')} "
                f"seeding={r.record.get('seeding')} "
                f"chunks={len(r.record.get('chunks', []))}"
            )
        rows.append(
            f"{r.engine:<17} {'yes' if r.ok else 'NO':<4} "
            f"{r.load_seconds:>8.1f} {r.duration_seconds:>8.2f} "
            f"{r.real_time_factor:>7.2f}  {detail}"
        )
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: ``python -m comparativa.generation.smoke``."""
    parser = argparse.ArgumentParser(
        prog="comparativa.generation.smoke",
        description="Generate one fixed line on each engine from local checkpoints.",
    )
    parser.add_argument(
        "-e",
        "--engine",
        action="append",
        dest="engines",
        help="Engine key; repeatable. Defaults to the four smoke engines.",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("out/smoke"),
        help="Directory for the generated wavs and smoke.json (default: out/smoke).",
    )
    parser.add_argument("--text", default=SMOKE_TEXT, help="Line to generate.")
    parser.add_argument(
        "--seed",
        type=int,
        default=SMOKE_SEED,
        help="RNG seed; use -1 to leave the RNG untouched.",
    )
    args = parser.parse_args(argv)

    keys = tuple(args.engines) if args.engines else SMOKE_ENGINE_KEYS
    seed = None if args.seed is not None and args.seed < 0 else args.seed

    results = run_smoke(keys, text=args.text, out_dir=args.out, seed=seed)

    args.out.mkdir(parents=True, exist_ok=True)
    summary = args.out / "smoke.json"
    summary.write_text(
        json.dumps(
            {"text": args.text, "seed": seed, "results": [r.to_dict() for r in results]},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(format_table(results))
    print(f"\nwrote {summary}")
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":  # pragma: no cover - exercised via the module CLI
    sys.exit(main())
