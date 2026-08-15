"""Voice audition: one fixed sentence per character × engine assignment.

``comparativa voices <project-dir> --audition`` exists to answer the question
``presets.yaml`` cannot: the assignment table says HUNTER gets ``ryan`` on
qwen3, but is ``ryan`` *right* for HUNTER? Generating the same sentence for
every character on one engine makes the whole cast comparable in one listen,
and generating it across engines shows how much of a character's identity
survives the engine change (on the single-voice engines: none of it — every
character sounds identical, which is exactly what conditions D and E look like
and is worth hearing before the bench run rather than after).

Each engine is loaded once and every character is generated through it before
moving on. Lines are loudness-normalized with the same meter and target the
episode assembler uses (RD-3), so an audition is directly comparable to the
episodes it predicts.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final

from .assembly import TARGET_LUFS, normalize_line
from .encode import write_wav
from .engines import EngineError, LineRequest, load_engine, spec

#: The sentence every audition speaks. Neutral in content and register so the
#: comparison is about the voice, not the reading; long enough (>0.4 s of
#: speech by a wide margin) to be measurable by the BS.1770 meter.
AUDITION_TEXT: Final = (
    "This is Granville, and every voice in this town has something it would "
    "rather not say out loud."
)

#: Fixed seed so re-auditioning the same cast produces the same takes.
AUDITION_SEED: Final = 20260815

#: Filename for the audition index inside the output directory.
AUDITION_MANIFEST: Final = "audition.json"


@dataclass
class AuditionTake:
    """One character's audition on one engine."""

    engine: str
    character: str
    voice: str | None
    ok: bool
    path: str | None = None
    error: str = ""
    record: dict[str, Any] | None = None

    @property
    def duration_seconds(self) -> float:
        return float((self.record or {}).get("duration_seconds", 0.0))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "engine": self.engine,
            "character": self.character,
            "voice": self.voice,
            "ok": self.ok,
        }
        if self.path:
            d["wav"] = self.path
        if self.error:
            d["error"] = self.error
        if self.record is not None:
            d["line"] = self.record
        return d


@dataclass
class AuditionRun:
    """Every take from one audition, plus what produced it."""

    text: str
    seed: int | None
    target_lufs: float | None
    takes: list[AuditionTake] = field(default_factory=list)
    load_seconds: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(take.ok for take in self.takes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_by": "comparativa voices --audition",
            "text": self.text,
            "seed": self.seed,
            "target_lufs": self.target_lufs,
            "load_seconds": {k: round(v, 3) for k, v in self.load_seconds.items()},
            "takes": [take.to_dict() for take in self.takes],
        }


#: Called with each finished take, for progress output.
ProgressHook = Callable[[AuditionTake], None]


def _safe_name(character: str) -> str:
    """A filesystem-safe filename stem for a character name."""
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in character.strip()]
    return "".join(keep).strip("_") or "character"


def run_audition(
    voices_by_character: dict[str, dict[str, str | None]],
    *,
    engine_keys: Sequence[str],
    characters: Iterable[str] | None = None,
    out_dir: str | Path = "out/audition",
    text: str = AUDITION_TEXT,
    seed: int | None = AUDITION_SEED,
    target_lufs: float | None = TARGET_LUFS,
    progress: ProgressHook | None = None,
) -> AuditionRun:
    """Generate ``text`` once per character per engine.

    ``voices_by_character`` maps character -> {engine key -> voice}, i.e. the
    ``assignments`` section of ``presets.yaml``. Engines are loaded one at a
    time and never reloaded; a failure on one engine is recorded and the rest
    of the run continues, because a partial audition is still useful.
    """
    directory = Path(out_dir).expanduser()
    names = list(characters) if characters is not None else list(voices_by_character)
    unknown = [n for n in names if n not in voices_by_character]
    if unknown:
        raise EngineError(
            "no voice assignment for: " + ", ".join(sorted(unknown))
        )

    run = AuditionRun(text=text, seed=seed, target_lufs=target_lufs)

    for engine_key in engine_keys:
        engine_spec = spec(engine_key)
        started = time.perf_counter()
        try:
            engine = load_engine(engine_key)
        except EngineError as exc:
            run.load_seconds[engine_key] = time.perf_counter() - started
            for character in names:
                take = AuditionTake(
                    engine=engine_key,
                    character=character,
                    voice=voices_by_character[character].get(engine_key),
                    ok=False,
                    error=str(exc),
                )
                run.takes.append(take)
                if progress is not None:
                    progress(take)
            continue
        run.load_seconds[engine_key] = engine.load_seconds

        for position, character in enumerate(names):
            voice = voices_by_character[character].get(engine_key)
            take = AuditionTake(
                engine=engine_key, character=character, voice=voice, ok=False
            )
            try:
                result = engine.generate_line(
                    LineRequest(
                        text=text,
                        voice=voice,
                        seed=None if seed is None else seed + position,
                        label=f"audition:{engine_key}:{character}",
                    )
                )
            except (EngineError, RuntimeError, ValueError) as exc:  # noqa: BLE001
                take.error = f"generation failed: {exc}"
                run.takes.append(take)
                if progress is not None:
                    progress(take)
                continue

            audio = result.audio
            loudness = normalize_line(
                audio, result.sample_rate, target_lufs=target_lufs
            )
            path = directory / engine_key / f"{_safe_name(character)}.wav"
            write_wav(loudness.audio, result.sample_rate, path)

            record = result.to_dict()
            record["loudness"] = loudness.to_dict()
            take.ok = result.duration_seconds > 0 and not result.truncated
            take.path = str(path)
            take.record = record
            if not take.ok:
                take.error = "duration sanity check failed"
            run.takes.append(take)
            if progress is not None:
                progress(take)

    return run


def write_audition_manifest(run: AuditionRun, out_dir: str | Path) -> Path:
    """Write ``audition.json`` next to the generated takes."""
    directory = Path(out_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / AUDITION_MANIFEST
    path.write_text(
        json.dumps(run.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def format_table(run: AuditionRun) -> str:
    """A fixed-width summary of an audition run."""
    header = f"{'ENGINE':<17} {'CHARACTER':<18} {'VOICE':<14} {'OK':<4} {'AUDIO S':>8}  DETAIL"
    rows = [header, "-" * len(header)]
    for take in run.takes:
        rows.append(
            f"{take.engine:<17} {take.character:<18} {(take.voice or '-'):<14} "
            f"{('yes' if take.ok else 'NO'):<4} {take.duration_seconds:>8.2f}  {take.error}".rstrip()
        )
    return "\n".join(rows)


__all__ = [
    "AUDITION_MANIFEST",
    "AUDITION_SEED",
    "AUDITION_TEXT",
    "AuditionRun",
    "AuditionTake",
    "format_table",
    "run_audition",
    "write_audition_manifest",
]
