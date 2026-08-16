"""Blinding: build a randomized, filename-blinded listening set + key file.

Discovers every rendered episode clip under a bench directory
(``bench/<condition>/<episode>/`` with a sibling ``manifest.json`` and/or
``metrics.json`` next to the audio — the Sortie 6/7/8 schema), assigns each
clip an opaque id, copies the audio into a blinded set directory in
randomized order, and writes two artifacts:

``key.csv``
    Opaque id -> condition/episode/engine/stack/checkpoint/original path.
    Kept in the output directory *next to*, not *inside*, ``set/`` — a
    listener working through ``set/`` never needs to open it.
``scoring_sheet.csv``
    One blank row per opaque id with the FR-13 score columns (naturalness,
    prosody, artifacts, voice consistency across lines, character
    distinctness) plus a free-text notes column. Contains no condition or
    episode information — that is exactly what stays blind.

``comparativa report`` (:mod:`.scoring`) reads both back to unblind.
"""

from __future__ import annotations

import csv
import json
import random
import secrets
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Filenames written under a listening-set output directory.
KEY_FILENAME = "key.csv"
SCORING_SHEET_FILENAME = "scoring_sheet.csv"
SET_DIRNAME = "set"

#: Key file columns, in write order.
KEY_FIELDS: tuple[str, ...] = (
    "opaque_id",
    "condition",
    "episode",
    "engine",
    "stack",
    "checkpoint",
    "original_filename",
    "original_path",
)

#: FR-13 scoring-sheet score columns (plus a free-text ``notes`` column).
SCORE_COLUMNS: tuple[str, ...] = (
    "naturalness",
    "prosody",
    "artifacts",
    "voice_consistency",
    "character_distinctness",
)
SCORING_SHEET_FIELDS: tuple[str, ...] = ("opaque_id", *SCORE_COLUMNS, "notes")


class BlindError(RuntimeError):
    """The listening set could not be built."""


@dataclass(frozen=True)
class Clip:
    """One condition x episode audio clip discovered under a bench dir."""

    condition: str
    episode: str
    audio_path: Path
    manifest_path: Path | None = None
    metrics_path: Path | None = None
    engine: str | None = None
    stack: str | None = None
    checkpoint: str | None = None


@dataclass(frozen=True)
class ListenSet:
    """The result of :func:`blind`: where everything landed."""

    bench_dir: Path
    out_dir: Path
    set_dir: Path
    key_path: Path
    scoring_sheet_path: Path
    key_rows: tuple[dict[str, str], ...] = field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.key_rows)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _load_metrics_entry(path: Path, *, condition: str, episode: str) -> dict[str, Any]:
    """Load the one entry that matches ``(condition, episode)`` from a
    ``metrics.json``.

    ``metrics.json`` is written as a schema-version-1 **envelope** —
    ``{"schema_version": 1, ..., "entries": [...]}`` (see
    :mod:`comparativa.bench.metrics`) — and a single file may hold more than
    one entry (e.g. Sortie 8's ``bench/A/metrics.json`` covering both
    episodes). Reading the envelope itself as if it *were* an entry silently
    yields ``None`` for every field (``engine``, ``checkpoint``, ``stack``)
    since none of those keys exist at the envelope's top level. This also
    tolerates a bare single-entry dict (no ``entries`` wrapper), the shape
    older fixtures use.
    """
    doc = _load_json(path)
    entries = doc.get("entries")
    if isinstance(entries, list):
        dict_entries = [e for e in entries if isinstance(e, dict)]
        for entry in dict_entries:
            if entry.get("condition") == condition and entry.get("episode") == episode:
                return entry
        # No exact (condition, episode) match — fall back to the first entry
        # rather than an empty dict, so the run still gets *some* fields.
        return dict_entries[0] if dict_entries else {}
    return doc


def _find_audio(episode_dir: Path, manifest: dict[str, Any]) -> Path | None:
    """Prefer the manifest's recorded ``.wav`` output; fall back to a glob."""
    outputs = manifest.get("outputs")
    if isinstance(outputs, dict) and outputs.get("wav"):
        candidate = Path(str(outputs["wav"]))
        if not candidate.is_absolute():
            candidate = episode_dir / candidate.name
        if candidate.exists():
            return candidate
    wavs = sorted(p for p in episode_dir.glob("*.wav"))
    return wavs[0] if wavs else None


def discover_clips(bench_dir: str | Path) -> list[Clip]:
    """Find every condition/episode audio clip under ``bench_dir``.

    A clip only needs an audio file plus *either* a ``manifest.json`` or a
    ``metrics.json`` sibling to be discovered — tolerant of conditions built
    by a different pipeline (e.g. the Swift condition A/F wrappers), which
    may not write the Python manifest schema at all.
    """
    bench_dir = Path(bench_dir)
    episode_dirs: set[Path] = set()
    for pattern in ("*/*/manifest.json", "*/*/metrics.json"):
        episode_dirs.update(p.parent for p in bench_dir.glob(pattern))

    clips: list[Clip] = []
    for episode_dir in sorted(episode_dirs):
        condition = episode_dir.parent.name
        episode = episode_dir.name
        manifest_path = episode_dir / "manifest.json"
        metrics_path = episode_dir / "metrics.json"
        manifest = _load_json(manifest_path)
        metrics = _load_metrics_entry(metrics_path, condition=condition, episode=episode)

        audio_path = _find_audio(episode_dir, manifest)
        if audio_path is None:
            continue

        engine_key = None
        checkpoint = None
        manifest_engine = manifest.get("engine")
        if isinstance(manifest_engine, dict):
            engine_key = manifest_engine.get("key")
            checkpoint = manifest_engine.get("checkpoint")
        engine_key = metrics.get("engine", engine_key)
        checkpoint = metrics.get("checkpoint", checkpoint)
        stack = metrics.get("stack")

        clips.append(
            Clip(
                condition=condition,
                episode=episode,
                audio_path=audio_path,
                manifest_path=manifest_path if manifest_path.exists() else None,
                metrics_path=metrics_path if metrics_path.exists() else None,
                engine=str(engine_key) if engine_key is not None else None,
                stack=str(stack) if stack is not None else None,
                checkpoint=str(checkpoint) if checkpoint is not None else None,
            )
        )
    return clips


def _new_opaque_id(used: set[str]) -> str:
    while True:
        token = f"clip-{secrets.token_hex(4)}"
        if token not in used:
            used.add(token)
            return token


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def blind(bench_dir: str | Path, out_dir: str | Path, *, seed: int | None = None) -> ListenSet:
    """Build a randomized, filename-blinded listening set under ``out_dir``.

    Copies each discovered clip's audio into ``out_dir/set/<opaque-id>.<ext>``
    in a random order, writes ``out_dir/key.csv`` (the unblinding key) and
    ``out_dir/scoring_sheet.csv`` (blank FR-13 score columns, one row per
    clip, same random order). Raises :class:`BlindError` if no clips are
    found.
    """
    bench_dir = Path(bench_dir)
    out_dir = Path(out_dir)
    clips = discover_clips(bench_dir)
    if not clips:
        raise BlindError(f"no rendered audio found under {bench_dir}")

    order = list(clips)
    random.Random(seed).shuffle(order)

    set_dir = out_dir / SET_DIRNAME
    set_dir.mkdir(parents=True, exist_ok=True)

    used_ids: set[str] = set()
    key_rows: list[dict[str, str]] = []
    for clip in order:
        opaque_id = _new_opaque_id(used_ids)
        suffix = clip.audio_path.suffix or ".wav"
        blinded_path = set_dir / f"{opaque_id}{suffix}"
        shutil.copyfile(clip.audio_path, blinded_path)
        key_rows.append(
            {
                "opaque_id": opaque_id,
                "condition": clip.condition,
                "episode": clip.episode,
                "engine": clip.engine or "",
                "stack": clip.stack or "",
                "checkpoint": clip.checkpoint or "",
                "original_filename": clip.audio_path.name,
                "original_path": str(clip.audio_path),
            }
        )

    key_path = out_dir / KEY_FILENAME
    _write_csv(key_path, KEY_FIELDS, key_rows)

    scoring_rows = [
        {"opaque_id": row["opaque_id"], **{col: "" for col in SCORE_COLUMNS}, "notes": ""}
        for row in key_rows
    ]
    scoring_sheet_path = out_dir / SCORING_SHEET_FILENAME
    _write_csv(scoring_sheet_path, SCORING_SHEET_FIELDS, scoring_rows)

    return ListenSet(
        bench_dir=bench_dir,
        out_dir=out_dir,
        set_dir=set_dir,
        key_path=key_path,
        scoring_sheet_path=scoring_sheet_path,
        key_rows=tuple(key_rows),
    )


__all__ = [
    "KEY_FIELDS",
    "KEY_FILENAME",
    "SCORE_COLUMNS",
    "SCORING_SHEET_FIELDS",
    "SCORING_SHEET_FILENAME",
    "SET_DIRNAME",
    "BlindError",
    "Clip",
    "ListenSet",
    "blind",
    "discover_clips",
]
