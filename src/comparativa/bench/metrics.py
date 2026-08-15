"""``metrics.json`` — the cross-stack performance record (FR-12). **Frozen.**

One *entry* describes one (condition, episode) run on one stack. Sortie 7 emits
Python-stack entries (conditions C/D/E, optionally T); **Sortie 8 emits
Swift-stack entries (conditions A and F) into the same schema**, which is why
nothing here is Python-only: an entry needs a condition id, a stack, an episode
id, a speech policy, five performance numbers, and a tool-version dict whose
keys are the writer's business (``produciesta`` version on the Swift side,
``mlx_audio`` on the Python side).

Document layout — a list of entries, so one file can hold one run
(``bench/<cond>/<episode>/metrics.json``) or a whole condition
(``bench/<cond>/metrics.json``, which is what Sortie 8's wrapper writes)::

    {
      "schema_version": 1,
      "generated_by": "comparativa bench",
      "written_at": "2026-08-15T...",
      "entries": [ { ... }, ... ]
    }

:func:`load_entries` reads either layout and de-duplicates on
``(stack, condition, episode)``, so a reader may glob ``**/metrics.json`` under
the bench root without double-counting a run that appears in both files.

Entry schema (keys marked ``*`` are required by :func:`validate_entry`)::

    condition*        "A" | "C" | "D" | "E" | "F" | "T"
    stack*            "python" | "swift"
    label             human-readable condition description
    engine*           engine key ("qwen3-1.7b", ...) or the Swift tool's engine
                      name; may be null only if checkpoint is set
    checkpoint*       HF repo id actually loaded (null if genuinely unknowable)
    voices            how voices were chosen ("presets", ".vox", ...)
    episode*          episode id (the .fountain stem — the join key across
                      stacks, so both stacks MUST use the same string)
    episode_path      absolute path to the screenplay
    episode_sha256    screenplay hash, so two stacks can prove same input
    speech_policy*    "produciesta-parity" for every round-1 bench condition
    status*           "ok" | "failed" | "skipped"
    performance*      wall_seconds*, real_time_factor*, model_load_seconds*,
                      peak_rss_bytes*, generate_seconds, process_wall_seconds,
                      start_rss_bytes, rss_samples
    audio*            audio_seconds*, duration_seconds, gap_seconds, sample_rate
    lines*            line_count*, placed_line_count, script_line_count,
                      truncated_lines, truncation_retry_lines, overrun_lines
    loudness*         meter, target_lufs, mean_output_lufs*,
                      mean_shortfall_db*, max_shortfall_db*,
                      peak_limited_lines, unnormalized_lines
    tool_versions*    free-form {name: version-or-null}
    host              platform/machine/cpu/memory of the measuring machine
    run               started_at, finished_at, command, log
    outputs           {wav, m4a, manifest} paths
    notes             list of free-form strings (e.g. "seeding unavailable")

Every ``performance``/``audio``/``lines``/``loudness`` key is *present* in an
entry even when its value is ``null``: a Swift run that does not normalize
loudness still carries ``mean_shortfall_db: null`` so the report's table has a
cell rather than a hole.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Iterable, Sequence

#: Bumped only on a breaking change. Frozen for the whole of round 1.
METRICS_SCHEMA_VERSION: Final = 1

#: Filename written in every run directory (and by Sortie 8 per condition).
METRICS_FILENAME: Final = "metrics.json"

#: Aggregate of one ``bench`` invocation, at the bench root.
SUMMARY_FILENAME: Final = "summary.json"

#: Entry ``status`` values.
STATUS_OK: Final = "ok"
STATUS_FAILED: Final = "failed"
STATUS_SKIPPED: Final = "skipped"

#: Top-level keys every entry must carry.
REQUIRED_KEYS: Final[tuple[str, ...]] = (
    "condition",
    "stack",
    "engine",
    "checkpoint",
    "episode",
    "speech_policy",
    "status",
    "performance",
    "audio",
    "lines",
    "loudness",
    "tool_versions",
)

#: Required keys inside each nested group.
REQUIRED_GROUP_KEYS: Final[dict[str, tuple[str, ...]]] = {
    "performance": (
        "wall_seconds",
        "real_time_factor",
        "model_load_seconds",
        "peak_rss_bytes",
    ),
    "audio": ("audio_seconds",),
    "lines": ("line_count",),
    "loudness": ("mean_output_lufs", "mean_shortfall_db", "max_shortfall_db"),
}


class MetricsError(ValueError):
    """A metrics entry does not satisfy the frozen schema."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _round(value: Any, digits: int = 3) -> Any:
    return round(float(value), digits) if isinstance(value, (int, float)) else None


# ---------------------------------------------------------------------------
# Building entries
# ---------------------------------------------------------------------------


def make_entry(
    *,
    condition: str,
    stack: str,
    episode: str,
    speech_policy: str,
    engine: str | None = None,
    checkpoint: str | None = None,
    wall_seconds: float | None,
    real_time_factor: float | None,
    model_load_seconds: float | None,
    peak_rss_bytes: int | None,
    audio_seconds: float | None,
    line_count: int | None,
    tool_versions: dict[str, Any],
    label: str | None = None,
    voices: str | None = None,
    status: str = STATUS_OK,
    episode_path: str | None = None,
    episode_sha256: str | None = None,
    performance_extra: dict[str, Any] | None = None,
    audio_extra: dict[str, Any] | None = None,
    lines_extra: dict[str, Any] | None = None,
    loudness: dict[str, Any] | None = None,
    host: dict[str, Any] | None = None,
    run: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    notes: Sequence[str] = (),
    error: str | None = None,
) -> dict[str, Any]:
    """Build one schema-conforming entry.

    This is the constructor **both stacks** use. Everything a Python run happens
    to know (chunk counts, truncation retries, loudness shortfalls) arrives
    through the ``*_extra`` dicts, so a Swift caller can omit all of it and
    still produce a valid entry.
    """
    loud = dict(loudness or {})
    entry: dict[str, Any] = {
        "condition": condition,
        "stack": stack,
        "label": label,
        "engine": engine,
        "checkpoint": checkpoint,
        "voices": voices,
        "episode": episode,
        "episode_path": episode_path,
        "episode_sha256": episode_sha256,
        "speech_policy": speech_policy,
        "status": status,
        "error": error,
        "performance": {
            "wall_seconds": _round(wall_seconds),
            "real_time_factor": _round(real_time_factor),
            "model_load_seconds": _round(model_load_seconds),
            "peak_rss_bytes": int(peak_rss_bytes) if peak_rss_bytes else None,
            **(performance_extra or {}),
        },
        "audio": {
            "audio_seconds": _round(audio_seconds),
            **(audio_extra or {}),
        },
        "lines": {
            "line_count": int(line_count) if line_count is not None else None,
            **(lines_extra or {}),
        },
        "loudness": {
            "meter": loud.get("meter"),
            "target_lufs": loud.get("target_lufs"),
            "mean_output_lufs": loud.get("mean_output_lufs"),
            "mean_shortfall_db": loud.get("mean_shortfall_db"),
            "max_shortfall_db": loud.get("max_shortfall_db"),
            "peak_limited_lines": loud.get("peak_limited_lines"),
            "unnormalized_lines": loud.get("unnormalized_lines"),
        },
        "tool_versions": dict(tool_versions),
        "host": host or {},
        "run": run or {},
        "outputs": outputs or {},
        "notes": list(notes),
    }
    validate_entry(entry)
    return entry


def entry_from_manifest(
    manifest: dict[str, Any],
    *,
    condition: str,
    episode: str,
    stack: str,
    tool_versions: dict[str, Any],
    peak_rss_bytes: int | None,
    label: str | None = None,
    voices: str | None = None,
    status: str = STATUS_OK,
    performance_extra: dict[str, Any] | None = None,
    host: dict[str, Any] | None = None,
    run: dict[str, Any] | None = None,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Build an entry from a ``manifest.json`` (schema_version 1).

    Wall-clock, RTF, and model-load time are taken **from the manifest's
    ``totals``**, never re-derived here — a supervisor ruling, and the reason
    the two documents can be trusted against each other.
    """
    totals = manifest.get("totals", {})
    engine = manifest.get("engine", {})
    episode_doc = manifest.get("episode", {})
    assembly = manifest.get("assembly", {})

    return make_entry(
        condition=condition,
        stack=stack,
        label=label,
        engine=engine.get("key"),
        checkpoint=engine.get("checkpoint"),
        voices=voices,
        episode=episode,
        episode_path=episode_doc.get("path"),
        episode_sha256=episode_doc.get("sha256"),
        speech_policy=manifest.get("speech_policy"),
        status=status,
        wall_seconds=totals.get("wall_seconds"),
        real_time_factor=totals.get("real_time_factor"),
        model_load_seconds=totals.get("load_seconds"),
        peak_rss_bytes=peak_rss_bytes,
        audio_seconds=totals.get("audio_seconds"),
        line_count=totals.get("line_count"),
        tool_versions=tool_versions,
        performance_extra={
            "generate_seconds": totals.get("generate_seconds"),
            **(performance_extra or {}),
        },
        audio_extra={
            "duration_seconds": totals.get("duration_seconds"),
            "gap_seconds": totals.get("gap_seconds"),
            "sample_rate": engine.get("sample_rate"),
        },
        lines_extra={
            "placed_line_count": totals.get("placed_line_count"),
            "script_line_count": manifest.get("script_line_count"),
            "truncated_lines": totals.get("truncated_lines"),
            "truncation_retry_lines": totals.get("truncation_retry_lines"),
            "overrun_lines": totals.get("overrun_lines"),
        },
        loudness={
            "meter": assembly.get("loudness_meter"),
            "target_lufs": assembly.get("target_lufs"),
            "mean_output_lufs": totals.get("mean_output_lufs"),
            "mean_shortfall_db": totals.get("mean_loudness_shortfall_db"),
            "max_shortfall_db": totals.get("max_loudness_shortfall_db"),
            "peak_limited_lines": totals.get("peak_limited_lines"),
            "unnormalized_lines": totals.get("unnormalized_lines"),
        },
        host=host,
        run=run,
        outputs=manifest.get("outputs", {}),
        notes=notes,
    )


def failed_entry(
    *,
    condition: str,
    stack: str,
    episode: str,
    speech_policy: str,
    tool_versions: dict[str, Any],
    error: str,
    engine: str | None = None,
    checkpoint: str | None = None,
    label: str | None = None,
    voices: str | None = None,
    episode_path: str | None = None,
    peak_rss_bytes: int | None = None,
    host: dict[str, Any] | None = None,
    run: dict[str, Any] | None = None,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """An entry recording that a run failed, so the matrix shows the hole."""
    return make_entry(
        condition=condition,
        stack=stack,
        label=label,
        engine=engine,
        checkpoint=checkpoint,
        voices=voices,
        episode=episode,
        episode_path=episode_path,
        speech_policy=speech_policy,
        status=STATUS_FAILED,
        error=error,
        wall_seconds=None,
        real_time_factor=None,
        model_load_seconds=None,
        peak_rss_bytes=peak_rss_bytes,
        audio_seconds=None,
        line_count=None,
        tool_versions=tool_versions,
        host=host,
        run=run,
        notes=notes,
    )


def validate_entry(entry: dict[str, Any]) -> None:
    """Raise :class:`MetricsError` unless ``entry`` satisfies the schema."""
    missing = [key for key in REQUIRED_KEYS if key not in entry]
    if missing:
        raise MetricsError(f"metrics entry is missing: {', '.join(missing)}")
    for group, keys in REQUIRED_GROUP_KEYS.items():
        section = entry.get(group)
        if not isinstance(section, dict):
            raise MetricsError(f"metrics entry field {group!r} must be an object")
        absent = [key for key in keys if key not in section]
        if absent:
            raise MetricsError(
                f"metrics entry {group!r} is missing: {', '.join(absent)}"
            )
    if entry["status"] not in (STATUS_OK, STATUS_FAILED, STATUS_SKIPPED):
        raise MetricsError(f"unknown metrics status {entry['status']!r}")
    if entry["engine"] is None and entry["checkpoint"] is None:
        raise MetricsError(
            "metrics entry needs at least one of engine / checkpoint so the "
            "report can say what produced the audio"
        )


# ---------------------------------------------------------------------------
# Documents on disk
# ---------------------------------------------------------------------------


def document(
    entries: Iterable[dict[str, Any]],
    *,
    generated_by: str = "comparativa bench",
) -> dict[str, Any]:
    """Wrap ``entries`` in the versioned document envelope."""
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "generated_by": generated_by,
        "written_at": _now(),
        "entries": list(entries),
    }


def write_metrics(
    path: str | Path,
    entries: Iterable[dict[str, Any]],
    *,
    generated_by: str = "comparativa bench",
) -> Path:
    """Write a metrics document, creating parent directories as needed."""
    target = Path(path).expanduser()
    if target.is_dir():
        target = target / METRICS_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    doc = document(entries, generated_by=generated_by)
    for entry in doc["entries"]:
        validate_entry(entry)
    target.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return target


def read_metrics(path: str | Path) -> list[dict[str, Any]]:
    """Read one metrics document's entries (tolerates a bare entry list)."""
    doc = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(doc, list):
        return list(doc)
    entries = doc.get("entries")
    if entries is None:
        raise MetricsError(f"{path}: not a metrics document (no 'entries')")
    return list(entries)


def entry_key(entry: dict[str, Any]) -> tuple[str, str, str]:
    """The identity of a run: stack, condition, episode."""
    return (
        str(entry.get("stack")),
        str(entry.get("condition")),
        str(entry.get("episode")),
    )


def load_entries(bench_dir: str | Path) -> list[dict[str, Any]]:
    """Every metrics entry under ``bench_dir``, de-duplicated.

    Globs both ``metrics.json`` and ``*/metrics.json`` at any depth, so it finds
    per-run files (``bench/C/<episode>/metrics.json``) and condition-level files
    (``bench/A/metrics.json``, Sortie 8) alike. Two files describing the same
    ``(stack, condition, episode)`` collapse to the deepest — i.e. the per-run
    file wins over an aggregate that merely repeats it.
    """
    root = Path(bench_dir).expanduser()
    found: dict[tuple[str, str, str], dict[str, Any]] = {}
    depth: dict[tuple[str, str, str], int] = {}
    paths = sorted(root.rglob(METRICS_FILENAME)) + (
        [root / METRICS_FILENAME] if (root / METRICS_FILENAME).is_file() else []
    )
    for path in dict.fromkeys(paths):
        try:
            entries = read_metrics(path)
        except (MetricsError, json.JSONDecodeError, OSError):
            continue
        level = len(path.relative_to(root).parts)
        for entry in entries:
            key = entry_key(entry)
            if key not in found or level > depth[key]:
                found[key] = entry
                depth[key] = level
    return [found[key] for key in sorted(found)]


__all__ = [
    "METRICS_FILENAME",
    "METRICS_SCHEMA_VERSION",
    "REQUIRED_GROUP_KEYS",
    "REQUIRED_KEYS",
    "STATUS_FAILED",
    "STATUS_OK",
    "STATUS_SKIPPED",
    "SUMMARY_FILENAME",
    "MetricsError",
    "document",
    "entry_from_manifest",
    "entry_key",
    "failed_entry",
    "load_entries",
    "make_entry",
    "read_metrics",
    "validate_entry",
    "write_metrics",
]
