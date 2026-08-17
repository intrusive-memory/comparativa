"""Tabulate ``bench/<condition>/<episode>/metrics.json`` into a results table.

Sortie 7 (Python conditions C/D/E) and Sortie 8 (Swift conditions A/F) both
write ``metrics.json`` in a schema shared across stacks: ``condition``,
``stack``, ``engine``, ``episode``, ``wall_seconds``, ``rtf``,
``peak_rss_bytes``. That schema was being finalized concurrently with this
sortie, so this module renders whatever numeric fields a ``metrics.json``
actually presents, in addition to the known ones, rather than hard-failing on
extras or missing keys.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .tables import markdown_table

#: The performance fields every condition is expected to record.
KNOWN_FIELDS: tuple[str, ...] = (
    "condition",
    "stack",
    "engine",
    "episode",
    "wall_seconds",
    "rtf",
    "peak_rss_bytes",
)

METRICS_FILENAME = "metrics.json"


@dataclass(frozen=True)
class MetricsRecord:
    """One ``metrics.json`` file, plus the condition/episode it belongs to."""

    condition: str
    episode: str
    path: Path
    data: dict[str, Any] = field(default_factory=dict)


def _resolve_entry(raw: Any, dir_condition: str, dir_episode: str) -> dict[str, Any]:
    """Unwrap a schema_version-1 envelope (``{"entries": [...]}}``) to the
    entry for this run directory.

    Sortie 7/8 write ``metrics.json`` as an envelope, not a bare entry
    (see :mod:`comparativa.bench.metrics`); reading the envelope itself as
    if it *were* an entry silently yields ``None`` for every field (``stack``,
    ``engine``, ``performance``, ...) since none of those keys exist at the
    envelope's top level. This also tolerates a bare single-entry dict (no
    ``entries`` wrapper), the shape older fixtures use, mirroring
    :func:`comparativa.eval.blind._resolve_entry`.
    """
    if not isinstance(raw, dict):
        return {"_error": "metrics.json is not a JSON object"}
    entries = raw.get("entries")
    if isinstance(entries, list):
        dict_entries = [e for e in entries if isinstance(e, dict)]
        for entry in dict_entries:
            if entry.get("condition") == dir_condition and entry.get("episode") == dir_episode:
                return entry
        # No exact (condition, episode) match — fall back to the first entry
        # rather than an empty dict, so the run still gets *some* fields.
        return dict_entries[0] if dict_entries else {}
    return raw


def discover_metrics(bench_dir: str | Path) -> list[MetricsRecord]:
    """Find every ``metrics.json`` under ``bench_dir``.

    The condition and episode are read from the resolved entry when present,
    falling back to the directory names (``bench/<condition>/<episode>/
    metrics.json``) so a minimal or malformed file still gets a row instead
    of vanishing from the table.
    """
    bench_dir = Path(bench_dir)
    records: list[MetricsRecord] = []
    for path in sorted(bench_dir.glob(f"*/*/{METRICS_FILENAME}")):
        episode_dir = path.parent
        dir_condition = episode_dir.parent.name
        dir_episode = episode_dir.name
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            records.append(
                MetricsRecord(
                    condition=dir_condition,
                    episode=dir_episode,
                    path=path,
                    data={"_error": str(exc)},
                )
            )
            continue
        data = _resolve_entry(raw, dir_condition, dir_episode)
        condition = str(data.get("condition") or dir_condition)
        episode = str(data.get("episode") or dir_episode)
        records.append(MetricsRecord(condition=condition, episode=episode, path=path, data=data))
    return records


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def extra_numeric_fields(records: list[MetricsRecord]) -> list[str]:
    """Numeric top-level fields present beyond :data:`KNOWN_FIELDS`."""
    extras: set[str] = set()
    for record in records:
        for key, value in record.data.items():
            if key in KNOWN_FIELDS or key.startswith("_"):
                continue
            if _is_number(value):
                extras.add(key)
    return sorted(extras)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _perf(data: dict[str, Any], flat_key: str, nested_key: str) -> Any:
    """A performance number, tolerant of both the frozen schema (nested
    under ``performance``) and a flat bare-entry shape (older fixtures).

    Checks the flat key first so an explicit top-level override still wins,
    then falls back to ``performance.<nested_key>`` — where the frozen
    ``metrics.json`` schema (docs/BENCH.md §4) actually carries
    ``wall_seconds``, ``real_time_factor``, and ``peak_rss_bytes``.
    """
    if flat_key in data:
        return data[flat_key]
    performance = data.get("performance")
    if isinstance(performance, dict):
        return performance.get(nested_key)
    return None


def render_performance_table(records: list[MetricsRecord]) -> str:
    """Render the performance table: known columns + any extra numeric ones."""
    if not records:
        return "_no metrics.json files found under the bench dir yet._"

    extras = extra_numeric_fields(records)
    has_errors = any("_error" in record.data for record in records)
    headers = [
        "condition",
        "episode",
        "stack",
        "engine",
        "wall_seconds",
        "rtf",
        "peak_rss_bytes",
        *extras,
        *(["error"] if has_errors else []),
    ]
    rows: list[list[str]] = []
    for record in sorted(records, key=lambda r: (r.condition, r.episode)):
        data = record.data
        row = [
            record.condition,
            record.episode,
            _fmt(data.get("stack")),
            _fmt(data.get("engine")),
            _fmt(_perf(data, "wall_seconds", "wall_seconds")),
            _fmt(_perf(data, "rtf", "real_time_factor")),
            _fmt(_perf(data, "peak_rss_bytes", "peak_rss_bytes")),
        ]
        row.extend(_fmt(data.get(name)) for name in extras)
        if has_errors:
            row.append(str(data.get("_error", "")))
        rows.append(row)

    return markdown_table(headers, rows)


__all__ = [
    "KNOWN_FIELDS",
    "METRICS_FILENAME",
    "MetricsRecord",
    "discover_metrics",
    "extra_numeric_fields",
    "render_performance_table",
]
