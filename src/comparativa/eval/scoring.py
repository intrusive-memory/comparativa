"""Unblind a filled-in ``scoring_sheet.csv`` via ``key.csv`` and aggregate it.

``comparativa listen`` (:mod:`.blind`) hands the listener a ``set/`` of
opaque-id clips and a blank ``scoring_sheet.csv``. Once a human fills in the
FR-13 score columns, ``comparativa report`` reads the scoring sheet back
alongside the ``key.csv`` it was blinded with, joins the two on ``opaque_id``,
and aggregates the numeric columns per condition.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .blind import SCORE_COLUMNS
from .tables import markdown_table


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_key(path: str | Path) -> dict[str, dict[str, str]]:
    """``opaque_id -> key row`` from a ``key.csv`` written by ``listen``."""
    return {row["opaque_id"]: row for row in _read_csv(path) if row.get("opaque_id")}


def load_scores(path: str | Path) -> list[dict[str, str]]:
    """The rows of a (possibly partially filled-in) ``scoring_sheet.csv``."""
    return _read_csv(path)


def unblind_scores(
    score_rows: list[dict[str, str]], key_by_id: dict[str, dict[str, str]]
) -> list[dict[str, str]]:
    """Join scored rows to their key row on ``opaque_id``.

    Rows whose ``opaque_id`` is missing from the key are dropped rather than
    raising — a scoring sheet edited by hand can pick up stray rows or typos,
    and one bad row shouldn't sink the whole report.
    """
    merged: list[dict[str, str]] = []
    for row in score_rows:
        key_row = key_by_id.get(row.get("opaque_id", ""))
        if key_row is None:
            continue
        merged.append({**key_row, **row})
    return merged


def aggregate_by_condition(merged_rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    """``condition -> {"n": scored-row count, "means": {score column -> mean}}``.

    ``n`` counts rows scored for that condition at all (a row with only some
    columns filled in still counts); each column's mean is over its own
    non-blank, numeric cells only.
    """
    counts: dict[str, int] = {}
    sums: dict[str, dict[str, list[float]]] = {}
    for row in merged_rows:
        filled = {
            column: row[column]
            for column in SCORE_COLUMNS
            if row.get(column) is not None and str(row[column]).strip() != ""
        }
        if not filled:
            # A blinded clip with no score filled in yet isn't "scored" —
            # skip it so a freshly generated, still-blank scoring_sheet.csv
            # renders as "empty until human scores arrive", not as N clips
            # of hollow zero-column scores.
            continue
        condition = row.get("condition") or "(unknown)"
        counts[condition] = counts.get(condition, 0) + 1
        for column, raw in filled.items():
            try:
                value = float(raw)
            except ValueError:
                continue
            sums.setdefault(condition, {}).setdefault(column, []).append(value)
    return {
        condition: {
            "n": counts[condition],
            "means": {
                column: sum(values) / len(values)
                for column, values in sums.get(condition, {}).items()
            },
        }
        for condition in counts
    }


def render_listening_score_table(aggregates: dict[str, dict[str, Any]]) -> str:
    """Render the per-condition mean-score table (FR-13 columns)."""
    if not aggregates:
        return (
            "_empty until human scores arrive — fill in `scoring_sheet.csv` and "
            "re-run `comparativa report --scores ... --key ...`._"
        )
    headers = ["condition", "n", *SCORE_COLUMNS]
    rows: list[list[str]] = []
    for condition in sorted(aggregates):
        entry = aggregates[condition]
        means = entry["means"]
        row = [condition, str(entry["n"])]
        row.extend(
            f"{means[column]:.2f}" if column in means else "—" for column in SCORE_COLUMNS
        )
        rows.append(row)
    return markdown_table(headers, rows)


__all__ = [
    "aggregate_by_condition",
    "load_key",
    "load_scores",
    "render_listening_score_table",
    "unblind_scores",
]
