"""``comparativa report`` — tabulate bench metrics + listening scores into ``REPORT.md``.

Renders results tables from every ``bench/<condition>/<episode>/metrics.json``
(Sorties 7-8) and the ``scoring_sheet.csv`` + ``key.csv`` a human filled in
after ``comparativa listen`` (Sortie 9, :mod:`.blind`), unblinding the scores
via the key. The verdict sections below the tables are **templated
placeholders** — Sortie 11 is the one that writes the actual per-hypothesis
verdicts once real human listening scores exist (EXECUTION_PLAN.md Sortie 11);
this sortie only has to prove the skeleton renders correctly from fixture
data.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from .metrics import discover_metrics, render_performance_table
from .proxy import objective_proxy_status
from .scoring import (
    aggregate_by_condition,
    load_key,
    load_scores,
    render_listening_score_table,
    unblind_scores,
)

#: The two caveat rows the supervisor requires carried forward into every
#: report, verbatim in substance (EXECUTION_PLAN.md, Sortie 9 handoff).
CAVEAT_ROWS: tuple[str, ...] = (
    "Condition C has zero English female qwen3 presets — heavy voice reuse "
    "and accent artifacts are expected wherever it voices a female character.",
    "Loudness: many lines cannot reach -16 LUFS integrated without clipping; "
    "the peak-limited-line counts and shortfall stats are recorded per line "
    "and per episode in each condition's manifest.json / metrics.json "
    "(see docs/ASSEMBLY.md §3).",
)

#: The three scored hypotheses (EXECUTION_PLAN.md § Conditions Matrix), each
#: with its templated description. Sortie 11 fills in the ``Verdict:`` line.
VERDICT_SECTIONS: tuple[tuple[str, str], ...] = (
    (
        "swift-port",
        "E vs F is the clean pair — same checkpoint (Soprano-80M-bf16), same "
        "single built-in voice, only the stack differs. A vs C also bears on "
        "this hypothesis but **confounds** the port with voice design (A uses "
        "production `.vox` voices, C uses auto-assigned built-in presets) and "
        "must be read with that confound stated.",
    ),
    (
        "model-ceiling",
        "C vs D/E probes whether a larger/different checkpoint (qwen3-1.7b vs "
        "chatterbox vs Soprano-80M) actually sounds better, independent of "
        "stack.",
    ),
    (
        "voice-design",
        "Partial evidence only this round: condition B (qwen3 cloned from "
        "`.vox` reference audio) is deferred to the follow-on custom-voices "
        "mission, so a definitive voice-design verdict is not possible from "
        "round-1 defaults-only data alone.",
    ),
)

REPORT_TITLE = "# REPORT — OPERATION BATTLING BARDS round 1"


def _listening_scores(
    scores_path: Path | None, key_path: Path | None
) -> tuple[dict[str, dict], int]:
    """Unblind + aggregate scores; ``(aggregates, scored-row-count)``."""
    if not scores_path or not key_path:
        return {}, 0
    if not Path(scores_path).exists() or not Path(key_path).exists():
        return {}, 0
    key_by_id = load_key(key_path)
    score_rows = load_scores(scores_path)
    merged = unblind_scores(score_rows, key_by_id)
    aggregates = aggregate_by_condition(merged)
    scored_n = sum(entry["n"] for entry in aggregates.values())
    return aggregates, scored_n


def render_report(
    bench_dir: str | Path,
    *,
    scores_path: str | Path | None = None,
    key_path: str | Path | None = None,
) -> str:
    """Render the full ``REPORT.md`` skeleton as a single Markdown string."""
    bench_dir = Path(bench_dir)
    metrics_records = discover_metrics(bench_dir)
    performance_table = render_performance_table(metrics_records)

    aggregates, scored_n = _listening_scores(
        Path(scores_path) if scores_path else None,
        Path(key_path) if key_path else None,
    )
    listening_table = render_listening_score_table(aggregates)

    _, proxy_status = objective_proxy_status()

    lines: list[str] = [
        REPORT_TITLE,
        "",
        f"_generated {_dt.datetime.now().isoformat(timespec='seconds')} by "
        "`comparativa report`_",
        "",
        f"bench dir: `{bench_dir}`",
        "",
        "## Caveats",
        "",
        *(f"- {row}" for row in CAVEAT_ROWS),
        "",
        "## Performance",
        "",
        performance_table,
        "",
        "## Listening scores",
        "",
        (
            f"_{scored_n} scored clip(s) unblinded via the key file._"
            if scored_n
            else (
                "_empty until human scores arrive — run `comparativa listen`, "
                "fill in `scoring_sheet.csv`, then re-run `comparativa report`._"
            )
        ),
        "",
        listening_table,
        "",
        "## Objective proxy — STT WER round-trip",
        "",
        f"objective proxy: {proxy_status}",
        "",
        "## Verdicts",
        "",
        "_Skeleton only — Sortie 11 writes the actual verdicts once human "
        "listening scores exist (EXECUTION_PLAN.md Sortie 11). Nothing below "
        "this line is a finding._",
        "",
    ]
    for name, description in VERDICT_SECTIONS:
        lines.extend(
            [
                f"### {name}",
                "",
                description,
                "",
                "**Verdict:** _pending — insufficient evidence until Sortie 10 "
                "(execution) and Sortie 11 (verdict) run._",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def write_report(
    bench_dir: str | Path,
    out_path: str | Path,
    *,
    scores_path: str | Path | None = None,
    key_path: str | Path | None = None,
) -> Path:
    """Render and write ``REPORT.md``; returns the path written."""
    text = render_report(bench_dir, scores_path=scores_path, key_path=key_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path


__all__ = [
    "CAVEAT_ROWS",
    "REPORT_TITLE",
    "VERDICT_SECTIONS",
    "render_report",
    "write_report",
]
