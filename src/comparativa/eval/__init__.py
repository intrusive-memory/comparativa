"""Evaluation tooling: blinded listening sets and the ``REPORT.md`` skeleton.

Two commands, one round-trip:

* **``comparativa listen``** (:mod:`.blind`) discovers every rendered clip
  under a bench directory, copies each into a randomized, filename-blinded
  ``set/`` directory, and writes ``key.csv`` (the unblinding key, stored
  separately from the listening set) and ``scoring_sheet.csv`` (blank FR-13
  score columns: naturalness, prosody, artifacts, voice consistency across
  lines, character distinctness).
* **``comparativa report``** (:mod:`.report`) tabulates every
  ``metrics.json`` into a performance table, unblinds a filled-in
  ``scoring_sheet.csv`` via ``key.csv`` into a listening-score table
  (:mod:`.scoring`), records the STT WER round-trip objective proxy's status
  (:mod:`.proxy` — included if ``mlx_whisper`` is importable, dropped with a
  reason otherwise), and renders the ``REPORT.md`` skeleton, including the
  templated verdict sections Sortie 11 fills in later.

Typical use::

    from comparativa.eval import blind, write_report

    listen_set = blind("bench/", "bench/listen", seed=20260815)
    write_report(
        "bench/",
        "REPORT.md",
        scores_path=listen_set.scoring_sheet_path,
        key_path=listen_set.key_path,
    )
"""

from .blind import (
    KEY_FIELDS,
    KEY_FILENAME,
    SCORE_COLUMNS,
    SCORING_SHEET_FIELDS,
    SCORING_SHEET_FILENAME,
    SET_DIRNAME,
    BlindError,
    Clip,
    ListenSet,
    blind,
    discover_clips,
)
from .metrics import (
    KNOWN_FIELDS,
    METRICS_FILENAME,
    MetricsRecord,
    discover_metrics,
    render_performance_table,
)
from .proxy import DROPPED_REASON, mlx_whisper_available, objective_proxy_status
from .report import CAVEAT_ROWS, VERDICT_SECTIONS, render_report, write_report
from .scoring import (
    aggregate_by_condition,
    load_key,
    load_scores,
    render_listening_score_table,
    unblind_scores,
)
from .tables import markdown_table

__all__ = [
    "CAVEAT_ROWS",
    "DROPPED_REASON",
    "KEY_FIELDS",
    "KEY_FILENAME",
    "KNOWN_FIELDS",
    "METRICS_FILENAME",
    "SCORE_COLUMNS",
    "SCORING_SHEET_FIELDS",
    "SCORING_SHEET_FILENAME",
    "SET_DIRNAME",
    "VERDICT_SECTIONS",
    "BlindError",
    "Clip",
    "ListenSet",
    "MetricsRecord",
    "aggregate_by_condition",
    "blind",
    "discover_clips",
    "discover_metrics",
    "load_key",
    "load_scores",
    "markdown_table",
    "mlx_whisper_available",
    "objective_proxy_status",
    "render_listening_score_table",
    "render_performance_table",
    "render_report",
    "unblind_scores",
    "write_report",
]
