"""``comparativa report`` — tabulate metrics + scores into a ``REPORT.md`` skeleton.

    comparativa report bench/ -o REPORT.md

Reads every ``bench/<condition>/<episode>/metrics.json`` plus the
``scoring_sheet.csv``/``key.csv`` pair written by ``comparativa listen``
(default location: ``<bench-dir>/listen/``), and writes the rendered
``REPORT.md`` (see :mod:`.report`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .listen_command import DEFAULT_OUT_SUBDIR
from .report import write_report

DEFAULT_REPORT_FILENAME = "REPORT.md"


def configure(parser: argparse.ArgumentParser) -> None:
    """Register the ``report`` subcommand's arguments."""
    parser.add_argument(
        "bench_dir",
        nargs="?",
        metavar="BENCH_DIR",
        help="Bench directory produced by `comparativa bench` (e.g. bench/). Omit to print this help.",
    )
    parser.add_argument(
        "-o",
        "--out",
        metavar="PATH",
        default=None,
        help=(
            f"Where to write {DEFAULT_REPORT_FILENAME} "
            f"(default: <bench-dir's parent>/{DEFAULT_REPORT_FILENAME})."
        ),
    )
    parser.add_argument(
        "--scores",
        metavar="PATH",
        default=None,
        help=(
            "scoring_sheet.csv with human scores filled in "
            f"(default: <bench-dir>/{DEFAULT_OUT_SUBDIR}/scoring_sheet.csv)."
        ),
    )
    parser.add_argument(
        "--key",
        metavar="PATH",
        default=None,
        help=(
            "key.csv written by `comparativa listen` "
            f"(default: <bench-dir>/{DEFAULT_OUT_SUBDIR}/key.csv)."
        ),
    )


def handle(args: argparse.Namespace) -> int:
    """Run ``comparativa report``. Returns the process exit code."""
    if not args.bench_dir:
        args.parser.print_help()
        return 0

    bench_dir = Path(args.bench_dir)
    out_path = Path(args.out) if args.out else bench_dir.parent / DEFAULT_REPORT_FILENAME
    listen_dir = bench_dir / DEFAULT_OUT_SUBDIR
    scores_path = Path(args.scores) if args.scores else listen_dir / "scoring_sheet.csv"
    key_path = Path(args.key) if args.key else listen_dir / "key.csv"

    try:
        written = write_report(bench_dir, out_path, scores_path=scores_path, key_path=key_path)
    except OSError as exc:
        print(f"comparativa report: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {written}")
    return 0


__all__ = ["configure", "handle"]
