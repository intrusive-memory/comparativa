"""``comparativa listen`` — build a blinded listening set + scoring sheet.

    comparativa listen bench/ -o bench/listen --seed 20260815

Discovers every rendered clip under a bench directory (:mod:`.blind`), copies
each into ``<out>/set/<opaque-id>.wav`` in randomized order, and writes
``<out>/key.csv`` (the unblinding key) and ``<out>/scoring_sheet.csv`` (blank
FR-13 score columns for a human listener to fill in).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .blind import BlindError, blind

#: Default output directory, relative to the bench dir.
DEFAULT_OUT_SUBDIR = "listen"


def configure(parser: argparse.ArgumentParser) -> None:
    """Register the ``listen`` subcommand's arguments."""
    parser.add_argument(
        "bench_dir",
        nargs="?",
        metavar="BENCH_DIR",
        help="Bench directory produced by `comparativa bench` (e.g. bench/). Omit to print this help.",
    )
    parser.add_argument(
        "-o",
        "--out",
        metavar="DIR",
        default=None,
        help=f"Output directory for the listening set (default: <bench-dir>/{DEFAULT_OUT_SUBDIR}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="RNG seed for the blinding order (default: unseeded/random each run).",
    )


def handle(args: argparse.Namespace) -> int:
    """Run ``comparativa listen``. Returns the process exit code."""
    if not args.bench_dir:
        args.parser.print_help()
        return 0

    bench_dir = Path(args.bench_dir)
    out_dir = Path(args.out) if args.out else bench_dir / DEFAULT_OUT_SUBDIR

    try:
        listen_set = blind(bench_dir, out_dir, seed=args.seed)
    except BlindError as exc:
        print(f"comparativa listen: {exc}", file=sys.stderr)
        return 1

    print(f"{len(listen_set)} clip(s) blinded")
    print(f"set:     {listen_set.set_dir}")
    print(f"key:     {listen_set.key_path}")
    print(f"scoring: {listen_set.scoring_sheet_path}")
    return 0


__all__ = ["configure", "handle"]
