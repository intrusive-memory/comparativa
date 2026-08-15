"""``comparativa parse`` — emit a screenplay's element stream as JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .fountain import FountainParseError, parse_file


def configure(parser: argparse.ArgumentParser) -> None:
    """Register the ``parse`` subcommand's arguments."""
    parser.add_argument(
        "episode",
        nargs="?",
        metavar="EPISODE",
        help="Path to a .fountain screenplay. Omit to print this help.",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="Write JSON here instead of stdout.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent width; 0 for a single compact line (default: 2).",
    )


def handle(args: argparse.Namespace) -> int:
    """Run ``comparativa parse``. Returns the process exit code."""
    if not args.episode:
        args.parser.print_help()
        return 0

    try:
        stream = parse_file(args.episode)
    except FountainParseError as exc:
        print(f"comparativa parse: {exc}", file=sys.stderr)
        return 1

    payload = json.dumps(
        stream.to_dict(),
        indent=args.indent or None,
        ensure_ascii=False,
    )

    if args.output:
        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0
