"""Console entry point for ``comparativa``.

Sortie 1 scaffold: every subcommand is a stub that prints its own help and
exits 0. Later sorties replace the ``_stub`` handlers with real
implementations; the command surface itself is fixed here.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__

#: The six subcommands the CLI exposes, in the order they appear in help.
SUBCOMMANDS: tuple[tuple[str, str], ...] = (
    ("parse", "Parse a Fountain screenplay into a JSON element stream."),
    ("voices", "Build the character roster and per-engine voice assignments."),
    ("generate", "Generate episode audio for one engine condition."),
    ("bench", "Run the benchmark matrix across conditions and episodes."),
    ("listen", "Build a blinded listening set and scoring sheet."),
    ("report", "Tabulate metrics and scores into a report."),
)

#: Just the names, for callers (and tests) that only need the command list.
SUBCOMMAND_NAMES: tuple[str, ...] = tuple(name for name, _ in SUBCOMMANDS)


def _stub(parser: argparse.ArgumentParser) -> int:
    """Placeholder handler: print the subcommand's help and succeed."""
    parser.print_help()
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the full argument parser, including all six subcommands."""
    parser = argparse.ArgumentParser(
        prog="comparativa",
        description=(
            "Fountain-to-speech reference pipeline for comparing the Python "
            "MLX stack against the Swift stack."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"comparativa {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    for name, help_text in SUBCOMMANDS:
        sub = subparsers.add_parser(name, help=help_text, description=help_text)
        sub.set_defaults(handler=_stub, parser=sub)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "command", None) is None:
        parser.print_help()
        return 0

    handler = args.handler
    return handler(args.parser)


if __name__ == "__main__":  # pragma: no cover - exercised via the console script
    sys.exit(main())
