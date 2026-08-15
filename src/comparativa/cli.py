"""Console entry point for ``comparativa``.

The command surface is fixed by :data:`SUBCOMMANDS`. A subcommand starts life
as a stub that prints its own help and exits 0; a sortie wires up the real
implementation by adding an entry to :data:`_WIRED`, which supplies a
``configure(parser)`` hook for the subcommand's arguments and a
``handle(args) -> int`` handler. Wiring is imported lazily so that ``--help``
never pays for a heavyweight module import.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable, Sequence

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


#: Subcommand name -> module implementing ``configure`` and ``handle``.
#: Each sortie adds its own entry as it lands.
_WIRED: dict[str, str] = {
    "parse": "comparativa.parsing.command",
    "voices": "comparativa.voices.command",
}


def _stub(args: argparse.Namespace) -> int:
    """Placeholder handler: print the subcommand's help and succeed."""
    args.parser.print_help()
    return 0


#: A wired subcommand's ``(configure, handle)`` pair.
Wiring = tuple[
    Callable[[argparse.ArgumentParser], None],
    Callable[[argparse.Namespace], int],
]


def _wiring(name: str) -> Wiring | None:
    """Import the module implementing ``name``, or ``None`` if it is a stub."""
    module_path = _WIRED.get(name)
    if module_path is None:
        return None
    module = importlib.import_module(module_path)
    return module.configure, module.handle


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
        wiring = _wiring(name)
        if wiring is None:
            sub.set_defaults(handler=_stub, parser=sub)
        else:
            configure, handle = wiring
            configure(sub)
            sub.set_defaults(handler=handle, parser=sub)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "command", None) is None:
        parser.print_help()
        return 0

    handler = args.handler
    return handler(args)


if __name__ == "__main__":  # pragma: no cover - exercised via the console script
    sys.exit(main())
