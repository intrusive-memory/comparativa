"""``comparativa voices`` — roster + per-engine default-voice assignments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .assign import PRESETS_FILENAME, Assignments, assign, dump_yaml, to_document, write_presets
from .catalog import ENGINES, ENGINE_KEYS, engine as get_engine
from .roster import RosterError, load_roster

#: Repository root, where the committed ``presets.yaml`` lives.
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PRESETS_PATH = REPO_ROOT / PRESETS_FILENAME


def configure(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the ``voices`` arguments to the subparser built in ``cli.py``."""
    parser.add_argument(
        "project_dir",
        nargs="?",
        metavar="PROJECT_DIR",
        help="Podcast project directory containing CAST.md (read-only). Omit to print this help.",
    )
    parser.add_argument(
        "--engines",
        default=",".join(ENGINE_KEYS),
        help=f"Comma-separated engine keys (default: all — {', '.join(ENGINE_KEYS)}).",
    )
    parser.add_argument(
        "--write",
        nargs="?",
        const=str(DEFAULT_PRESETS_PATH),
        default=None,
        metavar="PATH",
        help=f"Write the assignments to PATH (default: {DEFAULT_PRESETS_PATH}).",
    )
    parser.add_argument(
        "--format",
        choices=("table", "yaml"),
        default="table",
        help="Output format for stdout (default: table).",
    )
    return parser


def render_table(assignments: Assignments) -> str:
    """Render character -> per-engine voice as a fixed-width table."""
    headers = ["CHARACTER", *(k.upper() for k in assignments.engine_keys)]
    rows: list[list[str]] = []
    for character in assignments.characters:
        rows.append(
            [
                character.character,
                *(character.voices.get(key) or "UNRESOLVED" for key in assignments.engine_keys),
            ]
        )

    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows)) if rows else len(headers[column])
        for column in range(len(headers))
    ]
    lines = [
        "  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))).rstrip(),
        "  ".join("-" * widths[i] for i in range(len(headers))),
    ]
    lines.extend(
        "  ".join(row[i].ljust(widths[i]) for i in range(len(headers))).rstrip() for row in rows
    )
    return "\n".join(lines)


def handle(args: argparse.Namespace) -> int:
    """Handler for ``comparativa voices``."""
    if not args.project_dir:
        args.parser.print_help()
        return 0

    try:
        roster = load_roster(args.project_dir)
    except RosterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    keys = [key.strip() for key in str(args.engines).split(",") if key.strip()]
    try:
        engines = tuple(get_engine(key) for key in keys) if keys else ENGINES
    except KeyError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 2

    assignments = assign(roster, engines)

    if args.format == "yaml":
        print(dump_yaml(to_document(assignments)), end="")
    else:
        print(f"{roster.cast_file}  ({len(roster)} characters, {len(engines)} engines)")
        print(render_table(assignments))
        for character in assignments.characters:
            for note in character.notes:
                print(f"note: {character.character}: {note}")

    if args.write:
        written = write_presets(assignments, args.write)
        print(f"wrote {written}")

    unresolved = assignments.unresolved
    if unresolved:
        for character in unresolved:
            print(
                f"error: {character.character} has no voice for: "
                f"{', '.join(character.unresolved_engines)}",
                file=sys.stderr,
            )
        return 1

    return 0
