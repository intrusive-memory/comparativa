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

#: Where ``--audition`` writes its takes (gitignored, like every audio output).
DEFAULT_AUDITION_DIR = "out/audition"


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

    audition = parser.add_argument_group("audition")
    audition.add_argument(
        "--audition",
        action="store_true",
        help=(
            "Generate one fixed sentence per character x engine assignment "
            "(loads each engine's checkpoint; slow)."
        ),
    )
    audition.add_argument(
        "--audition-out",
        metavar="DIR",
        default=DEFAULT_AUDITION_DIR,
        help=f"Directory for the audition wavs (default: {DEFAULT_AUDITION_DIR}).",
    )
    audition.add_argument(
        "--audition-text",
        metavar="TEXT",
        default=None,
        help="Sentence to audition (default: the built-in fixed sentence).",
    )
    audition.add_argument(
        "--audition-characters",
        metavar="NAMES",
        default=None,
        help="Comma-separated subset of characters to audition (default: the whole cast).",
    )
    audition.add_argument(
        "--audition-seed",
        type=int,
        default=None,
        help="Base RNG seed for the audition; -1 leaves the RNG untouched.",
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

    if args.audition:
        return run_audition_from_args(args, assignments)

    return 0


def run_audition_from_args(args: argparse.Namespace, assignments: Assignments) -> int:
    """Handle ``--audition``: generate one fixed sentence per character × engine.

    Imported lazily so the plain ``voices`` path never pays for mlx-audio.
    """
    from ..generation.audition import (
        AUDITION_SEED,
        AUDITION_TEXT,
        format_table,
        run_audition,
        write_audition_manifest,
    )

    names = None
    if args.audition_characters:
        names = [n.strip() for n in str(args.audition_characters).split(",") if n.strip()]

    by_character = {c.character: dict(c.voices) for c in assignments.characters}
    seed = AUDITION_SEED if args.audition_seed is None else args.audition_seed
    if seed is not None and seed < 0:
        seed = None

    print()
    try:
        run = run_audition(
            by_character,
            engine_keys=assignments.engine_keys,
            characters=names,
            out_dir=args.audition_out,
            text=args.audition_text or AUDITION_TEXT,
            seed=seed,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"error: audition failed: {exc}", file=sys.stderr)
        return 1

    print(format_table(run))
    print(f"\nwrote {write_audition_manifest(run, args.audition_out)}")
    return 0 if run.ok else 1
