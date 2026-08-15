"""``comparativa parse`` — emit a screenplay's element stream as JSON.

The payload is the verbatim element stream (Sortie 2, FR-1) plus the speech
projection (Sortie 3, FR-2/FR-3/FR-4):

* each element gains ``spoken`` — ``true`` only when it produced a spoken line;
* ``speech`` carries the prepared per-line TTS text, resolved characters, and
  non-spoken performance directions;
* ``unresolved_cues`` lists every character cue that matched no ``CAST.md``
  character. It is empty on a healthy parse.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .cues import CastIndex, find_cast_file, load_cast_index
from .fountain import FountainParseError, parse_file
from .speech import FR3_POLICY, POLICIES, prepare_script


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
    parser.add_argument(
        "--cast",
        metavar="PATH",
        help=(
            "CAST.md to resolve character cues against. Default: the nearest "
            "CAST.md at or above the episode's directory."
        ),
    )
    parser.add_argument(
        "--no-cast",
        action="store_true",
        help="Skip CAST.md resolution entirely (cues are used as normalised).",
    )
    parser.add_argument(
        "--speech-policy",
        choices=sorted(POLICIES),
        default=FR3_POLICY.name,
        help=(
            "Which elements are speech. 'fr3' (default) voices dialogue only; "
            "'produciesta-parity' also narrates action, scene headings, "
            "centered text and transitions, as the Swift stack does."
        ),
    )
    parser.add_argument(
        "--strict-cues",
        action="store_true",
        help="Exit non-zero when any cue fails to resolve against CAST.md.",
    )


def _resolve_cast(args: argparse.Namespace) -> CastIndex | None:
    """Load the CAST.md index selected by the CLI flags, or ``None``."""
    if args.no_cast:
        return None
    cast_file = (
        Path(args.cast).expanduser()
        if args.cast
        else find_cast_file(args.episode)
    )
    if cast_file is None:
        return None
    return load_cast_index(cast_file)


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

    try:
        cast = _resolve_cast(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"comparativa parse: {exc}", file=sys.stderr)
        return 1

    script = prepare_script(stream, cast, policy=POLICIES[args.speech_policy])

    payload = stream.to_dict()
    for element in payload["elements"]:
        element["spoken"] = element["index"] in script.spoken_elements
    payload["speech"] = script.to_dict()
    payload["unresolved_cues"] = [c.to_dict() for c in script.unresolved_cues]

    text = json.dumps(payload, indent=args.indent or None, ensure_ascii=False)
    if args.output:
        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)

    if args.strict_cues and script.unresolved_cues:
        names = sorted({c.normalized for c in script.unresolved_cues})
        print(
            f"comparativa parse: {len(script.unresolved_cues)} unresolved "
            f"cue(s): {', '.join(names)}",
            file=sys.stderr,
        )
        return 1
    return 0
