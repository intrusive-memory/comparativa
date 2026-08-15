"""``comparativa generate`` — one engine condition, one episode, end to end.

Voices always come from the committed ``presets.yaml`` (RD-2: round 1 is
defaults-only, there are no cloning flags). One run writes into one output
directory: ``<out>/<episode>.wav``, ``<out>/<episode>.m4a``, and
``<out>/manifest.json``.

    HF_HUB_OFFLINE=1 uv run comparativa generate \\
        ~/Projects/podcasts/granville/episodes/episode_1_01_cold_open.fountain \\
        --engine qwen3-0.6b -o out/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..parsing.speech import FR3_POLICY, POLICIES
from .assembly import (
    LINE_GAP_SECONDS,
    PEAK_CEILING,
    SCENE_GAP_SECONDS,
    TARGET_LUFS,
    AssemblyOptions,
)
from .encode import EncodeError, M4A_BITRATE
from .engines import ENGINE_KEYS, EngineError
from .episode import (
    DEFAULT_PRESETS_PATH,
    DEFAULT_SEED,
    EpisodePlan,
    GenerationError,
    PlannedLine,
    build_plan,
    generate_episode,
    write_outputs,
)


def configure(parser: argparse.ArgumentParser) -> None:
    """Register the ``generate`` subcommand's arguments."""
    parser.add_argument(
        "episode",
        nargs="?",
        metavar="EPISODE",
        help="Path to a .fountain screenplay. Omit to print this help.",
    )
    parser.add_argument(
        "--engine",
        choices=ENGINE_KEYS,
        help=f"Engine key ({', '.join(ENGINE_KEYS)}).",
    )
    parser.add_argument(
        "-o",
        "--out",
        metavar="DIR",
        default="out",
        help="Output directory for the .wav, .m4a and manifest.json (default: out).",
    )
    parser.add_argument(
        "--name",
        metavar="STEM",
        help="Basename for the audio files (default: the episode's filename stem).",
    )

    voices = parser.add_argument_group("voices")
    voices.add_argument(
        "--presets",
        metavar="PATH",
        default=str(DEFAULT_PRESETS_PATH),
        help=f"presets.yaml with the per-engine voice assignments (default: {DEFAULT_PRESETS_PATH}).",
    )
    voices.add_argument(
        "--cast",
        metavar="PATH",
        help="CAST.md to resolve cues against (default: nearest to the episode).",
    )
    voices.add_argument(
        "--no-cast",
        action="store_true",
        help="Skip CAST.md resolution (cues are used as normalised).",
    )

    text = parser.add_argument_group("text preparation")
    text.add_argument(
        "--speech-policy",
        choices=sorted(POLICIES),
        default=FR3_POLICY.name,
        help=(
            "Which elements are speech. 'fr3' (default) voices dialogue only; "
            "'produciesta-parity' also narrates action, scene headings, centered "
            "text and transitions, as the Swift stack does. Recorded in the manifest."
        ),
    )

    timeline = parser.add_argument_group("assembly")
    timeline.add_argument(
        "--line-gap",
        type=float,
        default=LINE_GAP_SECONDS,
        metavar="SECONDS",
        help=f"Silence between consecutive lines (default: {LINE_GAP_SECONDS}).",
    )
    timeline.add_argument(
        "--scene-gap",
        type=float,
        default=SCENE_GAP_SECONDS,
        metavar="SECONDS",
        help=f"Silence at a scene change (default: {SCENE_GAP_SECONDS}).",
    )
    timeline.add_argument(
        "--lufs",
        type=float,
        default=TARGET_LUFS,
        metavar="LUFS",
        help=f"Per-line integrated loudness target (default: {TARGET_LUFS}, RD-3).",
    )
    timeline.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip loudness normalization; keep each line at its generated level.",
    )
    timeline.add_argument(
        "--peak-ceiling",
        type=float,
        default=PEAK_CEILING,
        metavar="PEAK",
        help=f"Sample-peak guard applied after the loudness gain (default: {PEAK_CEILING}).",
    )
    timeline.add_argument(
        "--no-trim-seams",
        action="store_true",
        help="Keep each line's model head/tail silence instead of tightening internal seams.",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--no-m4a",
        action="store_true",
        help="Write only the .wav (skips afconvert).",
    )
    output.add_argument(
        "--m4a-bitrate",
        type=int,
        default=M4A_BITRATE,
        metavar="BPS",
        help=f"AAC bitrate for the .m4a (default: {M4A_BITRATE}).",
    )

    run = parser.add_argument_group("run control")
    run.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Base RNG seed; use -1 to leave the RNG untouched (default: {DEFAULT_SEED}).",
    )
    run.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="Generate only the first N spoken lines (smoke runs).",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the episode and print the summary without loading a model.",
    )
    run.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print per-line progress.",
    )


def assembly_options(args: argparse.Namespace) -> AssemblyOptions:
    """Build :class:`AssemblyOptions` from the parsed CLI arguments."""
    return AssemblyOptions(
        line_gap_seconds=args.line_gap,
        scene_gap_seconds=args.scene_gap,
        target_lufs=None if args.no_normalize else args.lufs,
        peak_ceiling=args.peak_ceiling,
        trim_seams=not args.no_trim_seams,
    )


def plan_from_args(args: argparse.Namespace) -> EpisodePlan:
    """Build the episode plan described by the CLI arguments."""
    return build_plan(
        args.episode,
        args.engine,
        presets_path=args.presets,
        cast_path=args.cast,
        use_cast=not args.no_cast,
        speech_policy=args.speech_policy,
        seed=None if args.seed is not None and args.seed < 0 else args.seed,
        limit=args.limit,
    )


def _progress(total: int):
    """A progress hook printing one line per generated line."""

    def hook(line: PlannedLine, record: dict) -> None:
        duration = float(record.get("duration_seconds", 0.0))
        rtf = record.get("real_time_factor")
        flags = "".join(
            flag
            for key, flag in (
                ("truncated", "T"),
                ("overrun", "O"),
                ("truncation_retry", "R"),
            )
            if record.get(key)
        )
        print(
            f"[{line.index + 1:>4}/{total}] {line.character:<16} "
            f"{duration:>6.2f}s  rtf {float(rtf or 0.0):>5.2f}  {flags}",
            flush=True,
        )

    return hook


def handle(args: argparse.Namespace) -> int:
    """Run ``comparativa generate``. Returns the process exit code."""
    if not args.episode:
        args.parser.print_help()
        return 0
    if not args.engine:
        print(
            f"comparativa generate: --engine is required ({', '.join(ENGINE_KEYS)})",
            file=sys.stderr,
        )
        return 2

    try:
        plan = plan_from_args(args)
    except (GenerationError, KeyError, OSError, ValueError, RuntimeError) as exc:
        print(f"comparativa generate: {exc}", file=sys.stderr)
        return 1

    if not args.quiet or args.dry_run:
        print(plan.summary())
        print()

    if plan.unresolved_cues:
        names = sorted({c["normalized"] for c in plan.unresolved_cues})
        print(
            f"comparativa generate: warning: {len(plan.unresolved_cues)} unresolved "
            f"cue(s): {', '.join(names)}",
            file=sys.stderr,
        )

    if args.dry_run:
        return 0

    if not plan.lines:
        print("comparativa generate: no spoken lines to generate", file=sys.stderr)
        return 1

    try:
        result = generate_episode(
            plan,
            options=assembly_options(args),
            progress=None if args.quiet else _progress(len(plan.lines)),
        )
    except (GenerationError, EngineError) as exc:
        print(f"comparativa generate: {exc}", file=sys.stderr)
        return 1

    try:
        outputs = write_outputs(
            result,
            args.out,
            name=args.name,
            m4a=not args.no_m4a,
            m4a_bitrate=args.m4a_bitrate,
        )
    except (EncodeError, OSError) as exc:
        print(f"comparativa generate: {exc}", file=sys.stderr)
        return 1

    totals = result.manifest["totals"]
    print()
    print(
        f"{totals['line_count']} lines  "
        f"{totals['duration_seconds']:.2f}s audio "
        f"({totals['gap_seconds']:.2f}s gaps)  "
        f"generated in {totals['wall_seconds']:.1f}s  "
        f"RTF {totals['real_time_factor']}"
    )
    if totals["truncation_retry_lines"] or totals["truncated_lines"]:
        print(
            f"truncation: {totals['truncated_lines']} line(s) still short after "
            f"{totals['truncation_retry_lines']} retry/retries"
        )
    for key in ("wav", "m4a", "manifest"):
        if key in outputs:
            print(f"wrote {outputs[key]}")

    return 0


__all__ = ["assembly_options", "configure", "handle", "plan_from_args"]
