"""``comparativa bench`` — the round-1 conditions matrix, one run per cell.

    HF_HUB_OFFLINE=1 uv run comparativa bench ~/Projects/podcasts/granville \\
        --episodes episode_1_01a_bumper_donnie_and_arnie_1 --conditions C,E

Writes ``bench/<condition>/<episode>/`` — audio, ``manifest.json``,
``metrics.json`` — plus ``bench/summary.json`` for the invocation. ``--dry-run``
resolves and *plans* every cell (parse, cue resolution, voice assignment)
without loading a checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..generation.episode import DEFAULT_PRESETS_PATH, DEFAULT_SEED, GenerationError
from ..parsing.speech import POLICIES
from .conditions import (
    BENCH_ROOT,
    BENCH_SPEECH_POLICY,
    DEFAULT_CONDITIONS,
    ConditionError,
    matrix_table,
    parse_conditions,
)
from .metrics import STATUS_FAILED, STATUS_OK, SUMMARY_FILENAME, write_metrics
from .runner import (
    BenchError,
    BenchPlan,
    condition_presets_path,
    execute,
    plan_runs,
    resolve_episodes,
)


def configure(parser: argparse.ArgumentParser) -> None:
    """Register the ``bench`` subcommand's arguments."""
    parser.add_argument(
        "project_dir",
        nargs="?",
        metavar="PROJECT_DIR",
        help="Podcast project directory (read-only). Omit to print this help.",
    )
    parser.add_argument(
        "--episodes",
        metavar="IDS",
        help="Comma-separated episode ids (the .fountain stems), or 'all'.",
    )
    parser.add_argument(
        "--conditions",
        metavar="IDS",
        default=",".join(DEFAULT_CONDITIONS),
        help=(
            "Comma-separated condition ids, or 'all' "
            f"(default: {','.join(DEFAULT_CONDITIONS)}). "
            "A and F are produced by the Swift wrappers, not by bench."
        ),
    )
    parser.add_argument(
        "-o",
        "--out",
        metavar="DIR",
        default=BENCH_ROOT,
        help=f"Benchmark root; runs land in DIR/<cond>/<episode>/ (default: {BENCH_ROOT}).",
    )
    parser.add_argument(
        "--speech-policy",
        choices=sorted(POLICIES),
        default=BENCH_SPEECH_POLICY,
        help=(
            "Which elements are speech. Defaults to 'produciesta-parity' for "
            "every bench condition (supervisor ruling: the Swift baseline "
            "narrates action and sluglines, so A-vs-C is only valid at parity). "
            "Recorded in every manifest and metrics entry."
        ),
    )
    parser.add_argument(
        "--presets",
        metavar="PATH",
        default=str(DEFAULT_PRESETS_PATH),
        help=f"presets.yaml used to validate voice assignments (default: {DEFAULT_PRESETS_PATH}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Base RNG seed handed to every run (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        metavar="SECONDS",
        help="Kill a run that exceeds SECONDS (default: no limit).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run cells that already have a metrics.json (default: skip them).",
    )
    parser.add_argument(
        "--no-m4a",
        action="store_true",
        help="Write only the .wav for each run (skips afconvert).",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the matrix at the first failing cell.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and plan every cell without loading a model or writing audio.",
    )
    parser.add_argument(
        "--list-conditions",
        action="store_true",
        help="Print the conditions matrix and exit.",
    )


def _dry_run(plan: BenchPlan, *, presets: str | None) -> int:
    """Validate every cell: resolution, parse, cue mapping, voice assignment."""
    from ..generation.episode import build_plan

    print(plan.summary())
    print()
    if plan.external:
        for cond in plan.external:
            print(
                f"note: condition {cond.id} ({cond.label}) is produced by the "
                f"Swift wrapper (Sortie 8), not by bench — skipping"
            )
        print()

    header = (
        f"{'COND':<5} {'EPISODE':<44} {'ENGINE':<16} {'LINES':>6}  {'STATUS':<9} OUT"
    )
    print(header)
    print(f"{'-' * 5} {'-' * 44} {'-' * 16} {'-' * 6}  {'-' * 9} {'-' * 20}")

    failures = 0
    for run in plan.runs:
        engine = run.condition.engine or "-"
        # A condition that pins its own presets file (B/G -> presets-cloned.yaml)
        # must be validated against it, exactly as the child will run.
        run_presets = condition_presets_path(run.condition) or presets
        try:
            episode_plan = build_plan(
                run.episode.path,
                engine,
                presets_path=run_presets,
                speech_policy=run.speech_policy,
                seed=run.seed,
            )
        except (GenerationError, KeyError, OSError, ValueError, RuntimeError) as exc:
            failures += 1
            print(
                f"{run.condition.id:<5} {run.episode.id:<44} {engine:<16} "
                f"{'-':>6}  {'ERROR':<9} {exc}"
            )
            continue

        status = "existing" if run.done else "planned"
        print(
            f"{run.condition.id:<5} {run.episode.id:<44} {engine:<16} "
            f"{len(episode_plan.lines):>6}  {status:<9} {run.out_dir}"
        )
        if episode_plan.unresolved_cues:
            names = sorted({c["normalized"] for c in episode_plan.unresolved_cues})
            print(
                f"      warning: {len(episode_plan.unresolved_cues)} unresolved "
                f"cue(s): {', '.join(names)}",
                file=sys.stderr,
            )

    print()
    print(
        f"{len(plan.runs)} cell(s) planned, {failures} unresolvable "
        f"(dry run: no model was loaded, no audio was written)"
    )
    return 1 if failures else 0


def handle(args: argparse.Namespace) -> int:
    """Run ``comparativa bench``. Returns the process exit code."""
    if args.list_conditions:
        print(matrix_table())
        return 0
    if not args.project_dir:
        args.parser.print_help()
        return 0
    if not args.episodes:
        print(
            "comparativa bench: --episodes is required (ids, or 'all')",
            file=sys.stderr,
        )
        return 2

    try:
        conditions = parse_conditions(args.conditions)
        episodes = resolve_episodes(args.project_dir, args.episodes)
        plan = plan_runs(
            args.project_dir,
            conditions,
            episodes,
            out_root=args.out,
            speech_policy=args.speech_policy,
            seed=args.seed,
        )
    except (BenchError, ConditionError) as exc:
        print(f"comparativa bench: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        return _dry_run(plan, presets=args.presets)

    if not plan.runs:
        names = ", ".join(c.id for c in plan.conditions)
        print(
            f"comparativa bench: nothing to run — condition(s) {names} are "
            "produced by the Swift wrappers (Sortie 8), not by bench",
            file=sys.stderr,
        )
        return 2

    print(plan.summary())
    print()
    for cond in plan.external:
        print(f"note: condition {cond.id} is Sortie 8's (Swift) — skipping")

    def announce(run, command) -> None:
        print(f"==> {run.key}  ({run.condition.engine})", flush=True)

    outcomes = []
    for run in plan.runs:
        outcome = execute(
            run,
            m4a=not args.no_m4a,
            force=args.force,
            timeout=args.timeout,
            on_start=announce,
        )
        outcomes.append(outcome)
        print(outcome.line(), flush=True)
        if outcome.status == STATUS_FAILED and args.stop_on_error:
            print("comparativa bench: stopping at the first failure", file=sys.stderr)
            break

    entries = [o.entry for o in outcomes if o.entry is not None]
    if entries:
        summary_path = write_metrics(Path(plan.out_root) / SUMMARY_FILENAME, entries)
        print()
        print(f"wrote {summary_path}")

    completed = sum(1 for o in outcomes if o.status == STATUS_OK)
    failed = sum(1 for o in outcomes if o.status == STATUS_FAILED)
    skipped = len(outcomes) - completed - failed
    print(
        f"{completed} run(s) completed, {skipped} skipped, {failed} failed "
        f"across {len(plan.conditions)} condition(s)"
    )
    return 1 if failed else 0


__all__ = ["configure", "handle"]
