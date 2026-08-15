"""Orchestrating ``generate`` once per (condition, episode) into ``bench/``.

Layout, fixed by supervisor ruling — one run per output directory::

    bench/<condition>/<episode>/<episode>.wav
                               /<episode>.m4a
                               /manifest.json
                               /metrics.json
                               /generate.log
    bench/summary.json          # every entry from this invocation

Each run is a **subprocess** (``python -m comparativa generate ...``), for three
reasons: peak RSS is only meaningful when a run does not inherit the previous
condition's resident checkpoint (see :mod:`.perf`); a model that dies taking the
interpreter with it costs one condition rather than the whole matrix; and the
child's stdout — including ``chatterbox-turbo``'s tqdm bar — goes to
``generate.log`` instead of the operator's terminal.

The child runs with ``HF_HUB_OFFLINE=1``: nothing in a benchmark run may block
on, or be timed against, a network fetch.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ..generation.episode import DEFAULT_SEED, MANIFEST_FILENAME
from .conditions import (
    BENCH_SPEECH_POLICY,
    Condition,
    ConditionError,
    STACK_PYTHON,
)
from .metrics import (
    METRICS_FILENAME,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_SKIPPED,
    entry_from_manifest,
    failed_entry,
    write_metrics,
)
from .perf import PeakRSS, host_info, python_tool_versions

#: Where the child writes its (possibly tqdm-infested) output.
LOG_FILENAME = "generate.log"

#: Environment forced on every child process.
CHILD_ENV: dict[str, str] = {
    # No downloads, ever, inside a timed run.
    "HF_HUB_OFFLINE": "1",
    # chatterbox-turbo draws a tqdm bar; silence it at the source as well as
    # redirecting it, so the log stays readable too.
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TQDM_DISABLE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONUNBUFFERED": "1",
}

#: Where episodes live inside a podcast project.
EPISODES_DIRNAME = "episodes"

EPISODE_SUFFIX = ".fountain"


class BenchError(RuntimeError):
    """A benchmark run could not be planned."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Episode:
    """One resolved screenplay. ``id`` is the cross-stack join key."""

    id: str
    path: Path


def episodes_dir(project_dir: str | Path) -> Path:
    project = Path(project_dir).expanduser()
    if not project.is_dir():
        raise BenchError(f"project directory not found: {project}")
    directory = project / EPISODES_DIRNAME
    if directory.is_dir():
        return directory
    if project.name == EPISODES_DIRNAME:
        return project
    raise BenchError(f"no {EPISODES_DIRNAME}/ directory in {project}")


def available_episodes(project_dir: str | Path) -> list[Episode]:
    """Every ``.fountain`` in the project, in filename order."""
    return [
        Episode(path.stem, path)
        for path in sorted(episodes_dir(project_dir).glob(f"*{EPISODE_SUFFIX}"))
    ]


def resolve_episode(project_dir: str | Path, token: str) -> Episode:
    """Resolve one ``--episodes`` token to a screenplay.

    Accepts, in order: a path; an episode id (the ``.fountain`` stem); a
    filename; and finally a unique substring of an id, so the operator can type
    ``bumper_donnie_and_arnie_1`` instead of the full stem.
    """
    token = token.strip()
    if not token:
        raise BenchError("empty episode id")

    direct = Path(token).expanduser()
    if direct.is_file():
        return Episode(direct.stem, direct.resolve())

    directory = episodes_dir(project_dir)
    for candidate in (directory / token, directory / f"{token}{EPISODE_SUFFIX}"):
        if candidate.is_file():
            return Episode(candidate.stem, candidate.resolve())

    known = available_episodes(project_dir)
    matches = [ep for ep in known if token.lower() in ep.id.lower()]
    if len(matches) == 1:
        return Episode(matches[0].id, matches[0].path.resolve())
    if len(matches) > 1:
        raise BenchError(
            f"episode {token!r} is ambiguous: {', '.join(ep.id for ep in matches)}"
        )
    raise BenchError(
        f"episode {token!r} not found in {directory}; available: "
        + ", ".join(ep.id for ep in known)
    )


def resolve_episodes(project_dir: str | Path, spec: str | Sequence[str] | None) -> list[Episode]:
    """Resolve a ``--episodes`` specification. ``all`` selects the whole project."""
    if spec is None:
        raise BenchError("no episodes requested (--episodes)")
    tokens = (
        [t.strip() for t in spec.split(",")] if isinstance(spec, str) else list(spec)
    )
    tokens = [t for t in tokens if t]
    if not tokens:
        raise BenchError("no episodes requested (--episodes)")
    if len(tokens) == 1 and tokens[0].lower() == "all":
        found = available_episodes(project_dir)
        if not found:
            raise BenchError(f"no {EPISODE_SUFFIX} files in {episodes_dir(project_dir)}")
        return found

    resolved: dict[str, Episode] = {}
    for token in tokens:
        episode = resolve_episode(project_dir, token)
        resolved.setdefault(episode.id, episode)
    return list(resolved.values())


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchRun:
    """One (condition, episode) cell of the matrix."""

    condition: Condition
    episode: Episode
    out_dir: Path
    speech_policy: str = BENCH_SPEECH_POLICY
    seed: int = DEFAULT_SEED

    @property
    def key(self) -> str:
        return f"{self.condition.id}/{self.episode.id}"

    @property
    def manifest_path(self) -> Path:
        return self.out_dir / MANIFEST_FILENAME

    @property
    def metrics_path(self) -> Path:
        return self.out_dir / METRICS_FILENAME

    @property
    def log_path(self) -> Path:
        return self.out_dir / LOG_FILENAME

    @property
    def done(self) -> bool:
        """True when this cell already holds a completed run."""
        return self.metrics_path.is_file() and self.manifest_path.is_file()


@dataclass
class BenchPlan:
    """Every cell ``bench`` was asked for, plus where it will write."""

    project_dir: Path
    out_root: Path
    conditions: list[Condition]
    episodes: list[Episode]
    runs: list[BenchRun] = field(default_factory=list)
    speech_policy: str = BENCH_SPEECH_POLICY
    seed: int = DEFAULT_SEED

    @property
    def external(self) -> list[Condition]:
        """Requested conditions that Sortie 8's Swift wrappers own."""
        return [c for c in self.conditions if not c.runnable]

    def summary(self) -> str:
        rows = [
            f"project:       {self.project_dir}",
            f"out:           {self.out_root}",
            f"speech policy: {self.speech_policy}  (supervisor ruling: all bench "
            "conditions narrate like the Swift stack)",
            f"seed:          {self.seed}",
            f"conditions:    {', '.join(c.id for c in self.conditions)}",
            f"episodes:      {', '.join(e.id for e in self.episodes)}",
            f"runs:          {len(self.runs)}",
        ]
        if self.external:
            rows.append(
                "external:      "
                + ", ".join(
                    f"{c.id} ({c.stack}; Sortie 8)" for c in self.external
                )
            )
        return "\n".join(rows)


def plan_runs(
    project_dir: str | Path,
    conditions: Sequence[Condition],
    episodes: Sequence[Episode],
    *,
    out_root: str | Path,
    speech_policy: str = BENCH_SPEECH_POLICY,
    seed: int = DEFAULT_SEED,
) -> BenchPlan:
    """Lay out the matrix. Nothing is parsed, loaded, or written here."""
    if not conditions:
        raise ConditionError("no conditions requested")
    if not episodes:
        raise BenchError("no episodes requested")

    root = Path(out_root).expanduser()
    plan = BenchPlan(
        project_dir=Path(project_dir).expanduser(),
        out_root=root,
        conditions=list(conditions),
        episodes=list(episodes),
        speech_policy=speech_policy,
        seed=seed,
    )
    for cond in plan.conditions:
        if not cond.runnable:
            continue
        for episode in plan.episodes:
            plan.runs.append(
                BenchRun(
                    condition=cond,
                    episode=episode,
                    out_dir=root / cond.id / episode.id,
                    speech_policy=speech_policy,
                    seed=seed,
                )
            )
    return plan


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@dataclass
class RunOutcome:
    """What happened to one cell."""

    run: BenchRun
    status: str
    entry: dict[str, Any] | None = None
    returncode: int | None = None
    process_wall_seconds: float = 0.0
    peak_rss_bytes: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in (STATUS_OK, STATUS_SKIPPED)

    def line(self) -> str:
        if self.status == STATUS_SKIPPED:
            return f"{self.run.key}: skipped (already run; --force to redo)"
        if self.status == STATUS_FAILED:
            return f"{self.run.key}: FAILED ({self.error})"
        perf = (self.entry or {}).get("performance", {})
        audio = (self.entry or {}).get("audio", {})
        lines = (self.entry or {}).get("lines", {})
        rss_mb = (perf.get("peak_rss_bytes") or 0) / (1024 * 1024)
        return (
            f"{self.run.key}: {lines.get('line_count')} lines  "
            f"{audio.get('audio_seconds')}s audio  "
            f"wall {perf.get('wall_seconds')}s  "
            f"RTF {perf.get('real_time_factor')}  "
            f"load {perf.get('model_load_seconds')}s  "
            f"peak RSS {rss_mb:.0f} MiB"
        )


def child_command(run: BenchRun, *, m4a: bool = True) -> list[str]:
    """The ``generate`` invocation for one cell."""
    if run.condition.engine is None:  # pragma: no cover - guarded by plan_runs
        raise BenchError(f"condition {run.condition.id} has no Python engine")
    command = [
        sys.executable,
        "-m",
        "comparativa",
        "generate",
        str(run.episode.path),
        "--engine",
        run.condition.engine,
        "--out",
        str(run.out_dir),
        "--name",
        run.episode.id,
        "--speech-policy",
        run.speech_policy,
        "--seed",
        str(run.seed),
    ]
    if not m4a:
        command.append("--no-m4a")
    return command


def child_env() -> dict[str, str]:
    """The child's environment: the parent's, plus the offline/quiet overrides."""
    env = dict(os.environ)
    env.update(CHILD_ENV)
    return env


def execute(
    run: BenchRun,
    *,
    m4a: bool = True,
    force: bool = False,
    timeout: float | None = None,
    on_start: Any = None,
) -> RunOutcome:
    """Generate one cell and write its ``metrics.json``."""
    if run.done and not force:
        return RunOutcome(run=run, status=STATUS_SKIPPED)

    run.out_dir.mkdir(parents=True, exist_ok=True)
    # A re-run must not leave the previous attempt's metrics behind if it dies.
    if run.metrics_path.is_file():
        run.metrics_path.unlink()

    command = child_command(run, m4a=m4a)
    started_at = _now()
    started = time.perf_counter()
    if on_start is not None:
        on_start(run, command)

    versions = python_tool_versions()
    host = host_info()

    with run.log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(command)}\n\n")
        log.flush()
        child = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=child_env(),
            cwd=str(Path.cwd()),
        )
        sampler = PeakRSS(child.pid).start()
        try:
            returncode = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
            returncode = -9
        finally:
            rss = sampler.stop()

    process_wall = time.perf_counter() - started
    run_info = {
        "started_at": started_at,
        "finished_at": _now(),
        "command": command,
        "returncode": returncode,
        "log": str(run.log_path),
        "rss_sampler_interval": sampler.interval,
    }

    if returncode != 0 or not run.manifest_path.is_file():
        tail = _log_tail(run.log_path)
        error = f"generate exited {returncode}" + (f": {tail}" if tail else "")
        entry = failed_entry(
            condition=run.condition.id,
            stack=STACK_PYTHON,
            label=run.condition.label,
            engine=run.condition.engine,
            checkpoint=run.condition.checkpoint,
            voices=run.condition.voices,
            episode=run.episode.id,
            episode_path=str(run.episode.path),
            speech_policy=run.speech_policy,
            tool_versions=versions,
            peak_rss_bytes=rss.peak_rss_bytes or None,
            error=error,
            host=host,
            run=run_info,
        )
        write_metrics(run.metrics_path, [entry])
        return RunOutcome(
            run=run,
            status=STATUS_FAILED,
            entry=entry,
            returncode=returncode,
            process_wall_seconds=process_wall,
            peak_rss_bytes=rss.peak_rss_bytes,
            error=error,
        )

    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    entry = entry_from_manifest(
        manifest,
        condition=run.condition.id,
        episode=run.episode.id,
        stack=STACK_PYTHON,
        label=run.condition.label,
        voices=run.condition.voices,
        tool_versions=versions,
        peak_rss_bytes=rss.peak_rss_bytes or None,
        performance_extra={
            "process_wall_seconds": round(process_wall, 3),
            "start_rss_bytes": rss.first_rss_bytes or None,
            "rss_samples": rss.samples,
        },
        host=host,
        run=run_info,
    )
    write_metrics(run.metrics_path, [entry])
    return RunOutcome(
        run=run,
        status=STATUS_OK,
        entry=entry,
        returncode=returncode,
        process_wall_seconds=process_wall,
        peak_rss_bytes=rss.peak_rss_bytes,
    )


def _log_tail(path: Path, *, lines: int = 3, width: int = 400) -> str:
    """The last few log lines, for a failure message."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:  # pragma: no cover - the log was just written
        return ""
    tail = [line for line in text.strip().splitlines() if line.strip()][-lines:]
    return " | ".join(tail)[:width]


__all__ = [
    "CHILD_ENV",
    "EPISODES_DIRNAME",
    "LOG_FILENAME",
    "BenchError",
    "BenchPlan",
    "BenchRun",
    "Episode",
    "RunOutcome",
    "available_episodes",
    "child_command",
    "child_env",
    "episodes_dir",
    "execute",
    "plan_runs",
    "resolve_episode",
    "resolve_episodes",
]
