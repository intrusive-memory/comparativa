"""Benchmark: the round-1 conditions matrix and its performance record.

Three pieces:

* :mod:`.conditions` — the matrix as config (A, C, D, E, F, and the optional T
  speed probe), shared by this sortie's Python runs and Sortie 8's Swift ones.
* :mod:`.metrics` — the **frozen** ``metrics.json`` schema. Both stacks build
  entries with :func:`~comparativa.bench.metrics.make_entry`; nothing in it is
  Python-only.
* :mod:`.runner` — resolves episodes, lays out ``bench/<cond>/<episode>/``, and
  runs ``generate`` once per cell in its own subprocess so that peak RSS
  (:mod:`.perf`) measures one condition rather than the sum of all of them.

Typical use::

    uv run comparativa bench ~/Projects/podcasts/granville \\
        --episodes episode_1_01a_bumper_donnie_and_arnie_1 --conditions C,E
"""

from .conditions import (
    BENCH_ROOT,
    BENCH_SPEECH_POLICY,
    CONDITION_IDS,
    CONDITIONS,
    DEFAULT_CONDITIONS,
    RUNNER_BENCH,
    RUNNER_EXTERNAL,
    STACK_PYTHON,
    STACK_SWIFT,
    Condition,
    ConditionError,
    condition,
    matrix_table,
    parse_conditions,
)
from .metrics import (
    METRICS_FILENAME,
    METRICS_SCHEMA_VERSION,
    REQUIRED_GROUP_KEYS,
    REQUIRED_KEYS,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_SKIPPED,
    SUMMARY_FILENAME,
    MetricsError,
    document,
    entry_from_manifest,
    failed_entry,
    load_entries,
    make_entry,
    read_metrics,
    validate_entry,
    write_metrics,
)
from .perf import PeakRSS, RSSSample, host_info, python_tool_versions
from .runner import (
    CHILD_ENV,
    LOG_FILENAME,
    BenchError,
    BenchPlan,
    BenchRun,
    Episode,
    RunOutcome,
    available_episodes,
    child_command,
    child_env,
    execute,
    plan_runs,
    resolve_episode,
    resolve_episodes,
)

__all__ = [
    "BENCH_ROOT",
    "BENCH_SPEECH_POLICY",
    "CHILD_ENV",
    "CONDITIONS",
    "CONDITION_IDS",
    "DEFAULT_CONDITIONS",
    "LOG_FILENAME",
    "METRICS_FILENAME",
    "METRICS_SCHEMA_VERSION",
    "REQUIRED_GROUP_KEYS",
    "REQUIRED_KEYS",
    "RUNNER_BENCH",
    "RUNNER_EXTERNAL",
    "STACK_PYTHON",
    "STACK_SWIFT",
    "STATUS_FAILED",
    "STATUS_OK",
    "STATUS_SKIPPED",
    "SUMMARY_FILENAME",
    "BenchError",
    "BenchPlan",
    "BenchRun",
    "Condition",
    "ConditionError",
    "Episode",
    "MetricsError",
    "PeakRSS",
    "RSSSample",
    "RunOutcome",
    "available_episodes",
    "child_command",
    "child_env",
    "condition",
    "document",
    "entry_from_manifest",
    "execute",
    "failed_entry",
    "host_info",
    "load_entries",
    "make_entry",
    "matrix_table",
    "parse_conditions",
    "plan_runs",
    "python_tool_versions",
    "read_metrics",
    "resolve_episode",
    "resolve_episodes",
    "validate_entry",
    "write_metrics",
]
