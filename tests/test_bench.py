"""Tests for the bench runner, the conditions matrix, and ``metrics.json``.

Nothing here loads a checkpoint. The dry-run path (``--dry-run``) is the one the
execution plan names, and it exercises the whole resolution chain — condition
ids, episode ids, parse, cue resolution, and voice assignment — without a model.
The metrics tests pin the **frozen** schema: Sortie 8's Swift-side entries must
satisfy exactly these assertions, so a change that breaks
``test_a_swift_entry_needs_no_python_only_fields`` is a schema break.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from comparativa.bench import (
    CONDITIONS,
    DEFAULT_CONDITIONS,
    METRICS_SCHEMA_VERSION,
    STATUS_FAILED,
    STATUS_OK,
    BenchError,
    ConditionError,
    Episode,
    MetricsError,
    PeakRSS,
    available_episodes,
    child_command,
    child_env,
    condition,
    entry_from_manifest,
    failed_entry,
    load_entries,
    make_entry,
    matrix_table,
    parse_conditions,
    plan_runs,
    python_tool_versions,
    resolve_episode,
    resolve_episodes,
    validate_entry,
    write_metrics,
)
from comparativa.bench.conditions import BENCH_SPEECH_POLICY
from comparativa.cli import main as cli_main
from comparativa.generation.engines import ENGINE_SPECS

from conftest import CORPUS_ROOT

GRANVILLE = CORPUS_ROOT
BUMPER_ID = "episode_1_01a_bumper_donnie_and_arnie_1"
BUMPER = GRANVILLE / "episodes" / f"{BUMPER_ID}.fountain"

corpus = pytest.mark.skipif(
    not BUMPER.is_file(), reason=f"mission corpus not present at {GRANVILLE}"
)


# ---------------------------------------------------------------------------
# Task 1: the conditions matrix as config
# ---------------------------------------------------------------------------


def test_the_round_one_matrix_is_encoded_exactly_as_the_plan_states():
    assert CONDITIONS["C"].engine == "qwen3-1.7b"
    assert CONDITIONS["D"].engine == "chatterbox"
    assert CONDITIONS["E"].engine == "soprano"
    for cid in ("C", "D", "E"):
        cond = CONDITIONS[cid]
        assert cond.stack == "python"
        assert cond.runnable
        assert cond.checkpoint == ENGINE_SPECS[cond.engine].checkpoint
    # RD-1: E and F must name the same checkpoint or the port pair is not a pair.
    assert CONDITIONS["E"].checkpoint == CONDITIONS["F"].checkpoint
    assert CONDITIONS["F"].checkpoint == "mlx-community/Soprano-80M-bf16"


def test_the_swift_conditions_are_declared_but_not_runnable_here():
    for cid in ("A", "F"):
        cond = CONDITIONS[cid]
        assert cond.stack == "swift"
        assert not cond.runnable  # Sortie 8 owns these


def test_condition_b_is_the_vox_cloned_qwen3_pair_for_a():
    """Round 2: B un-defers the RD-2 cloned condition against A's voices."""
    cond = CONDITIONS["B"]
    assert cond.stack == "python"
    assert cond.runnable
    assert cond.engine == "qwen3-1.7b-clone"
    assert cond.checkpoint == "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
    assert cond.presets == "presets-cloned.yaml"
    assert "B" in DEFAULT_CONDITIONS


def test_condition_g_is_the_optional_chatterbox_clone_probe():
    cond = CONDITIONS["G"]
    assert cond.engine == "chatterbox"
    assert cond.optional
    assert cond.presets == "presets-cloned.yaml"
    assert "G" not in DEFAULT_CONDITIONS


def test_the_turbo_speed_probe_is_optional_and_off_by_default():
    assert CONDITIONS["T"].engine == "chatterbox-turbo"
    assert CONDITIONS["T"].optional
    assert "T" not in DEFAULT_CONDITIONS
    assert DEFAULT_CONDITIONS == ("B", "C", "D", "E")


def test_conditions_parse_in_matrix_order_and_deduplicate():
    assert [c.id for c in parse_conditions("E,C,E")] == ["C", "E"]
    assert [c.id for c in parse_conditions("c,d")] == ["C", "D"]
    assert [c.id for c in parse_conditions(None)] == list(DEFAULT_CONDITIONS)
    assert [c.id for c in parse_conditions("all")] == ["A", "B", "C", "D", "E"]


def test_an_unknown_condition_is_a_clear_error():
    with pytest.raises(ConditionError, match="unknown condition"):
        parse_conditions("C,Z")
    with pytest.raises(ConditionError, match="unknown condition"):
        condition("H")


def test_the_matrix_table_lists_every_condition():
    table = matrix_table()
    for cid in CONDITIONS:
        assert f"\n{cid}  " in f"\n{table}"


# ---------------------------------------------------------------------------
# Task 3: episode + output-directory resolution
# ---------------------------------------------------------------------------


@corpus
def test_an_episode_resolves_by_id_filename_path_and_unique_substring():
    by_id = resolve_episode(GRANVILLE, BUMPER_ID)
    assert by_id == Episode(BUMPER_ID, BUMPER.resolve())
    assert resolve_episode(GRANVILLE, f"{BUMPER_ID}.fountain") == by_id
    assert resolve_episode(GRANVILLE, str(BUMPER)) == by_id
    assert resolve_episode(GRANVILLE, "bumper_donnie_and_arnie_1") == by_id


def test_an_ambiguous_episode_substring_is_refused(tmp_path):
    # The frozen corpus (docs/CORPUS_PIN.md) intentionally carries only the
    # two episodes this mission benchmarks, so it can't exercise a genuine
    # substring collision on its own; the live granville tree has several
    # "bumper" episodes but is explicitly out of scope post-freeze. Ambiguity
    # resolution is a content-independent property of resolve_episode, so a
    # synthetic two-file project proves it without depending on either tree.
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    (episodes / "episode_1_01a_bumper_donnie_and_arnie_1.fountain").write_text("")
    (episodes / "episode_1_02a_bumper_donnie_and_arnie_2.fountain").write_text("")
    with pytest.raises(BenchError, match="ambiguous"):
        resolve_episode(tmp_path, "bumper")


@corpus
def test_an_unknown_episode_is_refused():
    with pytest.raises(BenchError, match="not found"):
        resolve_episode(GRANVILLE, "episode_9_99_nope")


@corpus
def test_episode_lists_deduplicate_and_all_selects_the_project():
    assert [e.id for e in resolve_episodes(GRANVILLE, f"{BUMPER_ID},{BUMPER_ID}")] == [
        BUMPER_ID
    ]
    every = resolve_episodes(GRANVILLE, "all")
    assert len(every) == len(available_episodes(GRANVILLE)) > 1
    assert BUMPER_ID in {e.id for e in every}


def test_a_missing_project_directory_is_refused(tmp_path):
    with pytest.raises(BenchError, match="project directory not found"):
        resolve_episodes(tmp_path / "nope", "anything")


@corpus
def test_the_plan_lays_out_one_directory_per_condition_and_episode(tmp_path):
    plan = plan_runs(
        GRANVILLE,
        parse_conditions("C,E"),
        resolve_episodes(GRANVILLE, BUMPER_ID),
        out_root=tmp_path / "bench",
    )
    assert [run.out_dir for run in plan.runs] == [
        tmp_path / "bench" / "C" / BUMPER_ID,
        tmp_path / "bench" / "E" / BUMPER_ID,
    ]
    assert all(run.speech_policy == BENCH_SPEECH_POLICY for run in plan.runs)
    assert all(not run.done for run in plan.runs)


@corpus
def test_swift_conditions_are_planned_as_external_not_as_runs(tmp_path):
    plan = plan_runs(
        GRANVILLE,
        parse_conditions("A,C,F"),
        resolve_episodes(GRANVILLE, BUMPER_ID),
        out_root=tmp_path / "bench",
    )
    assert [run.condition.id for run in plan.runs] == ["C"]
    assert [c.id for c in plan.external] == ["A", "F"]


# ---------------------------------------------------------------------------
# Task 3: the child invocation
# ---------------------------------------------------------------------------


@corpus
def test_the_child_runs_generate_offline_at_parity_with_a_recorded_seed(tmp_path):
    plan = plan_runs(
        GRANVILLE,
        parse_conditions("C"),
        resolve_episodes(GRANVILLE, BUMPER_ID),
        out_root=tmp_path / "bench",
        seed=1234,
    )
    command = child_command(plan.runs[0])

    assert command[:4] == [sys.executable, "-m", "comparativa", "generate"]
    assert "--engine" in command and command[command.index("--engine") + 1] == "qwen3-1.7b"
    # SUPERVISOR RULING: every bench condition narrates like the Swift stack.
    assert command[command.index("--speech-policy") + 1] == BENCH_SPEECH_POLICY
    assert command[command.index("--seed") + 1] == "1234"
    assert command[command.index("--name") + 1] == BUMPER_ID
    assert str(plan.runs[0].out_dir) in command

    env = child_env()
    assert env["HF_HUB_OFFLINE"] == "1"  # no downloads inside a timed run
    assert env["TQDM_DISABLE"] == "1"  # chatterbox-turbo draws a bar
    assert env["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"


# ---------------------------------------------------------------------------
# Task 2: peak RSS
# ---------------------------------------------------------------------------


def test_peak_rss_samples_a_real_child_process():
    import subprocess
    import time

    # The child allocates, announces readiness on stdout, then blocks reading
    # stdin. That gives the sampler a deterministic window to observe the
    # allocation instead of racing the child's exit — closing stdin is what
    # releases it, not a fixed sleep.
    child = subprocess.Popen(  # noqa: S603 - fixed argv
        [
            sys.executable,
            "-c",
            "import sys\n"
            "x = bytearray(64 * 1024 * 1024)\n"
            "print(len(x), flush=True)\n"
            "sys.stdin.readline()\n",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    sampler = PeakRSS(child.pid, interval=0.005).start()
    try:
        ready = child.stdout.readline()
        assert ready.strip() == str(64 * 1024 * 1024)

        # Let the sampler take a few ticks while the child is parked on stdin.
        deadline = time.monotonic() + 1.0
        while sampler.result.samples < 3 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert sampler.result.samples >= 3, "sampler never observed the child"
    finally:
        child.stdin.close()
        child.wait(timeout=5)

    result = sampler.stop()

    assert result.ok
    assert result.peak_rss_bytes > 8 * 1024 * 1024
    assert result.first_rss_bytes > 0


# ---------------------------------------------------------------------------
# Task 2: the frozen metrics schema
# ---------------------------------------------------------------------------

MANIFEST = {
    "schema_version": 1,
    "speech_policy": "produciesta-parity",
    "episode": {"path": "/tmp/e.fountain", "name": "e", "sha256": "abc123"},
    "engine": {
        "key": "soprano",
        "checkpoint": "mlx-community/Soprano-80M-bf16",
        "sample_rate": 32000,
    },
    "assembly": {"target_lufs": -16.0, "loudness_meter": "pyloudnorm.Meter"},
    "script_line_count": 15,
    "outputs": {"wav": "/tmp/e.wav", "manifest": "/tmp/manifest.json"},
    "totals": {
        "line_count": 15,
        "placed_line_count": 15,
        "duration_seconds": 61.5,
        "audio_seconds": 57.75,
        "gap_seconds": 3.75,
        "generate_seconds": 30.25,
        "real_time_factor": 0.492,
        "load_seconds": 4.125,
        "wall_seconds": 34.5,
        "truncated_lines": 1,
        "overrun_lines": 0,
        "truncation_retry_lines": 2,
        "peak_limited_lines": 9,
        "unnormalized_lines": 0,
        "mean_output_lufs": -16.8,
        "max_loudness_shortfall_db": 2.1,
        "mean_loudness_shortfall_db": 0.7,
    },
    "lines": [],
}


def _entry_from_fixture(**kwargs):
    return entry_from_manifest(
        MANIFEST,
        condition="E",
        episode="e",
        stack="python",
        tool_versions=python_tool_versions(),
        peak_rss_bytes=1234567,
        **kwargs,
    )


def test_metrics_take_wall_rtf_and_load_from_the_manifest_totals():
    """Supervisor ruling: read ``totals``, never re-derive."""
    entry = _entry_from_fixture()
    perf = entry["performance"]
    assert perf["wall_seconds"] == MANIFEST["totals"]["wall_seconds"]
    assert perf["real_time_factor"] == MANIFEST["totals"]["real_time_factor"]
    assert perf["model_load_seconds"] == MANIFEST["totals"]["load_seconds"]
    assert perf["peak_rss_bytes"] == 1234567  # the one number bench adds


def test_an_entry_carries_every_field_the_report_and_sortie_eight_need():
    entry = _entry_from_fixture()

    assert entry["condition"] == "E"
    assert entry["stack"] == "python"
    assert entry["engine"] == "soprano"
    assert entry["checkpoint"] == "mlx-community/Soprano-80M-bf16"
    assert entry["episode"] == "e"
    assert entry["episode_sha256"] == "abc123"
    assert entry["speech_policy"] == "produciesta-parity"
    assert entry["status"] == STATUS_OK
    assert entry["audio"]["audio_seconds"] == 57.75
    assert entry["audio"]["sample_rate"] == 32000
    assert entry["lines"]["line_count"] == 15
    assert entry["lines"]["truncation_retry_lines"] == 2
    assert entry["loudness"]["mean_output_lufs"] == -16.8
    assert entry["loudness"]["mean_shortfall_db"] == 0.7
    assert entry["loudness"]["max_shortfall_db"] == 2.1
    assert entry["tool_versions"]["comparativa"]
    assert entry["tool_versions"]["mlx_audio"] == "0.4.8"


def test_a_swift_entry_needs_no_python_only_fields():
    """Sortie 8's condition-A entry must be expressible in the frozen schema."""
    entry = make_entry(
        condition="A",
        stack="swift",
        label="Produciesta (Swift), Qwen3 1.7B, production .vox voices",
        engine="produciesta",
        checkpoint="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
        voices=".vox",
        episode="episode_1_01a_bumper_donnie_and_arnie_1",
        speech_policy="produciesta-parity",
        wall_seconds=91.4,
        real_time_factor=1.31,
        model_load_seconds=6.2,
        peak_rss_bytes=5_400_000_000,
        audio_seconds=69.8,
        line_count=15,
        loudness={"mean_output_lufs": -19.4},
        tool_versions={"produciesta": "1.4.2", "macos": "26.0"},
        notes=["peak RSS from /usr/bin/time -l"],
    )

    validate_entry(entry)
    # Keys the Python side fills in are present and null, so the report's table
    # has a cell rather than a hole.
    assert entry["loudness"]["mean_shortfall_db"] is None
    assert entry["loudness"]["max_shortfall_db"] is None
    assert entry["performance"]["peak_rss_bytes"] == 5_400_000_000
    assert entry["lines"]["line_count"] == 15
    assert "produciesta" in entry["tool_versions"]


def test_an_entry_missing_a_required_field_is_rejected():
    entry = _entry_from_fixture()
    del entry["performance"]["peak_rss_bytes"]
    with pytest.raises(MetricsError, match="peak_rss_bytes"):
        validate_entry(entry)

    entry = _entry_from_fixture()
    del entry["speech_policy"]
    with pytest.raises(MetricsError, match="speech_policy"):
        validate_entry(entry)


def test_an_entry_without_an_engine_or_checkpoint_is_rejected():
    with pytest.raises(MetricsError, match="engine / checkpoint"):
        make_entry(
            condition="A",
            stack="swift",
            episode="e",
            speech_policy="produciesta-parity",
            wall_seconds=1.0,
            real_time_factor=1.0,
            model_load_seconds=1.0,
            peak_rss_bytes=1,
            audio_seconds=1.0,
            line_count=1,
            tool_versions={},
        )


def test_a_failed_run_is_recorded_rather_than_vanishing():
    entry = failed_entry(
        condition="D",
        stack="python",
        episode="e",
        speech_policy="produciesta-parity",
        engine="chatterbox",
        checkpoint="mlx-community/chatterbox-fp16",
        tool_versions=python_tool_versions(),
        error="generate exited 1",
    )
    validate_entry(entry)
    assert entry["status"] == STATUS_FAILED
    assert entry["performance"]["wall_seconds"] is None
    assert "exited 1" in entry["error"]


def test_metrics_documents_round_trip_and_deduplicate_across_layouts(tmp_path):
    entry = _entry_from_fixture()
    per_run = write_metrics(tmp_path / "E" / "e" / "metrics.json", [entry])
    # Sortie 8's layout: one file per condition, several entries.
    write_metrics(tmp_path / "A" / "metrics.json", [entry, dict(entry, condition="A")])

    document = json.loads(per_run.read_text(encoding="utf-8"))
    assert document["schema_version"] == METRICS_SCHEMA_VERSION
    assert len(document["entries"]) == 1

    entries = load_entries(tmp_path)
    keys = {(e["stack"], e["condition"], e["episode"]) for e in entries}
    assert keys == {("python", "E", "e"), ("python", "A", "e")}
    assert len(entries) == 2  # the duplicate E entry collapsed


# ---------------------------------------------------------------------------
# Task 4: --dry-run
# ---------------------------------------------------------------------------


@corpus
def test_dry_run_resolves_every_cell_without_generating(tmp_path, capsys):
    out = tmp_path / "bench"
    code = cli_main(
        [
            "bench",
            str(GRANVILLE),
            "--episodes",
            BUMPER_ID,
            "--conditions",
            "C,D,E",
            "--out",
            str(out),
            "--dry-run",
        ]
    )
    assert code == 0

    printed = capsys.readouterr().out
    assert "produciesta-parity" in printed  # the policy is stated up front
    assert "3 cell(s) planned, 0 unresolvable" in printed
    for cid, engine in (("C", "qwen3-1.7b"), ("D", "chatterbox"), ("E", "soprano")):
        assert engine in printed
        assert str(out / cid / BUMPER_ID) in printed

    # Nothing was written: not into the bench root, not into the corpus.
    assert not out.exists()


@corpus
def test_dry_run_reports_the_planned_line_count_at_the_bench_policy(tmp_path, capsys):
    from comparativa.generation.episode import spoken_line_count

    # An isolated --out: a cell that already holds a real run reports
    # "existing" rather than "planned", and this test is about the count.
    cli_main(
        [
            "bench", str(GRANVILLE), "--episodes", BUMPER_ID,
            "--conditions", "E", "--out", str(tmp_path / "bench"), "--dry-run",
        ]
    )
    expected = spoken_line_count(BUMPER, speech_policy=BENCH_SPEECH_POLICY)
    printed = capsys.readouterr().out
    assert f"{expected:>6}  planned" in printed
    # Parity narrates action and sluglines, so it must exceed the FR-3 count.
    assert expected > spoken_line_count(BUMPER, speech_policy="fr3")


@corpus
def test_dry_run_names_the_swift_conditions_as_sortie_eights(capsys):
    code = cli_main(
        [
            "bench", str(GRANVILLE), "--episodes", BUMPER_ID,
            "--conditions", "A,E", "--dry-run",
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "Sortie 8" in printed
    assert "1 cell(s) planned" in printed


@corpus
def test_an_unknown_condition_or_episode_is_a_usage_error(capsys):
    assert (
        cli_main(["bench", str(GRANVILLE), "--episodes", BUMPER_ID, "--conditions", "Z"])
        == 2
    )
    assert "unknown condition" in capsys.readouterr().err

    assert (
        cli_main(["bench", str(GRANVILLE), "--episodes", "nope", "--conditions", "E"])
        == 2
    )
    assert "not found" in capsys.readouterr().err


@corpus
def test_asking_only_for_swift_conditions_refuses_to_run(capsys):
    assert (
        cli_main(["bench", str(GRANVILLE), "--episodes", BUMPER_ID, "--conditions", "A,F"])
        == 2
    )
    assert "Sortie 8" in capsys.readouterr().err


def test_bench_without_arguments_prints_help(capsys):
    assert cli_main(["bench"]) == 0
    assert "usage: comparativa bench" in capsys.readouterr().out


def test_bench_without_episodes_is_a_usage_error(capsys):
    assert cli_main(["bench", str(GRANVILLE)]) == 2
    assert "--episodes is required" in capsys.readouterr().err


def test_list_conditions_prints_the_matrix(capsys):
    assert cli_main(["bench", "--list-conditions"]) == 0
    printed = capsys.readouterr().out
    assert "qwen3-1.7b" in printed and "external" in printed
