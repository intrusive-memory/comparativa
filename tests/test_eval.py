"""Tests for ``comparativa listen`` / ``comparativa report`` (Sortie 9).

Builds a synthetic ``bench/<condition>/<episode>/`` fixture — tiny wavs plus
minimal ``manifest.json``/``metrics.json`` files, one condition (``A``)
deliberately missing ``manifest.json`` to model a Swift-side condition that
doesn't write the Python manifest schema — and round-trips it through
``listen`` (blinding) and ``report`` (tabulation + REPORT.md skeleton).

No real bench run, no network, no MLX checkpoints: everything here is
fixture data written directly to a pytest ``tmp_path``.
"""

from __future__ import annotations

import csv
import json
import wave
from pathlib import Path

import pytest

from comparativa.cli import main as cli_main
from comparativa.eval import (
    CAVEAT_ROWS,
    VERDICT_SECTIONS,
    BlindError,
    aggregate_by_condition,
    blind,
    discover_clips,
    discover_metrics,
    load_key,
    load_scores,
    mlx_whisper_available,
    objective_proxy_status,
    render_performance_table,
    render_report,
    unblind_scores,
    write_report,
)
from comparativa.eval.blind import SCORE_COLUMNS
from comparativa.eval.proxy import DROPPED_REASON


def _write_wav(path: Path, *, seconds: float = 0.2, sample_rate: int = 24000) -> None:
    """A tiny silent mono 16-bit wav — enough to be a real, openable file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * n_frames)


def _make_condition(
    bench_dir: Path,
    condition: str,
    episode: str,
    *,
    stack: str,
    engine: str,
    checkpoint: str,
    write_manifest: bool = True,
    extra_metric: float | None = None,
) -> Path:
    episode_dir = bench_dir / condition / episode
    wav_path = episode_dir / f"{episode}.wav"
    _write_wav(wav_path)

    if write_manifest:
        manifest = {
            "schema_version": 1,
            "episode": {"name": episode},
            "engine": {"key": engine, "checkpoint": checkpoint},
            "outputs": {"wav": str(wav_path)},
        }
        (episode_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    metrics = {
        "condition": condition,
        "stack": stack,
        "engine": engine,
        "episode": episode,
        "wall_seconds": 12.5,
        "rtf": 0.8,
        "peak_rss_bytes": 123456789,
    }
    if extra_metric is not None:
        metrics["model_load_seconds"] = extra_metric
    (episode_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    return wav_path


@pytest.fixture
def bench_dir(tmp_path: Path) -> Path:
    bench = tmp_path / "bench"
    _make_condition(
        bench,
        "C",
        "ep1",
        stack="python",
        engine="qwen3-1.7b",
        checkpoint="Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
        extra_metric=3.2,
    )
    _make_condition(
        bench,
        "D",
        "ep1",
        stack="python",
        engine="chatterbox",
        checkpoint="chatterbox-fp16",
    )
    # Condition A: Swift side, no manifest.json (models Sortie 8's wrapper).
    _make_condition(
        bench,
        "A",
        "ep1",
        stack="swift",
        engine="qwen3-1.7b",
        checkpoint="Qwen3-TTS-12Hz-1.7B-bf16",
        write_manifest=False,
    )
    return bench


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def test_discover_clips_finds_every_condition_including_manifestless(bench_dir: Path) -> None:
    clips = discover_clips(bench_dir)
    assert {c.condition for c in clips} == {"A", "C", "D"}
    by_condition = {c.condition: c for c in clips}
    assert by_condition["A"].audio_path.exists()
    assert by_condition["A"].stack == "swift"
    assert by_condition["C"].engine == "qwen3-1.7b"


def test_discover_clips_empty_bench_dir_yields_nothing(tmp_path: Path) -> None:
    assert discover_clips(tmp_path / "empty-bench") == []


def test_discover_clips_resolves_stack_engine_checkpoint_from_envelope(
    tmp_path: Path,
) -> None:
    """``metrics.json`` is a schema_version-1 envelope (``{"entries": [...]}``),
    not a bare entry. Reading it as a bare entry silently yields ``None`` for
    ``engine``/``stack``/``checkpoint`` in the key file — the Sortie 9 bug
    this test guards against.
    """
    bench = tmp_path / "bench"
    episode_dir = bench / "C" / "ep1"
    _write_wav(episode_dir / "ep1.wav")
    envelope = {
        "schema_version": 1,
        "generated_by": "comparativa bench",
        "written_at": "2026-08-15T23:53:35+00:00",
        "entries": [
            {
                "condition": "C",
                "stack": "python",
                "engine": "qwen3-1.7b",
                "checkpoint": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
                "episode": "ep1",
            }
        ],
    }
    (episode_dir / "metrics.json").write_text(json.dumps(envelope), encoding="utf-8")

    clips = discover_clips(bench)
    assert len(clips) == 1
    clip = clips[0]
    assert clip.stack == "python"
    assert clip.engine == "qwen3-1.7b"
    assert clip.checkpoint == "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"


def test_discover_clips_envelope_with_multiple_entries_matches_by_episode(
    tmp_path: Path,
) -> None:
    """A single metrics.json (e.g. Sortie 8's per-condition file) can hold
    entries for more than one episode; discover_clips must pick the entry
    matching this episode dir, not just the first one in the list.
    """
    bench = tmp_path / "bench"
    episode_dir = bench / "A" / "ep2"
    _write_wav(episode_dir / "ep2.wav")
    envelope = {
        "schema_version": 1,
        "entries": [
            {
                "condition": "A",
                "stack": "swift",
                "engine": "produciesta",
                "checkpoint": "ckpt-ep1",
                "episode": "ep1",
            },
            {
                "condition": "A",
                "stack": "swift",
                "engine": "produciesta",
                "checkpoint": "ckpt-ep2",
                "episode": "ep2",
            },
        ],
    }
    (episode_dir / "metrics.json").write_text(json.dumps(envelope), encoding="utf-8")

    clips = discover_clips(bench)
    assert len(clips) == 1
    assert clips[0].checkpoint == "ckpt-ep2"


# ---------------------------------------------------------------------------
# listen / blinding
# ---------------------------------------------------------------------------


def test_blind_produces_recoverable_opaque_ids(bench_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "listen-out"
    listen_set = blind(bench_dir, out_dir, seed=20260815)

    assert len(listen_set) == 3
    blinded_files = sorted(listen_set.set_dir.iterdir())
    assert len(blinded_files) == 3

    key_by_id = load_key(listen_set.key_path)
    assert set(key_by_id) == {row["opaque_id"] for row in listen_set.key_rows}

    # No filename in the blinded set leaks its condition or episode name.
    for blinded_file in blinded_files:
        assert blinded_file.stem in key_by_id
        assert "C" != blinded_file.stem and "D" != blinded_file.stem

    # Each key row recovers the original condition and byte-identical audio.
    conditions_seen = set()
    for opaque_id, row in key_by_id.items():
        conditions_seen.add(row["condition"])
        blinded_path = listen_set.set_dir / f"{opaque_id}.wav"
        assert blinded_path.exists()
        original = Path(row["original_path"])
        assert blinded_path.read_bytes() == original.read_bytes()
    assert conditions_seen == {"A", "C", "D"}


def test_blind_seed_is_deterministic(bench_dir: Path, tmp_path: Path) -> None:
    set1 = blind(bench_dir, tmp_path / "out1", seed=42)
    set2 = blind(bench_dir, tmp_path / "out2", seed=42)
    ids1 = [row["opaque_id"] for row in set1.key_rows]
    ids2 = [row["opaque_id"] for row in set2.key_rows]
    conditions1 = [row["condition"] for row in set1.key_rows]
    conditions2 = [row["condition"] for row in set2.key_rows]
    # Opaque ids themselves are random (secrets.token_hex), but the *order*
    # clips were assigned ids in is seeded and must match run to run.
    assert len(ids1) == len(ids2) == 3
    assert conditions1 == conditions2


def test_blind_scoring_sheet_has_fr13_columns_and_is_blind(
    bench_dir: Path, tmp_path: Path
) -> None:
    listen_set = blind(bench_dir, tmp_path / "listen-out", seed=1)
    with listen_set.scoring_sheet_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    for row in rows:
        assert "condition" not in row
        assert "episode" not in row
        for column in SCORE_COLUMNS:
            assert row[column] == ""
        assert set(row) == {"opaque_id", *SCORE_COLUMNS, "notes"}


def test_blind_raises_on_empty_bench_dir(tmp_path: Path) -> None:
    with pytest.raises(BlindError):
        blind(tmp_path / "no-such-bench", tmp_path / "out")


# ---------------------------------------------------------------------------
# metrics tabulation
# ---------------------------------------------------------------------------


def test_discover_metrics_and_extra_field_tolerance(bench_dir: Path) -> None:
    records = discover_metrics(bench_dir)
    assert len(records) == 3
    table = render_performance_table(records)
    assert "condition" in table
    assert "wall_seconds" in table
    # The extra field on condition C's metrics.json must not crash rendering
    # and must show up as its own column (tolerant of Sortie 7's schema).
    assert "model_load_seconds" in table
    for condition in ("A", "C", "D"):
        assert f"| {condition} |" in table


def test_render_performance_table_empty() -> None:
    assert "no metrics.json" in render_performance_table([])


# ---------------------------------------------------------------------------
# scoring round-trip
# ---------------------------------------------------------------------------


def test_scoring_round_trip_aggregates_by_condition(bench_dir: Path, tmp_path: Path) -> None:
    listen_set = blind(bench_dir, tmp_path / "listen-out", seed=7)
    key_by_id = load_key(listen_set.key_path)

    # Fill in the scoring sheet as a human would.
    rows_by_condition = {}
    for opaque_id, row in key_by_id.items():
        rows_by_condition.setdefault(row["condition"], []).append(opaque_id)

    scores_path = tmp_path / "filled_scores.csv"
    with scores_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["opaque_id", *SCORE_COLUMNS, "notes"])
        writer.writeheader()
        for opaque_id, row in key_by_id.items():
            value = 4.0 if row["condition"] == "C" else 2.0
            writer.writerow(
                {
                    "opaque_id": opaque_id,
                    **{col: value for col in SCORE_COLUMNS},
                    "notes": "",
                }
            )

    score_rows = load_scores(scores_path)
    merged = unblind_scores(score_rows, key_by_id)
    assert len(merged) == 3
    aggregates = aggregate_by_condition(merged)
    assert aggregates["C"]["means"]["naturalness"] == pytest.approx(4.0)
    assert aggregates["D"]["means"]["naturalness"] == pytest.approx(2.0)
    assert aggregates["C"]["n"] == 1


def test_unblind_scores_drops_unknown_opaque_ids(bench_dir: Path, tmp_path: Path) -> None:
    listen_set = blind(bench_dir, tmp_path / "listen-out", seed=3)
    key_by_id = load_key(listen_set.key_path)
    bogus_rows = [{"opaque_id": "clip-doesnotexist", "naturalness": "5"}]
    assert unblind_scores(bogus_rows, key_by_id) == []


# ---------------------------------------------------------------------------
# objective proxy (mlx-whisper dropped path)
# ---------------------------------------------------------------------------


def test_objective_proxy_reports_dropped_or_included_consistently() -> None:
    available, status = objective_proxy_status()
    assert available == mlx_whisper_available()
    if available:
        assert status.startswith("included")
    else:
        assert status == f"dropped — {DROPPED_REASON}"


# ---------------------------------------------------------------------------
# report rendering
# ---------------------------------------------------------------------------


def test_render_report_contains_tables_caveats_and_verdict_skeleton(bench_dir: Path) -> None:
    text = render_report(bench_dir)
    for condition in ("A", "C", "D"):
        assert f"| {condition} |" in text
    for caveat in CAVEAT_ROWS:
        assert caveat in text
    for name, _description in VERDICT_SECTIONS:
        assert f"### {name}" in text
    assert "objective proxy:" in text
    assert "dropped" in text or "included" in text
    assert "empty until human scores arrive" in text


def test_render_report_with_scores_shows_aggregated_table(
    bench_dir: Path, tmp_path: Path
) -> None:
    listen_set = blind(bench_dir, tmp_path / "listen-out", seed=99)
    key_by_id = load_key(listen_set.key_path)

    scores_path = tmp_path / "scores.csv"
    with scores_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["opaque_id", *SCORE_COLUMNS, "notes"])
        writer.writeheader()
        for opaque_id in key_by_id:
            writer.writerow(
                {"opaque_id": opaque_id, **{col: "3" for col in SCORE_COLUMNS}, "notes": "fine"}
            )

    text = render_report(bench_dir, scores_path=scores_path, key_path=listen_set.key_path)
    assert "scored clip(s) unblinded" in text
    assert "3.00" in text


def test_write_report_writes_file(bench_dir: Path, tmp_path: Path) -> None:
    out_path = tmp_path / "REPORT.md"
    written = write_report(bench_dir, out_path)
    assert written == out_path
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8").startswith("# REPORT")


# ---------------------------------------------------------------------------
# CLI end-to-end
# ---------------------------------------------------------------------------


def test_cli_listen_then_report_round_trip(bench_dir: Path, tmp_path: Path) -> None:
    listen_out = tmp_path / "cli-listen"
    exit_code = cli_main(
        ["listen", str(bench_dir), "-o", str(listen_out), "--seed", "1234"]
    )
    assert exit_code == 0
    assert (listen_out / "key.csv").exists()
    assert (listen_out / "scoring_sheet.csv").exists()
    assert (listen_out / "set").is_dir()
    assert len(list((listen_out / "set").iterdir())) == 3

    report_out = tmp_path / "REPORT.md"
    exit_code = cli_main(
        [
            "report",
            str(bench_dir),
            "-o",
            str(report_out),
            "--scores",
            str(listen_out / "scoring_sheet.csv"),
            "--key",
            str(listen_out / "key.csv"),
        ]
    )
    assert exit_code == 0
    assert report_out.exists()
    text = report_out.read_text(encoding="utf-8")
    assert "## Performance" in text
    assert "## Listening scores" in text
    assert "## Verdicts" in text


def test_cli_listen_without_bench_dir_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["listen"]) == 0
    assert "comparativa listen" in capsys.readouterr().out


def test_cli_report_without_bench_dir_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["report"]) == 0
    assert "comparativa report" in capsys.readouterr().out


def test_cli_listen_on_empty_bench_dir_fails_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli_main(["listen", str(tmp_path / "no-bench")])
    assert exit_code == 1
    assert "comparativa listen:" in capsys.readouterr().err
