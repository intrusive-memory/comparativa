"""Unit and integration tests for episode assembly and ``generate`` (Sortie 6).

Everything here except :func:`test_full_cold_open_episode` runs against a fake
model — a sine generator whose length tracks the text — so the assembly maths,
the manifest schema, the plan, and the CLI are all covered in under a second.

The one real test (marked ``smoke``, skipped unless ``--smoke`` or
``COMPARATIVA_SMOKE=1``) generates the entire cold open on ``qwen3-0.6b`` and
checks the two invariants the execution plan names: the episode's duration is
the sum of its line durations plus its gaps, and the manifest has exactly one
record per spoken line of the parse.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from comparativa.cli import main as cli_main
from comparativa.generation import (
    AssemblyError,
    AssemblyOptions,
    Engine,
    GenerationError,
    LineAudio,
    afconvert_available,
    assemble,
    build_plan,
    generate_episode,
    manifest_line_count,
    normalize_line,
    spec,
    spoken_line_count,
    trim_edges,
    write_outputs,
)
from comparativa.generation.assembly import LOUDNESS_BLOCK_SECONDS, SILENCE_THRESHOLD
from comparativa.generation.episode import DEFAULT_PRESETS_PATH, SEED_STRIDE

from conftest import CORPUS_ROOT

GRANVILLE = CORPUS_ROOT
COLD_OPEN = GRANVILLE / "episodes" / "episode_1_01_cold_open.fountain"
BUMPER = GRANVILLE / "episodes" / "episode_1_01a_bumper_donnie_and_arnie_1.fountain"

#: The engine the integration test uses (size-degradation probe; fastest qwen3).
TEST_ENGINE = "qwen3-0.6b"

#: One policy, used for both sides of every line-count comparison.
TEST_POLICY = "fr3"

corpus = pytest.mark.skipif(
    not COLD_OPEN.is_file(), reason=f"mission corpus not present at {GRANVILLE}"
)


# ---------------------------------------------------------------------------
# Fake model: a sine, padded with the silence a real model emits
# ---------------------------------------------------------------------------

SAMPLE_RATE = 24000
PAD_SECONDS = 0.1


class _Result:
    def __init__(self, audio: np.ndarray) -> None:
        self.audio = audio


class SineModel:
    """Emits ``len(text) * seconds_per_char`` of tone between silent pads."""

    sample_rate = SAMPLE_RATE

    def __init__(self, *, seconds_per_char: float = 0.05, amplitude: float = 0.2) -> None:
        self.seconds_per_char = seconds_per_char
        self.amplitude = amplitude
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        voiced = max(1, int(len(kwargs["text"]) * self.seconds_per_char * self.sample_rate))
        t = np.arange(voiced, dtype=np.float32) / self.sample_rate
        tone = (self.amplitude * np.sin(2.0 * np.pi * 200.0 * t)).astype(np.float32)
        pad = np.zeros(int(PAD_SECONDS * self.sample_rate), dtype=np.float32)
        yield _Result(np.concatenate([pad, tone, pad]))


def fake_engine(key: str = TEST_ENGINE, **kwargs) -> Engine:
    return Engine(spec(key), SineModel(**kwargs))


def tone(seconds: float, *, amplitude: float = 0.2, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    return (amplitude * np.sin(2.0 * np.pi * 200.0 * t)).astype(np.float32)


def silence(seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


# ---------------------------------------------------------------------------
# Task 1a: seam trimming (Produciesta WAVAssembler parity)
# ---------------------------------------------------------------------------


def test_trim_removes_pads_and_keeps_an_8ms_guard():
    audio = np.concatenate([silence(0.5), tone(1.0), silence(0.5)])
    trimmed = trim_edges(audio, SAMPLE_RATE)

    # 0.5 s of pad minus the 8 ms guard is removed from each edge.
    assert trimmed.head_seconds == pytest.approx(0.492, abs=0.002)
    assert trimmed.tail_seconds == pytest.approx(0.492, abs=0.002)
    # 1 s of tone plus two 8 ms guards (± a few samples of zero crossing).
    assert trimmed.audio.size == pytest.approx(int(1.016 * SAMPLE_RATE), abs=16)


def test_trim_respects_which_edges_were_asked_for():
    audio = np.concatenate([silence(0.5), tone(1.0), silence(0.5)])

    head_only = trim_edges(audio, SAMPLE_RATE, head=True, tail=False)
    assert head_only.head_seconds > 0.4
    assert head_only.tail_seconds == 0.0

    neither = trim_edges(audio, SAMPLE_RATE, head=False, tail=False)
    assert neither.audio.size == audio.size


def test_a_wholly_silent_internal_seam_vanishes_but_an_edge_span_survives():
    quiet = np.full(SAMPLE_RATE, SILENCE_THRESHOLD / 2, dtype=np.float32)
    assert trim_edges(quiet, SAMPLE_RATE, head=True, tail=True).audio.size == 0
    assert trim_edges(quiet, SAMPLE_RATE, head=False, tail=True).audio.size == quiet.size


# ---------------------------------------------------------------------------
# Task 1b: loudness (RD-3)
# ---------------------------------------------------------------------------


def test_normalization_hits_the_target_when_headroom_allows():
    result = normalize_line(tone(3.0, amplitude=0.02), SAMPLE_RATE, target_lufs=-30.0)
    assert result.applied
    assert result.output_lufs == pytest.approx(-30.0, abs=0.3)
    assert not result.peak_limited


def test_peak_guard_prevents_clipping_and_records_the_shortfall():
    result = normalize_line(tone(3.0, amplitude=0.9), SAMPLE_RATE, target_lufs=0.0)
    assert result.applied
    assert result.peak_limited
    assert result.peak_after <= 0.99 + 1e-6
    assert result.applied_gain_db < result.requested_gain_db
    assert result.to_dict()["shortfall_db"] > 0


def test_a_line_shorter_than_the_bs1770_block_is_left_alone():
    short = tone(LOUDNESS_BLOCK_SECONDS / 2)
    result = normalize_line(short, SAMPLE_RATE)
    assert not result.applied
    assert "BS.1770" in result.reason
    assert np.array_equal(result.audio, short)


def test_normalization_can_be_disabled():
    audio = tone(1.0)
    result = normalize_line(audio, SAMPLE_RATE, target_lufs=None)
    assert not result.applied
    assert np.array_equal(result.audio, audio)


# ---------------------------------------------------------------------------
# Task 1c: the timeline
# ---------------------------------------------------------------------------


def _line(seconds: float, scene: int, index: int) -> LineAudio:
    return LineAudio(
        audio=tone(seconds),
        sample_rate=SAMPLE_RATE,
        scene_index=scene,
        record={"index": index},
    )


NO_LOUDNESS = AssemblyOptions(target_lufs=None, trim_seams=False)


def test_gaps_are_inter_line_within_a_scene_and_inter_scene_across_one():
    episode = assemble(
        [_line(1.0, 1, 0), _line(1.0, 1, 1), _line(1.0, 2, 2)],
        options=NO_LOUDNESS,
    )
    gaps = [line["gap_before_seconds"] for line in episode.lines]
    assert gaps == [0.0, NO_LOUDNESS.line_gap_seconds, NO_LOUDNESS.scene_gap_seconds]
    assert episode.gap_seconds == pytest.approx(
        NO_LOUDNESS.line_gap_seconds + NO_LOUDNESS.scene_gap_seconds
    )


def test_duration_is_exactly_the_sum_of_lines_and_gaps():
    episode = assemble(
        [_line(1.0, 1, 0), _line(2.0, 1, 1), _line(0.5, 2, 2)],
        options=NO_LOUDNESS,
    )
    accounted = sum(
        line["assembled_duration_seconds"] + line["gap_before_seconds"]
        for line in episode.lines
    )
    assert episode.duration_seconds == pytest.approx(accounted, abs=0.002)
    assert episode.duration_seconds == pytest.approx(3.5 + 0.25 + 0.75, abs=0.002)


def test_offsets_advance_monotonically_and_start_at_zero():
    episode = assemble([_line(1.0, 1, i) for i in range(4)], options=NO_LOUDNESS)
    offsets = [line["offset_seconds"] for line in episode.lines]
    assert offsets[0] == 0.0
    assert offsets == sorted(offsets)
    assert offsets[-1] == pytest.approx(3 * (1.0 + 0.25), abs=0.002)


def test_internal_seams_are_trimmed_but_the_outer_envelope_is_kept():
    padded = [
        LineAudio(
            audio=np.concatenate([silence(0.5), tone(1.0), silence(0.5)]),
            sample_rate=SAMPLE_RATE,
            scene_index=1,
            record={"index": i},
        )
        for i in range(3)
    ]
    episode = assemble(padded, options=AssemblyOptions(target_lufs=None))
    trims = [line["trim"] for line in episode.lines]
    assert trims[0]["head_seconds"] == 0.0  # first line keeps its natural head
    assert trims[0]["tail_seconds"] > 0.4
    assert trims[1]["head_seconds"] > 0.4 and trims[1]["tail_seconds"] > 0.4
    assert trims[2]["tail_seconds"] == 0.0  # last line keeps its natural tail


def test_a_line_that_generated_no_audio_stays_in_the_manifest_and_costs_no_time():
    lines = [
        _line(1.0, 1, 0),
        LineAudio(np.zeros(0, dtype=np.float32), SAMPLE_RATE, 1, {"index": 1}),
        _line(1.0, 1, 2),
    ]
    episode = assemble(lines, options=NO_LOUDNESS)
    assert len(episode.lines) == 3
    assert episode.lines[1]["assembled_duration_seconds"] == 0.0
    assert episode.lines[1]["placed"] is False
    assert episode.duration_seconds == pytest.approx(2.0 + 0.25, abs=0.002)


def test_mixed_sample_rates_are_refused():
    lines = [_line(1.0, 1, 0), LineAudio(tone(1.0), 32000, 1, {"index": 1})]
    with pytest.raises(AssemblyError, match="mixed sample rates"):
        assemble(lines, options=NO_LOUDNESS)


# ---------------------------------------------------------------------------
# Task 3: the plan
# ---------------------------------------------------------------------------


@corpus
def test_plan_line_count_matches_the_parse_spoken_line_count():
    plan = build_plan(COLD_OPEN, TEST_ENGINE, speech_policy=TEST_POLICY)
    assert len(plan.lines) == spoken_line_count(COLD_OPEN, speech_policy=TEST_POLICY)
    assert plan.script_line_count == len(plan.lines)
    assert plan.unresolved_cues == []


@corpus
def test_plan_resolves_every_voice_from_presets_yaml():
    plan = build_plan(COLD_OPEN, TEST_ENGINE, speech_policy=TEST_POLICY)
    assert plan.presets_path == DEFAULT_PRESETS_PATH
    assert set(plan.voice_by_character) == set(plan.characters)
    assert all(voice for voice in plan.voice_by_character.values())
    for line in plan.lines:
        assert line.voice == plan.voice_by_character[line.character]


@corpus
def test_single_voice_engines_need_no_preset_assignment():
    plan = build_plan(COLD_OPEN, "soprano", speech_policy=TEST_POLICY, presets_path=None)
    assert plan.lines
    assert all(line.voice is None for line in plan.lines)


@corpus
def test_a_preset_engine_without_assignments_is_a_clear_error():
    with pytest.raises(GenerationError, match="preset voice per character"):
        build_plan(COLD_OPEN, TEST_ENGINE, speech_policy=TEST_POLICY, presets_path=None)


@corpus
def test_seeds_are_deterministic_and_never_collide_between_lines():
    plan = build_plan(COLD_OPEN, TEST_ENGINE, speech_policy=TEST_POLICY, seed=7)
    seeds = [line.seed for line in plan.lines]
    assert seeds[:3] == [7, 7 + SEED_STRIDE, 7 + 2 * SEED_STRIDE]
    assert len(set(seeds)) == len(seeds)

    unseeded = build_plan(COLD_OPEN, TEST_ENGINE, speech_policy=TEST_POLICY, seed=None)
    assert all(line.seed is None for line in unseeded.lines)


@corpus
def test_limit_truncates_the_plan_but_records_the_full_script_count():
    plan = build_plan(COLD_OPEN, TEST_ENGINE, speech_policy=TEST_POLICY, limit=5)
    assert len(plan.lines) == 5
    assert plan.script_line_count > 5


@corpus
def test_the_speech_policy_changes_what_the_episode_contains():
    fr3 = build_plan(COLD_OPEN, TEST_ENGINE, speech_policy="fr3")
    parity = build_plan(COLD_OPEN, TEST_ENGINE, speech_policy="produciesta-parity")
    assert len(parity.lines) > len(fr3.lines)
    assert fr3.speech_policy == "fr3"
    assert parity.speech_policy == "produciesta-parity"


# ---------------------------------------------------------------------------
# Task 2: the manifest
# ---------------------------------------------------------------------------


@corpus
def test_manifest_records_the_full_provenance_of_a_run():
    plan = build_plan(COLD_OPEN, TEST_ENGINE, speech_policy=TEST_POLICY, limit=4)
    manifest = generate_episode(plan, engine=fake_engine()).manifest

    assert manifest["schema_version"] == 1
    # SUPERVISOR RULING: the active speech policy must be recorded, because it
    # changes what the episode contains and therefore what it is comparable to.
    assert manifest["speech_policy"] == TEST_POLICY
    assert manifest["text_policy"] == "parity"
    assert manifest["episode"]["sha256"]
    assert manifest["engine"]["key"] == TEST_ENGINE
    assert manifest["engine"]["checkpoint"] == spec(TEST_ENGINE).checkpoint
    assert manifest["engine"]["sampling"]["temperature"] == pytest.approx(0.7)
    assert manifest["voices"]["presets_sha256"]
    assert manifest["seed"]["base"] and manifest["seed"]["stride"] == SEED_STRIDE
    assert manifest["assembly"]["target_lufs"] == -16.0
    assert manifest["script_line_count"] == spoken_line_count(
        COLD_OPEN, speech_policy=TEST_POLICY
    )


@corpus
def test_every_line_record_carries_what_the_report_needs():
    plan = build_plan(COLD_OPEN, TEST_ENGINE, speech_policy=TEST_POLICY, limit=4)
    manifest = generate_episode(plan, engine=fake_engine()).manifest

    required = {
        "index", "character", "text", "scene_index", "element_index",
        "engine", "checkpoint", "voice", "sampling", "seed", "seeding",
        "duration_seconds", "generate_seconds", "real_time_factor", "chunks",
        "offset_seconds", "assembled_duration_seconds", "gap_before_seconds",
        "trim", "loudness",
    }
    for record in manifest["lines"]:
        assert required <= set(record), sorted(required - set(record))
    assert [r["index"] for r in manifest["lines"]] == list(range(4))


@corpus
def test_totals_are_internally_consistent():
    plan = build_plan(COLD_OPEN, TEST_ENGINE, speech_policy=TEST_POLICY, limit=6)
    result = generate_episode(plan, engine=fake_engine())
    totals = result.manifest["totals"]

    assert totals["line_count"] == 6
    accounted = sum(
        line["assembled_duration_seconds"] + line["gap_before_seconds"]
        for line in result.manifest["lines"]
    )
    assert totals["duration_seconds"] == pytest.approx(accounted, abs=0.01)
    assert totals["audio_seconds"] + totals["gap_seconds"] == pytest.approx(
        totals["duration_seconds"], abs=0.01
    )


# ---------------------------------------------------------------------------
# Task 2/3: outputs
# ---------------------------------------------------------------------------


@corpus
def test_write_outputs_produces_wav_manifest_and_m4a(tmp_path):
    plan = build_plan(COLD_OPEN, TEST_ENGINE, speech_policy=TEST_POLICY, limit=3)
    result = generate_episode(plan, engine=fake_engine())
    outputs = write_outputs(result, tmp_path, m4a=afconvert_available())

    assert Path(outputs["wav"]).is_file()
    assert Path(outputs["manifest"]).name == "manifest.json"
    document = json.loads(Path(outputs["manifest"]).read_text(encoding="utf-8"))
    assert manifest_line_count(document) == 3
    assert document["outputs"]["wav"] == outputs["wav"]

    if afconvert_available():
        assert Path(outputs["m4a"]).is_file()
        assert Path(outputs["m4a"]).stat().st_size > 0


@corpus
def test_written_wav_length_matches_the_manifest_duration(tmp_path):
    import soundfile as sf

    plan = build_plan(COLD_OPEN, TEST_ENGINE, speech_policy=TEST_POLICY, limit=3)
    result = generate_episode(plan, engine=fake_engine())
    outputs = write_outputs(result, tmp_path, m4a=False)

    info = sf.info(outputs["wav"])
    assert info.samplerate == result.sample_rate
    assert info.channels == 1
    assert info.frames / info.samplerate == pytest.approx(
        result.manifest["totals"]["duration_seconds"], abs=0.01
    )


# ---------------------------------------------------------------------------
# Task 3: the CLI
# ---------------------------------------------------------------------------


@corpus
def test_generate_dry_run_exits_zero_without_loading_a_model(capsys):
    code = cli_main(
        ["generate", str(COLD_OPEN), "--engine", TEST_ENGINE, "--dry-run"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "qwen3-0.6b" in out and "speech policy: fr3" in out


@corpus
def test_generate_dry_run_honours_the_speech_policy_flag(capsys):
    assert (
        cli_main(
            [
                "generate", str(BUMPER), "--engine", TEST_ENGINE,
                "--speech-policy", "produciesta-parity", "--dry-run",
            ]
        )
        == 0
    )
    assert "speech policy: produciesta-parity" in capsys.readouterr().out


def test_generate_without_an_engine_is_a_usage_error(capsys):
    assert cli_main(["generate", str(COLD_OPEN)]) == 2
    assert "--engine is required" in capsys.readouterr().err


def test_generate_with_no_episode_prints_help(capsys):
    assert cli_main(["generate"]) == 0
    assert "usage: comparativa generate" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Task 4: audition
# ---------------------------------------------------------------------------


def test_audition_generates_one_take_per_character_per_engine(tmp_path, monkeypatch):
    from comparativa.generation import audition as audition_module

    monkeypatch.setattr(
        audition_module, "load_engine", lambda key, **kw: fake_engine(key)
    )
    run = audition_module.run_audition(
        {"HUNTER": {"qwen3-0.6b": "ryan"}, "JOANN": {"qwen3-0.6b": "ono_anna"}},
        engine_keys=["qwen3-0.6b"],
        out_dir=tmp_path,
    )

    assert run.ok
    assert len(run.takes) == 2
    assert {t.character for t in run.takes} == {"HUNTER", "JOANN"}
    for take in run.takes:
        assert Path(take.path).is_file()
        assert take.record["voice"] == take.voice
        assert "loudness" in take.record

    path = audition_module.write_audition_manifest(run, tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["text"] == audition_module.AUDITION_TEXT
    assert len(document["takes"]) == 2


def test_audition_can_be_limited_to_a_subset(tmp_path, monkeypatch):
    from comparativa.generation import audition as audition_module

    monkeypatch.setattr(
        audition_module, "load_engine", lambda key, **kw: fake_engine(key)
    )
    run = audition_module.run_audition(
        {"HUNTER": {"qwen3-0.6b": "ryan"}, "JOANN": {"qwen3-0.6b": "ono_anna"}},
        engine_keys=["qwen3-0.6b"],
        characters=["JOANN"],
        out_dir=tmp_path,
    )
    assert [t.character for t in run.takes] == ["JOANN"]


def test_voices_audition_flags_are_wired():
    """``--audition`` and its options live on the ``voices`` subcommand."""
    from comparativa.cli import build_parser

    action = next(
        sub
        for sub in build_parser()._subparsers._group_actions  # noqa: SLF001
        if hasattr(sub, "choices")
    )
    options = set(action.choices["voices"]._option_string_actions)  # noqa: SLF001
    assert {
        "--audition",
        "--audition-out",
        "--audition-text",
        "--audition-characters",
        "--audition-seed",
    } <= options


# ---------------------------------------------------------------------------
# Task 5: the integration test (real checkpoint; slow)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@corpus
def test_full_cold_open_episode(tmp_path):
    """The whole cold open on ``qwen3-0.6b``, start to finish.

    Two assertions, both from the execution plan:

    * episode duration ≈ Σ(line durations) + gaps, within 0.1 s per line;
    * manifest line count == the spoken-line count ``parse`` reports.

    Both sides of the line-count comparison use the same speech policy, so a
    policy change can never make this pass for the wrong reason.
    """
    plan = build_plan(COLD_OPEN, TEST_ENGINE, speech_policy=TEST_POLICY)
    result = generate_episode(plan)
    outputs = write_outputs(result, tmp_path, m4a=afconvert_available())

    expected_lines = spoken_line_count(COLD_OPEN, speech_policy=TEST_POLICY)
    assert manifest_line_count(outputs["manifest"]) == expected_lines
    assert result.manifest["totals"]["line_count"] == expected_lines
    assert result.manifest["speech_policy"] == TEST_POLICY

    lines = result.manifest["lines"]
    accounted = sum(
        line["assembled_duration_seconds"] + line["gap_before_seconds"] for line in lines
    )
    tolerance = 0.1 * len(lines)
    assert result.duration_seconds == pytest.approx(accounted, abs=tolerance)

    # Trimming and normalization only ever shorten a line, never lengthen it.
    raw = sum(line["duration_seconds"] for line in lines)
    placed = sum(line["assembled_duration_seconds"] for line in lines)
    assert placed <= raw + 1e-6

    assert Path(outputs["wav"]).is_file()
    if afconvert_available():
        assert Path(outputs["m4a"]).is_file()
