"""Unit tests for the engine layer (Sortie 5).

Nothing here loads a checkpoint: the model is a stub whose output duration is a
function of the text it is given, which is exactly what the truncation detector
keys on. The heavy end of Sortie 5 lives in ``tests/smoke/test_engine_smoke.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from comparativa.generation import (
    ENGINE_KEYS,
    ENGINE_SPECS,
    QWEN3_SAMPLING,
    QWEN3_TOP_K_DISABLED,
    SEEDING_UNAVAILABLE,
    SMOKE_ENGINE_KEYS,
    SWIFT_PARITY,
    SWIFT_SOURCE,
    Engine,
    EngineError,
    LineRequest,
    SamplingParams,
    check_duration,
    expected_duration,
    spec,
)
from comparativa.generation.smoke import SMOKE_TEXT, SMOKE_VOICE
from comparativa.generation.truncation import WORDS_PER_SECOND
from comparativa.voices import catalog

REPO_ROOT = Path(__file__).resolve().parents[1]
PARITY_DOC = REPO_ROOT / "docs" / "SAMPLING_PARITY.md"
SWIFT_FILE = Path(SWIFT_SOURCE.replace("~", str(Path.home())))


# ---------------------------------------------------------------------------
# Fake model
# ---------------------------------------------------------------------------


class _Result:
    def __init__(self, audio: np.ndarray) -> None:
        self.audio = audio


class FakeModel:
    """A model whose audio length tracks the text, with scriptable failures.

    ``seconds_per_char`` sets the nominal rate. ``fail_first`` makes the first
    call to each distinct text return a fraction of that (a truncation), so the
    retry path can be exercised deterministically.
    """

    sample_rate = 24000

    def __init__(
        self,
        *,
        seconds_per_char: float = 0.06,
        fail_first: bool = False,
        fail_factor: float = 0.1,
    ) -> None:
        self.seconds_per_char = seconds_per_char
        self.fail_first = fail_first
        self.fail_factor = fail_factor
        self.calls: list[dict] = []
        self._seen: set[str] = set()

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        text = kwargs["text"]
        factor = 1.0
        if self.fail_first and text not in self._seen:
            factor = self.fail_factor
        self._seen.add(text)
        samples = int(len(text) * self.seconds_per_char * factor * self.sample_rate)
        yield _Result(np.full(samples, 0.01, dtype=np.float32))


def make_engine(key: str = "qwen3-1.7b", **model_kwargs) -> Engine:
    return Engine(spec(key), FakeModel(**model_kwargs))


# ---------------------------------------------------------------------------
# Sampling parity (task 1)
# ---------------------------------------------------------------------------


def test_qwen3_defaults_are_the_swift_defaults():
    assert QWEN3_SAMPLING.temperature == pytest.approx(0.7)
    assert QWEN3_SAMPLING.top_p == pytest.approx(0.9)
    assert QWEN3_SAMPLING.repetition_penalty == pytest.approx(1.3)
    assert QWEN3_SAMPLING.max_tokens == 16384
    # Swift has no top-k knob at all; mlx-audio defaults it to 50, so we must
    # explicitly disable it or the two stacks silently diverge.
    assert QWEN3_SAMPLING.top_k == QWEN3_TOP_K_DISABLED == 0
    assert "GenerationSettings.swift" in QWEN3_SAMPLING.source


@pytest.mark.skipif(
    not SWIFT_FILE.exists(), reason=f"read-only Swift source not present: {SWIFT_FILE}"
)
def test_parity_values_match_their_cited_swift_lines():
    """Every recorded value is actually on the line it cites."""
    lines = SWIFT_FILE.read_text(encoding="utf-8").splitlines()
    for parity in SWIFT_PARITY:
        default_line = lines[parity.default_line - 1]
        assert parity.swift_name in default_line, (
            f"{parity.swift_name} not on {SWIFT_FILE.name}:{parity.default_line}: "
            f"{default_line!r}"
        )
        rendered = "true" if parity.value is True else str(parity.value)
        assert rendered in default_line, (
            f"{parity.swift_name} default {parity.value} not on line "
            f"{parity.default_line}: {default_line!r}"
        )
        declaration = lines[parity.declaration_line - 1]
        assert parity.swift_name in declaration


def test_sampling_parity_doc_lists_every_parameter():
    assert PARITY_DOC.exists(), "docs/SAMPLING_PARITY.md is a Sortie 5 exit criterion"
    text = PARITY_DOC.read_text(encoding="utf-8")
    for parity in SWIFT_PARITY:
        assert parity.swift_name in text
        assert re.search(rf"\b{parity.default_line}\b", text), (
            f"source line {parity.default_line} for {parity.swift_name} is not cited"
        )
        rendered = "true" if parity.value is True else str(parity.value)
        assert rendered in text


def test_sampling_params_round_trip():
    params = SamplingParams(
        temperature=0.5, top_p=0.8, max_tokens=64, extra={"min_p": 0.1}
    )
    d = params.to_dict()
    assert d == {
        "temperature": 0.5,
        "top_p": 0.8,
        "max_tokens": 64,
        "extra": {"min_p": 0.1},
    }
    assert params.evolve(temperature=0.9).temperature == 0.9
    assert params.temperature == 0.5  # frozen


# ---------------------------------------------------------------------------
# Specs and capabilities (task 2)
# ---------------------------------------------------------------------------


def test_specs_cover_the_catalog_engines():
    # Round 2: the spec table is a superset of the round-1 catalog — the clone
    # engines have no catalog entry because their voices come from .vox files.
    assert set(catalog.ENGINE_KEYS) <= set(ENGINE_KEYS)
    for key in catalog.ENGINE_KEYS:
        assert ENGINE_SPECS[key].checkpoint == catalog.engine(key).checkpoint
    extras = set(ENGINE_KEYS) - set(catalog.ENGINE_KEYS)
    assert extras == {"qwen3-1.7b-clone", "qwen3-0.6b-clone"}
    for key in extras:
        assert ENGINE_SPECS[key].clone_voices
        assert ENGINE_SPECS[key].voices == ()  # tolerant, not a KeyError


def test_smoke_engines_are_the_four_required_by_the_plan():
    assert SMOKE_ENGINE_KEYS == ("qwen3-1.7b", "qwen3-0.6b", "chatterbox", "soprano")


def test_soprano_uses_the_rd1_checkpoint():
    assert spec("soprano").checkpoint == "mlx-community/Soprano-80M-bf16"


@pytest.mark.parametrize("key", catalog.ENGINE_KEYS)
def test_capability_flags_agree_with_the_voice_catalog(key):
    s = spec(key)
    caps = s.capabilities()
    assert caps["preset_voices"] is catalog.engine(key).multi_voice
    assert caps["voice_count"] == len(catalog.engine(key).voices)
    assert caps["seeding"] is True
    assert caps["seeding_note"]


@pytest.mark.parametrize("key", ("qwen3-1.7b-clone", "qwen3-0.6b-clone"))
def test_clone_engine_capabilities(key):
    caps = spec(key).capabilities()
    assert caps["clone_voices"] is True
    assert caps["clone_needs_ref_text"] is True
    assert caps["preset_voices"] is False
    assert caps["voice_count"] == 0
    assert caps["seeding"] is True


def test_chatterbox_family_clones_without_ref_text():
    for key in ("chatterbox", "chatterbox-turbo"):
        caps = spec(key).capabilities()
        assert caps["clone_voices"] is True
        assert caps["clone_needs_ref_text"] is False
    assert spec("soprano").capabilities()["clone_voices"] is False


def test_unknown_engine_raises_with_the_known_list():
    with pytest.raises(KeyError, match="qwen3-1.7b"):
        spec("nope")


# ---------------------------------------------------------------------------
# Voice resolution
# ---------------------------------------------------------------------------


def test_preset_engine_requires_a_voice():
    engine = make_engine("qwen3-1.7b")
    with pytest.raises(EngineError, match="requires one"):
        engine.generate_line(LineRequest(text="Hello there."))


def test_preset_engine_rejects_an_unknown_voice():
    engine = make_engine("qwen3-1.7b")
    with pytest.raises(EngineError, match="not a preset"):
        engine.resolve_voice("hal9000")


def test_preset_engine_accepts_a_catalog_voice():
    engine = make_engine("qwen3-1.7b")
    assert engine.resolve_voice("ryan") == "ryan"
    # Even the dialect-forcing presets exist; assignment policy is Sortie 4's job.
    assert engine.resolve_voice("eric") == "eric"


def test_single_voice_engine_normalizes_whatever_it_is_given():
    engine = make_engine("soprano")
    assert engine.resolve_voice(None) == catalog.DEFAULT_VOICE
    assert engine.resolve_voice("ryan") == catalog.DEFAULT_VOICE


# ---------------------------------------------------------------------------
# Duration sanity check (task 4)
# ---------------------------------------------------------------------------


def test_expected_duration_uses_the_word_rate():
    text = "one two three four five six seven eight nine ten"
    assert expected_duration(text) >= 10 / WORDS_PER_SECOND


def test_check_duration_accepts_a_plausible_duration():
    check = check_duration(SMOKE_TEXT, expected_duration(SMOKE_TEXT))
    assert check.ok
    assert not check.truncated and not check.overrun
    assert check.to_dict()["ratio"] == pytest.approx(1.0, abs=0.01)


def test_check_duration_flags_a_truncation():
    check = check_duration(SMOKE_TEXT, 0.4)
    assert check.truncated and not check.ok
    assert "expected" in check.reason
    assert check.to_dict()["truncated"] is True


def test_check_duration_flags_silence():
    check = check_duration(SMOKE_TEXT, 0.0)
    assert check.truncated
    assert "silent" in check.reason


def test_check_duration_flags_an_overrun():
    check = check_duration(SMOKE_TEXT, expected_duration(SMOKE_TEXT) * 6)
    assert check.overrun and not check.truncated and not check.ok


# ---------------------------------------------------------------------------
# Generation, chunking, retry (tasks 3 and 4)
# ---------------------------------------------------------------------------


def test_generate_line_records_sampling_seed_and_duration():
    engine = make_engine("qwen3-1.7b")
    result = engine.generate_line(
        LineRequest(text=SMOKE_TEXT, voice=SMOKE_VOICE, seed=11, label="l0")
    )
    assert result.duration_seconds > 0
    assert not result.truncated and not result.retried
    record = result.to_dict()
    assert record["engine"] == "qwen3-1.7b"
    assert record["voice"] == SMOKE_VOICE
    assert record["seed"] == 11
    assert record["sampling"]["temperature"] == pytest.approx(0.7)
    assert record["sampling"]["top_k"] == 0
    assert record["label"] == "l0"
    assert len(record["chunks"]) == 1


def test_qwen3_call_passes_parity_parameters_and_the_instruct():
    engine = make_engine("qwen3-1.7b")
    engine.generate_line(
        LineRequest(text="Say it plainly.", voice="ryan", direction="sotto")
    )
    call = engine.model.calls[0]
    assert call["temperature"] == pytest.approx(0.7)
    assert call["top_p"] == pytest.approx(0.9)
    assert call["top_k"] == 0
    assert call["repetition_penalty"] == pytest.approx(1.3)
    assert call["max_tokens"] == 16384
    assert call["instruct"] == "sotto"
    assert call["lang_code"] == "english"
    assert call["voice"] == "ryan"


def test_engines_without_the_instruct_capability_drop_the_direction():
    engine = make_engine("chatterbox")
    result = engine.generate_line(
        LineRequest(text="Say it plainly.", direction="sotto")
    )
    assert result.direction is None
    assert "instruct" not in engine.model.calls[0]


def test_chatterbox_call_uses_its_own_defaults_and_extras():
    engine = make_engine("chatterbox")
    engine.generate_line(LineRequest(text="Say it plainly."))
    call = engine.model.calls[0]
    assert call["temperature"] == pytest.approx(0.8)
    assert call["max_new_tokens"] == 1000
    assert call["min_p"] == pytest.approx(0.05)
    assert call["cfg_weight"] == pytest.approx(0.5)


def test_soprano_call_has_no_top_k_or_repetition_penalty():
    engine = make_engine("soprano")
    engine.generate_line(LineRequest(text="Say it plainly."))
    call = engine.model.calls[0]
    assert set(call) == {"text", "temperature", "top_p", "max_tokens", "verbose"}
    assert call["temperature"] == pytest.approx(0.3)


def test_seeding_is_recorded_per_line():
    engine = make_engine("soprano")
    seeded = engine.generate_line(LineRequest(text="Hello there.", seed=5))
    assert seeded.seeding.startswith("mlx.core.random.seed")
    unseeded = engine.generate_line(LineRequest(text="Hello there."))
    assert unseeded.seeding == "unseeded"


def test_seeding_unavailable_is_recorded_when_the_engine_cannot_be_seeded():
    from dataclasses import replace

    unseedable = replace(spec("soprano"), seeding=False)
    engine = Engine(unseedable, FakeModel())
    result = engine.generate_line(LineRequest(text="Hello there.", seed=5))
    assert result.seeding == SEEDING_UNAVAILABLE
    assert result.to_dict()["seeding"] == SEEDING_UNAVAILABLE


def test_long_lines_are_chunked_for_chatterbox():
    engine = make_engine("chatterbox")
    long_text = " ".join(
        f"This is sentence number {n} of a very long paragraph." for n in range(20)
    )
    chunks = engine.plan_chunks(long_text)
    assert len(chunks) > 1
    result = engine.generate_line(LineRequest(text=long_text))
    assert len(result.chunks) == len(chunks)
    assert len(engine.model.calls) == len(chunks)
    # Inter-chunk pauses land in the audio.
    joined = sum(c.duration_seconds for c in result.chunks)
    pauses = (len(chunks) - 1) * spec("chatterbox").chunk_pause_seconds
    assert result.duration_seconds == pytest.approx(joined + pauses, abs=0.05)


def test_engines_that_chunk_themselves_get_the_whole_line():
    engine = make_engine("soprano")
    long_text = " ".join(
        f"This is sentence number {n} of a very long paragraph." for n in range(20)
    )
    assert engine.plan_chunks(long_text) == [long_text]
    engine.generate_line(LineRequest(text=long_text))
    assert len(engine.model.calls) == 1


def test_breath_segments_are_generated_separately_and_spliced():
    engine = make_engine("chatterbox")
    request = LineRequest(
        text="First span here. Second span here.",
        segments=("First span here.", "Second span here."),
        breath_gap_seconds=0.15,
    )
    result = engine.generate_line(request)
    assert len(engine.model.calls) == 2
    joined = sum(c.duration_seconds for c in result.chunks)
    assert result.duration_seconds == pytest.approx(joined + 0.15, abs=0.02)


def test_a_truncated_generation_is_retried_once_and_flagged():
    engine = make_engine("chatterbox", fail_first=True)
    result = engine.generate_line(LineRequest(text=SMOKE_TEXT, seed=3))
    assert result.retried
    assert not result.truncated  # the retry produced a sane duration
    record = result.to_dict()
    assert record["truncation_retry"] is True
    assert record["chunks"][0]["retried"] is True
    assert record["chunks"][0]["first_attempt_seconds"] < record["duration_seconds"]
    assert len(engine.model.calls) == 2


def test_a_persistently_truncated_generation_stays_flagged():
    engine = make_engine("chatterbox", seconds_per_char=0.005)
    result = engine.generate_line(LineRequest(text=SMOKE_TEXT, seed=3))
    assert result.retried and result.truncated
    assert result.to_dict()["truncated"] is True
    assert len(engine.model.calls) == 2  # exactly one retry, never a loop


def test_empty_text_generates_nothing():
    engine = make_engine("soprano")
    result = engine.generate_line(LineRequest(text="   "))
    assert result.audio.size == 0
    assert result.chunks == []
    assert engine.model.calls == []
