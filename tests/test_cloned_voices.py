"""Round-2 tests: cloned-voice assignment, resolution, and engine dispatch.

Covers ``comparativa.voices.cloned`` (the ``presets-cloned.yaml`` document),
the engine layer's clone paths (with fake models — no checkpoint loads), and
the episode planner's schema-2 resolution. The real-model end of round 2 lives
in ``tests/smoke/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from comparativa.generation.engines import (
    CLONE_ENGINE_KEYS,
    CloneVoice,
    Engine,
    EngineError,
    LineRequest,
    spec,
)
from comparativa.generation.episode import GenerationError, build_plan
from comparativa.voices.assign import dump_yaml
from comparativa.voices.cloned import (
    CLONED_ENGINE_KEYS,
    CLONE_REF_MODEL_SIZE,
    build_cloned_document,
    is_cloned_document,
    resolve_voice_entry,
    unresolved_characters,
)
from comparativa.voices.roster import load_roster

from conftest import CORPUS_ROOT
from test_vox import make_vox

requires_granville_vox = pytest.mark.skipif(
    not (CORPUS_ROOT / "voices" / "ARCHER.vox").is_file(),
    reason="mission corpus not available (see docs/CORPUS_PIN.md)",
)

REF_TEXT = "Hello, this is a reference."


# ---------------------------------------------------------------------------
# Synthetic project fixture
# ---------------------------------------------------------------------------


CAST_MD = """---
type: cast
schemaVersion: 1
cast:
- character: TESTY
  voicePrompt: Male. 30s. A steady synthetic baritone.
  voices:
    voxalta:
    - voices/TESTY.vox
- character: GHOST
  voicePrompt: Female. 40s. Has no .vox bundle yet.
  voices: {}
---
"""


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    (tmp_path / "CAST.md").write_text(CAST_MD, encoding="utf-8")
    (tmp_path / "voices").mkdir()
    make_vox(tmp_path / "voices" / "TESTY.vox", name="TESTY", ref_text=REF_TEXT)
    return tmp_path


# ---------------------------------------------------------------------------
# Document building
# ---------------------------------------------------------------------------


def test_cloned_document_covers_every_engine_and_is_deterministic(project):
    roster = load_roster(project)
    doc = build_cloned_document(roster)

    assert is_cloned_document(doc)
    assert doc["mode"] == "cloned"
    assert tuple(doc["engines"]) == CLONED_ENGINE_KEYS

    testy = doc["assignments"]["TESTY"]["engines"]
    for key in ("qwen3-1.7b-clone", "qwen3-0.6b-clone", "chatterbox", "chatterbox-turbo"):
        entry = testy[key]
        assert entry["mode"] == "clone"
        assert entry["vox"] == "voices/TESTY.vox"
        assert entry["model_size"] == CLONE_REF_MODEL_SIZE[key]
        assert entry["ref_text"] == REF_TEXT
        assert entry["ref_seconds"] > 5.0
    assert testy["soprano"] == {"mode": "default", "voice": "default"}

    # Regenerating from the same inputs must be byte-identical (stale detection).
    assert dump_yaml(doc) == dump_yaml(build_cloned_document(roster))


def test_character_without_vox_falls_back_or_goes_unresolved(project):
    doc = build_cloned_document(load_roster(project))
    ghost = doc["assignments"]["GHOST"]["engines"]

    # No preset fallback exists on the Base checkpoints.
    assert ghost["qwen3-1.7b-clone"] is None
    assert ghost["qwen3-0.6b-clone"] is None
    # The chatterbox family and soprano keep their built-in voice.
    assert ghost["chatterbox"] == {"mode": "default", "voice": "default"}
    assert ghost["chatterbox-turbo"] == {"mode": "default", "voice": "default"}
    assert ghost["soprano"] == {"mode": "default", "voice": "default"}

    problems = unresolved_characters(doc)
    assert problems == {"GHOST": ["qwen3-0.6b-clone", "qwen3-1.7b-clone"]}


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_resolve_clone_entry_loads_the_reference(project):
    doc = build_cloned_document(load_roster(project))
    resolved = resolve_voice_entry(
        doc, "TESTY", "qwen3-1.7b-clone", project_dir=project
    )
    assert resolved is not None and resolved.mode == "clone"
    clone = resolved.clone
    assert clone is not None
    assert clone.name == "vox:TESTY"
    assert clone.ref_text == REF_TEXT
    assert clone.sample_rate == 24000
    assert clone.seconds > 5.0
    assert resolved.provenance["vox_sha256"] == doc["assignments"]["TESTY"]["engines"][
        "qwen3-1.7b-clone"
    ]["vox_sha256"]
    assert "stale" not in resolved.provenance


def test_resolve_handles_schema_one_and_missing_entries(project):
    round1 = {
        "schema_version": 1,
        "assignments": {"ARCHER": {"engines": {"qwen3-1.7b": "ryan"}}},
    }
    assert not is_cloned_document(round1)
    resolved = resolve_voice_entry(round1, "ARCHER", "qwen3-1.7b")
    assert resolved is not None
    assert (resolved.mode, resolved.voice, resolved.clone) == ("preset", "ryan", None)
    assert resolve_voice_entry(round1, "ARCHER", "qwen3-1.7b-clone") is None
    assert resolve_voice_entry(round1, "NOBODY", "qwen3-1.7b") is None

    doc = build_cloned_document(load_roster(project))
    default = resolve_voice_entry(doc, "GHOST", "chatterbox", project_dir=project)
    assert default is not None
    assert (default.mode, default.voice, default.clone) == ("default", "default", None)
    assert resolve_voice_entry(doc, "GHOST", "qwen3-1.7b-clone", project_dir=project) is None


# ---------------------------------------------------------------------------
# Engine dispatch (fake models, no checkpoints)
# ---------------------------------------------------------------------------


class _Result:
    def __init__(self, audio: np.ndarray) -> None:
        self.audio = audio


def _tone(seconds: float = 1.0, sample_rate: int = 24000) -> np.ndarray:
    return np.full(int(seconds * sample_rate), 0.01, dtype=np.float32)


class FakeQwenBase:
    sample_rate = 24000

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        yield _Result(_tone(max(0.5, len(kwargs["text"]) * 0.06)))


class FakeChatterbox:
    sample_rate = 24000

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.prepared: list[tuple] = []

    def prepare_conditionals(self, ref_wav, ref_sr, exaggeration=0.5):
        self.prepared.append((ref_sr, exaggeration))
        return object()

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        yield _Result(_tone(max(0.5, len(kwargs["text"]) * 0.06)))


class FakeTurbo:
    sample_rate = 24000

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.prepared = 0
        self._conds = None

    def prepare_conditionals(self, ref_audio, sample_rate=None, exaggeration=0.5):
        self.prepared += 1
        self._conds = ("conds", self.prepared)

    def generate(self, **kwargs):
        assert self._conds is not None
        self.calls.append({**kwargs, "conds_used": self._conds})
        yield _Result(_tone(max(0.5, len(kwargs["text"]) * 0.06)))


def make_clone(name: str = "vox:TESTY", *, ref_text: str | None = REF_TEXT) -> CloneVoice:
    return CloneVoice(
        name=name, audio=_tone(6.0), sample_rate=24000, ref_text=ref_text
    )


def test_clone_engine_keys_are_the_expected_four():
    assert set(CLONE_ENGINE_KEYS) == {
        "qwen3-1.7b-clone",
        "qwen3-0.6b-clone",
        "chatterbox",
        "chatterbox-turbo",
    }


def test_qwen3_clone_passes_ref_audio_and_text_and_no_preset_voice():
    model = FakeQwenBase()
    engine = Engine(spec("qwen3-1.7b-clone"), model)
    result = engine.generate_line(
        LineRequest(text="Say the line.", clone=make_clone(), seed=7)
    )
    assert result.voice == "vox:TESTY"
    call = model.calls[0]
    assert call["ref_text"] == REF_TEXT
    assert call["ref_audio"] is not None
    assert "voice" not in call and "instruct" not in call
    # The recorded penalty is the mlx-audio ICL floor, not the Swift 1.3.
    assert result.sampling["repetition_penalty"] == pytest.approx(1.5)


def test_qwen3_clone_requires_a_clone_with_ref_text():
    engine = Engine(spec("qwen3-1.7b-clone"), FakeQwenBase())
    with pytest.raises(EngineError, match="requires a CloneVoice"):
        engine.generate_line(LineRequest(text="Hello."))
    with pytest.raises(EngineError, match="ref_text"):
        engine.generate_line(
            LineRequest(text="Hello.", clone=make_clone(ref_text=None))
        )


def test_non_clone_engine_rejects_a_clone_request():
    engine = Engine(spec("soprano"), FakeQwenBase())
    with pytest.raises(EngineError, match="cannot clone"):
        engine.generate_line(LineRequest(text="Hello.", clone=make_clone()))


def test_chatterbox_clone_prepares_conditionals_once_per_voice():
    model = FakeChatterbox()
    engine = Engine(spec("chatterbox"), model)
    a, b = make_clone("vox:A"), make_clone("vox:B")

    engine.generate_line(LineRequest(text="Line one.", clone=a))
    engine.generate_line(LineRequest(text="Line two.", clone=a))
    engine.generate_line(LineRequest(text="Line three.", clone=b))
    engine.generate_line(LineRequest(text="Line four.", clone=a))

    assert len(model.prepared) == 2  # one encode per distinct voice
    assert all(call["conds"] is not None for call in model.calls)
    # Without a clone the engine keeps the round-1 builtin-voice behaviour.
    engine.generate_line(LineRequest(text="Line five."))
    assert model.calls[-1]["conds"] is None


def test_turbo_clone_restores_cached_conditionals_between_voices():
    model = FakeTurbo()
    engine = Engine(spec("chatterbox-turbo"), model)
    a, b = make_clone("vox:A"), make_clone("vox:B")

    engine.generate_line(LineRequest(text="One.", clone=a))
    engine.generate_line(LineRequest(text="Two.", clone=b))
    engine.generate_line(LineRequest(text="Three.", clone=a))

    assert model.prepared == 2
    assert model.calls[0]["conds_used"] == ("conds", 1)
    assert model.calls[1]["conds_used"] == ("conds", 2)
    assert model.calls[2]["conds_used"] == ("conds", 1)  # cache hit, not re-prepared


def test_qwen3_clone_rejects_a_sample_rate_mismatch():
    engine = Engine(spec("qwen3-1.7b-clone"), FakeQwenBase())
    clone = CloneVoice(
        name="vox:BAD", audio=_tone(6.0), sample_rate=44100, ref_text=REF_TEXT
    )
    with pytest.raises(EngineError, match="44100"):
        engine.generate_line(LineRequest(text="Hello.", clone=clone))


# ---------------------------------------------------------------------------
# Episode planning with a schema-2 presets file
# ---------------------------------------------------------------------------


EPISODE = """INT. TEST KITCHEN - DAY

TESTY
Someone has to taste this.

TESTY
And it will not be me.
"""


def test_build_plan_resolves_clones_from_a_cloned_presets_file(project, tmp_path):
    doc = build_cloned_document(load_roster(project))
    presets_path = tmp_path / "presets-cloned.yaml"
    presets_path.write_text(dump_yaml(doc), encoding="utf-8")
    episode = tmp_path / "episode.fountain"
    episode.write_text(EPISODE, encoding="utf-8")

    plan = build_plan(
        episode,
        "qwen3-1.7b-clone",
        presets_path=presets_path,
        cast_path=project / "CAST.md",
    )
    assert plan.voices_mode == "cloned"
    assert plan.voice_by_character == {"TESTY": "vox:TESTY"}
    assert plan.clone_by_character["TESTY"].ref_text == REF_TEXT
    assert all(line.clone is not None for line in plan.lines)
    assert plan.voice_provenance["TESTY"]["ref_seconds"] > 5.0


def test_build_plan_fails_loudly_for_an_unresolved_clone_character(project, tmp_path):
    doc = build_cloned_document(load_roster(project))
    presets_path = tmp_path / "presets-cloned.yaml"
    presets_path.write_text(dump_yaml(doc), encoding="utf-8")
    episode = tmp_path / "episode.fountain"
    episode.write_text(EPISODE.replace("TESTY", "GHOST"), encoding="utf-8")

    with pytest.raises(GenerationError, match="GHOST"):
        build_plan(
            episode,
            "qwen3-1.7b-clone",
            presets_path=presets_path,
            cast_path=project / "CAST.md",
        )


def test_round_one_presets_still_plan_identically(tmp_path):
    """Schema-1 resolution is untouched: a preset engine plans as in round 1."""
    round1 = {
        "schema_version": 1,
        "assignments": {"TESTY": {"engines": {"qwen3-1.7b": "ryan"}}},
    }
    presets_path = tmp_path / "presets.yaml"
    presets_path.write_text(yaml.safe_dump(round1), encoding="utf-8")
    episode = tmp_path / "episode.fountain"
    episode.write_text(EPISODE, encoding="utf-8")

    plan = build_plan(episode, "qwen3-1.7b", presets_path=presets_path, use_cast=False)
    assert plan.voices_mode == "defaults"
    assert plan.voice_by_character == {"TESTY": "ryan"}
    assert plan.clone_by_character == {}
    assert all(line.clone is None for line in plan.lines)


# ---------------------------------------------------------------------------
# Corpus-pinned reality checks
# ---------------------------------------------------------------------------


@requires_granville_vox
def test_frozen_corpus_cast_is_fully_cloneable_on_the_headline_engine():
    doc = build_cloned_document(load_roster(CORPUS_ROOT))
    assignments = doc["assignments"]
    assert len(assignments) >= 25
    for character, record in assignments.items():
        entry = record["engines"]["qwen3-1.7b-clone"]
        assert entry is not None, f"{character} unresolved on qwen3-1.7b-clone"
        assert entry["mode"] == "clone"
        assert entry["ref_text"].strip()
        # KEVIN's production reference is 4.96s; everything else is 5s+.
        assert entry["ref_seconds"] > 4.5
    assert unresolved_characters(doc) == {}
    # The one short reference is flagged for the turbo engine, not hidden.
    kevin_notes = " ".join(assignments["KEVIN"].get("notes") or [])
    assert "under the documented >5s guidance" in kevin_notes
    # narrator.vox is an older export with only a 1.7b entry; the 0.6b clone
    # engine substitutes the 1.7b reference and says so.
    narrator = assignments["NARRATOR"]["engines"]["qwen3-0.6b-clone"]
    assert narrator["model_size"] == "1.7b"
    assert narrator["size_substituted"] is True
