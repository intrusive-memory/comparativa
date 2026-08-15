"""Sortie 4 tests: CAST.md roster, voice catalog, and presets.yaml coverage."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from comparativa.cli import main
from comparativa.voices import (
    ENGINE_KEYS,
    ENGINES,
    Roster,
    RosterError,
    assign,
    classify,
    dump_yaml,
    load_presets,
    load_roster,
    parse_cast_markdown,
    to_document,
    voice_for,
)
from comparativa.voices.command import DEFAULT_PRESETS_PATH

GRANVILLE = Path("~/Projects/podcasts/granville").expanduser()
REPO_ROOT = Path(__file__).resolve().parents[1]

requires_granville = pytest.mark.skipif(
    not (GRANVILLE / "CAST.md").is_file(),
    reason="granville corpus not available on this machine",
)


@pytest.fixture(scope="module")
def granville_roster() -> Roster:
    return load_roster(GRANVILLE)


@pytest.fixture(scope="module")
def presets() -> dict:
    return load_presets(DEFAULT_PRESETS_PATH)


# --- roster parsing ----------------------------------------------------------

SAMPLE_CAST = """---
type: cast
schemaVersion: 1
cast:
- character: ADA
  voicePrompt: Female, 30s. A warm alto.
  voices:
    voxalta:
    - voices/ADA.vox
- character: BOB
  voicePrompt: Male, 70s. An elderly bass.
---

# notes below the frontmatter are ignored
"""


def test_parse_cast_markdown_reads_frontmatter():
    members = parse_cast_markdown(SAMPLE_CAST)
    assert [m.character for m in members] == ["ADA", "BOB"]
    assert members[0].voxalta_paths == ("voices/ADA.vox",)
    assert members[1].voxalta_paths == ()


@pytest.mark.parametrize(
    "text",
    [
        "no frontmatter here",
        "---\ntype: cast\n",  # unterminated
        "---\ntype: cast\ncast: []\n---\n",  # empty cast
        "---\ntype: cast\ncast:\n- voicePrompt: nameless\n---\n",  # no character
    ],
)
def test_parse_cast_markdown_rejects_bad_input(text):
    with pytest.raises(RosterError):
        parse_cast_markdown(text)


def test_load_roster_missing_project(tmp_path):
    with pytest.raises(RosterError):
        load_roster(tmp_path / "nope")


# --- catalog -----------------------------------------------------------------


def test_every_engine_offers_at_least_one_assignable_voice():
    for engine in ENGINES:
        assert engine.assignable_voices, f"{engine.key} has no assignable voice"


def test_single_voice_engines_are_marked_and_have_exactly_one_voice():
    for key in ("chatterbox", "chatterbox-turbo", "soprano"):
        engine = next(e for e in ENGINES if e.key == key)
        assert engine.multi_voice is False
        assert len(engine.assignable_voices) == 1


def test_qwen3_preset_speakers_match_the_checkpoint_spk_id():
    engine = next(e for e in ENGINES if e.key == "qwen3-1.7b")
    assert [v.name for v in engine.voices] == [
        "serena",
        "vivian",
        "uncle_fu",
        "ryan",
        "aiden",
        "ono_anna",
        "sohee",
        "eric",
        "dylan",
    ]
    # Dialect speakers force a Chinese language id and must not be auto-assigned.
    assert {v.name for v in engine.voices if not v.assignable} == {"eric", "dylan"}


# --- classification ----------------------------------------------------------


@pytest.mark.parametrize(
    "character,prompt,gender,age_band",
    [
        ("PRINCIPAL", "Female, A gravelly, lilting baritone", "female", "adult"),
        ("TOBY", "Male child, 10 years old. A bright boyish treble.", "male", "child"),
        ("LEROY", "Male, 70s. The Rhodes boy threw up in his pool.", "male", "elder"),
        ("CYRUS", "Male, 30s to 40s. Archer's older brother.", "male", "adult"),
        ("PREDATOR MOM 3", "The voice scrapes like gravel underfoot.", "female", "adult"),
        ("MARSHALL", "Low, gravelly rasp, a tremor in the low notes.", "male", "adult"),
    ],
)
def test_classification_rules(character, prompt, gender, age_band):
    from comparativa.voices import CastMember

    result = classify(CastMember(character=character, voice_prompt=prompt))
    assert (result.gender, result.age_band) == (gender, age_band)


def test_assignment_is_deterministic(granville_roster):
    first = assign(granville_roster)
    second = assign(granville_roster)
    assert [c.voices for c in first.characters] == [c.voices for c in second.characters]


def test_two_town_chorus_elders_get_distinct_qwen3_voices(granville_roster):
    assignments = assign(granville_roster)
    by_name = {c.character: c.voices for c in assignments.characters}
    assert by_name["DUKE"]["qwen3-1.7b"] != by_name["LEROY"]["qwen3-1.7b"]


def test_child_character_is_not_given_the_elder_preset(granville_roster):
    assignments = assign(granville_roster)
    toby = next(c for c in assignments.characters if c.character == "TOBY")
    assert toby.voices["qwen3-1.7b"] != "uncle_fu"


# --- presets.yaml ------------------------------------------------------------


@requires_granville
def test_presets_file_is_committed_and_current(granville_roster, presets):
    """The committed presets.yaml matches a fresh regeneration from CAST.md."""
    assert DEFAULT_PRESETS_PATH.is_file()
    regenerated = dump_yaml(to_document(assign(granville_roster)))
    assert DEFAULT_PRESETS_PATH.read_text(encoding="utf-8") == regenerated


@requires_granville
def test_every_cast_member_has_every_engine_assigned(granville_roster, presets):
    """Sortie 4 exit criterion: full cast x engine matrix, no gaps."""
    assert set(presets["assignments"]) == set(granville_roster.character_names)
    assert len(presets["assignments"]) == len(granville_roster)

    for character in granville_roster.character_names:
        for engine_key in ENGINE_KEYS:
            voice = voice_for(presets, character, engine_key)
            assert voice, f"{character} has no voice for {engine_key}"
            engine = next(e for e in ENGINES if e.key == engine_key)
            option = engine.voice(voice)
            assert option is not None, f"{character}: {voice!r} is not a {engine_key} voice"
            assert option.assignable, f"{character}: {voice!r} is excluded from assignment"


@requires_granville
def test_presets_record_voxalta_paths_without_reading_them(presets):
    """RD-2: .vox paths are recorded for the future cloning mission only."""
    archer = presets["assignments"]["ARCHER"]
    assert archer["voxalta"] == ["voices/ARCHER.vox"]


@requires_granville
def test_presets_describe_every_engine(presets):
    assert set(presets["engines"]) == set(ENGINE_KEYS)
    for key, entry in presets["engines"].items():
        assert entry["checkpoint"]
        assert entry["voice_list_source"]
        assert entry["voices"]


# --- CLI ---------------------------------------------------------------------


@requires_granville
def test_voices_command_exits_zero(capsys):
    assert main(["voices", str(GRANVILLE)]) == 0
    out = capsys.readouterr().out
    assert "ARCHER" in out and "QWEN3-1.7B" in out


@requires_granville
def test_voices_command_yaml_format_round_trips(capsys):
    assert main(["voices", str(GRANVILLE), "--format", "yaml"]) == 0
    import yaml

    document = yaml.safe_load(capsys.readouterr().out)
    assert len(document["assignments"]) == 25


def test_voices_command_rejects_missing_project(tmp_path, capsys):
    assert main(["voices", str(tmp_path / "absent")]) == 2
    assert "error:" in capsys.readouterr().err


def test_voices_command_rejects_unknown_engine(capsys):
    assert main(["voices", str(GRANVILLE), "--engines", "nope"]) == 2
    assert "unknown engine" in capsys.readouterr().err


@requires_granville
def test_voices_command_writes_presets_to_a_given_path(tmp_path):
    target = tmp_path / "presets.yaml"
    assert main(["voices", str(GRANVILLE), "--write", str(target)]) == 0
    written = load_presets(target)
    assert len(written["assignments"]) == 25


@requires_granville
def test_voices_command_does_not_write_inside_the_corpus():
    """Exit criterion: no writes inside the granville tree."""
    before = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=GRANVILLE,
        capture_output=True,
        text=True,
    ).stdout
    assert main(["voices", str(GRANVILLE)]) == 0
    after = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=GRANVILLE,
        capture_output=True,
        text=True,
    ).stdout
    assert before == after


def test_cli_module_is_importable_without_mlx():
    """`voices` must not drag in a model runtime (fast CLI startup)."""
    code = (
        "import sys, comparativa.voices.command;"
        "assert 'mlx' not in sys.modules, sorted(m for m in sys.modules if 'mlx' in m)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
