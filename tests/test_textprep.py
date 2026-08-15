"""Sortie 3 — cue normalisation (FR-2), speech classification (FR-3), text prep (FR-4).

The strongest test in this file is :func:`test_produciesta_parity_reproduces_the_shipped_transcript`:
the granville tree ships the `.vtt` Produciesta emitted for the cold open, and
each cue in it is the exact string the Swift stack handed to the TTS backend.
Under the ``produciesta-parity`` speech policy our prepared text must equal it
line for line — that is what "matched to Produciesta's semantics" (FR-4) means
in practice.
"""

from __future__ import annotations

import html
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from comparativa.parsing import textprep
from comparativa.parsing.command import configure, handle
from comparativa.parsing.cues import (
    NARRATOR,
    CastIndex,
    find_cast_file,
    load_cast_index,
    normalize_cue,
    strip_cue_extensions,
)
from comparativa.parsing.elements import ElementType
from comparativa.parsing.fountain import parse_file, parse_text
from comparativa.parsing.speech import (
    FR3_POLICY,
    PRODUCIESTA_POLICY,
    prepare_script,
)
from comparativa.parsing.textprep import (
    DIVERGENCE_IDS,
    STRICT_PARITY,
    expand_slugline,
    parenthetical_direction,
    prepare,
    shot_prompt,
    split_at_sentences,
    split_sentences,
)

GRANVILLE = Path("~/Projects/podcasts/granville").expanduser()
EPISODES = GRANVILLE / "episodes"
COLD_OPEN = EPISODES / "episode_1_01_cold_open.fountain"
BUMPER = EPISODES / "episode_1_01a_bumper_donnie_and_arnie_1.fountain"
CORPUS = (COLD_OPEN, BUMPER)
CAST_MD = GRANVILLE / "CAST.md"
DOC = Path(__file__).resolve().parents[1] / "docs" / "TEXT_PREP.md"

requires_corpus = pytest.mark.skipif(
    not COLD_OPEN.is_file() or not CAST_MD.is_file(),
    reason="granville corpus not present",
)


# ---------------------------------------------------------------------------
# FR-2 — cue normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("RAY", "RAY"),
        ("RAY (CONT'D)", "RAY"),
        ("RAY (CONT’D)", "RAY"),
        ("JOANN (V.O.)", "JOANN"),
        ("HUNTER (O.S.)", "HUNTER"),
        ("RAY (V.O.) (CONT'D)", "RAY"),
        ("  toby  ", "TOBY"),
        ("@McKenzie", "MCKENZIE"),
        ("JOANN^", "JOANN"),
        ("PREDATOR MOM 1", "PREDATOR MOM 1"),
    ],
)
def test_normalize_cue_strips_extensions_and_uppercases(
    raw: str, expected: str
) -> None:
    assert normalize_cue(raw) == expected


def test_strip_cue_extensions_reports_what_it_removed() -> None:
    name, extensions = strip_cue_extensions("RAY (V.O.) (CONT'D)")
    assert name == "RAY"
    assert extensions == ("(V.O.)", "(CONT'D)")


def test_narrator_resolves_even_without_a_roster_entry() -> None:
    index = CastIndex.from_names(["RAY", "HUNTER"])
    assert index.resolve("NARRATOR") == NARRATOR
    assert index.resolve("RAY (CONT'D)") == "RAY"
    assert index.resolve("NOBODY") is None


@requires_corpus
def test_find_cast_file_walks_up_from_the_episode() -> None:
    assert find_cast_file(COLD_OPEN) == CAST_MD.resolve()


# ---------------------------------------------------------------------------
# FR-2 — every corpus cue resolves (the sortie's stated acceptance test)
# ---------------------------------------------------------------------------


@requires_corpus
@pytest.mark.parametrize("episode", CORPUS, ids=lambda p: p.stem)
def test_every_corpus_cue_resolves_to_a_cast_member(episode: Path) -> None:
    cast = load_cast_index(CAST_MD)
    script = prepare_script(parse_file(episode), cast)
    assert script.unresolved_cues == []


@requires_corpus
@pytest.mark.parametrize("episode", CORPUS, ids=lambda p: p.stem)
def test_every_spoken_line_names_a_roster_character(episode: Path) -> None:
    cast = load_cast_index(CAST_MD)
    roster = set(cast.names) | {NARRATOR}
    script = prepare_script(parse_file(episode), cast)
    assert script.lines
    assert {line.character for line in script.lines} <= roster


@requires_corpus
def test_cold_open_speaking_characters() -> None:
    script = prepare_script(parse_file(COLD_OPEN), load_cast_index(CAST_MD))
    assert set(script.characters) == {
        "NARRATOR",
        "RAY",
        "HUNTER",
        "JOANN",
        "KEVIN",
        "TOBY",
        "GARETH",
    }


@requires_corpus
def test_bumper_speaking_characters() -> None:
    script = prepare_script(parse_file(BUMPER), load_cast_index(CAST_MD))
    assert set(script.characters) == {"NARRATOR", "KEVIN"}


@requires_corpus
def test_contd_cue_keeps_its_raw_form_but_resolves_to_the_character() -> None:
    script = prepare_script(parse_file(COLD_OPEN), load_cast_index(CAST_MD))
    contd = [line for line in script.lines if line.raw_cue == "RAY (CONT'D)"]
    assert contd, "the cold open has a RAY (CONT'D) cue"
    assert all(line.character == "RAY" for line in contd)


# ---------------------------------------------------------------------------
# FR-3 — speech classification
# ---------------------------------------------------------------------------


SNIPPET = """\
INT. KITCHEN - DAY

Ray slams the door.

[[ NOTE: a production note ]]

RAY
(angrily, to Hunter)
Get out.

> THE END <
"""


def test_fr3_marks_only_dialogue_as_spoken() -> None:
    script = prepare_script(parse_text(SNIPPET), CastIndex.from_names(["RAY"]))
    assert [(line.character, line.text) for line in script.lines] == [
        ("RAY", "Get out.")
    ]


def test_fr3_preserves_non_spoken_elements_in_the_stream() -> None:
    stream = parse_text(SNIPPET)
    script = prepare_script(stream, CastIndex.from_names(["RAY"]))
    kinds = {e.type for e in stream.elements if e.index not in script.spoken_elements}
    assert ElementType.ACTION in kinds
    assert ElementType.SCENE_HEADING in kinds
    assert ElementType.CENTERED in kinds
    # ...and nothing was dropped from the stream itself.
    assert len(stream.elements) > len(script.lines)


@requires_corpus
def test_notes_are_never_spoken_under_either_policy() -> None:
    stream = parse_file(COLD_OPEN)
    notes = [e for e in stream.elements if e.type is ElementType.NOTE]
    assert notes, "the cold open has standalone [[notes]]"
    cast = load_cast_index(CAST_MD)
    for policy in (FR3_POLICY, PRODUCIESTA_POLICY):
        spoken = prepare_script(stream, cast, policy=policy).spoken_elements
        assert not [n for n in notes if n.index in spoken]


@requires_corpus
def test_inline_notes_are_stripped_from_narrated_action() -> None:
    script = prepare_script(
        parse_file(COLD_OPEN), load_cast_index(CAST_MD), policy=PRODUCIESTA_POLICY
    )
    assert all("[[" not in line.text for line in script.lines)
    assert all("AUDIO DRAMA" not in line.text for line in script.lines)


def test_produciesta_policy_narrates_action_and_headings() -> None:
    script = prepare_script(
        parse_text(SNIPPET),
        CastIndex.from_names(["RAY"]),
        policy=PRODUCIESTA_POLICY,
    )
    texts = [(line.character, line.text) for line in script.lines]
    assert ("NARRATOR", "INTERIOR. KITCHEN. DAY") in texts
    assert ("NARRATOR", "Ray slams the door.") in texts
    assert ("RAY", "Get out.") in texts
    # The `[[note]]` stays silent under both policies.
    assert all("production note" not in t for _, t in texts)


def test_shot_prompt_panels_are_silent_under_fr3_and_voiced_under_parity() -> None:
    source = 'RAY\nHello.\n\n[[<shot prompt="a wide of the porch"/>]]\n'
    cast = CastIndex.from_names(["RAY"])
    assert [l.text for l in prepare_script(parse_text(source), cast).lines] == [
        "Hello."
    ]
    parity = prepare_script(parse_text(source), cast, policy=PRODUCIESTA_POLICY)
    assert "SHOT PROMPT: a wide of the porch" in [l.text for l in parity.lines]


def test_shot_prompt_with_an_empty_prompt_stays_silent() -> None:
    assert shot_prompt('<shot prompt=""/>') is None
    assert shot_prompt("<pause seconds=\"1\"/>") is None


# ---------------------------------------------------------------------------
# FR-4 — individual transforms, each matched to a Swift source
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("INT. HOUSE - DAY", "INTERIOR. HOUSE. DAY"),
        ("EXT. PARKING LOT - NIGHT", "EXTERIOR. PARKING LOT. NIGHT"),
        ("EST. THE CITY - DAWN", "ESTABLISHING SHOT. THE CITY. DAWN"),
        ("INT./EXT. CAR - CONTINUOUS", "INTERIOR AND EXTERIOR. CAR. CONTINUOUS"),
        ("I/E. CAR - DAY", "INTERIOR AND EXTERIOR. CAR. DAY"),
        ("EXT. RHODES HOUSE — GRAVEL DRIVE — MAGIC HOUR",
         "EXTERIOR. RHODES HOUSE. GRAVEL DRIVE. MAGIC HOUR"),
        # Intra-word hyphens survive; only spaced separators convert.
        ("INT. WAL-MART - DAY", "INTERIOR. WAL-MART. DAY"),
        # Markers only count on a word boundary.
        ("INTERCUT: THE CHASE", "INTERCUT: THE CHASE"),
        ("INT. HOUSE. - DAY", "INTERIOR. HOUSE. DAY"),
        ("INT. HOUSE #12A#", "INTERIOR. HOUSE"),
        ("OVER BLACK", "OVER BLACK"),
        ("", ""),
    ],
)
def test_expand_slugline_matches_sluglinespeech(heading: str, expected: str) -> None:
    assert expand_slugline(heading) == expected


def test_parenthetical_direction_keeps_delivery_and_drops_blocking() -> None:
    assert parenthetical_direction(["(sotto, to Abbey)"]) == "sotto"
    assert parenthetical_direction(["(angrily)", "(beat)"]) == "angrily, beat"
    assert parenthetical_direction(["(looking out the window)"]) is None
    assert parenthetical_direction([]) is None


def test_stage_directions_are_removed() -> None:
    assert prepare("Hello {{wave}} world").text == "Hello world"
    assert prepare("Hello {{multi\nline}}world").text == "Hello world"


def test_inline_notes_are_removed_from_speech() -> None:
    assert prepare("Say [[an aside]] this.").text == "Say this."


def test_breath_markers_split_the_line_and_never_reach_the_model() -> None:
    long_a = "A" * 60
    long_b = "B" * 60
    prepared = prepare(f"{long_a} [[<breath/>]] {long_b}")
    assert prepared.segments == (long_a, long_b)
    assert prepared.text == f"{long_a} {long_b}"
    assert prepared.breath_markers == 1
    assert prepared.breath_gap_seconds == textprep.BREATH_GAP_SECONDS
    assert "breath" not in prepared.text


def test_runt_breath_spans_merge_into_a_neighbour() -> None:
    prepared = prepare("Well. [[<breath/>]] " + "C" * 60)
    assert len(prepared.segments) == 1
    assert prepared.segments[0].startswith("Well.")


def test_lyric_tildes_are_stripped_but_strict_parity_keeps_them() -> None:
    source = "~ Tell me\n~ if you need my love"
    assert prepare(source).text == "Tell me if you need my love"
    assert "~" in prepare(source, policy=STRICT_PARITY).text


def test_emphasis_is_preserved_by_default_and_removable_by_policy() -> None:
    source = "I was **HERE**. *I whispered*."
    assert prepare(source).text == source
    stripped = prepare(source, policy=textprep.DEFAULT_POLICY.evolve(
        strip_emphasis=True
    ))
    assert stripped.text == "I was HERE. I whispered."


def test_em_dashes_ellipses_and_all_caps_are_left_verbatim() -> None:
    source = "It's WHERE— ...on the corner… of Ball and Sack"
    assert prepare(source).text == source


def test_whitespace_is_collapsed_unless_strict_parity() -> None:
    assert prepare("a  b\n c").text == "a b c"
    assert prepare("a  b\n c", policy=STRICT_PARITY).text == "a  b\n c"


# ---------------------------------------------------------------------------
# FR-4 — chunking parity (SwiftVoxAlta VoiceLockManager)
# ---------------------------------------------------------------------------


def test_estimate_duration_uses_the_swift_rate() -> None:
    assert textprep.estimate_duration("x" * 421) == pytest.approx(23.155, abs=0.01)


def test_split_sentences_does_not_break_on_abbreviations_or_decimals() -> None:
    assert split_sentences("Dr. Smith paid 3.14 dollars. Then he left.") == [
        "Dr. Smith paid 3.14 dollars.",
        "Then he left.",
    ]


def test_split_at_sentences_packs_to_the_duration_budget() -> None:
    sentence = "This is a fairly long sentence used for chunk packing tests. "
    chunks = split_at_sentences(sentence * 12, max_duration=12.0)
    assert len(chunks) > 1
    assert all(len(c) >= textprep.MINIMUM_CHUNK_CHARS for c in chunks)
    assert "".join(c.replace(" ", "") for c in chunks) == (
        (sentence * 12).replace(" ", "")
    )


def test_split_at_sentences_returns_nothing_for_blank_input() -> None:
    assert split_at_sentences("   ") == []


# ---------------------------------------------------------------------------
# The parity proof: our prepared text == the strings Produciesta actually spoke
# ---------------------------------------------------------------------------


def _vtt_cues(path: Path) -> list[tuple[str, str]]:
    """(speaker, text) for every cue in a WebVTT transcript, in order."""
    body = path.read_text(encoding="utf-8")
    return [
        (m.group(1), html.unescape(m.group(2)))
        for m in re.finditer(r"<v ([^>]+)>(.*)</v>", body)
    ]


@requires_corpus
@pytest.mark.skipif(
    not (GRANVILLE / "audio" / "episode_1_01_cold_open.vtt").is_file(),
    reason="shipped Produciesta transcript not present",
)
def test_produciesta_parity_reproduces_the_shipped_transcript() -> None:
    reference = _vtt_cues(GRANVILLE / "audio" / "episode_1_01_cold_open.vtt")
    script = prepare_script(
        parse_file(COLD_OPEN),
        load_cast_index(CAST_MD),
        policy=PRODUCIESTA_POLICY,
    )
    produced = [(line.character, line.text) for line in script.lines]
    assert produced == reference


@requires_corpus
def test_fr3_is_a_strict_subset_of_the_parity_policy() -> None:
    cast = load_cast_index(CAST_MD)
    stream = parse_file(COLD_OPEN)
    fr3 = prepare_script(stream, cast, policy=FR3_POLICY)
    parity = prepare_script(stream, cast, policy=PRODUCIESTA_POLICY)
    assert len(fr3.lines) < len(parity.lines)
    parity_texts = {(l.character, l.text) for l in parity.lines}
    assert {(l.character, l.text) for l in fr3.lines} <= parity_texts


# ---------------------------------------------------------------------------
# CLI surface: unresolved_cues and the spoken flag
# ---------------------------------------------------------------------------


def _parse_json(tmp_path: Path, episode: Path, *extra: str) -> dict:
    out = tmp_path / "stream.json"
    argv = [str(episode), "-o", str(out), *extra]
    import argparse

    parser = argparse.ArgumentParser()
    configure(parser)
    args = parser.parse_args(argv)
    args.parser = parser
    assert handle(args) == 0
    return json.loads(out.read_text(encoding="utf-8"))


@requires_corpus
@pytest.mark.parametrize("episode", CORPUS, ids=lambda p: p.stem)
def test_parse_json_reports_an_empty_unresolved_cue_list(
    tmp_path: Path, episode: Path
) -> None:
    payload = _parse_json(tmp_path, episode)
    assert payload["unresolved_cues"] == []
    assert payload["speech"]["cast_source"] == str(CAST_MD.resolve())
    assert payload["speech"]["line_count"] > 0


@requires_corpus
def test_parse_json_flags_every_element_spoken_or_not(tmp_path: Path) -> None:
    payload = _parse_json(tmp_path, COLD_OPEN)
    assert all("spoken" in e for e in payload["elements"])
    spoken = [e for e in payload["elements"] if e["spoken"]]
    assert len(spoken) == payload["speech"]["line_count"]
    assert all(e["type"] == "dialogue" for e in spoken)


@requires_corpus
def test_parse_cli_exits_zero_and_emits_unresolved_cues() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "comparativa", "parse", str(COLD_OPEN)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["unresolved_cues"] == []


def test_strict_cues_fails_on_an_unknown_cue(tmp_path: Path) -> None:
    import argparse

    cast = tmp_path / "CAST.md"
    cast.write_text(
        "---\ncast:\n- character: RAY\n  voicePrompt: x\n---\n", encoding="utf-8"
    )
    episode = tmp_path / "ep.fountain"
    episode.write_text("RAY\nHi.\n\nSTRANGER\nHello.\n", encoding="utf-8")

    parser = argparse.ArgumentParser()
    configure(parser)
    args = parser.parse_args(
        [str(episode), "-o", str(tmp_path / "o.json"), "--strict-cues"]
    )
    args.parser = parser
    assert handle(args) == 1


def test_no_cast_skips_resolution(tmp_path: Path) -> None:
    import argparse

    episode = tmp_path / "ep.fountain"
    episode.write_text("STRANGER\nHello.\n", encoding="utf-8")
    parser = argparse.ArgumentParser()
    configure(parser)
    out = tmp_path / "o.json"
    args = parser.parse_args([str(episode), "-o", str(out), "--no-cast"])
    args.parser = parser
    assert handle(args) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["speech"]["cast_source"] is None
    assert payload["unresolved_cues"] == []
    assert payload["speech"]["lines"][0]["character"] == "STRANGER"


# ---------------------------------------------------------------------------
# Documentation is part of the deliverable
# ---------------------------------------------------------------------------


def test_text_prep_doc_exists_with_a_divergence_table() -> None:
    assert DOC.is_file(), "docs/TEXT_PREP.md is a Sortie 3 deliverable"
    body = DOC.read_text(encoding="utf-8")
    assert "## 4. Divergences" in body
    assert "| ID | Topic |" in body


def test_every_registered_divergence_appears_in_the_doc() -> None:
    body = DOC.read_text(encoding="utf-8")
    for identifier in DIVERGENCE_IDS:
        assert f"| {identifier} |" in body, f"{identifier} missing from TEXT_PREP.md"


@requires_corpus
def test_nothing_is_written_into_the_corpus_tree(tmp_path: Path) -> None:
    before = {p: p.stat().st_mtime_ns for p in GRANVILLE.rglob("*") if p.is_file()}
    _parse_json(tmp_path, COLD_OPEN)
    _parse_json(tmp_path, BUMPER)
    after = {p: p.stat().st_mtime_ns for p in GRANVILLE.rglob("*") if p.is_file()}
    assert before == after
