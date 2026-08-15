"""Sortie 2 tests: the Fountain element stream and the ``parse`` command.

Two tiers:

* **Unit tests** on inline Fountain snippets — always run, no external data.
* **Corpus tests** against the two granville episodes named in the execution
  plan. Those live outside the repo, so they skip when the corpus is absent;
  a final test asserts parsing never writes anything into that tree.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from comparativa.cli import main
from comparativa.parsing import ElementType, parse_file, parse_text
from comparativa.parsing.fountain import PARSER_NAME

CORPUS = Path("~/Projects/podcasts/granville/episodes").expanduser()
COLD_OPEN = CORPUS / "episode_1_01_cold_open.fountain"
BUMPER = CORPUS / "episode_1_01a_bumper_donnie_and_arnie_1.fountain"

requires_corpus = pytest.mark.skipif(
    not (COLD_OPEN.is_file() and BUMPER.is_file()),
    reason=f"granville corpus not present at {CORPUS}",
)

# Known anchors, read off the corpus files by hand.
COLD_OPEN_FIRST_DIALOGUE = (
    "Our story today opens on a dirty SUV. It's pulling a u-haul trailer down "
    'a two-lane country road. They pass a water tower that reads "GRANVILLE". '
    "Grain silos punctuate the otherwise flat land. Then they turn down a "
    "tree-lined gravel driveway."
)
COLD_OPEN_LAST_DIALOGUE = (
    "The windows go dark at the Rhodes place, one by one, the way they always "
    "have. The porch light stays on. Just in case."
)
BUMPER_FIRST_DIALOGUE_HEAD = "Somewhere west of Granville — call it Wichita —"
BUMPER_LAST_DIALOGUE_HEAD = "The two otters put their drawings side by side."


# ---------------------------------------------------------------------------
# Unit tests on inline snippets
# ---------------------------------------------------------------------------

SNIPPET = """\
[[ A standalone production note. ]]

>Episode 1<

EXT. PORCH - NIGHT

Two men lean on the rail.

RAY
(gruff)
Take the bypass. [[<breath/>]] Not the county road.

HUNTER^
Sure, Dad.
"""


def test_snippet_element_types_and_order() -> None:
    s = parse_text(SNIPPET, source="snippet")
    assert s.parser == PARSER_NAME
    types = [str(e.type) for e in s.elements]
    assert types == [
        "note",
        "centered",
        "scene_heading",
        "action",
        "character",
        "parenthetical",
        "dialogue",
        "character",
        "dialogue",
    ]
    assert [e.index for e in s.elements] == list(range(len(s.elements)))


def test_snippet_line_references_point_at_the_source() -> None:
    s = parse_text(SNIPPET, source="snippet")
    lines = SNIPPET.splitlines()
    for el in s.elements:
        assert el.line_ref_exact, f"{el.type} @ {el.index} lost its line ref"
        assert 1 <= el.start_line <= el.end_line <= len(lines)
    by_type = {str(e.type): e for e in s.elements}
    assert by_type["scene_heading"].start_line == 5
    assert lines[by_type["scene_heading"].start_line - 1] == "EXT. PORCH - NIGHT"
    assert by_type["parenthetical"].start_line == 10


def test_snippet_notes_are_split_standalone_from_inline() -> None:
    s = parse_text(SNIPPET, source="snippet")
    notes = s.of_type(ElementType.NOTE)
    assert [n.text for n in notes] == ["A standalone production note."]

    ray = next(e for e in s.dialogue if e.character == "RAY")
    # The host keeps its text verbatim; stripping notes is Sortie 3's job.
    assert "[[<breath/>]]" in ray.text
    assert [n.text for n in ray.notes] == ["<breath/>"]
    assert ray.text[ray.notes[0].start : ray.notes[0].end] == "[[<breath/>]]"


def test_snippet_dual_dialogue_cue_survives() -> None:
    """Regression: jouvence drops ``NAME^`` cues into action, losing the line."""
    s = parse_text(SNIPPET, source="snippet")
    hunter = [e for e in s.dialogue if e.character == "HUNTER"]
    assert [e.text for e in hunter] == ["Sure, Dad."]
    assert all(e.dual for e in hunter)
    assert not any("HUNTER^" in e.text for e in s.elements)


def test_dialogue_carries_its_character_across_a_mid_block_parenthetical() -> None:
    s = parse_text(
        "INT. KITCHEN - DAY\n\nTOBY\nShe's doing...\n(to his dad)\nWhat's that word?\n",
        source="snippet",
    )
    assert [e.character for e in s.dialogue] == ["TOBY", "TOBY"]
    assert [e.text for e in s.dialogue] == ["She's doing...", "What's that word?"]


def test_lyrics_lines_are_flagged() -> None:
    s = parse_text(
        "INT. CLUB - NIGHT\n\nKEVIN\n(singing)\n~ 1000 ways\n~ To fall in love\n",
        source="snippet",
    )
    lyric = [e for e in s.dialogue if e.lyric]
    assert len(lyric) == 1
    assert lyric[0].character == "KEVIN"


def test_stream_round_trips_through_json() -> None:
    s = parse_text(SNIPPET, source="snippet")
    payload = json.loads(json.dumps(s.to_dict()))
    assert payload["parser"] == PARSER_NAME
    assert payload["element_count"] == len(s.elements)
    assert payload["counts"]["dialogue"] == len(s.dialogue)
    assert [e["index"] for e in payload["elements"]] == list(range(len(s.elements)))


# ---------------------------------------------------------------------------
# Corpus tests
# ---------------------------------------------------------------------------


@requires_corpus
@pytest.mark.parametrize("episode", [COLD_OPEN, BUMPER], ids=["cold_open", "bumper"])
def test_corpus_episode_has_dialogue(episode: Path) -> None:
    s = parse_file(episode)
    assert s.parser == PARSER_NAME
    assert len(s.dialogue) > 0
    assert all(e.character for e in s.dialogue)


@requires_corpus
@pytest.mark.parametrize("episode", [COLD_OPEN, BUMPER], ids=["cold_open", "bumper"])
def test_corpus_element_ordering(episode: Path) -> None:
    """Indices are dense and source lines never run backwards."""
    s = parse_file(episode)
    assert [e.index for e in s.elements] == list(range(len(s.elements)))

    previous = 0
    for el in s.elements:
        assert el.start_line >= previous, f"element {el.index} went backwards"
        assert el.end_line >= el.start_line
        previous = el.start_line

    # Every parenthetical and dialogue line sits under a character cue.
    speech = (ElementType.CHARACTER, ElementType.PARENTHETICAL, ElementType.DIALOGUE)
    for i, el in enumerate(s.elements):
        if el.type in (ElementType.DIALOGUE, ElementType.PARENTHETICAL):
            assert i > 0 and s.elements[i - 1].type in speech


@requires_corpus
@pytest.mark.parametrize("episode", [COLD_OPEN, BUMPER], ids=["cold_open", "bumper"])
def test_corpus_line_references_are_exact(episode: Path) -> None:
    lines = episode.read_text(encoding="utf-8").splitlines()
    s = parse_file(episode)
    for el in s.elements:
        assert el.line_ref_exact, f"{el.type} @ {el.index}: {el.text[:40]!r}"
        assert 1 <= el.start_line <= el.end_line <= len(lines)
    # Single-line dialogue text must actually be on the line we point at.
    for el in s.dialogue:
        if el.start_line == el.end_line and not el.lyric:
            assert el.text == lines[el.start_line - 1].strip()


@requires_corpus
def test_cold_open_known_first_and_last_dialogue() -> None:
    s = parse_file(COLD_OPEN)
    dialogue = s.dialogue
    assert dialogue[0].character == "NARRATOR"
    assert dialogue[0].text == COLD_OPEN_FIRST_DIALOGUE
    assert dialogue[0].start_line == 11
    assert dialogue[-1].character == "NARRATOR"
    assert dialogue[-1].text == COLD_OPEN_LAST_DIALOGUE
    assert dialogue[-1].start_line == 670


@requires_corpus
def test_cold_open_structure() -> None:
    s = parse_file(COLD_OPEN)
    assert len(s.dialogue) == 189
    assert s.scene_headings[0] == "EST. SOMEWHERE IN THE MIDDLE OF THE COUNTRY - SUNSET"
    assert len(s.scene_headings) == 7
    assert [e.text for e in s.of_type(ElementType.CENTERED)] == [
        "Episode 1",
        '"On the Corner of Ball and Sack"',
    ]
    # Both bracketed production notes are promoted out of the action stream.
    notes = s.of_type(ElementType.NOTE)
    assert len(notes) == 2
    assert notes[0].start_line == 2
    assert notes[0].text.startswith("NOTE: AUDIO DRAMA")
    # The `[[<breath/>]]` markers stay attached to the dialogue that hosts them.
    hosts = [e for e in s.dialogue if e.notes]
    assert len(hosts) == 6
    assert all(n.text == "<breath/>" for e in hosts for n in e.notes)

    cues = {e.text for e in s.of_type(ElementType.CHARACTER)}
    assert {"NARRATOR", "RAY", "HUNTER", "JOANN", "KEVIN", "TOBY", "GARETH"} <= cues


@requires_corpus
def test_cold_open_dual_dialogue_line_is_not_lost() -> None:
    """``JOANN^`` on line 613 must yield a JOANN cue, not an action block."""
    s = parse_file(COLD_OPEN)
    dual = [e for e in s.elements if e.dual]
    assert [(str(e.type), e.text, e.start_line) for e in dual] == [
        ("character", "JOANN", 613),
        ("dialogue", "KEVIN!", 614),
    ]


@requires_corpus
def test_bumper_known_first_and_last_dialogue() -> None:
    s = parse_file(BUMPER)
    dialogue = s.dialogue
    assert len(dialogue) == 10
    assert dialogue[0].character == "NARRATOR"
    assert dialogue[0].text.startswith(BUMPER_FIRST_DIALOGUE_HEAD)
    assert dialogue[0].start_line == 11
    assert dialogue[-1].character == "NARRATOR"
    assert dialogue[-1].text.startswith(BUMPER_LAST_DIALOGUE_HEAD)
    assert dialogue[-1].start_line == 57


@requires_corpus
def test_bumper_title_page_and_sung_lines() -> None:
    s = parse_file(BUMPER)
    assert s.title == {
        "type": "bumper",
        "title": "The Adventures of Donnie & Arnie — Part One",
    }
    assert s.scene_headings == ["OVER BLACK", "FADE OUT"]
    lyric = [e for e in s.dialogue if e.lyric]
    assert len(lyric) == 2
    assert all(e.character == "KEVIN" for e in lyric)
    assert len(s.of_type(ElementType.NOTE)) == 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_parse_without_an_episode_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["parse"]) == 0
    assert "comparativa parse" in capsys.readouterr().out


def test_parse_reports_a_missing_file(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["parse", "/nonexistent/nope.fountain"]) == 1
    assert "comparativa parse:" in capsys.readouterr().err


def test_parse_emits_json_for_a_snippet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    episode = tmp_path / "snippet.fountain"
    episode.write_text(SNIPPET, encoding="utf-8")
    assert main(["parse", str(episode)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == str(episode)
    assert payload["counts"]["dialogue"] == 2


def test_parse_writes_to_an_output_file(tmp_path: Path) -> None:
    episode = tmp_path / "snippet.fountain"
    episode.write_text(SNIPPET, encoding="utf-8")
    out = tmp_path / "nested" / "stream.json"
    assert main(["parse", str(episode), "-o", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["element_count"] == len(payload["elements"])


@requires_corpus
def test_parse_command_on_the_cold_open(capsys: pytest.CaptureFixture[str]) -> None:
    """The sortie's headline exit criterion, exercised in-process."""
    assert main(["parse", str(COLD_OPEN)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["parser"] == PARSER_NAME
    dialogue = [e for e in payload["elements"] if e["type"] == "dialogue"]
    assert len(dialogue) > 0


# ---------------------------------------------------------------------------
# The corpus is read-only input
# ---------------------------------------------------------------------------


@requires_corpus
def test_parsing_never_writes_into_the_corpus_tree() -> None:
    root = CORPUS.parent

    def snapshot() -> dict[str, tuple[int, str]]:
        out: dict[str, tuple[int, str]] = {}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != ".git"]
            for name in filenames:
                p = Path(dirpath) / name
                try:
                    data = p.read_bytes()
                except OSError:  # pragma: no cover - symlinks, sockets
                    continue
                out[str(p)] = (len(data), hashlib.sha256(data).hexdigest())
        return out

    before = snapshot()
    parse_file(COLD_OPEN)
    parse_file(BUMPER)
    assert main(["parse", str(BUMPER), "-o", os.devnull]) == 0
    assert snapshot() == before
