"""The Fountain element stream data model (FR-1).

The element stream is the foundational data structure for every downstream
sortie: cue normalization, text preparation, generation, and assembly all
consume it. It is a *flat, ordered* list of typed elements, each carrying a
reference back to the source lines it came from.

Ordering is document order. Scene headings appear inline as elements rather
than as a nesting level, so that "the Nth thing in the screenplay" is a single
index into :attr:`ElementStream.elements`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ElementType(StrEnum):
    """Canonical element kinds emitted by the parser.

    The first seven are the types FR-1 requires; the remainder are Fountain
    constructs jouvence recognises that we pass through rather than discard.
    """

    SCENE_HEADING = "scene_heading"
    ACTION = "action"
    CHARACTER = "character"
    PARENTHETICAL = "parenthetical"
    DIALOGUE = "dialogue"
    NOTE = "note"
    CENTERED = "centered"
    TRANSITION = "transition"
    LYRICS = "lyrics"
    SECTION = "section"
    SYNOPSIS = "synopsis"
    PAGE_BREAK = "page_break"


#: Element types that carry a speaking character.
DIALOGUE_TYPES = frozenset({ElementType.DIALOGUE})


@dataclass(slots=True)
class InlineNote:
    """A ``[[note]]`` embedded inside another element's text.

    The host element keeps its text *verbatim* (the note is not stripped);
    removing notes from speech text is Sortie 3's job, not the parser's.
    ``start``/``end`` are character offsets into the host element's ``text``.
    """

    text: str
    start: int
    end: int
    line: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(slots=True)
class Element:
    """One element of the ordered stream."""

    index: int
    type: ElementType
    text: str
    #: 1-based inclusive source line range in the original ``.fountain`` file.
    start_line: int
    end_line: int
    #: Index of the enclosing scene (0 for anything before the first heading).
    scene_index: int
    #: Raw character cue this element belongs to (parentheticals + dialogue).
    character: str | None = None
    #: True when the cue was marked for dual dialogue with a trailing ``^``.
    dual: bool = False
    #: True when the dialogue body is Fountain lyrics (``~``-prefixed lines).
    lyric: bool = False
    #: ``[[notes]]`` found inside ``text`` (standalone notes are their own
    #: element and never appear here).
    notes: list[InlineNote] = field(default_factory=list)
    #: False when the source line range had to be inferred rather than matched.
    line_ref_exact: bool = True

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "index": self.index,
            "type": str(self.type),
            "text": self.text,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "scene_index": self.scene_index,
        }
        if self.character is not None:
            d["character"] = self.character
        if self.dual:
            d["dual"] = True
        if self.lyric:
            d["lyric"] = True
        if self.notes:
            d["notes"] = [n.to_dict() for n in self.notes]
        if not self.line_ref_exact:
            d["line_ref_exact"] = False
        return d


@dataclass(slots=True)
class ElementStream:
    """A parsed screenplay: title page values plus the ordered element list."""

    source: str
    parser: str
    parser_version: str
    title: dict[str, str]
    elements: list[Element]
    scene_headings: list[str] = field(default_factory=list)

    def of_type(self, *types: ElementType) -> list[Element]:
        """All elements matching any of ``types``, in document order."""
        wanted = set(types)
        return [e for e in self.elements if e.type in wanted]

    @property
    def dialogue(self) -> list[Element]:
        """Every dialogue element, in document order."""
        return self.of_type(ElementType.DIALOGUE)

    def counts(self) -> dict[str, int]:
        """Element count per type, in :class:`ElementType` declaration order."""
        out = {str(t): 0 for t in ElementType}
        for e in self.elements:
            out[str(e.type)] += 1
        return {k: v for k, v in out.items() if v}

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "parser": self.parser,
            "parser_version": self.parser_version,
            "title": self.title,
            "scene_headings": self.scene_headings,
            "element_count": len(self.elements),
            "counts": self.counts(),
            "elements": [e.to_dict() for e in self.elements],
        }
