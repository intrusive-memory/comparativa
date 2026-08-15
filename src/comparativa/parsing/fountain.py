"""jouvence-backed Fountain parser wrapper (FR-1).

**Active parser: jouvence 0.4.2.** It was validated against both granville
corpus episodes and classifies every construct they use correctly, so the
`fountain-tools` fallback contemplated by the execution plan was not needed
and is not installed.

jouvence is used only through its public API (:class:`jouvence.parser.
JouvenceParser` and the :mod:`jouvence.document` model). That model has three
gaps this wrapper closes, because downstream sorties depend on all three:

1. **No source-line references.** jouvence discards line numbers entirely.
   We re-attach them after the fact with a forward-only cursor that matches
   each element's first and last text line back against the source
   (:func:`_attach_line_refs`). Anything that fails to match is still emitted,
   flagged ``line_ref_exact=False``, so a parse never silently loses an
   element.
2. **No ``[[note]]`` support.** jouvence folds notes into action text. We
   promote note-only blocks to :attr:`~.elements.ElementType.NOTE` elements and
   record notes embedded in dialogue/action on the host element.
3. **Dual dialogue cues are dropped.** ``JOANN^`` does not match jouvence's
   character-cue regex, so the cue *and its dialogue* degrade into an action
   block — a lost spoken line. We strip the caret in a pre-pass and record
   which cues were dual.

The wrapper deliberately does no text preparation: element text is verbatim
(notes included). Cue normalisation and speech classification are Sortie 3.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import jouvence
from jouvence.document import (
    TYPE_ACTION,
    TYPE_CENTEREDACTION,
    TYPE_CHARACTER,
    TYPE_DIALOG,
    TYPE_LYRICS,
    TYPE_PAGEBREAK,
    TYPE_PARENTHETICAL,
    TYPE_SECTION,
    TYPE_SYNOPSIS,
    TYPE_TRANSITION,
    JouvenceDocument,
)
from jouvence.parser import JouvenceParser

from .elements import Element, ElementStream, ElementType, InlineNote

#: Name of the parser backend actually in use, recorded in the JSON output.
PARSER_NAME: Final = "jouvence"
PARSER_VERSION: Final = getattr(jouvence, "__version__", "unknown")

_TYPE_MAP: Final[dict[int, ElementType]] = {
    TYPE_ACTION: ElementType.ACTION,
    TYPE_CENTEREDACTION: ElementType.CENTERED,
    TYPE_CHARACTER: ElementType.CHARACTER,
    TYPE_DIALOG: ElementType.DIALOGUE,
    TYPE_PARENTHETICAL: ElementType.PARENTHETICAL,
    TYPE_TRANSITION: ElementType.TRANSITION,
    TYPE_LYRICS: ElementType.LYRICS,
    TYPE_PAGEBREAK: ElementType.PAGE_BREAK,
    TYPE_SECTION: ElementType.SECTION,
    TYPE_SYNOPSIS: ElementType.SYNOPSIS,
}

#: ``[[ ... ]]`` note, possibly spanning lines.
RE_NOTE: Final = re.compile(r"\[\[(.*?)\]\]", re.DOTALL)

#: jouvence's own character-cue shape, reused to validate de-careted cues.
RE_CUE: Final = re.compile(r"^\s*[A-Z][A-Z\-\._\s]+\s*(\(.*\))?$")

#: Fountain page break (``===``), which jouvence stores as a text-free element.
RE_PAGE_BREAK: Final = re.compile(r"^={3,}$")

#: Fountain markers jouvence strips from element text but that remain in the
#: source line, so line matching has to strip them too.
_FORCE_PREFIXES: Final = (".", "!", "@", "~", ">")


class FountainParseError(RuntimeError):
    """Raised when the backing parser cannot read the screenplay at all."""


# --------------------------------------------------------------------------
# Pre-pass: dual dialogue carets
# --------------------------------------------------------------------------


def _strip_dual_carets(lines: list[str]) -> tuple[list[str], set[int]]:
    """Remove trailing ``^`` from dual-dialogue cues, keeping line count fixed.

    Returns the rewritten lines and the set of 1-based line numbers that were
    marked dual. Line numbering is preserved exactly, so line references stay
    valid against the original file.
    """
    out = list(lines)
    dual: set[int] = set()
    for i, raw in enumerate(lines):
        body = raw.rstrip("\r\n").rstrip()
        if not body.endswith("^"):
            continue
        candidate = body[:-1].rstrip()
        if not RE_CUE.match(candidate.lstrip("@")):
            continue
        prev_blank = i == 0 or not lines[i - 1].strip()
        next_filled = i + 1 < len(lines) and bool(lines[i + 1].strip())
        if prev_blank and next_filled:
            out[i] = candidate + "\n"
            dual.add(i + 1)
    return out, dual


# --------------------------------------------------------------------------
# Line-reference attachment
# --------------------------------------------------------------------------


def _normalize(line: str) -> str:
    """Reduce a source line to the form jouvence would have stored as text."""
    s = line.strip()
    if len(s) >= 2 and s.startswith(">") and s.endswith("<"):
        return s[1:-1].strip()
    if s[:1] in _FORCE_PREFIXES:
        return s[1:].strip()
    return s


def _text_lines(text: str) -> list[str]:
    """The non-blank lines of an element's text, normalized for matching."""
    return [n for n in (_normalize(ln) for ln in text.splitlines()) if n]


def _find(needle: str, norm_lines: list[str], start: int) -> int:
    """Index of the first line at/after ``start`` matching ``needle``, or -1.

    Exact equality is tried across the whole remainder first; only if that
    fails do we accept a containment match, which covers the rare case where
    jouvence rewrote the line slightly.
    """
    for i in range(start, len(norm_lines)):
        if norm_lines[i] == needle:
            return i
    for i in range(start, len(norm_lines)):
        if needle and (needle in norm_lines[i] or norm_lines[i] in needle):
            return i
    return -1


def _attach_line_refs(elements: list[Element], source_lines: list[str]) -> None:
    """Fill in ``start_line``/``end_line`` on every element, in place.

    Uses a forward-only cursor so that repeated text (``KEVIN!`` said twice in
    a row, for instance) still lands on distinct lines.
    """
    norm = [_normalize(ln) for ln in source_lines]
    cursor = 0
    for el in elements:
        probes = _text_lines(el.text)
        if not probes:
            # Page breaks carry no text; find the `===` rule itself. Anything
            # else text-free is pinned to the cursor and flagged inexact.
            hit = -1
            if el.type is ElementType.PAGE_BREAK:
                for i in range(cursor, len(norm)):
                    if RE_PAGE_BREAK.match(norm[i]):
                        hit = i
                        break
            if hit < 0:
                el.start_line = min(cursor + 1, len(source_lines))
                el.end_line = el.start_line
                el.line_ref_exact = False
            else:
                el.start_line = el.end_line = hit + 1
                cursor = hit + 1
            continue

        first = _find(probes[0], norm, cursor)
        if first < 0:
            el.start_line = min(cursor + 1, len(source_lines))
            el.end_line = el.start_line
            el.line_ref_exact = False
            continue

        last = first if len(probes) == 1 else _find(probes[-1], norm, first)
        if last < first:
            last = first
            el.line_ref_exact = False

        el.start_line = first + 1
        el.end_line = last + 1
        cursor = last + 1


# --------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------


def _note_line(note_text: str, el: Element, source_lines: list[str]) -> int:
    """Best-effort source line for a note inside ``el``'s line span."""
    probe = note_text.strip().splitlines()[0].strip()[:40]
    for i in range(el.start_line - 1, min(el.end_line, len(source_lines))):
        if probe and probe in source_lines[i]:
            return i + 1
    return el.start_line


def _split_notes(elements: list[Element], source_lines: list[str]) -> list[Element]:
    """Promote note-only blocks to note elements; record inline notes on hosts.

    A block whose text is *nothing but* ``[[notes]]`` becomes one NOTE element
    per note. Otherwise the host keeps its verbatim text and the notes are
    attached to it, so no speech text is silently rewritten by the parser.
    """
    out: list[Element] = []
    for el in elements:
        matches = list(RE_NOTE.finditer(el.text))
        if not matches:
            out.append(el)
            continue

        remainder = RE_NOTE.sub("", el.text).strip()
        if not remainder and el.type in (ElementType.ACTION, ElementType.NOTE):
            for m in matches:
                note = Element(
                    index=0,
                    type=ElementType.NOTE,
                    text=m.group(1).strip(),
                    start_line=el.start_line,
                    end_line=el.end_line,
                    scene_index=el.scene_index,
                    line_ref_exact=el.line_ref_exact,
                )
                line = _note_line(note.text, el, source_lines)
                note.start_line = line
                note.end_line = line
                out.append(note)
            continue

        el.notes = [
            InlineNote(
                text=m.group(1).strip(),
                start=m.start(),
                end=m.end(),
                line=_note_line(m.group(1), el, source_lines),
            )
            for m in matches
        ]
        out.append(el)
    return out


# --------------------------------------------------------------------------
# Main entry points
# --------------------------------------------------------------------------


def _flatten(doc: JouvenceDocument) -> list[Element]:
    """Flatten jouvence's scene/paragraph tree into an ordered element list.

    Scene headings become elements in their own right. Whitespace-only action
    paragraphs — jouvence emits those to preserve blank-line runs — are dropped.
    Character cues are propagated onto the parentheticals and dialogue below
    them.
    """
    elements: list[Element] = []
    scene_index = -1
    for scene in doc.scenes:
        scene_index += 1
        if scene.header:
            elements.append(
                Element(
                    index=0,
                    type=ElementType.SCENE_HEADING,
                    text=scene.header.strip(),
                    start_line=0,
                    end_line=0,
                    scene_index=scene_index,
                )
            )
        current_cue: str | None = None
        for para in scene.paragraphs:
            el_type = _TYPE_MAP.get(para.type)
            if el_type is None:  # pragma: no cover - jouvence has no other types
                continue
            text = para.text or ""
            if el_type is ElementType.ACTION and not text.strip():
                continue
            speech_types = (
                ElementType.CHARACTER,
                ElementType.PARENTHETICAL,
                ElementType.DIALOGUE,
                ElementType.LYRICS,
            )
            if el_type is ElementType.CHARACTER:
                current_cue = text.strip()
            elif el_type not in speech_types:
                current_cue = None

            el = Element(
                index=0,
                type=el_type,
                text=text.strip(),
                start_line=0,
                end_line=0,
                scene_index=scene_index,
                character=current_cue if el_type in speech_types else None,
            )
            if el_type is ElementType.DIALOGUE and all(
                ln.lstrip().startswith("~") for ln in text.splitlines() if ln.strip()
            ):
                el.lyric = True
            elements.append(el)
    return elements


def parse_text(text: str, source: str = "<string>") -> ElementStream:
    """Parse Fountain source text into an :class:`ElementStream`."""
    source_lines = text.splitlines(keepends=True)
    rewritten, dual_lines = _strip_dual_carets(source_lines)

    try:
        doc = JouvenceParser().parseString("".join(rewritten))
    except Exception as exc:  # pragma: no cover - corpus parses cleanly
        raise FountainParseError(f"{PARSER_NAME} failed on {source}: {exc}") from exc

    elements = _flatten(doc)
    bare_lines = [ln.rstrip("\r\n") for ln in rewritten]
    _attach_line_refs(elements, bare_lines)
    elements = _split_notes(elements, bare_lines)

    # Propagate dual-dialogue marks from the cue down to its dialogue block.
    dual_active = False
    for el in elements:
        if el.type is ElementType.CHARACTER:
            dual_active = el.start_line in dual_lines
            el.dual = dual_active
        elif el.type in (
            ElementType.PARENTHETICAL,
            ElementType.DIALOGUE,
            ElementType.LYRICS,
        ):
            el.dual = dual_active
        else:
            dual_active = False

    for i, el in enumerate(elements):
        el.index = i

    return ElementStream(
        source=source,
        parser=PARSER_NAME,
        parser_version=PARSER_VERSION,
        title=dict(doc.title_values),
        elements=elements,
        scene_headings=[
            e.text for e in elements if e.type is ElementType.SCENE_HEADING
        ],
    )


def parse_file(path: str | Path) -> ElementStream:
    """Parse a ``.fountain`` file. The file is only ever opened for reading."""
    p = Path(path).expanduser()
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise FountainParseError(f"cannot read {p}: {exc}") from exc
    return parse_text(text, source=str(p))
