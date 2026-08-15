"""Speech classification and the prepared-speech stream (FR-2, FR-3, FR-4).

This is the layer between the verbatim element stream (Sortie 2) and the
generation layer (Sortie 5): it walks the stream once, decides what is spoken
and by whom, resolves every cue against ``CAST.md``, and produces the prepared
TTS text for each spoken line.

The walk mirrors Produciesta's ``VoicingPlanBuilder`` (speaker cursor, buffered
parentheticals folded into a non-spoken ``direction``, a cue reset on every
narrated element) so line boundaries match between the two stacks. What it
*voices* is governed by :class:`SpeechPolicy`:

* :data:`FR3_POLICY` (default) — only character dialogue is spoken. Action,
  scene headings, centered text, transitions, notes, and ``SHOT PROMPT``
  panels are ``spoken: false`` but stay in the stream for timing context. This
  is REQUIREMENTS.md FR-3, verbatim.
* :data:`PRODUCIESTA_POLICY` — additionally narrates action, scene headings,
  centered text, transitions, sections/synopses, and ``SHOT PROMPT`` panels in
  the NARRATOR voice, exactly as the Swift stack does.

The two produce very different episodes on the granville corpus (the shipped
Swift transcripts narrate the titles, every slugline, and every action line);
``docs/TEXT_PREP.md`` § Comparison validity spells out what that means for the
A-vs-C comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

from .cues import NARRATOR, CastIndex, UnresolvedCue, normalize_cue
from .elements import Element, ElementStream, ElementType
from .textprep import (
    DEFAULT_POLICY,
    PreparedText,
    TextPrepPolicy,
    parenthetical_direction,
    prepare,
    shot_prompt,
)

#: Types that are dialogue for a cued character.
_DIALOGUE_TYPES: Final = frozenset({ElementType.DIALOGUE, ElementType.LYRICS})

#: Types Produciesta voices in the NARRATOR voice.
_PRODUCIESTA_NARRATED: Final = frozenset(
    {
        ElementType.ACTION,
        ElementType.SCENE_HEADING,
        ElementType.CENTERED,
        ElementType.TRANSITION,
        ElementType.SECTION,
        ElementType.SYNOPSIS,
        # Only reached for lyrics with no character cue; cued lyrics are
        # dialogue and are handled by the speaker cursor first.
        ElementType.LYRICS,
    }
)


@dataclass(frozen=True)
class SpeechPolicy:
    """What counts as speech, and how its text is prepared."""

    name: str
    #: Non-dialogue element types voiced by NARRATOR. Empty under FR-3.
    narrated_types: frozenset[ElementType] = frozenset()
    #: Voice ``<shot prompt="..."/>`` notes as ``SHOT PROMPT: ...``.
    voice_shot_prompts: bool = False
    text: TextPrepPolicy = DEFAULT_POLICY


#: REQUIREMENTS.md FR-3: dialogue only. The default.
FR3_POLICY: Final = SpeechPolicy(name="fr3")

#: What the Swift stack actually voices (see the shipped granville `.vtt`s).
PRODUCIESTA_POLICY: Final = SpeechPolicy(
    name="produciesta-parity",
    narrated_types=_PRODUCIESTA_NARRATED,
    voice_shot_prompts=True,
)

#: Selectable by name from the CLI.
POLICIES: Final[dict[str, SpeechPolicy]] = {
    FR3_POLICY.name: FR3_POLICY,
    PRODUCIESTA_POLICY.name: PRODUCIESTA_POLICY,
}


@dataclass(frozen=True)
class SpokenLine:
    """One unit of speech: a character (or NARRATOR) saying prepared text."""

    #: Position among spoken lines, 0-based.
    index: int
    #: Index into :attr:`ElementStream.elements` this line came from.
    element_index: int
    element_type: str
    #: Resolved ``CAST.md`` character, or ``NARRATOR``.
    character: str
    #: The raw cue as written (``RAY (CONT'D)``), or ``None`` for narration.
    raw_cue: str | None
    prepared: PreparedText
    #: Non-spoken performance instruct derived from the parenthetical(s).
    direction: str | None
    scene_index: int
    start_line: int
    end_line: int
    lyric: bool = False
    dual: bool = False

    @property
    def text(self) -> str:
        return self.prepared.text

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "index": self.index,
            "element_index": self.element_index,
            "element_type": self.element_type,
            "character": self.character,
            "scene_index": self.scene_index,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }
        if self.raw_cue is not None:
            d["raw_cue"] = self.raw_cue
        d.update(self.prepared.to_dict())
        if self.direction is not None:
            d["direction"] = self.direction
        if self.lyric:
            d["lyric"] = True
        if self.dual:
            d["dual"] = True
        return d


@dataclass
class PreparedScript:
    """The speech projection of one screenplay."""

    policy: str
    text_policy: str
    #: Path to the ``CAST.md`` used for resolution, or ``None`` when skipped.
    cast_source: str | None
    lines: list[SpokenLine] = field(default_factory=list)
    #: Cues that matched no roster character. Empty is the success condition.
    unresolved_cues: list[UnresolvedCue] = field(default_factory=list)
    #: ``element_index`` -> True for elements that produced a spoken line.
    spoken_elements: set[int] = field(default_factory=set)

    @property
    def characters(self) -> dict[str, int]:
        """Spoken-line count per character, most-spoken first."""
        counts: dict[str, int] = {}
        for line in self.lines:
            counts[line.character] = counts.get(line.character, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "text_policy": self.text_policy,
            "cast_source": self.cast_source,
            "line_count": len(self.lines),
            "characters": self.characters,
            "lines": [line.to_dict() for line in self.lines],
        }


def prepare_script(
    stream: ElementStream,
    cast: CastIndex | None = None,
    *,
    policy: SpeechPolicy = FR3_POLICY,
) -> PreparedScript:
    """Walk ``stream`` and build its prepared-speech projection.

    ``cast`` resolves cues to canonical character names; when it is ``None``
    resolution is skipped entirely and cues are used as normalised
    (``unresolved_cues`` stays empty because nothing was checked — the caller
    reports ``cast_source: null`` so that is not mistaken for success).
    """
    script = PreparedScript(
        policy=policy.name,
        text_policy=policy.text.name,
        cast_source=cast.source if cast is not None else None,
    )

    speaker: str | None = None
    raw_cue: str | None = None
    pending: list[str] = []

    def add(
        element: Element,
        character: str,
        prepared: PreparedText,
        *,
        cue: str | None = None,
        direction: str | None = None,
    ) -> None:
        if prepared.is_empty:
            return
        script.lines.append(
            SpokenLine(
                index=len(script.lines),
                element_index=element.index,
                element_type=str(element.type),
                character=character,
                raw_cue=cue,
                prepared=prepared,
                direction=direction,
                scene_index=element.scene_index,
                start_line=element.start_line,
                end_line=element.end_line,
                lyric=element.lyric,
                dual=element.dual,
            )
        )
        script.spoken_elements.add(element.index)

    for element in stream.elements:
        if element.type is ElementType.CHARACTER:
            raw_cue = element.text.strip()
            resolved = cast.resolve(raw_cue) if cast is not None else None
            if resolved is None and cast is not None:
                script.unresolved_cues.append(
                    UnresolvedCue(
                        cue=raw_cue,
                        normalized=normalize_cue(raw_cue),
                        element_index=element.index,
                        line=element.start_line,
                    )
                )
            speaker = resolved or normalize_cue(raw_cue)
            pending.clear()
            continue

        if element.type is ElementType.PARENTHETICAL:
            trimmed = element.text.strip()
            if trimmed:
                pending.append(trimmed)
            continue

        if element.type in _DIALOGUE_TYPES and element.character is not None:
            direction = parenthetical_direction(
                pending, filter_blocking=policy.text.filter_blocking_directions
            )
            pending.clear()
            add(
                element,
                speaker or NARRATOR,
                prepare(element.text, policy=policy.text),
                cue=raw_cue,
                direction=direction,
            )
            continue

        if element.type is ElementType.NOTE:
            if policy.voice_shot_prompts:
                spoken = shot_prompt(element.text)
                if spoken:
                    add(element, NARRATOR, prepare(spoken, policy=policy.text))
            continue

        if element.type in policy.narrated_types:
            speaker = None
            raw_cue = None
            pending.clear()
            add(
                element,
                NARRATOR,
                prepare(
                    element.text,
                    policy=policy.text,
                    slugline=element.type is ElementType.SCENE_HEADING,
                ),
            )
            continue

        # Everything else is non-speech and resets the dialogue block, exactly
        # as a narrated element would in the Swift walk.
        if element.type not in (ElementType.PAGE_BREAK,):
            speaker = None
            raw_cue = None
            pending.clear()

    return script
