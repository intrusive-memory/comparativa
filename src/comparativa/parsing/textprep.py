"""TTS text preparation (FR-4), matched to Produciesta/SwiftVoxAlta semantics.

Every transform here was derived by reading the Swift stack (read-only) and
cross-checking it against the shipped granville transcripts
(``~/Projects/podcasts/granville/audio/*.vtt``), which record the exact strings
Produciesta handed to the TTS backend. ``docs/TEXT_PREP.md`` documents each
matched behaviour with its Swift source line, plus every divergence.

The Swift path, end to end, is:

``GuionElementSnapshot.speakableText`` (strip ``{{stage directions}}``, trim)
→ ``VoicingPlanBuilder`` (parenthetical → non-spoken ``instruct``; scene
heading → ``SluglineSpeech.expand``) → ``SwiftVoxAltaAdapter``
(``InlineBreath.mergedSegments`` splits on ``[[<breath/>]]``) → SwiftVoxAlta
(``VoiceLockManager.splitAtSentences`` duration-bounded chunking).

Nothing in that path touches em-dashes, ellipses, or ALL-CAPS words: they reach
the model verbatim. This module matches that (they are *deliberately* untouched),
and adds the small set of divergences listed in :data:`DIVERGENCES` and in
``docs/TEXT_PREP.md``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any, Final

# ---------------------------------------------------------------------------
# Patterns (each mirrors a named Swift regex; see docs/TEXT_PREP.md)
# ---------------------------------------------------------------------------

#: ``{{stage directions}}``, removed by ``GuionElementSnapshot.speakableText``.
#: ``re.DOTALL`` mirrors that regex's ``.dotMatchesLineSeparators``.
RE_STAGE_DIRECTION: Final = re.compile(r"\{\{.*?\}\}", re.DOTALL)

#: An inline ``[[<breath .../>]]`` marker — a split point, not spoken.
#: Mirrors ``InlineBreath.breathPattern``.
RE_BREATH: Final = re.compile(r"\[\[\s*<breath\b[^>]*/>\s*\]\]")

#: Any remaining inline ``[[note]]``.
RE_NOTE: Final = re.compile(r"\[\[.*?\]\]", re.DOTALL)

#: A Fountain lyric line's leading ``~``.
RE_LYRIC_PREFIX: Final = re.compile(r"^[ \t]*~[ \t]?", re.MULTILINE)

#: Fountain/Markdown emphasis runs (``***x***``, ``**x**``, ``*x*``, ``_x_``).
RE_EMPHASIS: Final = re.compile(r"(\*{1,3}|_)(?=\S)(.+?)(?<=\S)\1", re.DOTALL)

RE_WHITESPACE: Final = re.compile(r"\s+")

#: ``<shot prompt="..."/>`` glosa-av directive (Produciesta voices it as
#: ``SHOT PROMPT: <prompt>``; FR-3 marks it non-spoken by default).
RE_SHOT_PROMPT: Final = re.compile(
    r"<shot\b[^>]*\bprompt\s*=\s*([\"'])(?P<prompt>.*?)\1", re.DOTALL
)

# --- SluglineSpeech (ProduciestaCore/Pipeline/SluglineSpeech.swift) ---------

#: Spoken expansion for each INT./EXT./EST. marker (``SluglineSpeech.expansions``).
SLUGLINE_EXPANSIONS: Final[dict[str, str]] = {
    "INT": "INTERIOR",
    "EXT": "EXTERIOR",
    "EST": "ESTABLISHING SHOT",
    "INT/EXT": "INTERIOR AND EXTERIOR",
    "EXT/INT": "EXTERIOR AND INTERIOR",
    "I/E": "INTERIOR AND EXTERIOR",
    "E/I": "EXTERIOR AND INTERIOR",
}

RE_SLUG_MARKER: Final = re.compile(
    r"^\.?\s*(INT\.?/EXT\.?|EXT\.?/INT\.?|I/E\.?|E/I\.?|INT\.?|EXT\.?|EST\.?)(?=\s|$)",
    re.IGNORECASE,
)
RE_SLUG_SCENE_NUMBER: Final = re.compile(r"\s*#[0-9A-Za-z.)-]+#\s*$")
RE_SLUG_SEPARATOR: Final = re.compile(r"\s+[-–—]+\s+")
RE_SLUG_TRAILING_DASH: Final = re.compile(r"\s*[-–—]+\s*$")
RE_SLUG_REPEATED_PERIOD: Final = re.compile(r"\.(\s*\.)+")

# --- ParentheticalDirection (ProduciestaCore/Parsing/ParentheticalDirection.swift)

#: Lowercased prefixes marking a parenthetical segment as blocking/addressing
#: rather than delivery. Copied verbatim from ``blockingPrefixes``.
BLOCKING_PREFIXES: Final[tuple[str, ...]] = (
    "to ",
    "back to ",
    "to the ",
    "turning to",
    "turning back",
    "addressing",
    "aside to",
    "looking",
    "gesturing",
    "glancing",
    "off ",
    "pointing",
    "facing",
    "crossing to",
    "moving to",
    "into the",
    "over to",
)

# --- Timing / chunking constants (SwiftVoxAlta) ----------------------------

#: ``VoiceLockManager.estimatedSecondsPerChar``.
SECONDS_PER_CHAR: Final = 0.055
#: ``VoiceLockManager.minimumChunkSize``.
MINIMUM_CHUNK_CHARS: Final = 100
#: ``GenerationSettings.chunkTargetDuration`` default.
CHUNK_TARGET_SECONDS: Final = 12.0
#: ``GenerationSettings.chunkPauseDuration`` default.
CHUNK_PAUSE_SECONDS: Final = 0.25
#: ``InlineBreath.breathGapSeconds``.
BREATH_GAP_SECONDS: Final = 0.15
#: ``InlineBreath.minimumVoicedSpanLength``.
MINIMUM_SPAN_CHARS: Final = 40


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextPrepPolicy:
    """Which text transforms run, and with what constants.

    The defaults are the *parity* defaults: they reproduce the string
    Produciesta hands to the TTS backend wherever the Swift stack does a
    transform at all, and deviate only where the Swift behaviour is a
    documented artefact source (see ``docs/TEXT_PREP.md``).
    """

    name: str = "parity"
    #: Remove ``{{stage directions}}`` (matched: ``speakableText``).
    strip_stage_directions: bool = True
    #: Split dialogue on inline ``[[<breath/>]]`` markers (matched: ``InlineBreath``).
    split_on_breaths: bool = True
    breath_gap_seconds: float = BREATH_GAP_SECONDS
    minimum_span_chars: int = MINIMUM_SPAN_CHARS
    #: Remove any remaining inline ``[[note]]`` (matched: glosa strips them).
    strip_inline_notes: bool = True
    #: Remove Fountain lyric ``~`` prefixes (DIVERGENCE: Swift keeps them).
    strip_lyric_prefix: bool = True
    #: Remove ``*``/``**``/``_`` emphasis runs (DIVERGENCE when enabled;
    #: default off so both stacks speak the identical string).
    strip_emphasis: bool = False
    #: Collapse whitespace runs to one space (DIVERGENCE: Swift only trims).
    collapse_whitespace: bool = True
    #: Rewrite scene headings for the ear (matched: ``SluglineSpeech``).
    expand_sluglines: bool = True
    #: Drop blocking/addressing parenthetical segments (matched:
    #: ``ParentheticalDirection``).
    filter_blocking_directions: bool = True

    def evolve(self, **changes: Any) -> TextPrepPolicy:
        """A copy of this policy with ``changes`` applied."""
        return replace(self, **changes)


#: Byte-for-byte Produciesta parity: keep lyric tildes and emphasis, no
#: whitespace collapsing. Useful to quantify how much the divergences matter.
STRICT_PARITY: Final = TextPrepPolicy(
    name="strict-parity",
    strip_lyric_prefix=False,
    collapse_whitespace=False,
)

#: The default policy used by ``comparativa parse``.
DEFAULT_POLICY: Final = TextPrepPolicy()


# ---------------------------------------------------------------------------
# Prepared text
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedText:
    """The TTS-ready form of one screenplay element's text."""

    #: The whole line as one string (segments joined by a single space). This
    #: is what a single-shot engine should be handed.
    text: str
    #: The inter-breath spans, in order. ``len(segments) - 1`` breath gaps sit
    #: between them; an engine honouring breaths generates each span separately
    #: and splices :attr:`breath_gap_seconds` of silence at every seam.
    segments: tuple[str, ...]
    #: Silence to insert at each seam between consecutive segments.
    breath_gap_seconds: float = BREATH_GAP_SECONDS
    #: Number of ``[[<breath/>]]`` markers found before runt merging.
    breath_markers: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"text": self.text}
        if len(self.segments) > 1:
            d["segments"] = list(self.segments)
            d["breath_gap_seconds"] = self.breath_gap_seconds
        if self.breath_markers:
            d["breath_markers"] = self.breath_markers
        return d


# ---------------------------------------------------------------------------
# Individual transforms
# ---------------------------------------------------------------------------


def strip_stage_directions(text: str) -> str:
    """Remove ``{{stage directions}}`` (``speakableText``, matched)."""
    return RE_STAGE_DIRECTION.sub("", text)


def strip_inline_notes(text: str) -> str:
    """Remove any inline ``[[note]]`` left after breath splitting."""
    return RE_NOTE.sub("", text)


def strip_lyric_prefix(text: str) -> str:
    """Remove Fountain's leading ``~`` from each lyric line."""
    return RE_LYRIC_PREFIX.sub("", text)


def strip_emphasis(text: str) -> str:
    """Remove Fountain/Markdown emphasis runs, keeping the emphasised words."""
    previous = None
    out = text
    while out != previous:
        previous = out
        out = RE_EMPHASIS.sub(r"\2", out)
    return out


def collapse_whitespace(text: str) -> str:
    """Collapse every whitespace run to a single space and trim."""
    return RE_WHITESPACE.sub(" ", text).strip()


def expand_slugline(heading: str) -> str:
    """Rewrite a scene heading for the ear.

    Port of ``SluglineSpeech.expand``: ``EST. SOMEWHERE - SUNSET`` becomes
    ``ESTABLISHING SHOT. SOMEWHERE. SUNSET``. Headings with no recognised
    marker still get their dash separators converted.
    """
    text = heading.strip()
    if not text:
        return ""

    text = RE_SLUG_SCENE_NUMBER.sub("", text)

    match = RE_SLUG_MARKER.match(text)
    if match is not None:
        key = match.group(1).replace(".", "").upper()
        expansion = SLUGLINE_EXPANSIONS.get(key)
        if expansion is not None:
            text = expansion + "." + text[match.end() :]

    text = RE_SLUG_SEPARATOR.sub(". ", text)
    text = RE_SLUG_TRAILING_DASH.sub("", text)
    text = RE_SLUG_REPEATED_PERIOD.sub(".", text)
    return RE_WHITESPACE.sub(" ", text).strip()


def shot_prompt(note_body: str) -> str | None:
    """The ``SHOT PROMPT: ...`` line for a ``<shot prompt="..."/>`` note.

    Port of ``VoicingPlanBuilder.glosaSpokenLine``'s ``<shot>`` branch: an
    empty ``prompt`` is a glosa-av defaults declaration and stays silent.
    Attribute extraction here is a regex rather than the glosa-av parser —
    a documented divergence (no Python glosa-av binding exists).
    """
    body = note_body.strip()
    if not body.startswith("<shot"):
        return None
    match = RE_SHOT_PROMPT.search(body)
    if match is None:
        return None
    prompt = match.group("prompt").strip()
    return f"SHOT PROMPT: {prompt}" if prompt else None


def parenthetical_direction(
    parentheticals: Iterable[str], *, filter_blocking: bool = True
) -> str | None:
    """Turn a dialogue's parenthetical lines into a non-spoken TTS ``instruct``.

    Port of ``ParentheticalDirection.instruct``. Each parenthetical is
    unwrapped, split on commas, and every segment classified: delivery
    directions (``sotto``, ``angrily``) are kept, blocking/addressing ones
    (``to Abbey``, ``looking out the window``) are dropped. Kept segments are
    joined with ``", "``. Returns ``None`` when nothing survives.
    """
    kept: list[str] = []
    for raw in parentheticals:
        inner = str(raw).strip()
        if inner.startswith("("):
            inner = inner[1:]
        if inner.endswith(")"):
            inner = inner[:-1]
        for segment in inner.split(","):
            trimmed = segment.strip()
            if not trimmed:
                continue
            if filter_blocking and _is_blocking(trimmed):
                continue
            kept.append(trimmed)
    if not kept:
        return None
    return ", ".join(kept)


def _is_blocking(segment: str) -> bool:
    lower = segment.lower()
    return any(lower.startswith(prefix) for prefix in BLOCKING_PREFIXES)


# ---------------------------------------------------------------------------
# Breath splitting
# ---------------------------------------------------------------------------


def breath_segments(text: str) -> tuple[list[str], int]:
    """Split ``text`` on ``[[<breath/>]]`` markers (``InlineBreath.segments``).

    Returns the prose spans in order plus the number of markers found.
    """
    parts = RE_BREATH.split(text)
    return parts, len(parts) - 1


def merge_runt_segments(
    spans: list[str], minimum: int = MINIMUM_SPAN_CHARS
) -> list[str]:
    """Merge spans shorter than ``minimum`` into a neighbour.

    Port of ``InlineBreath.mergedSegments``: forward by preference, backward at
    the end of the line. Empty spans are left alone (they voice nothing, so
    they cannot clip, and their gap is an authored pause).
    """
    out = list(spans)
    if len(out) <= 1:
        return out

    def length(s: str) -> int:
        return len(s.strip())

    def join(a: str, b: str) -> str:
        seam_has_space = (a[-1:].isspace()) or (b[:1].isspace())
        return a + b if seam_has_space else a + " " + b

    i = 0
    while len(out) > 1 and i < len(out):
        if 0 < length(out[i]) < minimum:
            if i + 1 < len(out) and length(out[i + 1]) > 0:
                out[i + 1] = join(out[i], out[i + 1])
                del out[i]
                continue
            if i > 0 and length(out[i - 1]) > 0:
                out[i - 1] = join(out[i - 1], out[i])
                del out[i]
                i -= 1
                continue
        i += 1
    return out


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


def prepare(
    text: str,
    *,
    policy: TextPrepPolicy = DEFAULT_POLICY,
    slugline: bool = False,
) -> PreparedText:
    """Prepare one element's verbatim text for TTS.

    Order of operations mirrors the Swift pipeline: stage directions first
    (``speakableText``), then the slugline rewrite (``VoicingPlanBuilder``),
    then the breath split (``SwiftVoxAltaAdapter``), then per-span cleanup and
    trimming (the adapter trims each span before generating).
    """
    body = text
    if policy.strip_stage_directions:
        body = strip_stage_directions(body)
    if slugline and policy.expand_sluglines:
        body = expand_slugline(body)

    if policy.split_on_breaths:
        spans, markers = breath_segments(body)
        spans = merge_runt_segments(spans, policy.minimum_span_chars)
    else:
        spans, markers = [body], 0

    cleaned: list[str] = []
    for span in spans:
        piece = span
        if policy.strip_inline_notes:
            piece = strip_inline_notes(piece)
        if policy.strip_lyric_prefix:
            piece = strip_lyric_prefix(piece)
        if policy.strip_emphasis:
            piece = strip_emphasis(piece)
        piece = collapse_whitespace(piece) if policy.collapse_whitespace else piece.strip()
        if piece:
            cleaned.append(piece)

    return PreparedText(
        text=" ".join(cleaned),
        segments=tuple(cleaned),
        breath_gap_seconds=policy.breath_gap_seconds,
        breath_markers=markers,
    )


# ---------------------------------------------------------------------------
# Duration estimation + sentence chunking (SwiftVoxAlta parity)
# ---------------------------------------------------------------------------


def estimate_duration(text: str) -> float:
    """Estimated spoken duration in seconds (``VoiceLockManager.estimateDuration``)."""
    return len(text) * SECONDS_PER_CHAR


#: Sentence boundary candidate: terminal punctuation, optional closing
#: quotes/brackets, then whitespace. Stand-in for Foundation's ICU
#: ``.bySentences`` segmenter (see the divergence table in ``docs/TEXT_PREP.md``).
RE_SENTENCE_BREAK: Final = re.compile(r"""[.!?…]+["'”’)\]]*\s+""")

#: Trailing word before a candidate break, used to veto abbreviations.
RE_BREAK_LEAD: Final = re.compile(r"([A-Za-z][A-Za-z.]*)[.!?…]+[\"'”’)\]]*\s*$")

#: Words that end in a period without ending a sentence.
ABBREVIATIONS: Final[frozenset[str]] = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "rev", "sr", "jr", "st",
        "vs", "etc", "eg", "ie", "no", "fig", "inc", "ltd", "co",
        "dept", "approx", "est", "min", "max",
    }
)


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, ICU-approximately.

    A candidate break is vetoed when the preceding token is a known
    abbreviation or a single-letter initial, which is what ICU's segmenter
    gets right and a bare regex gets wrong.
    """
    out: list[str] = []
    start = 0
    for match in RE_SENTENCE_BREAK.finditer(text):
        head = text[start : match.end()]
        lead = RE_BREAK_LEAD.search(head)
        word = lead.group(1).lower().replace(".", "") if lead else ""
        if word in ABBREVIATIONS or (len(word) == 1 and word.isalpha()):
            continue
        piece = head.strip()
        if piece:
            out.append(piece)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def split_at_sentences(
    text: str,
    max_duration: float = CHUNK_TARGET_SECONDS,
    minimum_chunk_chars: int = MINIMUM_CHUNK_CHARS,
) -> list[str]:
    """Pack sentences into duration-bounded chunks.

    Port of ``VoiceLockManager.splitAtSentences``: greedy packing to
    ``max_duration`` estimated seconds, with runt chunks (< ``minimum_chunk_chars``)
    merged forward, and a trailing runt merged backward.
    """
    sentences = split_sentences(text)
    if not sentences:
        trimmed = text.strip()
        return [trimmed] if trimmed else []

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = sentence if not current else current + " " + sentence
        if estimate_duration(candidate) > max_duration and current:
            if len(current) >= minimum_chunk_chars:
                chunks.append(current)
                current = sentence
            else:
                current = candidate
        else:
            current = candidate

    if current:
        if len(current) < minimum_chunk_chars and chunks:
            chunks.append(chunks.pop() + " " + current)
        else:
            chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Divergence register (mirrored by docs/TEXT_PREP.md)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Divergence:
    """One documented difference from the Swift stack's text preparation."""

    id: str
    topic: str
    swift: str
    python: str
    reason: str


#: Every intentional divergence. ``docs/TEXT_PREP.md`` renders this as a table;
#: ``tests/test_textprep.py`` asserts the two stay in sync.
DIVERGENCES: Final[tuple[Divergence, ...]] = (
    Divergence(
        id="D-1",
        topic="Action / scene heading / centered / transition narration",
        swift=(
            "Voiced by NARRATOR (VoicingPlan.swift:319-356); scene headings are "
            "rewritten by SluglineSpeech first."
        ),
        python=(
            "spoken=false under the default FR-3 policy; available via the "
            "`produciesta-parity` speech policy."
        ),
        reason=(
            "REQUIREMENTS.md FR-3 mandates it. This is the single largest "
            "behavioural gap and it changes what condition C contains relative "
            "to condition A — see docs/TEXT_PREP.md § Comparison validity."
        ),
    ),
    Divergence(
        id="D-2",
        topic="Fountain lyric `~` prefix",
        swift=(
            "Kept in elementText and spoken (FountainParser.swift:228-237; "
            "speakableText does not strip it)."
        ),
        python="Stripped (`strip_lyric_prefix`, on by default).",
        reason=(
            "A literal tilde is not speech; Qwen3 either vocalises or garbles "
            "it. The bumper's sung lines are the only affected corpus text."
        ),
    ),
    Divergence(
        id="D-3",
        topic="Fountain/Markdown emphasis (`*x*`, `**x**`, `_x_`)",
        swift=(
            "Kept verbatim — confirmed in the shipped transcript: "
            "`I'm the one that was **HERE**.` "
            "(audio/episode_1_01_cold_open.vtt:182)."
        ),
        python="Kept verbatim by default; `strip_emphasis=True` removes them.",
        reason=(
            "MATCHED by default so both stacks speak the identical string. "
            "Flagged because the Swift behaviour is an artefact source, not a "
            "design choice."
        ),
    ),
    Divergence(
        id="D-4",
        topic="Internal whitespace",
        swift=(
            "Only the outer ends are trimmed; removing a mid-string `{{...}}` "
            "leaves a double space (GuionElementSnapshot.swift:198-213)."
        ),
        python="All whitespace runs collapsed to one space.",
        reason="Cosmetic; keeps prepared text stable for hashing and diffing.",
    ),
    Divergence(
        id="D-5",
        topic="Sentence segmentation for chunking",
        swift=(
            "Foundation/ICU `enumerateSubstrings(.bySentences)` "
            "(VoiceLockManager.swift:394-403)."
        ),
        python="Regex approximation (`split_sentences`).",
        reason=(
            "No ICU sentence segmenter in the Python stdlib. Differs on exotic "
            "abbreviations; chunk boundaries can therefore differ on long lines."
        ),
    ),
    Divergence(
        id="D-6",
        topic="`<shot prompt=...>` attribute extraction",
        swift="glosa-av `GlosaParser` (VoicingPlan.swift:387-406).",
        python="Regex (`shot_prompt`).",
        reason=(
            "No Python glosa-av binding. Neither corpus episode contains a "
            "`<shot>` note, so this is untested against real data."
        ),
    ),
    Divergence(
        id="D-7",
        topic="Empty breath spans",
        swift=(
            "A leading/trailing/adjacent-breath empty span is preserved and "
            "contributes only its silence gap (SwiftVoxAltaAdapter.swift:127-141)."
        ),
        python="Empty spans are dropped, so their gap is lost.",
        reason=(
            "Simplification; no corpus line has a leading, trailing, or "
            "doubled breath marker."
        ),
    ),
)

#: Divergence ids, for tests.
DIVERGENCE_IDS: Final[tuple[str, ...]] = tuple(d.id for d in DIVERGENCES)
