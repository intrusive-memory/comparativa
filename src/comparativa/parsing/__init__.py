"""Fountain parsing: the ordered element stream every later stage consumes,
plus its speech projection (resolved characters + prepared TTS text)."""

from .cues import (
    NARRATOR,
    CastIndex,
    UnresolvedCue,
    find_cast_file,
    load_cast_index,
    load_cast_index_for,
    normalize_cue,
)
from .elements import Element, ElementStream, ElementType, InlineNote
from .fountain import (
    PARSER_NAME,
    PARSER_VERSION,
    FountainParseError,
    parse_file,
    parse_text,
)
from .speech import (
    FR3_POLICY,
    POLICIES,
    PRODUCIESTA_POLICY,
    PreparedScript,
    SpeechPolicy,
    SpokenLine,
    prepare_script,
)
from .textprep import (
    DEFAULT_POLICY,
    DIVERGENCES,
    STRICT_PARITY,
    PreparedText,
    TextPrepPolicy,
    prepare,
    split_at_sentences,
)

__all__ = [
    "DEFAULT_POLICY",
    "DIVERGENCES",
    "FR3_POLICY",
    "NARRATOR",
    "PARSER_NAME",
    "PARSER_VERSION",
    "POLICIES",
    "PRODUCIESTA_POLICY",
    "STRICT_PARITY",
    "CastIndex",
    "Element",
    "ElementStream",
    "ElementType",
    "FountainParseError",
    "InlineNote",
    "PreparedScript",
    "PreparedText",
    "SpeechPolicy",
    "SpokenLine",
    "TextPrepPolicy",
    "UnresolvedCue",
    "find_cast_file",
    "load_cast_index",
    "load_cast_index_for",
    "normalize_cue",
    "parse_file",
    "parse_text",
    "prepare",
    "prepare_script",
    "split_at_sentences",
]
