"""Character roster and default-voice assignment (Sortie 4).

Round 1 is defaults-only per Resolved Decision RD-2: every engine uses its
built-in voices, deterministically auto-assigned per character and recorded in
the committed ``presets.yaml``.
"""

from __future__ import annotations

from .assign import (
    PRESETS_FILENAME,
    SCHEMA_VERSION,
    Assignments,
    CharacterAssignment,
    Classification,
    assign,
    classify,
    dump_yaml,
    load_presets,
    to_document,
    voice_for,
    write_presets,
)
from .catalog import DEFAULT_VOICE, ENGINE_KEYS, ENGINES, Engine, Voice, engine
from .roster import CastMember, Roster, RosterError, load_roster, parse_cast_markdown

__all__ = [
    "Assignments",
    "CastMember",
    "CharacterAssignment",
    "Classification",
    "DEFAULT_VOICE",
    "ENGINES",
    "ENGINE_KEYS",
    "Engine",
    "PRESETS_FILENAME",
    "Roster",
    "RosterError",
    "SCHEMA_VERSION",
    "Voice",
    "assign",
    "classify",
    "dump_yaml",
    "engine",
    "load_presets",
    "load_roster",
    "parse_cast_markdown",
    "to_document",
    "voice_for",
    "write_presets",
]
