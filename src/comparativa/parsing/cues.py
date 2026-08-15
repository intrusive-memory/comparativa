"""Character-cue normalisation and CAST.md resolution (FR-2).

A Fountain character cue is written for a reader: ``RAY (CONT'D)``,
``JOANN (V.O.)``, ``@McKenzie``. The generation layer needs the *character*,
not the cue, so every cue is reduced to a canonical name and then resolved
against the project's ``CAST.md`` roster.

Normalisation deliberately mirrors Produciesta's
``CharacterNameNormalizer.normalize`` (``ProduciestaCore/Domain/
CharacterNameNormalizer.swift``): strip *every* trailing parenthetical —
repeatedly, so ``RAY (V.O.) (CONT'D)`` collapses — then trim and uppercase.
The execution plan names ``(CONT'D)``/``(V.O.)``/``(O.S.)`` specifically;
stripping all trailing parentheticals is the Swift behaviour and a strict
superset of that list, so the two stacks agree on every cue either would
handle.

``NARRATOR`` is a first-class speaking character: it resolves even when the
roster omits it, because narrated elements synthesise a NARRATOR speaker.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

#: The synthetic-but-first-class narrator name (Produciesta:
#: ``VoicingPlan.narratorName``).
NARRATOR: Final = "NARRATOR"

#: A trailing ``(...)`` extension on a cue, with any surrounding whitespace.
RE_TRAILING_PARENTHETICAL: Final = re.compile(r"\s*\([^)]*\)\s*$")

#: Runs of whitespace inside a cue, collapsed so ``RAY   BOB`` folds to one.
RE_WHITESPACE: Final = re.compile(r"\s+")

#: The file a project's roster lives in.
CAST_FILENAME: Final = "CAST.md"


def strip_cue_extensions(raw: str) -> tuple[str, tuple[str, ...]]:
    """Split a raw cue into its bare name and its trailing extensions.

    >>> strip_cue_extensions("RAY (CONT'D)")
    ("RAY", ("(CONT'D)",))
    """
    name = raw.strip()
    # Fountain markers that can ride along on a cue line.
    if name.startswith("@"):
        name = name[1:]
    name = name.rstrip("^").rstrip()

    extensions: list[str] = []
    while True:
        match = RE_TRAILING_PARENTHETICAL.search(name)
        if match is None:
            break
        extensions.append(match.group(0).strip())
        name = name[: match.start()]
    extensions.reverse()
    return name.strip(), tuple(extensions)


def normalize_cue(raw: str) -> str:
    """Canonicalise a raw character cue to an uppercase, extension-free name."""
    name, _ = strip_cue_extensions(raw)
    return RE_WHITESPACE.sub(" ", name).strip().upper()


def _fold(name: str) -> str:
    """The lookup key for a character name: caps, single spaces, no punctuation."""
    folded = RE_WHITESPACE.sub(" ", name).strip().upper()
    return re.sub(r"[.\-_']", "", folded)


@dataclass(frozen=True)
class UnresolvedCue:
    """A cue that did not match any roster character."""

    cue: str
    """The raw cue exactly as written in the screenplay."""
    normalized: str
    """The cue after :func:`normalize_cue`."""
    element_index: int
    line: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "cue": self.cue,
            "normalized": self.normalized,
            "element_index": self.element_index,
            "line": self.line,
        }


@dataclass(frozen=True)
class CastIndex:
    """A resolver from normalised cue to canonical ``CAST.md`` character name."""

    names: tuple[str, ...]
    source: str | None = None
    _lookup: dict[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def from_names(
        cls, names: Iterable[str], *, source: str | None = None
    ) -> CastIndex:
        """Build an index from an iterable of canonical character names."""
        ordered = tuple(str(n).strip() for n in names if str(n).strip())
        lookup: dict[str, str] = {}
        for name in ordered:
            lookup.setdefault(_fold(name), name)
        # NARRATOR is always resolvable, roster or not.
        lookup.setdefault(_fold(NARRATOR), NARRATOR)
        return cls(names=ordered, source=source, _lookup=lookup)

    def resolve(self, cue: str) -> str | None:
        """Canonical character for ``cue``, or ``None`` when unresolved."""
        return self._lookup.get(_fold(normalize_cue(cue)))

    def __contains__(self, cue: object) -> bool:
        return self.resolve(str(cue)) is not None


def find_cast_file(episode: str | Path) -> Path | None:
    """Locate the ``CAST.md`` governing ``episode``.

    Walks up from the screenplay (``<project>/episodes/ep.fountain`` is the
    granville layout) until a ``CAST.md`` turns up, stopping at the filesystem
    root. Returns ``None`` when the screenplay is not inside a cast project.
    """
    path = Path(episode).expanduser().resolve()
    start = path.parent if path.is_file() or path.suffix else path
    for directory in (start, *start.parents):
        candidate = directory / CAST_FILENAME
        if candidate.is_file():
            return candidate
    return None


def load_cast_index(cast_file: str | Path) -> CastIndex:
    """Read a ``CAST.md`` and build the resolver (read-only).

    The roster parser lives in :mod:`comparativa.voices.roster` (FR-5); it is
    imported lazily so ``parse`` keeps working — with cue resolution skipped —
    if that module is unavailable.
    """
    from ..voices.roster import parse_cast_markdown

    path = Path(cast_file).expanduser()
    members = parse_cast_markdown(
        path.read_text(encoding="utf-8"), source=str(path)
    )
    return CastIndex.from_names(
        (m.character for m in members), source=str(path)
    )


def load_cast_index_for(episode: str | Path) -> CastIndex | None:
    """Find and load the ``CAST.md`` for ``episode``, or ``None`` if there is none."""
    cast_file = find_cast_file(episode)
    if cast_file is None:
        return None
    return load_cast_index(cast_file)
