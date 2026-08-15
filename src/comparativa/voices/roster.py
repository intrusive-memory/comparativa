"""Parse a podcast project's ``CAST.md`` into a character roster (FR-5).

``CAST.md`` is a markdown file whose YAML frontmatter carries a ``cast:`` list::

    ---
    type: cast
    schemaVersion: 1
    cast:
    - character: ARCHER
      voicePrompt: Male. 20's. Low, gravelly rasp...
      voices:
        voxalta:
        - voices/ARCHER.vox
    ---

Round 1 is defaults-only (RD-2): the ``voices.voxalta`` paths are *recorded* on
the roster for the future cloning mission but are never opened, and no file
inside the project tree is ever written.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CAST_FILENAME = "CAST.md"


class RosterError(RuntimeError):
    """Raised when ``CAST.md`` is missing or structurally unusable."""


@dataclass(frozen=True)
class CastMember:
    """One speaking character from ``CAST.md``."""

    character: str
    voice_prompt: str
    #: ``voices.voxalta`` entries, verbatim and project-relative. Never opened.
    voxalta_paths: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Roster:
    """The full cast of one project."""

    project_dir: Path
    cast_file: Path
    cast_sha256: str
    members: tuple[CastMember, ...]

    def __iter__(self):
        return iter(self.members)

    def __len__(self) -> int:
        return len(self.members)

    @property
    def character_names(self) -> tuple[str, ...]:
        return tuple(m.character for m in self.members)


def _split_frontmatter(text: str) -> str:
    """Return the YAML frontmatter block of a markdown document."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise RosterError("CAST.md does not start with a '---' frontmatter fence")
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "\n".join(lines[1:index])
    raise RosterError("CAST.md frontmatter is not terminated by a closing '---'")


def parse_cast_markdown(text: str, *, source: str = "CAST.md") -> tuple[CastMember, ...]:
    """Parse ``CAST.md`` content into cast members, preserving file order."""
    try:
        data = yaml.safe_load(_split_frontmatter(text))
    except yaml.YAMLError as exc:  # pragma: no cover - depends on corrupt input
        raise RosterError(f"{source}: frontmatter is not valid YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise RosterError(f"{source}: frontmatter is not a YAML mapping")

    entries = data.get("cast")
    if not isinstance(entries, list) or not entries:
        raise RosterError(f"{source}: frontmatter has no non-empty 'cast:' list")

    members: list[CastMember] = []
    seen: set[str] = set()
    for position, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise RosterError(f"{source}: cast entry #{position} is not a mapping")
        name = str(entry.get("character") or "").strip()
        if not name:
            raise RosterError(f"{source}: cast entry #{position} has no 'character'")
        if name in seen:
            raise RosterError(f"{source}: duplicate cast entry for {name!r}")
        seen.add(name)

        prompt = str(entry.get("voicePrompt") or "").strip()
        voices = entry.get("voices")
        voxalta: tuple[str, ...] = ()
        if isinstance(voices, dict):
            raw = voices.get("voxalta")
            if isinstance(raw, str):
                voxalta = (raw,)
            elif isinstance(raw, list):
                voxalta = tuple(str(p) for p in raw)

        members.append(
            CastMember(character=name, voice_prompt=prompt, voxalta_paths=voxalta)
        )

    return tuple(members)


def load_roster(project_dir: str | Path) -> Roster:
    """Read ``<project_dir>/CAST.md`` and build the roster (read-only)."""
    project = Path(project_dir).expanduser()
    if not project.is_dir():
        raise RosterError(f"project directory not found: {project}")

    cast_file = project / CAST_FILENAME
    if not cast_file.is_file():
        raise RosterError(f"no {CAST_FILENAME} in {project}")

    raw = cast_file.read_bytes()
    members = parse_cast_markdown(raw.decode("utf-8"), source=str(cast_file))
    return Roster(
        project_dir=project,
        cast_file=cast_file,
        cast_sha256=hashlib.sha256(raw).hexdigest(),
        members=members,
    )
