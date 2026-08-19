"""Cloned-voice assignment: ``CAST.md`` + ``.vox`` bundles → ``presets-cloned.yaml``.

Round 2 of the comparison. Round 1 (RD-2) gave every Python condition its
engine's built-in voices; this module builds the **cloned** counterpart, where
every engine that can condition on a reference clip speaks in the character's
production voice, taken from the same ``.vox`` bundles the Swift stack uses:

* ``qwen3-1.7b-clone`` / ``qwen3-0.6b-clone`` — Base-checkpoint ICL cloning
  from the bundle's size-matched ``sample-audio.wav`` plus the transcript in
  its ``clone-prompt.bin`` header. This is the same clone mechanism SwiftVoxAlta
  applies, which is what makes condition **B** (Python qwen3, ``.vox``-cloned)
  the clean counterpart to condition A.
* ``chatterbox`` / ``chatterbox-turbo`` — reference-audio conditioning from the
  1.7b sample clip (6+ seconds, satisfying turbo's >5 s requirement). No
  transcript is needed.
* ``soprano`` — cannot clone (its one voice is baked in); every character keeps
  the default voice, recorded as such.

A character without a usable ``.vox`` falls back per engine: the chatterbox
family and soprano fall back to their built-in voice; the qwen3 clone engines
have **no** fallback (the Base checkpoints expose no preset speakers), so the
character is recorded unresolved and planning an episode that voices them
fails loudly rather than silently recasting.

The document (``schema_version: 2``, ``mode: cloned``) is deterministic: byte
identical when regenerated from the same ``CAST.md`` and ``.vox`` files, so a
stale ``presets-cloned.yaml`` shows up as a real diff.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from .roster import Roster
from .vox import VoxBundle, VoxError, find_vox

if TYPE_CHECKING:  # imported lazily at runtime to avoid a package cycle
    from ..generation.engines import CloneVoice

SCHEMA_VERSION_CLONED: Final = 2
PRESETS_CLONED_FILENAME: Final = "presets-cloned.yaml"

#: Which ``.vox`` model-size entry each clone-capable engine conditions on.
#: The chatterbox family just needs a good reference clip; the 1.7b sample is
#: the higher-fidelity render of the same voice, so both sizes use it.
CLONE_REF_MODEL_SIZE: Final[dict[str, str]] = {
    "qwen3-1.7b-clone": "1.7b",
    "qwen3-0.6b-clone": "0.6b",
    "chatterbox": "1.7b",
    "chatterbox-turbo": "1.7b",
    "dia": "1.7b",
    "csm": "1.7b",
    "higgs": "1.7b",
}

#: Engines covered by a cloned document, in matrix order.
CLONED_ENGINE_KEYS: Final[tuple[str, ...]] = (
    "qwen3-1.7b-clone",
    "qwen3-0.6b-clone",
    "chatterbox",
    "chatterbox-turbo",
    "dia",
    "csm",
    "higgs",
    "soprano",
)

#: Engines whose clone path needs the reference transcript.
_NEEDS_REF_TEXT: Final = frozenset(
    {"qwen3-1.7b-clone", "qwen3-0.6b-clone", "dia", "csm", "higgs"}
)


class ClonedVoicesError(RuntimeError):
    """A cloned-voice document cannot be built or resolved."""


def _engine_spec(key: str):
    from ..generation.engines import spec

    return spec(key)


def is_cloned_document(document: dict[str, Any]) -> bool:
    """True when a presets document is a round-2 cloned one (schema 2)."""
    return int(document.get("schema_version") or 1) >= SCHEMA_VERSION_CLONED


# ---------------------------------------------------------------------------
# Building the document
# ---------------------------------------------------------------------------


def _home_relative(path: Path) -> str:
    try:
        return "~/" + str(path.resolve().relative_to(Path.home()))
    except ValueError:
        return str(path)


def _clone_entry(
    bundle: VoxBundle, vox_rel: str, engine_key: str
) -> tuple[dict[str, Any] | None, str | None]:
    """One character × engine clone entry, or ``(None, reason)``.

    When the preferred model size is missing (older exports carry only 1.7b),
    another size's reference is substituted — the sample is just audio plus a
    transcript, usable by any clone engine — and the substitution is noted.
    """
    preferred = CLONE_REF_MODEL_SIZE[engine_key]
    available = bundle.model_sizes()
    if not available:
        return None, f"{bundle.path.name} has no qwen3-tts entries at all"
    size = preferred if preferred in available else available[0]
    substituted = size != preferred
    meta = bundle.clone_prompt(size)
    reference = bundle.reference_audio(size)
    if engine_key in _NEEDS_REF_TEXT and not meta.ref_text.strip():
        return None, (
            f"{bundle.path.name} qwen3-tts/{size} clone-prompt has an empty "
            "refText; ICL cloning needs the transcript"
        )
    entry: dict[str, Any] = {
        "mode": "clone",
        "vox": vox_rel,
        "vox_sha256": bundle.sha256(),
        "model_size": size,
        "member": reference.member,
        "ref_seconds": round(reference.seconds, 3),
        "ref_sample_rate": reference.sample_rate,
        "ref_text": meta.ref_text,
        "language": meta.language,
    }
    if substituted:
        entry["size_substituted"] = True
    return entry, None


def build_cloned_document(
    roster: Roster, engine_keys: tuple[str, ...] = CLONED_ENGINE_KEYS
) -> dict[str, Any]:
    """Build the ``presets-cloned.yaml`` document for a project's cast."""
    from ..generation.engines import ENGINE_SPECS

    unknown = [k for k in engine_keys if k not in ENGINE_SPECS]
    if unknown:
        raise ClonedVoicesError(f"unknown engine key(s): {', '.join(unknown)}")

    doc: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_CLONED,
        "mode": "cloned",
        "generated_by": "comparativa voices --mode cloned --write",
        "policy": (
            "Round 2 is cloned voices: every clone-capable engine conditions on the "
            "character's production .vox reference (the same bundles the Swift stack "
            "uses), qwen3 via Base-checkpoint ICL (ref audio + clone-prompt refText), "
            "the chatterbox family via reference audio alone. soprano cannot clone and "
            "keeps its single built-in voice."
        ),
        "source": {
            "project_dir": _home_relative(roster.project_dir),
            "cast_file": roster.cast_file.name,
            "cast_sha256": roster.cast_sha256,
            "cast_size": len(roster),
        },
        "engines": {},
        "assignments": {},
    }

    for key in engine_keys:
        engine_spec = _engine_spec(key)
        entry: dict[str, Any] = {
            "checkpoint": engine_spec.checkpoint,
            "clone_voices": engine_spec.clone_voices,
        }
        if engine_spec.clone_voices:
            entry["clone_source"] = (
                f".vox embeddings/qwen3-tts/{CLONE_REF_MODEL_SIZE[key]} "
                "(sample-audio.wav" + (
                    " + clone-prompt.bin refText)" if key in _NEEDS_REF_TEXT else ")"
                )
            )
        else:
            entry["note"] = "cannot clone; single built-in voice for every character"
        doc["engines"][key] = entry

    for member in roster.members:
        record: dict[str, Any] = {"engines": {}}
        notes: list[str] = []

        vox_path = find_vox(roster.project_dir, member.character, *member.voxalta_paths)
        bundle: VoxBundle | None = None
        if vox_path is not None:
            try:
                bundle = VoxBundle.open(vox_path)
            except VoxError as exc:
                notes.append(f"unreadable .vox ignored: {exc}")
        else:
            notes.append("no .vox bundle found for this character")

        vox_rel: str | None = None
        if vox_path is not None and bundle is not None:
            try:
                vox_rel = vox_path.resolve().relative_to(
                    roster.project_dir.resolve()
                ).as_posix()
            except ValueError:
                vox_rel = str(vox_path)
            record["vox"] = vox_rel

        for key in engine_keys:
            engine_spec = _engine_spec(key)
            if not engine_spec.clone_voices:
                record["engines"][key] = {"mode": "default", "voice": "default"}
                continue
            if bundle is None or vox_rel is None:
                if key in _NEEDS_REF_TEXT:
                    record["engines"][key] = None
                    notes.append(
                        f"{key}: unresolved -- no .vox and the Base checkpoint "
                        "has no preset voices to fall back to"
                    )
                else:
                    record["engines"][key] = {"mode": "default", "voice": "default"}
                    notes.append(f"{key}: no .vox; fell back to the built-in voice")
                continue

            entry, reason = _clone_entry(bundle, vox_rel, key)
            if entry is not None and entry.get("size_substituted"):
                notes.append(
                    f"{key}: no {CLONE_REF_MODEL_SIZE[key]} entry in "
                    f"{Path(vox_rel).name}; cloned from the "
                    f"{entry['model_size']} reference instead"
                )
            if entry is not None and key == "chatterbox-turbo" and entry["ref_seconds"] <= 5.0:
                notes.append(
                    f"{key}: reference is {entry['ref_seconds']}s, under the "
                    "documented >5s guidance (chatterbox_turbo.py "
                    "prepare_conditionals); cloned anyway, listen for drift"
                )
            if entry is None:
                if key in _NEEDS_REF_TEXT:
                    record["engines"][key] = None
                    notes.append(f"{key}: unresolved -- {reason}")
                else:
                    record["engines"][key] = {"mode": "default", "voice": "default"}
                    notes.append(f"{key}: {reason}; fell back to the built-in voice")
                continue
            record["engines"][key] = entry

        if notes:
            record["notes"] = notes
        doc["assignments"][member.character] = record

    return doc


def unresolved_characters(document: dict[str, Any]) -> dict[str, list[str]]:
    """Character → engines with a ``None`` entry (no voice possible)."""
    problems: dict[str, list[str]] = {}
    for character, record in (document.get("assignments") or {}).items():
        engines = record.get("engines") or {}
        missing = sorted(key for key, entry in engines.items() if entry is None)
        if missing:
            problems[character] = missing
    return problems


# ---------------------------------------------------------------------------
# Resolving at generation time
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedVoice:
    """One character × engine voice decision, ready for the engine layer."""

    #: ``"preset"`` | ``"default"`` | ``"clone"``.
    mode: str
    #: Preset/default voice name; ``None`` for clone mode.
    voice: str | None = None
    #: Loaded reference voice for clone mode.
    clone: "CloneVoice | None" = None
    #: Manifest-ready provenance (vox path, sha, member, ref seconds, ...).
    provenance: dict[str, Any] | None = None


#: ``.vox`` bundles opened during resolution, keyed by resolved path.
_BUNDLE_CACHE: dict[Path, VoxBundle] = {}


def _open_bundle(path: Path) -> VoxBundle:
    resolved = path.resolve()
    bundle = _BUNDLE_CACHE.get(resolved)
    if bundle is None:
        bundle = VoxBundle.open(resolved)
        _BUNDLE_CACHE[resolved] = bundle
    return bundle


def document_project_dir(document: dict[str, Any]) -> Path | None:
    """The project directory a cloned document's ``vox`` paths are relative to."""
    raw = (document.get("source") or {}).get("project_dir")
    return Path(str(raw)).expanduser() if raw else None


def resolve_voice_entry(
    document: dict[str, Any],
    character: str,
    engine_key: str,
    *,
    project_dir: str | Path | None = None,
) -> ResolvedVoice | None:
    """Resolve one character's voice for one engine, either schema.

    Schema 1 (round 1) entries are plain preset names; schema 2 entries are
    mode dicts. Returns ``None`` when the document assigns nothing — the
    caller decides whether that is fatal (it is for preset and clone engines).
    """
    record = (document.get("assignments") or {}).get(character)
    if not record:
        return None
    entry = (record.get("engines") or {}).get(engine_key)
    if entry is None:
        return None

    if isinstance(entry, str):  # schema 1: a bare preset name
        return ResolvedVoice(mode="preset", voice=entry)
    if not isinstance(entry, dict):
        raise ClonedVoicesError(
            f"{character}/{engine_key}: unrecognized presets entry {entry!r}"
        )

    mode = str(entry.get("mode") or "preset")
    if mode in ("preset", "default"):
        return ResolvedVoice(mode=mode, voice=entry.get("voice"))
    if mode != "clone":
        raise ClonedVoicesError(f"{character}/{engine_key}: unknown mode {mode!r}")

    from ..generation.engines import CloneVoice

    base = Path(project_dir).expanduser() if project_dir else document_project_dir(document)
    vox_value = Path(str(entry.get("vox") or ""))
    vox_path = vox_value if vox_value.is_absolute() else (base or Path(".")) / vox_value
    if not vox_path.is_file():
        raise ClonedVoicesError(
            f"{character}/{engine_key}: .vox not found at {vox_path} "
            "(regenerate presets-cloned.yaml or pass the right project dir)"
        )

    bundle = _open_bundle(vox_path)
    size = str(entry.get("model_size") or CLONE_REF_MODEL_SIZE.get(engine_key, "1.7b"))
    reference = bundle.reference_audio(size)
    ref_text = entry.get("ref_text")
    if ref_text is None and engine_key in _NEEDS_REF_TEXT:
        ref_text = bundle.clone_prompt(size).ref_text

    vox_sha = bundle.sha256()
    recorded_sha = entry.get("vox_sha256")
    provenance: dict[str, Any] = {
        "mode": "clone",
        "vox": str(vox_path),
        "vox_sha256": vox_sha,
        "model_size": size,
        "member": reference.member,
        "ref_seconds": round(reference.seconds, 3),
        "ref_sample_rate": reference.sample_rate,
        "ref_text": ref_text,
    }
    if recorded_sha and recorded_sha != vox_sha:
        provenance["vox_sha256_expected"] = recorded_sha
        provenance["stale"] = True

    clone = CloneVoice(
        name=f"vox:{character}",
        audio=reference.audio,
        sample_rate=reference.sample_rate,
        ref_text=str(ref_text) if ref_text is not None else None,
        source=f"{vox_path}#{reference.member}",
    )
    return ResolvedVoice(mode="clone", clone=clone, provenance=provenance)


__all__ = [
    "CLONED_ENGINE_KEYS",
    "CLONE_REF_MODEL_SIZE",
    "PRESETS_CLONED_FILENAME",
    "SCHEMA_VERSION_CLONED",
    "ClonedVoicesError",
    "ResolvedVoice",
    "build_cloned_document",
    "document_project_dir",
    "is_cloned_document",
    "resolve_voice_entry",
    "unresolved_characters",
]
