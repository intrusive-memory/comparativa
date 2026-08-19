"""Read SwiftVoxAlta ``.vox`` voice bundles (round 2: voice cloning).

A ``.vox`` file is a **zip archive** written by SwiftVoxAlta's ``VoxExporter``
(``SwiftVoxAlta/Sources/SwiftVoxAlta/VoxExporter.swift``) and produced for a
cast by the ``echada cast`` command. Round 1 recorded these paths and never
opened them (RD-2); this module is the round-2 reader. Everything here is
**read-only**: a ``.vox`` is never written, moved, or modified.

Bundle layout (vox_version 0.4.0, one entry per Qwen3 model size)::

    manifest.json
    embeddings/qwen3-tts/1.7b/sample-audio.wav    # engine-generated reference clip
    embeddings/qwen3-tts/1.7b/clone-prompt.bin    # serialized VoiceClonePrompt
    embeddings/qwen3-tts/0.6b/sample-audio.wav
    embeddings/qwen3-tts/0.6b/clone-prompt.bin

``clone-prompt.bin`` is ``VoiceClonePrompt.serialize()`` output
(``mlx-audio-swift/Sources/MLXAudioTTS/Models/Qwen3TTS/
Qwen3TTSVoiceClonePrompt.swift``)::

    [4 bytes little-endian metadata length]
    [JSON metadata: refText, language, hasEmbedding, refCodesSize, speakerDataSize]
    [refCodes safetensors]
    [speaker-embedding safetensors, when hasEmbedding]

The JSON header carries ``refText`` — the transcript of ``sample-audio.wav`` —
which is exactly what Qwen3 ICL voice cloning needs on the Python side
(``mlx_audio`` requires ``ref_audio`` *and* ``ref_text``). The safetensors
payloads are Swift-side caches of the encoded reference; Python mlx-audio
re-encodes the reference audio itself, so only the header is parsed here.

Per-language variants (``embeddings/qwen3-tts/<size>/<lang>/...``) exist in
newer exports; :meth:`VoxBundle.entry` returns the language-less default unless
a language is requested.
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

#: The Qwen3 model sizes a ``.vox`` may carry embeddings for.
MODEL_SIZES: Final[tuple[str, ...]] = ("1.7b", "0.6b")

#: Zip member templates (VoxExporter.swift: clonePromptPath / sample paths).
_MEMBER_ROOT: Final = "embeddings/qwen3-tts"


class VoxError(RuntimeError):
    """A ``.vox`` bundle is missing, malformed, or lacks a requested entry."""


@dataclass(frozen=True)
class ClonePromptMeta:
    """The JSON header of a ``clone-prompt.bin`` (transcript + provenance)."""

    #: Transcript of the reference audio — qwen3 ICL cloning's ``ref_text``.
    ref_text: str
    #: Language code the reference was encoded with (e.g. ``"english"``).
    language: str
    has_embedding: bool
    ref_codes_bytes: int
    speaker_bytes: int


@dataclass(frozen=True)
class VoxEntry:
    """One model size's cloning material inside a bundle."""

    model_size: str
    sample_member: str
    clone_prompt_member: str
    #: From the bundle manifest's matching ``embeddings`` record, when present.
    checkpoint: str | None = None


@dataclass(frozen=True)
class ReferenceAudio:
    """A decoded reference clip, ready to hand to an engine."""

    #: 1-D float32 mono waveform.
    audio: np.ndarray
    sample_rate: int
    member: str

    @property
    def seconds(self) -> float:
        return len(self.audio) / self.sample_rate if self.sample_rate else 0.0


def parse_clone_prompt_header(data: bytes) -> ClonePromptMeta:
    """Parse the JSON header of ``VoiceClonePrompt.serialize()`` output."""
    if len(data) < 4:
        raise VoxError("clone-prompt.bin is shorter than its 4-byte length header")
    (meta_len,) = struct.unpack_from("<I", data, 0)
    if len(data) < 4 + meta_len:
        raise VoxError(
            f"clone-prompt.bin metadata length {meta_len} exceeds file size {len(data)}"
        )
    try:
        meta = json.loads(data[4 : 4 + meta_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VoxError(f"clone-prompt.bin metadata is not valid JSON: {exc}") from exc

    try:
        return ClonePromptMeta(
            ref_text=str(meta["refText"]),
            language=str(meta["language"]),
            has_embedding=bool(meta["hasEmbedding"]),
            ref_codes_bytes=int(meta["refCodesSize"]),
            speaker_bytes=int(meta.get("speakerDataSize") or 0),
        )
    except KeyError as exc:
        raise VoxError(f"clone-prompt.bin metadata lacks required key {exc}") from exc


class VoxBundle:
    """A read-only view of one ``.vox`` file.

    Construct with :meth:`open`; the archive is read into memory once (bundles
    are well under a megabyte) so the source file is never held open.
    """

    def __init__(self, path: Path, manifest: dict[str, Any], members: dict[str, bytes]):
        self.path = path
        self.manifest = manifest
        self._members = members

    # -- construction ---------------------------------------------------------

    @classmethod
    def open(cls, path: str | Path) -> "VoxBundle":
        """Open a ``.vox`` bundle, validating its zip structure and manifest."""
        vox_path = Path(path).expanduser()
        if not vox_path.is_file():
            raise VoxError(f".vox not found: {vox_path}")
        try:
            with zipfile.ZipFile(vox_path) as archive:
                members = {name: archive.read(name) for name in archive.namelist()}
        except zipfile.BadZipFile as exc:
            raise VoxError(f"{vox_path} is not a zip archive: {exc}") from exc

        raw_manifest = members.get("manifest.json")
        if raw_manifest is None:
            raise VoxError(f"{vox_path} has no manifest.json")
        try:
            manifest = json.loads(raw_manifest.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VoxError(f"{vox_path}: manifest.json is not valid JSON: {exc}") from exc

        return cls(vox_path, manifest, members)

    # -- manifest accessors ----------------------------------------------------

    @property
    def voice_name(self) -> str:
        return str((self.manifest.get("voice") or {}).get("name") or self.path.stem)

    @property
    def voice_description(self) -> str:
        return str((self.manifest.get("voice") or {}).get("description") or "")

    @property
    def vox_version(self) -> str:
        return str(self.manifest.get("vox_version") or "")

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self.manifest.get("provenance") or {})

    def sha256(self) -> str:
        """Digest of the bundle file, for run provenance."""
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    # -- entries ---------------------------------------------------------------

    def model_sizes(self) -> tuple[str, ...]:
        """The model sizes this bundle has *both* a sample and a prompt for."""
        return tuple(size for size in MODEL_SIZES if self._has(size))

    def _member_names(self, model_size: str, language: str | None) -> tuple[str, str]:
        base = f"{_MEMBER_ROOT}/{model_size}"
        if language:
            base = f"{base}/{language}"
        return f"{base}/sample-audio.wav", f"{base}/clone-prompt.bin"

    def _has(self, model_size: str, language: str | None = None) -> bool:
        sample, prompt = self._member_names(model_size, language)
        return sample in self._members and prompt in self._members

    def entry(self, model_size: str, *, language: str | None = None) -> VoxEntry:
        """The cloning material for one model size (language-less default)."""
        sample, prompt = self._member_names(model_size, language)
        if sample not in self._members or prompt not in self._members:
            raise VoxError(
                f"{self.path.name} has no qwen3-tts/{model_size}"
                f"{'/' + language if language else ''} entry "
                f"(members: {', '.join(sorted(self._members))})"
            )
        checkpoint = None
        for record in (self.manifest.get("embeddings") or {}).values():
            if isinstance(record, dict) and record.get("file", "").replace("\\", "") == prompt:
                checkpoint = record.get("model")
                break
        return VoxEntry(
            model_size=model_size,
            sample_member=sample,
            clone_prompt_member=prompt,
            checkpoint=checkpoint,
        )

    def clone_prompt(self, model_size: str, *, language: str | None = None) -> ClonePromptMeta:
        """Parse the clone-prompt header (``refText``, language) for one size."""
        entry = self.entry(model_size, language=language)
        return parse_clone_prompt_header(self._members[entry.clone_prompt_member])

    def reference_audio(
        self, model_size: str, *, language: str | None = None
    ) -> ReferenceAudio:
        """Decode one size's ``sample-audio.wav`` into a mono float32 waveform."""
        import soundfile as sf

        entry = self.entry(model_size, language=language)
        data, sample_rate = sf.read(
            io.BytesIO(self._members[entry.sample_member]), dtype="float32"
        )
        audio = np.asarray(data, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return ReferenceAudio(
            audio=audio, sample_rate=int(sample_rate), member=entry.sample_member
        )


def find_vox(project_dir: str | Path, character: str, *paths: str) -> Path | None:
    """Locate a character's ``.vox`` under a project directory, read-only.

    ``paths`` are the project-relative ``voices.voxalta`` entries from
    ``CAST.md`` and are tried verbatim first. When none resolves, the project's
    ``voices/`` directory is scanned case-insensitively (the corpus has
    ``narrator.vox`` for the ``NARRATOR`` cue, and a case-sensitive filesystem
    must find it too).
    """
    project = Path(project_dir).expanduser()
    for rel in paths:
        candidate = project / rel
        if candidate.is_file():
            return candidate

    wanted = f"{character.replace(' ', '_')}.vox".lower()
    voices_dir = project / "voices"
    if voices_dir.is_dir():
        for candidate in sorted(voices_dir.iterdir()):
            if candidate.name.lower() == wanted and candidate.is_file():
                return candidate
    return None


__all__ = [
    "MODEL_SIZES",
    "ClonePromptMeta",
    "ReferenceAudio",
    "VoxBundle",
    "VoxEntry",
    "VoxError",
    "find_vox",
    "parse_clone_prompt_header",
]
