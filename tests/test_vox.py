"""Round-2 tests: the ``.vox`` bundle reader (``comparativa.voices.vox``).

The synthetic fixture reproduces the exact serialization the Swift side writes
(``VoxExporter.swift`` member layout, ``VoiceClonePrompt.serialize()`` binary
framing), so the reader is proven against the documented format without
needing the corpus. The corpus-pinned tests then hold the reader against a
real production bundle.
"""

from __future__ import annotations

import io
import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from comparativa.voices.vox import (
    MODEL_SIZES,
    VoxBundle,
    VoxError,
    find_vox,
    parse_clone_prompt_header,
)

from conftest import CORPUS_ROOT

requires_granville_vox = pytest.mark.skipif(
    not (CORPUS_ROOT / "voices" / "ARCHER.vox").is_file(),
    reason="mission corpus not available (see docs/CORPUS_PIN.md)",
)


# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------


def make_clone_prompt_bin(
    ref_text: str = "Hello, this is a reference.",
    language: str = "english",
    *,
    has_embedding: bool = True,
) -> bytes:
    """Bytes in the exact ``VoiceClonePrompt.serialize()`` framing."""
    ref_codes = b"\x00" * 64  # stands in for the refCodes safetensors payload
    speaker = b"\x01" * 32 if has_embedding else b""
    meta = json.dumps(
        {
            "hasEmbedding": has_embedding,
            "language": language,
            "refCodesSize": len(ref_codes),
            "refText": ref_text,
            "speakerDataSize": len(speaker),
        },
        sort_keys=True,
    ).encode("utf-8")
    return struct.pack("<I", len(meta)) + meta + ref_codes + speaker


def make_vox(
    path: Path,
    *,
    name: str = "TESTY",
    sizes: tuple[str, ...] = MODEL_SIZES,
    ref_text: str = "Hello, this is a reference.",
    sample_rate: int = 24000,
    seconds: float = 6.0,
) -> Path:
    """Write a structurally faithful ``.vox`` bundle for tests."""
    tone = (0.1 * np.sin(np.linspace(0, 440 * 2 * np.pi * seconds, int(sample_rate * seconds)))).astype(
        np.float32
    )
    wav = io.BytesIO()
    sf.write(wav, tone, sample_rate, format="WAV", subtype="PCM_16")

    manifest = {
        "created": "2026-08-09T12:52:49Z",
        "embeddings": {},
        "id": "00000000-0000-0000-0000-000000000000",
        "provenance": {"engine": "qwen3-tts", "method": "synthesized"},
        "voice": {"name": name, "description": "A synthetic test voice."},
        "vox_version": "0.4.0",
    }
    for size in sizes:
        slug = size.replace(".", "")
        manifest["embeddings"][f"qwen3-tts-{size}-clone-prompt"] = {
            "engine": "qwen3-tts",
            "file": f"embeddings/qwen3-tts/{size}/clone-prompt.bin",
            "format": "bin",
            "model": f"mlx-community/Qwen3-TTS-12Hz-{slug.upper()}-Base-bf16",
        }

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for size in sizes:
            archive.writestr(
                f"embeddings/qwen3-tts/{size}/sample-audio.wav", wav.getvalue()
            )
            archive.writestr(
                f"embeddings/qwen3-tts/{size}/clone-prompt.bin",
                make_clone_prompt_bin(ref_text),
            )
    return path


# ---------------------------------------------------------------------------
# clone-prompt.bin framing
# ---------------------------------------------------------------------------


def test_clone_prompt_header_round_trips():
    meta = parse_clone_prompt_header(make_clone_prompt_bin("Say the line.", "english"))
    assert meta.ref_text == "Say the line."
    assert meta.language == "english"
    assert meta.has_embedding is True
    assert meta.ref_codes_bytes == 64
    assert meta.speaker_bytes == 32


def test_clone_prompt_header_rejects_truncation_and_garbage():
    with pytest.raises(VoxError, match="shorter"):
        parse_clone_prompt_header(b"\x00")
    with pytest.raises(VoxError, match="exceeds"):
        parse_clone_prompt_header(struct.pack("<I", 999) + b"{}")
    with pytest.raises(VoxError, match="not valid JSON"):
        parse_clone_prompt_header(struct.pack("<I", 4) + b"nope")


# ---------------------------------------------------------------------------
# Bundle reading
# ---------------------------------------------------------------------------


def test_bundle_exposes_manifest_sizes_and_reference_audio(tmp_path):
    vox = make_vox(tmp_path / "TESTY.vox", seconds=6.0)
    bundle = VoxBundle.open(vox)

    assert bundle.voice_name == "TESTY"
    assert bundle.voice_description.startswith("A synthetic")
    assert bundle.vox_version == "0.4.0"
    assert bundle.model_sizes() == MODEL_SIZES
    assert len(bundle.sha256()) == 64

    entry = bundle.entry("1.7b")
    assert entry.sample_member == "embeddings/qwen3-tts/1.7b/sample-audio.wav"
    assert entry.checkpoint == "mlx-community/Qwen3-TTS-12Hz-17B-Base-bf16"

    reference = bundle.reference_audio("1.7b")
    assert reference.sample_rate == 24000
    assert reference.audio.dtype == np.float32
    assert reference.audio.ndim == 1
    assert reference.seconds == pytest.approx(6.0, abs=0.01)

    meta = bundle.clone_prompt("0.6b")
    assert meta.ref_text == "Hello, this is a reference."


def test_bundle_missing_size_is_a_clear_error(tmp_path):
    vox = make_vox(tmp_path / "ONESIZE.vox", sizes=("1.7b",))
    bundle = VoxBundle.open(vox)
    assert bundle.model_sizes() == ("1.7b",)
    with pytest.raises(VoxError, match="0.6b"):
        bundle.entry("0.6b")


def test_open_rejects_missing_and_non_zip_files(tmp_path):
    with pytest.raises(VoxError, match="not found"):
        VoxBundle.open(tmp_path / "ghost.vox")
    junk = tmp_path / "junk.vox"
    junk.write_bytes(b"this is not a zip")
    with pytest.raises(VoxError, match="not a zip"):
        VoxBundle.open(junk)


def test_find_vox_prefers_cast_paths_then_scans_case_insensitively(tmp_path):
    (tmp_path / "voices").mkdir()
    listed = make_vox(tmp_path / "voices" / "ARCHER.vox")
    lowercase = make_vox(tmp_path / "voices" / "narrator.vox")

    assert find_vox(tmp_path, "ARCHER", "voices/ARCHER.vox") == listed
    # CAST.md path wrong -> falls through to the case-insensitive scan.
    assert find_vox(tmp_path, "NARRATOR", "voices/NARRATOR_typo.vox") == lowercase
    # Spaces become underscores, as the exporter writes them.
    made = make_vox(tmp_path / "voices" / "PREDATOR_MOM_1.vox")
    assert find_vox(tmp_path, "PREDATOR MOM 1") == made
    assert find_vox(tmp_path, "NOBODY") is None


# ---------------------------------------------------------------------------
# Corpus-pinned reality checks
# ---------------------------------------------------------------------------


@requires_granville_vox
def test_archer_vox_reads_with_both_sizes_and_a_transcript():
    bundle = VoxBundle.open(CORPUS_ROOT / "voices" / "ARCHER.vox")
    assert bundle.voice_name == "ARCHER"
    assert set(bundle.model_sizes()) == {"1.7b", "0.6b"}
    for size in bundle.model_sizes():
        meta = bundle.clone_prompt(size)
        assert meta.ref_text.strip(), f"{size}: empty refText"
        reference = bundle.reference_audio(size)
        assert reference.sample_rate == 24000
        # chatterbox-turbo needs > 5 s of reference; production bundles have it.
        assert reference.seconds > 5.0


@requires_granville_vox
def test_every_frozen_corpus_vox_is_readable_and_cloneable():
    voices_dir = CORPUS_ROOT / "voices"
    bundles = sorted(voices_dir.glob("*.vox"))
    assert bundles, "frozen corpus has no .vox files"
    for path in bundles:
        bundle = VoxBundle.open(path)
        assert "1.7b" in bundle.model_sizes(), f"{path.name}: no 1.7b entry"
        assert bundle.clone_prompt("1.7b").ref_text.strip(), f"{path.name}: no refText"
