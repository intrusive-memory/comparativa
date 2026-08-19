"""Round-2 clone smoke tests — real checkpoints, real ``.vox`` references.

Skipped by default; see ``tests/conftest.py``. To run::

    HF_HUB_OFFLINE=1 uv run pytest tests/smoke --smoke -s

Each test clones one line in a frozen-corpus voice. The qwen3 test needs the
1.7B **Base** checkpoint in the local HF cache (the 0.6B Base is knowingly
absent and its engine is not smoked here); the chatterbox tests reuse the
round-1 checkpoints. Every test additionally needs the frozen corpus for its
``.vox`` reference and skips without it.
"""

from __future__ import annotations

import os

import pytest

from comparativa.generation.engines import EngineError, LineRequest, load_engine, spec
from comparativa.generation.smoke import SMOKE_SEED, SMOKE_TEXT
from comparativa.generation.truncation import expected_duration
from comparativa.voices.vox import VoxBundle

from conftest import CORPUS_ROOT

pytestmark = pytest.mark.smoke

ARCHER_VOX = CORPUS_ROOT / "voices" / "ARCHER.vox"

requires_corpus = pytest.mark.skipif(
    not ARCHER_VOX.is_file(),
    reason="mission corpus not available (see docs/CORPUS_PIN.md)",
)


@pytest.fixture(scope="module")
def offline_env():
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        pytest.skip(
            "set HF_HUB_OFFLINE=1 so the smoke run cannot download a checkpoint"
        )
    return True


@pytest.fixture(scope="module")
def archer_clone():
    from comparativa.generation.engines import CloneVoice

    bundle = VoxBundle.open(ARCHER_VOX)
    reference = bundle.reference_audio("1.7b")
    return CloneVoice(
        name="vox:ARCHER",
        audio=reference.audio,
        sample_rate=reference.sample_rate,
        ref_text=bundle.clone_prompt("1.7b").ref_text,
        source=f"{ARCHER_VOX}#{reference.member}",
    )


def _load(key: str):
    try:
        return load_engine(key)
    except EngineError as exc:
        pytest.skip(f"{key}: checkpoint not in the local cache ({exc})")


@requires_corpus
@pytest.mark.parametrize(
    "key",
    ("qwen3-1.7b-clone", "chatterbox", "chatterbox-turbo", "dia", "csm", "higgs"),
)
def test_engine_clones_one_line(key, offline_env, archer_clone):
    engine = _load(key)
    result = engine.generate_line(
        LineRequest(text=SMOKE_TEXT, clone=archer_clone, seed=SMOKE_SEED)
    )

    assert result.voice == "vox:ARCHER"
    assert result.checkpoint == spec(key).checkpoint
    expected = expected_duration(SMOKE_TEXT)
    assert 0.5 * expected <= result.duration_seconds <= 4.0 * expected, (
        f"{key}: {result.duration_seconds:.2f}s for an expected ~{expected:.2f}s"
    )
    assert result.chunks, "no chunk records were written"
    assert result.sampling == spec(key).sampling.to_dict()

    print(
        f"\n{key}: cloned {result.duration_seconds:.2f}s as vox:ARCHER, "
        f"rtf {result.real_time_factor:.2f}"
    )


@requires_corpus
def test_chatterbox_clone_reuses_prepared_conditionals(offline_env, archer_clone):
    """Two lines, one voice: the reference must be encoded exactly once."""
    engine = _load("chatterbox")
    engine.generate_line(LineRequest(text="First line.", clone=archer_clone, seed=1))
    assert set(engine._conds_cache) == {"vox:ARCHER"}
    first = engine._conds_cache["vox:ARCHER"]
    engine.generate_line(LineRequest(text="Second line.", clone=archer_clone, seed=2))
    assert engine._conds_cache["vox:ARCHER"] is first
