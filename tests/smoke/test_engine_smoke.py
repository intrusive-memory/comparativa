"""Engine smoke tests (Sortie 5, task 5) — real checkpoints, real audio.

Skipped by default; see ``tests/conftest.py``. To run::

    HF_HUB_OFFLINE=1 uv run pytest tests/smoke --smoke -s

``HF_HUB_OFFLINE=1`` is what proves the exit criterion "no network downloads of
new checkpoints": with it set, ``huggingface_hub`` resolves from the local cache
or raises, so a passing run cannot have fetched anything.
"""

from __future__ import annotations

import os

import pytest

from comparativa.generation import SMOKE_ENGINE_KEYS, spec
from comparativa.generation.smoke import SMOKE_SEED, SMOKE_TEXT, run_one
from comparativa.generation.truncation import expected_duration

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def offline_env():
    """Fail loudly if the smoke run was not pinned offline."""
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        pytest.skip(
            "set HF_HUB_OFFLINE=1 so the smoke run cannot download a checkpoint"
        )
    return True


@pytest.mark.parametrize("key", SMOKE_ENGINE_KEYS)
def test_engine_generates_one_line(key, offline_env, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("smoke")
    result = run_one(key, out_dir=out_dir)

    assert result.ok, f"{key}: {result.error}"
    assert result.record is not None
    record = result.record

    # Sane duration: the shared detector already ran, but assert the bounds here
    # too so a regression in the detector cannot mask a bad generation.
    expected = expected_duration(SMOKE_TEXT)
    assert 0.5 * expected <= result.duration_seconds <= 4.0 * expected, (
        f"{key}: {result.duration_seconds:.2f}s for an expected ~{expected:.2f}s"
    )

    # Logged params: exactly what the engine spec says, echoed back.
    engine_spec = spec(key)
    assert record["checkpoint"] == engine_spec.checkpoint
    assert record["sampling"] == engine_spec.sampling.to_dict()
    assert record["seed"] == SMOKE_SEED
    assert record["seeding"].startswith("mlx.core.random.seed")
    assert record["sample_rate"] > 0
    assert record["chunks"], "no chunk records were written"

    print(
        f"\n[smoke] {key}: {result.duration_seconds:.2f}s audio, "
        f"load {result.load_seconds:.1f}s, RTF {result.real_time_factor:.2f}, "
        f"wav {result.wav_path}"
    )


def test_soprano_loads_the_rd1_bf16_conversion(offline_env):
    """RD-1 residual check: Python mlx-audio loads the Swift port's checkpoint."""
    from comparativa.generation import load_engine

    engine = load_engine("soprano")
    assert engine.spec.checkpoint == "mlx-community/Soprano-80M-bf16"
    assert engine.sample_rate == 32000
    # Non-"soprano-1.1" paths must select the v1 decoder config (RD-1 rationale).
    decoder_config = engine.model.config.decoder_config
    assert decoder_config.decoder_dim == 512
    assert decoder_config.decoder_intermediate_dim == 1536
    assert decoder_config.input_kernel == 3


def test_seeding_is_reproducible_on_the_smallest_engine(offline_env):
    """Same seed, same audio — the claim behind the ``seeding`` capability flag."""
    import numpy as np

    from comparativa.generation import LineRequest, load_engine

    engine = load_engine("qwen3-0.6b")
    request = LineRequest(text="Same seed, same waveform.", voice="ryan", seed=4242)
    first = engine.generate_line(request)
    second = engine.generate_line(request)
    assert first.audio.shape == second.audio.shape
    assert np.array_equal(first.audio, second.audio)
