"""Generation: one interface over the four MLX TTS engines.

Sortie 5 covers the engine layer only — loading a checkpoint, generating a
single line with fixed and recorded sampling parameters, and detecting a
truncated or runaway generation. Episode assembly, loudness, and the manifest
belong to Sortie 6.

Typical use::

    from comparativa.generation import LineRequest, load_engine

    engine = load_engine("qwen3-1.7b")
    result = engine.generate_line(
        LineRequest(text=line.text, voice="ryan", direction=line.direction, seed=7)
    )
    result.audio           # 1-D float32 at result.sample_rate
    result.to_dict()       # the manifest record for this line
"""

from .engines import (
    ENGINE_KEYS,
    ENGINE_SPECS,
    QWEN3_LANGUAGE,
    SEEDING_MLX_GLOBAL,
    SEEDING_UNAVAILABLE,
    SMOKE_ENGINE_KEYS,
    ChunkRecord,
    Engine,
    EngineError,
    EngineSpec,
    LineRequest,
    LineResult,
    load_engine,
    spec,
)
from .sampling import (
    CHATTERBOX_SAMPLING,
    CHATTERBOX_TURBO_SAMPLING,
    QWEN3_SAMPLING,
    QWEN3_TOP_K_DISABLED,
    SOPRANO_SAMPLING,
    SWIFT_PARITY,
    SWIFT_PARITY_BY_NAME,
    SWIFT_SOURCE,
    ParityValue,
    SamplingParams,
)
from .truncation import (
    MAX_RATIO,
    MIN_RATIO,
    WORDS_PER_SECOND,
    DurationCheck,
    check_duration,
    expected_duration,
)

__all__ = [
    "CHATTERBOX_SAMPLING",
    "CHATTERBOX_TURBO_SAMPLING",
    "ENGINE_KEYS",
    "ENGINE_SPECS",
    "MAX_RATIO",
    "MIN_RATIO",
    "QWEN3_LANGUAGE",
    "QWEN3_SAMPLING",
    "QWEN3_TOP_K_DISABLED",
    "SEEDING_MLX_GLOBAL",
    "SEEDING_UNAVAILABLE",
    "SMOKE_ENGINE_KEYS",
    "SOPRANO_SAMPLING",
    "SWIFT_PARITY",
    "SWIFT_PARITY_BY_NAME",
    "SWIFT_SOURCE",
    "WORDS_PER_SECOND",
    "ChunkRecord",
    "DurationCheck",
    "Engine",
    "EngineError",
    "EngineSpec",
    "LineRequest",
    "LineResult",
    "ParityValue",
    "SamplingParams",
    "check_duration",
    "expected_duration",
    "load_engine",
    "spec",
]
