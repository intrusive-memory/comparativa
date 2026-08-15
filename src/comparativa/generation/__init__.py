"""Generation: MLX TTS engines, episode assembly, and the manifest.

Two layers, one package:

* **Engine layer** (Sortie 5) — :func:`load_engine` returns an :class:`Engine`
  that generates one line at a time with fixed, recorded sampling parameters
  and a duration sanity check.
* **Episode layer** (Sortie 6) — :func:`build_plan` resolves a screenplay into
  voiced lines, :func:`generate_episode` runs them all through one loaded
  engine and lays them on a timeline (:mod:`.assembly`: seam trimming, −16 LUFS
  per line, inter-line and inter-scene gaps), and :func:`write_outputs` writes
  the ``.wav``, ``.m4a``, and ``manifest.json``.

Typical use::

    from comparativa.generation import build_plan, generate_episode, write_outputs

    plan = build_plan("episode.fountain", "qwen3-0.6b")
    result = generate_episode(plan)
    write_outputs(result, "out/")

A single line, for smoke tests and auditions::

    engine = load_engine("qwen3-1.7b")
    result = engine.generate_line(
        LineRequest(text=line.text, voice="ryan", direction=line.direction, seed=7)
    )
    result.audio           # 1-D float32 at result.sample_rate
    result.to_dict()       # the per-line half of the manifest record
"""

from .assembly import (
    LINE_GAP_SECONDS,
    LOUDNESS_BLOCK_SECONDS,
    LOUDNESS_METER,
    PEAK_CEILING,
    SCENE_GAP_SECONDS,
    SILENCE_THRESHOLD,
    TARGET_LUFS,
    TRIM_GUARD_SECONDS,
    AssembledEpisode,
    AssemblyError,
    AssemblyOptions,
    LineAudio,
    LoudnessResult,
    TrimResult,
    assemble,
    measure_lufs,
    normalize_line,
    trim_edges,
)
from .audition import (
    AUDITION_SEED,
    AUDITION_TEXT,
    AuditionRun,
    AuditionTake,
    run_audition,
    write_audition_manifest,
)
from .encode import (
    AFCONVERT,
    M4A_BITRATE,
    EncodeError,
    afconvert_available,
    encode_m4a,
    write_wav,
)
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
from .episode import (
    DEFAULT_PRESETS_PATH,
    DEFAULT_SEED,
    MANIFEST_FILENAME,
    SCHEMA_VERSION,
    SEED_STRIDE,
    EpisodePlan,
    EpisodeResult,
    GenerationError,
    PlannedLine,
    build_manifest,
    build_plan,
    generate_episode,
    manifest_line_count,
    spoken_line_count,
    write_outputs,
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
    "AFCONVERT",
    "AUDITION_SEED",
    "AUDITION_TEXT",
    "CHATTERBOX_SAMPLING",
    "CHATTERBOX_TURBO_SAMPLING",
    "DEFAULT_PRESETS_PATH",
    "DEFAULT_SEED",
    "ENGINE_KEYS",
    "ENGINE_SPECS",
    "LINE_GAP_SECONDS",
    "LOUDNESS_BLOCK_SECONDS",
    "LOUDNESS_METER",
    "M4A_BITRATE",
    "MANIFEST_FILENAME",
    "MAX_RATIO",
    "MIN_RATIO",
    "PEAK_CEILING",
    "QWEN3_LANGUAGE",
    "QWEN3_SAMPLING",
    "QWEN3_TOP_K_DISABLED",
    "SCENE_GAP_SECONDS",
    "SCHEMA_VERSION",
    "SEED_STRIDE",
    "SEEDING_MLX_GLOBAL",
    "SEEDING_UNAVAILABLE",
    "SILENCE_THRESHOLD",
    "SMOKE_ENGINE_KEYS",
    "SOPRANO_SAMPLING",
    "SWIFT_PARITY",
    "SWIFT_PARITY_BY_NAME",
    "SWIFT_SOURCE",
    "TARGET_LUFS",
    "TRIM_GUARD_SECONDS",
    "WORDS_PER_SECOND",
    "AssembledEpisode",
    "AssemblyError",
    "AssemblyOptions",
    "AuditionRun",
    "AuditionTake",
    "ChunkRecord",
    "DurationCheck",
    "EncodeError",
    "Engine",
    "EngineError",
    "EngineSpec",
    "EpisodePlan",
    "EpisodeResult",
    "GenerationError",
    "LineAudio",
    "LineRequest",
    "LineResult",
    "LoudnessResult",
    "ParityValue",
    "PlannedLine",
    "SamplingParams",
    "TrimResult",
    "afconvert_available",
    "assemble",
    "build_manifest",
    "build_plan",
    "check_duration",
    "encode_m4a",
    "expected_duration",
    "generate_episode",
    "load_engine",
    "manifest_line_count",
    "measure_lufs",
    "normalize_line",
    "run_audition",
    "spec",
    "spoken_line_count",
    "trim_edges",
    "write_audition_manifest",
    "write_outputs",
    "write_wav",
]
