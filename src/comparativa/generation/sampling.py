"""Sampling parameters, and the Swift parity constants they come from.

The qwen3 defaults here are **not** mlx-audio's library defaults: they are
lifted verbatim from SwiftVoxAlta's ``GenerationSettings`` so that the Python
condition (C) and the Swift condition (A) sample the same way and the
comparison measures the *port*, not the knob settings (EXECUTION_PLAN.md
Risk §8.2). ``docs/SAMPLING_PARITY.md`` renders :data:`SWIFT_PARITY` as a table
with the source line of every value.

The other engines have no Swift analogue in this mission, so their defaults are
mlx-audio 0.4.8's own — recorded here with the line that declares them, so the
report can state exactly what was run.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Final

#: Read-only source of every Swift-parity value below.
SWIFT_SOURCE: Final = (
    "~/Projects/package-collection/pkg/SwiftVoxAlta/Sources/SwiftVoxAlta/"
    "GenerationSettings.swift"
)


@dataclass(frozen=True)
class ParityValue:
    """One sampling parameter traced back to its Swift declaration."""

    #: Python-side keyword this maps to (``temperature``, ``top_p``, ...).
    python_name: str
    #: Swift property name (``temperature``, ``topP``, ...).
    swift_name: str
    value: Any
    #: ``GenerationSettings.swift`` line that *declares* the property.
    declaration_line: int
    #: ``GenerationSettings.swift`` line that supplies the default value.
    default_line: int
    note: str = ""


#: Every value taken from ``GenerationSettings.init`` (lines 97-113), which is
#: what ``GenerationSettings.default`` (line 118) instantiates. Verified by
#: reading the file on 2026-08-15 at mission commit time.
SWIFT_PARITY: Final[tuple[ParityValue, ...]] = (
    ParityValue(
        python_name="temperature",
        swift_name="temperature",
        value=0.7,
        declaration_line=39,
        default_line=98,
        note="Swift doc comment calls 0.6-0.7 'balanced stability and naturalness'.",
    ),
    ParityValue(
        python_name="top_p",
        swift_name="topP",
        value=0.9,
        declaration_line=47,
        default_line=99,
        note="Nucleus threshold; Swift doc comment calls 0.9 'balanced'.",
    ),
    ParityValue(
        python_name="repetition_penalty",
        swift_name="repetitionPenalty",
        value=1.3,
        declaration_line=55,
        default_line=100,
        note="Top of the Swift 'light penalty (recommended)' band (1.1-1.3).",
    ),
    ParityValue(
        python_name="max_tokens",
        swift_name="maxTokens",
        value=16384,
        declaration_line=61,
        default_line=101,
        note="12 Hz token rate; the Swift doc comment reads ~22 minutes of audio.",
    ),
    ParityValue(
        python_name="__chunking_enabled",
        swift_name="enableAutoChunking",
        value=True,
        declaration_line=77,
        default_line=102,
        note=(
            "Not a sampling knob: it switches on the sentence chunker. "
            "comparativa applies the same chunking in "
            "comparativa.generation.engines."
        ),
    ),
    ParityValue(
        python_name="__chunk_target_seconds",
        swift_name="chunkTargetDuration",
        value=12.0,
        declaration_line=89,
        default_line=103,
        note="Mirrored by comparativa.parsing.textprep.CHUNK_TARGET_SECONDS.",
    ),
    ParityValue(
        python_name="__chunk_pause_seconds",
        swift_name="chunkPauseDuration",
        value=0.25,
        declaration_line=95,
        default_line=104,
        note="Mirrored by comparativa.parsing.textprep.CHUNK_PAUSE_SECONDS.",
    ),
)

#: Lookup by Python keyword.
SWIFT_PARITY_BY_NAME: Final[dict[str, ParityValue]] = {
    p.python_name: p for p in SWIFT_PARITY
}


def _swift(name: str) -> Any:
    return SWIFT_PARITY_BY_NAME[name].value


#: ``GenerationSettings`` has **no** top-k property, and the Swift Qwen3
#: generation entry point takes none either
#: (mlx-audio-swift ``Sources/MLXAudioTTS/Models/Qwen3TTS/
#: Qwen3TTSVoiceClonePrompt.swift:230-238`` -- temperature/topP/
#: repetitionPenalty/maxTokens only). Python mlx-audio defaults ``top_k=50``,
#: which would be a silent divergence, so we disable it: ``qwen3_tts.py:859``
#: and ``:934`` both gate on ``top_k > 0``.
QWEN3_TOP_K_DISABLED: Final = 0


@dataclass(frozen=True)
class SamplingParams:
    """The fixed, recorded sampling configuration for one generation call.

    Every field lands verbatim in the per-line record so a run can be replayed
    (FR-10). ``extra`` carries engine-specific knobs that have no cross-engine
    meaning (chatterbox's ``cfg_weight``, ``exaggeration``, ``min_p``).
    """

    temperature: float
    top_p: float
    max_tokens: int
    #: ``None`` when the engine exposes no top-k knob; ``0`` means "disabled".
    top_k: int | None = None
    #: ``None`` when the engine exposes no repetition penalty.
    repetition_penalty: float | None = None
    #: Engine-specific extras, recorded but not interpreted.
    extra: dict[str, float] = field(default_factory=dict)
    #: Where these numbers came from, quoted in the manifest and the report.
    source: str = ""

    def evolve(self, **changes: Any) -> SamplingParams:
        """A copy with ``changes`` applied."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.top_k is not None:
            d["top_k"] = self.top_k
        if self.repetition_penalty is not None:
            d["repetition_penalty"] = self.repetition_penalty
        if self.extra:
            d["extra"] = dict(self.extra)
        if self.source:
            d["source"] = self.source
        return d


#: Qwen3 (both sizes): SwiftVoxAlta parity, Risk §8.2.
QWEN3_SAMPLING: Final = SamplingParams(
    temperature=_swift("temperature"),
    top_p=_swift("top_p"),
    top_k=QWEN3_TOP_K_DISABLED,
    repetition_penalty=_swift("repetition_penalty"),
    max_tokens=_swift("max_tokens"),
    source=f"{SWIFT_SOURCE}:97-113 (GenerationSettings.init defaults)",
)

#: Chatterbox: mlx-audio's own defaults (`chatterbox.py:759-782`). No Swift
#: analogue exists in this mission, so nothing is being matched.
CHATTERBOX_SAMPLING: Final = SamplingParams(
    temperature=0.8,
    top_p=1.0,
    max_tokens=1000,
    repetition_penalty=1.2,
    extra={"min_p": 0.05, "cfg_weight": 0.5, "exaggeration": 0.1},
    source="mlx_audio/tts/models/chatterbox/chatterbox.py:759-782 (library defaults)",
)

#: Chatterbox-turbo: `chatterbox_turbo.py:780-797` library defaults.
CHATTERBOX_TURBO_SAMPLING: Final = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    top_k=1000,
    max_tokens=800,
    repetition_penalty=1.2,
    extra={"min_p": 0.0},
    source=(
        "mlx_audio/tts/models/chatterbox_turbo/chatterbox_turbo.py:780-797 "
        "(library defaults)"
    ),
)

#: Soprano: `soprano.py:362-371` library defaults. Soprano exposes neither a
#: top-k nor a repetition penalty.
SOPRANO_SAMPLING: Final = SamplingParams(
    temperature=0.3,
    top_p=0.95,
    max_tokens=512,
    source="mlx_audio/tts/models/soprano/soprano.py:362-371 (library defaults)",
)
