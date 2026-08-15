"""Episode assembly: seam trimming, loudness, gaps, and the timeline (FR-9).

The engine layer (Sortie 5) owns everything *inside* a line — chunk pauses
(0.25 s) and breath gaps (0.15 s) are already spliced into ``LineResult.audio``.
This module owns everything *between* lines:

1. **Seam trimming** — the model pads every generation with head and tail
   silence. Produciesta's ``WAVAssembler.trimSilence`` strips that padding at
   internal seams only, keeping the clip's outer envelope natural. Ported here
   with the same constants (amplitude 350/32768 ≈ −39.4 dBFS, 8 ms guard), but
   applied at *episode* seams: the first placed line keeps its head, the last
   keeps its tail, every other edge is tightened so the audible pause is
   governed by the configured gap rather than by the model's mood.
2. **Loudness** — per-line normalization to −16 LUFS integrated via
   ``pyloudnorm`` (RD-3), with a peak guard so a quiet line boosted by +12 dB
   cannot clip. Lines too short for BS.1770's 400 ms block are left alone and
   say so in the manifest.
3. **Gaps** — a configurable inter-line gap, and a larger inter-scene gap
   whenever ``scene_index`` changes between two placed lines.

Divergence from Produciesta (docs/ASSEMBLY.md § Divergences): the Swift stack
butts voiced clips together on the timeline and only ever inserts silence for an
authored ``<pause>`` (``TimelineComposer``). Comparativa defaults to a 0.25 s
inter-line and 0.75 s inter-scene gap; ``--line-gap 0 --scene-gap 0`` reproduces
the Swift timing exactly.

Durations are always derived from the assembled sample count, never by summing
the backend's reported per-line durations, so the manifest's offsets and the
bytes on disk cannot drift apart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Final, Sequence

import numpy as np

#: RD-3: integrated loudness target, per line.
TARGET_LUFS: Final = -16.0

#: Sample-peak ceiling applied after the loudness gain (≈ −0.09 dBFS). A peak
#: guard, not a limiter: the whole line is scaled, so no waveshaping happens.
PEAK_CEILING: Final = 0.99

#: Default silence between consecutive lines in the same scene.
LINE_GAP_SECONDS: Final = 0.25

#: Default silence at a scene change.
SCENE_GAP_SECONDS: Final = 0.75

#: ``WAVAssembler.silenceThreshold`` (350 of int16 full scale ≈ −39.4 dBFS).
SILENCE_THRESHOLD: Final = 350.0 / 32768.0

#: ``WAVAssembler.guardMillis`` — speech kept either side of a trimmed edge.
TRIM_GUARD_SECONDS: Final = 0.008

#: ITU-R BS.1770 block size used by ``pyloudnorm.Meter``. Audio shorter than
#: this cannot be measured at all.
LOUDNESS_BLOCK_SECONDS: Final = 0.400

#: Written to the manifest so a reader knows which meter produced the numbers.
LOUDNESS_METER: Final = "pyloudnorm.Meter (ITU-R BS.1770-4, 400 ms block)"


class AssemblyError(RuntimeError):
    """The lines handed to :func:`assemble` cannot be laid on one timeline."""


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssemblyOptions:
    """Everything the caller can tune about the episode timeline."""

    line_gap_seconds: float = LINE_GAP_SECONDS
    scene_gap_seconds: float = SCENE_GAP_SECONDS
    #: ``None`` disables loudness normalization entirely.
    target_lufs: float | None = TARGET_LUFS
    peak_ceiling: float = PEAK_CEILING
    #: Trim model head/tail silence at internal seams (Produciesta parity).
    trim_seams: bool = True
    silence_threshold: float = SILENCE_THRESHOLD
    trim_guard_seconds: float = TRIM_GUARD_SECONDS

    def to_dict(self) -> dict[str, Any]:
        threshold_dbfs = (
            20.0 * math.log10(self.silence_threshold)
            if self.silence_threshold > 0
            else float("-inf")
        )
        return {
            "line_gap_seconds": self.line_gap_seconds,
            "scene_gap_seconds": self.scene_gap_seconds,
            "target_lufs": self.target_lufs,
            "loudness_meter": LOUDNESS_METER if self.target_lufs is not None else None,
            "peak_ceiling": self.peak_ceiling,
            "trim_seams": self.trim_seams,
            "silence_threshold": round(self.silence_threshold, 6),
            "silence_threshold_dbfs": round(threshold_dbfs, 2),
            "trim_guard_seconds": self.trim_guard_seconds,
        }


# ---------------------------------------------------------------------------
# Seam trimming
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrimResult:
    """A trimmed waveform plus how much was removed from each edge."""

    audio: np.ndarray
    head_seconds: float = 0.0
    tail_seconds: float = 0.0

    @property
    def trimmed(self) -> bool:
        return self.head_seconds > 0.0 or self.tail_seconds > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "head_seconds": round(self.head_seconds, 4),
            "tail_seconds": round(self.tail_seconds, 4),
        }


def trim_edges(
    audio: np.ndarray,
    sample_rate: int,
    *,
    head: bool = True,
    tail: bool = True,
    threshold: float = SILENCE_THRESHOLD,
    guard_seconds: float = TRIM_GUARD_SECONDS,
) -> TrimResult:
    """Strip near-silence from the requested edges of one line.

    Port of ``WAVAssembler.trimSilence``. A wholly silent span vanishes when
    both edges are trimmed (it is an internal seam contributing nothing) and is
    kept untouched otherwise, so the episode's outer envelope is preserved.
    """
    n = int(audio.size)
    if n == 0 or not (head or tail):
        return TrimResult(audio)

    loud = np.flatnonzero(np.abs(audio) > threshold)
    if loud.size == 0:
        if head and tail:
            return TrimResult(audio[:0], n / sample_rate, 0.0)
        return TrimResult(audio)

    guard = int(round(guard_seconds * sample_rate))
    start = max(0, int(loud[0]) - guard) if head else 0
    end = min(n, int(loud[-1]) + 1 + guard) if tail else n
    if start >= end:  # pragma: no cover - guard against a degenerate span
        return TrimResult(audio)

    return TrimResult(
        audio[start:end],
        start / sample_rate,
        (n - end) / sample_rate,
    )


# ---------------------------------------------------------------------------
# Loudness (RD-3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoudnessResult:
    """One line's loudness measurement and the gain that was applied."""

    audio: np.ndarray
    applied: bool
    input_lufs: float | None = None
    output_lufs: float | None = None
    #: Gain the loudness target asked for, before the peak guard.
    requested_gain_db: float = 0.0
    #: Gain actually applied (differs from the request when peak-guarded).
    applied_gain_db: float = 0.0
    peak_before: float = 0.0
    peak_after: float = 0.0
    peak_limited: bool = False
    #: Why nothing was applied, when :attr:`applied` is False.
    reason: str = ""

    @property
    def shortfall_db(self) -> float:
        """How far under the loudness target the peak guard left this line.

        Non-zero only when the gain the target asked for would have clipped.
        Speech at −16 LUFS with a natural (unlimited) crest factor routinely
        needs more headroom than 0 dBFS leaves, so this is expected rather than
        exceptional — see ``docs/ASSEMBLY.md`` § Loudness.
        """
        return max(0.0, self.requested_gain_db - self.applied_gain_db)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "normalized": self.applied,
            "peak_before": round(self.peak_before, 5),
            "peak_after": round(self.peak_after, 5),
        }
        if self.input_lufs is not None:
            d["input_lufs"] = round(self.input_lufs, 2)
        if self.output_lufs is not None:
            d["output_lufs"] = round(self.output_lufs, 2)
        if self.applied:
            d["requested_gain_db"] = round(self.requested_gain_db, 2)
            d["applied_gain_db"] = round(self.applied_gain_db, 2)
        if self.peak_limited:
            d["peak_limited"] = True
            d["shortfall_db"] = round(self.shortfall_db, 2)
        if self.reason:
            d["reason"] = self.reason
        return d


#: One ``pyloudnorm.Meter`` per sample rate; construction builds filter state.
_METERS: dict[int, Any] = {}


def meter(sample_rate: int) -> Any:
    """A cached ``pyloudnorm.Meter`` for ``sample_rate``."""
    existing = _METERS.get(sample_rate)
    if existing is not None:
        return existing
    import pyloudnorm

    made = pyloudnorm.Meter(sample_rate)
    _METERS[sample_rate] = made
    return made


def measure_lufs(audio: np.ndarray, sample_rate: int) -> float | None:
    """Integrated loudness in LUFS, or ``None`` when it cannot be measured."""
    if audio.size < int(LOUDNESS_BLOCK_SECONDS * sample_rate) + 1:
        return None
    value = float(meter(sample_rate).integrated_loudness(audio.astype(np.float64)))
    return value if math.isfinite(value) else None


def normalize_line(
    audio: np.ndarray,
    sample_rate: int,
    *,
    target_lufs: float | None = TARGET_LUFS,
    peak_ceiling: float = PEAK_CEILING,
) -> LoudnessResult:
    """Normalize one line to ``target_lufs`` integrated, guarding the peak.

    Never raises: a line that cannot be measured (too short for the BS.1770
    block, or gated to silence) comes back untouched with the reason recorded,
    because dropping a line from an episode over a meter limitation would be a
    far worse outcome than an unnormalized whisper.
    """
    peak_before = float(np.max(np.abs(audio))) if audio.size else 0.0

    if target_lufs is None:
        return LoudnessResult(
            audio, False, peak_before=peak_before, peak_after=peak_before,
            reason="loudness normalization disabled",
        )
    if peak_before <= 0.0:
        return LoudnessResult(
            audio, False, peak_before=peak_before, peak_after=peak_before,
            reason="line is digital silence",
        )

    input_lufs = measure_lufs(audio, sample_rate)
    if input_lufs is None:
        return LoudnessResult(
            audio, False, peak_before=peak_before, peak_after=peak_before,
            reason=(
                f"shorter than the {LOUDNESS_BLOCK_SECONDS:.3f} s BS.1770 block, "
                "or gated to silence; left at its generated level"
            ),
        )

    requested_gain_db = target_lufs - input_lufs
    gained = audio.astype(np.float32) * float(10.0 ** (requested_gain_db / 20.0))

    peak_gained = float(np.max(np.abs(gained)))
    applied_gain_db = requested_gain_db
    peak_limited = False
    if peak_ceiling > 0.0 and peak_gained > peak_ceiling:
        gained = gained * (peak_ceiling / peak_gained)
        applied_gain_db = requested_gain_db + 20.0 * math.log10(peak_ceiling / peak_gained)
        peak_limited = True

    gained = gained.astype(np.float32, copy=False)
    output_lufs = measure_lufs(gained, sample_rate)

    return LoudnessResult(
        audio=gained,
        applied=True,
        input_lufs=input_lufs,
        output_lufs=output_lufs,
        requested_gain_db=requested_gain_db,
        applied_gain_db=applied_gain_db,
        peak_before=peak_before,
        peak_after=float(np.max(np.abs(gained))),
        peak_limited=peak_limited,
    )


# ---------------------------------------------------------------------------
# The timeline
# ---------------------------------------------------------------------------


@dataclass
class LineAudio:
    """One generated line entering assembly."""

    audio: np.ndarray
    sample_rate: int
    #: Scene the line belongs to; a change triggers the inter-scene gap.
    scene_index: int = 0
    #: The line's manifest record so far (Sortie 5's ``LineResult.to_dict()``
    #: plus the planned-line metadata). Assembly fields are added to a copy.
    record: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssembledEpisode:
    """The concatenated episode plus its per-line timeline records."""

    audio: np.ndarray
    sample_rate: int
    lines: list[dict[str, Any]]
    options: AssemblyOptions
    #: Total silence inserted *between* lines (excludes intra-line pauses).
    gap_seconds: float = 0.0
    #: Total placed line audio, post-trim.
    audio_seconds: float = 0.0

    @property
    def duration_seconds(self) -> float:
        return self.audio.size / self.sample_rate if self.sample_rate else 0.0

    @property
    def placed_line_count(self) -> int:
        return sum(1 for line in self.lines if line.get("assembled_duration_seconds", 0.0) > 0)

    def totals(self) -> dict[str, Any]:
        """Episode-level timing, truncation, and loudness totals."""
        generate_seconds = sum(float(line.get("generate_seconds", 0.0)) for line in self.lines)
        duration = self.duration_seconds
        loudness = [line["loudness"] for line in self.lines if "loudness" in line]
        measured = [m["output_lufs"] for m in loudness if m.get("output_lufs") is not None]
        shortfalls = [float(m.get("shortfall_db", 0.0)) for m in loudness if m.get("normalized")]
        return {
            "line_count": len(self.lines),
            "placed_line_count": self.placed_line_count,
            "duration_seconds": round(duration, 3),
            "audio_seconds": round(self.audio_seconds, 3),
            "gap_seconds": round(self.gap_seconds, 3),
            "generate_seconds": round(generate_seconds, 3),
            "real_time_factor": round(generate_seconds / duration, 3) if duration > 0 else None,
            "truncated_lines": sum(1 for line in self.lines if line.get("truncated")),
            "overrun_lines": sum(1 for line in self.lines if line.get("overrun")),
            "truncation_retry_lines": sum(
                1 for line in self.lines if line.get("truncation_retry")
            ),
            "peak_limited_lines": sum(1 for m in loudness if m.get("peak_limited")),
            "unnormalized_lines": sum(1 for m in loudness if not m.get("normalized")),
            "mean_output_lufs": round(sum(measured) / len(measured), 2) if measured else None,
            "max_loudness_shortfall_db": round(max(shortfalls), 2) if shortfalls else 0.0,
            "mean_loudness_shortfall_db": (
                round(sum(shortfalls) / len(shortfalls), 2) if shortfalls else 0.0
            ),
        }


def _silence(seconds: float, sample_rate: int) -> np.ndarray:
    return np.zeros(max(0, int(round(seconds * sample_rate))), dtype=np.float32)


def assemble(
    lines: Sequence[LineAudio],
    *,
    options: AssemblyOptions = AssemblyOptions(),
    sample_rate: int | None = None,
) -> AssembledEpisode:
    """Lay ``lines`` end to end with gaps, trimming and loudness applied.

    Every input line appears in :attr:`AssembledEpisode.lines`, including ones
    that generated no audio — the manifest's line count is the *script's* line
    count, not the count of lines that happened to produce samples. A silent
    line occupies no time and contributes no gap.
    """
    rates = {int(line.sample_rate) for line in lines if line.sample_rate}
    if sample_rate is None:
        if len(rates) > 1:
            raise AssemblyError(
                "cannot assemble lines with mixed sample rates: "
                + ", ".join(str(r) for r in sorted(rates))
                + " (load one engine per episode)"
            )
        sample_rate = rates.pop() if rates else 24000
    elif rates - {int(sample_rate)}:
        raise AssemblyError(
            f"lines at {sorted(rates)} Hz cannot be assembled at {sample_rate} Hz"
        )

    with_audio = [i for i, line in enumerate(lines) if line.audio.size > 0]
    first = with_audio[0] if with_audio else None
    last = with_audio[-1] if with_audio else None

    pieces: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    cursor = 0  # samples placed so far
    gap_samples = 0
    audio_samples = 0
    placed = 0
    previous_scene: int | None = None

    for index, line in enumerate(lines):
        record = dict(line.record)

        if line.audio.size == 0:
            record.update(
                {
                    "offset_seconds": round(cursor / sample_rate, 3),
                    "assembled_duration_seconds": 0.0,
                    "gap_before_seconds": 0.0,
                    "placed": False,
                }
            )
            records.append(record)
            continue

        gap_seconds = 0.0
        if placed > 0:
            gap_seconds = (
                options.scene_gap_seconds
                if previous_scene is not None and line.scene_index != previous_scene
                else options.line_gap_seconds
            )
            gap = _silence(gap_seconds, sample_rate)
            if gap.size:
                pieces.append(gap)
                cursor += gap.size
                gap_samples += gap.size
            gap_seconds = gap.size / sample_rate

        trim = trim_edges(
            line.audio,
            sample_rate,
            head=options.trim_seams and index != first,
            tail=options.trim_seams and index != last,
            threshold=options.silence_threshold,
            guard_seconds=options.trim_guard_seconds,
        )
        loudness = normalize_line(
            trim.audio,
            sample_rate,
            target_lufs=options.target_lufs,
            peak_ceiling=options.peak_ceiling,
        )
        placed_audio = np.asarray(loudness.audio, dtype=np.float32)

        record.update(
            {
                "offset_seconds": round(cursor / sample_rate, 3),
                "assembled_duration_seconds": round(placed_audio.size / sample_rate, 3),
                "gap_before_seconds": round(gap_seconds, 3),
                "trim": trim.to_dict(),
                "loudness": loudness.to_dict(),
            }
        )
        records.append(record)

        pieces.append(placed_audio)
        cursor += placed_audio.size
        audio_samples += placed_audio.size
        placed += 1
        previous_scene = line.scene_index

    audio = (
        np.concatenate(pieces).astype(np.float32, copy=False)
        if pieces
        else np.zeros(0, dtype=np.float32)
    )
    return AssembledEpisode(
        audio=audio,
        sample_rate=sample_rate,
        lines=records,
        options=options,
        gap_seconds=gap_samples / sample_rate,
        audio_seconds=audio_samples / sample_rate,
    )
