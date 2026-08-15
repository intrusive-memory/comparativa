"""Duration sanity checking (FR-8, Risk §8.3).

Autoregressive TTS fails in two directions, and both are silent: the model can
emit its stop token early (a *truncated* line — the tail of the sentence is
simply missing), or it can loop and babble past the text (an *overrun*). Neither
raises; both produce a well-formed wav. The only cheap signal available without
an ASR round-trip is the audio's duration against what the text should take to
say, so that is what this module checks.

Two independent estimates are used:

* **word rate** — ``words / WORDS_PER_SECOND``. The primary estimate; it is what
  the execution plan asks for ("duration sanity check vs word count") and it
  degrades gracefully on punctuation-heavy dialogue.
* **character rate** — ``comparativa.parsing.textprep.estimate_duration``, i.e.
  SwiftVoxAlta's ``estimatedSecondsPerChar`` (0.055). Reported alongside so the
  two stacks agree on what "too short" means.

The expected duration is the larger of the two; a line is flagged when the
measured audio falls below :data:`MIN_RATIO` of it (or rises above
:data:`MAX_RATIO`). The thresholds are deliberately loose — this is a detector
for *gross* failure, not a prosody judge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from ..parsing.textprep import estimate_duration

#: Speaking rate used for the word-count estimate (~168 wpm, unhurried podcast
#: narration). Lower than a rushed read on purpose: it makes the truncation
#: threshold conservative rather than trigger-happy.
WORDS_PER_SECOND: Final = 2.8

#: Audio shorter than this fraction of the expected duration is truncated.
MIN_RATIO: Final = 0.5

#: Audio longer than this multiple of the expected duration is an overrun
#: (repetition-loop babble, or a hallucinated tail).
MAX_RATIO: Final = 4.0

#: Never flag anything below this absolute duration as an overrun; one-word
#: lines have almost no signal and their leading/trailing silence dominates.
MIN_EXPECTED_SECONDS: Final = 0.30

#: Audio at or below this is empty for practical purposes.
SILENCE_SECONDS: Final = 0.05


def word_count(text: str) -> int:
    """Whitespace-delimited word count."""
    return len(text.split())


def expected_duration(text: str) -> float:
    """Expected spoken duration in seconds for ``text``.

    ``max`` of the word-rate and character-rate estimates, floored at
    :data:`MIN_EXPECTED_SECONDS`.
    """
    by_words = word_count(text) / WORDS_PER_SECOND
    by_chars = estimate_duration(text)
    return max(by_words, by_chars, MIN_EXPECTED_SECONDS)


@dataclass(frozen=True)
class DurationCheck:
    """The verdict on one generated span."""

    text_chars: int
    text_words: int
    expected_seconds: float
    actual_seconds: float
    #: ``actual / expected``.
    ratio: float
    truncated: bool
    overrun: bool
    #: Human-readable reason when something is wrong, else ``""``.
    reason: str = ""

    @property
    def ok(self) -> bool:
        return not (self.truncated or self.overrun)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "expected_seconds": round(self.expected_seconds, 3),
            "actual_seconds": round(self.actual_seconds, 3),
            "ratio": round(self.ratio, 3),
            "words": self.text_words,
        }
        if self.truncated:
            d["truncated"] = True
        if self.overrun:
            d["overrun"] = True
        if self.reason:
            d["reason"] = self.reason
        return d


def check_duration(
    text: str,
    actual_seconds: float,
    *,
    min_ratio: float = MIN_RATIO,
    max_ratio: float = MAX_RATIO,
) -> DurationCheck:
    """Compare ``actual_seconds`` of audio against what ``text`` should take."""
    expected = expected_duration(text)
    ratio = actual_seconds / expected if expected > 0 else 0.0

    truncated = False
    overrun = False
    reason = ""

    if actual_seconds <= SILENCE_SECONDS:
        truncated = True
        reason = f"produced {actual_seconds:.3f}s of audio (effectively silent)"
    elif ratio < min_ratio:
        truncated = True
        reason = (
            f"{actual_seconds:.2f}s of audio for {word_count(text)} words "
            f"(expected ~{expected:.2f}s, ratio {ratio:.2f} < {min_ratio})"
        )
    elif ratio > max_ratio:
        overrun = True
        reason = (
            f"{actual_seconds:.2f}s of audio for {word_count(text)} words "
            f"(expected ~{expected:.2f}s, ratio {ratio:.2f} > {max_ratio})"
        )

    return DurationCheck(
        text_chars=len(text),
        text_words=word_count(text),
        expected_seconds=expected,
        actual_seconds=actual_seconds,
        ratio=ratio,
        truncated=truncated,
        overrun=overrun,
        reason=reason,
    )
