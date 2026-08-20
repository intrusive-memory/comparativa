"""Unified engine interface over ``mlx-audio==0.4.8`` (FR-7, FR-8, FR-10).

Four TTS families live behind one call. They disagree about almost everything —
argument names (``max_tokens`` vs ``max_new_tokens``), which knobs exist (only
qwen3 has top-k; only qwen3 and chatterbox have a repetition penalty), whether a
voice can be chosen at all, and whether the model splits long text itself. This
module hides that behind :meth:`Engine.generate_line` and, crucially, *records*
what it did: every line comes back with the exact sampling parameters, the seed,
the chunk boundaries, and the duration check, which is what the manifest in
Sortie 6 and the report in Sortie 11 are built from.

Capability flags (:class:`EngineSpec`) are the honest summary of what each
engine can do this round:

================ ============= ============== ========= ========== ==========
engine           preset voices instruct       seeding   we chunk   clones
================ ============= ============== ========= ========== ==========
qwen3-1.7b       yes (9 spk)   yes            yes       yes        no
qwen3-0.6b       yes (9 spk)   yes            yes       yes        no
qwen3-1.7b-clone no            no             yes       yes        yes (ICL)
qwen3-0.6b-clone no            no             yes       yes        yes (ICL)
chatterbox       no (1 voice)  no             yes       yes        yes (ref)
chatterbox-turbo no (1 voice)  no             yes       no (built in) yes (ref)
soprano          no (1 voice)  no             yes       no (built in) no
================ ============= ============== ========= ========== ==========

"Clones" (round 2) means the engine can take a **reference clip** and speak in
that voice: the qwen3 ``*-clone`` engines run the Base checkpoints' ICL path
(``ref_audio`` + ``ref_text``, both mandatory), and the chatterbox family
conditions on reference audio alone. Reference material comes from the
``.vox`` bundles (:mod:`comparativa.voices.vox`) via a :class:`CloneVoice` on
the request. Prepared conditioning is cached per clone name so an episode pays
the reference-encoding cost once per character, not once per line.

"Seeding" means ``mx.random.seed`` is effective: every one of these engines
samples through MLX's global RNG (``mlx_lm.sample_utils.make_sampler`` for
soprano and chatterbox's T3, ``categorical_sampling`` for qwen3, plus
``mx.random.normal`` in chatterbox's flow matching), so seeding before the call
pins the sample stream. Nothing here exposes a per-call generator; if a future
mlx-audio version breaks the global-RNG assumption, set ``seeding=False`` on the
spec and :class:`LineResult` will record ``"seeding unavailable"``.

Nothing in this module downloads: checkpoints are resolved through
``huggingface_hub``'s cache, and the smoke path runs with ``HF_HUB_OFFLINE=1``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from ..parsing.textprep import (
    BREATH_GAP_SECONDS,
    CHUNK_PAUSE_SECONDS,
    CHUNK_TARGET_SECONDS,
    estimate_duration,
    split_at_sentences,
)
from ..voices import catalog
from .sampling import (
    CHATTERBOX_SAMPLING,
    CHATTERBOX_TURBO_SAMPLING,
    CSM_SAMPLING,
    DIA_SAMPLING,
    HIGGS_SAMPLING,
    KOKORO_SAMPLING,
    ORPHEUS_SAMPLING,
    QWEN3_ICL_SAMPLING,
    QWEN3_SAMPLING,
    SOPRANO_SAMPLING,
    SamplingParams,
)
from .truncation import DurationCheck, check_duration

#: Recorded in the line record when the engine cannot be seeded (FR-10).
SEEDING_UNAVAILABLE: Final = "seeding unavailable"

#: Recorded when ``mx.random.seed`` was applied.
SEEDING_MLX_GLOBAL: Final = "mlx.core.random.seed (global RNG)"

#: Language token handed to qwen3. SwiftVoxAlta resolves ``en_US``/``en_GB`` to
#: ``TTSLanguage.english`` -> model name ``"english"``
#: (SwiftVoxAlta/TTSLanguage.swift:25,41-47); the granville corpus is English,
#: so this is the parity value. ``"auto"`` emits no language token at all.
QWEN3_LANGUAGE: Final = "english"


class EngineError(RuntimeError):
    """An engine could not be loaded or could not generate."""


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineSpec:
    """Static description of one engine: checkpoint, defaults, capabilities."""

    key: str
    #: mlx-audio model family this engine is driven as.
    family: str
    checkpoint: str
    sampling: SamplingParams

    # --- capability flags ---------------------------------------------------
    #: True when several built-in voices exist and one must be selected.
    preset_voices: bool = False
    #: True when the engine can speak in a reference clip's voice (round 2).
    clone_voices: bool = False
    #: True when cloning also requires the reference transcript (qwen3 ICL).
    clone_needs_ref_text: bool = False
    #: True when a non-spoken performance direction can be passed through.
    accepts_instruct: bool = False
    #: True when ``mx.random.seed`` pins this engine's sampling.
    seeding: bool = True
    #: Why seeding is (un)available, recorded verbatim in the line record.
    seeding_note: str = ""
    #: True when *this* layer splits long text; False when the model does it.
    chunks_long_lines: bool = False

    chunk_target_seconds: float = CHUNK_TARGET_SECONDS
    chunk_pause_seconds: float = CHUNK_PAUSE_SECONDS
    #: Nominal output sample rate; the loaded model is authoritative.
    sample_rate: int = 24000
    notes: str = ""

    @property
    def voices(self) -> tuple[catalog.Voice, ...]:
        """The built-in voices, from Sortie 4's catalog.

        The round-2 clone engines have no catalog entry — their voices come
        from ``.vox`` references, not from a preset list — so they report none.
        """
        try:
            return catalog.engine(self.key).voices
        except KeyError:
            return ()

    @property
    def default_voice(self) -> str | None:
        """The voice to use when the caller names none."""
        if not self.preset_voices:
            return catalog.DEFAULT_VOICE
        return None

    def capabilities(self) -> dict[str, Any]:
        """Capability flags, for the manifest and ``docs/SAMPLING_PARITY.md``."""
        return {
            "preset_voices": self.preset_voices,
            "voice_count": len(self.voices),
            "clone_voices": self.clone_voices,
            "clone_needs_ref_text": self.clone_needs_ref_text,
            "accepts_instruct": self.accepts_instruct,
            "seeding": self.seeding,
            "seeding_note": self.seeding_note or (
                SEEDING_MLX_GLOBAL if self.seeding else SEEDING_UNAVAILABLE
            ),
            "chunks_long_lines": self.chunks_long_lines,
            "chunk_target_seconds": self.chunk_target_seconds,
        }


_QWEN3_NOTES = (
    "CustomVoice checkpoint: `voice` is a preset speaker id and is required; "
    "`instruct` carries the parenthetical direction. Sampling is SwiftVoxAlta "
    "parity (docs/SAMPLING_PARITY.md), including top_k=0 to disable a knob the "
    "Swift path does not have."
)

ENGINE_SPECS: Final[dict[str, EngineSpec]] = {
    "qwen3-1.7b": EngineSpec(
        key="qwen3-1.7b",
        family="qwen3_tts",
        checkpoint="mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
        sampling=QWEN3_SAMPLING,
        preset_voices=True,
        accepts_instruct=True,
        chunks_long_lines=True,
        sample_rate=24000,
        notes=_QWEN3_NOTES,
    ),
    "qwen3-0.6b": EngineSpec(
        key="qwen3-0.6b",
        family="qwen3_tts",
        checkpoint="mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16",
        sampling=QWEN3_SAMPLING,
        preset_voices=True,
        accepts_instruct=True,
        chunks_long_lines=True,
        sample_rate=24000,
        notes=_QWEN3_NOTES,
    ),
    "qwen3-1.7b-clone": EngineSpec(
        key="qwen3-1.7b-clone",
        family="qwen3_tts",
        checkpoint="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
        sampling=QWEN3_ICL_SAMPLING,
        preset_voices=False,
        clone_voices=True,
        clone_needs_ref_text=True,
        accepts_instruct=False,
        chunks_long_lines=True,
        sample_rate=24000,
        notes=(
            "Base checkpoint, ICL voice cloning: every line needs a CloneVoice "
            "with ref audio AND ref text (qwen3_tts.py routes to _generate_icl "
            "only when both are present). This is the same clone path "
            "SwiftVoxAlta uses for .vox voices, so condition B vs A measures "
            "the port, not the voice design. The Base path takes no instruct; "
            "parentheticals are dropped (recorded in the manifest as "
            "accepts_instruct=false). mlx-audio floors the ICL repetition "
            "penalty at 1.5 -- QWEN3_ICL_SAMPLING records the floored value."
        ),
    ),
    "qwen3-0.6b-clone": EngineSpec(
        key="qwen3-0.6b-clone",
        family="qwen3_tts",
        checkpoint="mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
        sampling=QWEN3_ICL_SAMPLING,
        preset_voices=False,
        clone_voices=True,
        clone_needs_ref_text=True,
        accepts_instruct=False,
        chunks_long_lines=True,
        sample_rate=24000,
        notes=(
            "Size-degradation clone probe. NOTE: the 0.6B-Base checkpoint is "
            "not in the local HF cache as of 2026-08-19 (refs only, no blobs); "
            "loading it offline fails cleanly until it is prefetched with "
            "`hf download mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16`."
        ),
    ),
    "chatterbox": EngineSpec(
        key="chatterbox",
        family="chatterbox",
        checkpoint="mlx-community/chatterbox-fp16",
        sampling=CHATTERBOX_SAMPLING,
        preset_voices=False,
        clone_voices=True,
        accepts_instruct=False,
        chunks_long_lines=True,
        sample_rate=24000,
        notes=(
            "Single built-in voice (the checkpoint's conds.safetensors). Its "
            "`generate` only splits on `split_pattern='\\n'` "
            "(chatterbox.py:759-782), so a long single-paragraph line would run "
            "into the 1000-token cap (~40 s at the 25 Hz S3 rate) unsplit -- "
            "this layer chunks it at sentence boundaries instead (Risk §8.3)."
        ),
    ),
    "chatterbox-turbo": EngineSpec(
        key="chatterbox-turbo",
        family="chatterbox_turbo",
        checkpoint="mlx-community/chatterbox-turbo-fp16",
        sampling=CHATTERBOX_TURBO_SAMPLING,
        preset_voices=False,
        clone_voices=True,
        accepts_instruct=False,
        chunks_long_lines=False,
        sample_rate=24000,
        notes=(
            "Optional speed probe. Splits itself on sentence boundaries "
            "(`split_pattern=r'(?<=[.!?])\\s+'`, chatterbox_turbo.py:791)."
        ),
    ),
    "dia": EngineSpec(
        key="dia",
        family="dia",
        checkpoint="mlx-community/Dia-1.6B-fp16",
        sampling=DIA_SAMPLING,
        preset_voices=False,
        clone_voices=True,
        clone_needs_ref_text=True,
        accepts_instruct=False,
        chunks_long_lines=True,
        sample_rate=44100,
        notes=(
            "Nari Labs (South Korea), trained from scratch — no LLM backbone. "
            "Dialogue-native: text is speaker-tagged; single-character lines "
            "are sent as '[S1] <text>' with the reference as '[S1] <ref_text>'. "
            "Clones from ref audio + transcript. 44.1 kHz output; the .vox "
            "24 kHz reference is resampled by the clone cache."
        ),
    ),
    "csm": EngineSpec(
        key="csm",
        family="sesame",
        checkpoint="mlx-community/csm-1b-fp16",
        sampling=CSM_SAMPLING,
        preset_voices=False,
        clone_voices=True,
        clone_needs_ref_text=True,
        accepts_instruct=False,
        chunks_long_lines=True,
        sample_rate=24000,
        notes=(
            "Sesame AI (US), Llama backbone. Conversational model; clones from "
            "ref audio + transcript (sesame.py generate ref_audio/ref_text)."
        ),
    ),
    "higgs": EngineSpec(
        key="higgs",
        family="higgs_audio",
        checkpoint="mlx-community/higgs-audio-v2-3B-mlx-q8",
        sampling=HIGGS_SAMPLING,
        preset_voices=False,
        clone_voices=True,
        clone_needs_ref_text=True,
        accepts_instruct=False,
        chunks_long_lines=True,
        sample_rate=24000,
        notes=(
            "Boson AI (US), Llama 3.2 backbone. Only quantized MLX conversions "
            "exist (q8 used — a stated quality caveat vs the fp16 conditions). "
            "Clones from 24 kHz ref audio + transcript; the codec loads from "
            "mlx-community/higgs-audio-v2-tokenizer (higgs_audio/model.py:41)."
        ),
    ),
    "kokoro": EngineSpec(
        key="kokoro",
        family="kokoro",
        checkpoint="mlx-community/Kokoro-82M-bf16",
        sampling=KOKORO_SAMPLING,
        preset_voices=True,
        accepts_instruct=False,
        chunks_long_lines=True,
        sample_rate=24000,
        notes=(
            "Indie (hexgrad, US), StyleTTS2 lineage, from scratch. Presets "
            "control: ~28 English voices, no cloning, no sampling knobs "
            "(deterministic). Hard 510-phoneme cap per call "
            "(kokoro/pipeline.py:337) -- our 12 s chunker keeps well under it. "
            "lang_code is derived from the voice prefix (af_->a, bf_->b)."
        ),
    ),
    "orpheus": EngineSpec(
        key="orpheus",
        family="llama",
        checkpoint="mlx-community/orpheus-3b-0.1-ft-bf16",
        sampling=ORPHEUS_SAMPLING,
        preset_voices=True,
        accepts_instruct=False,
        chunks_long_lines=True,
        sample_rate=24000,
        notes=(
            "Canopy Labs (US), Llama backbone, 8 preset voices. 1200-token cap "
            "is ~14 s of audio, so chunking is mandatory. Also supports "
            "ref-audio cloning (unused this round; the -ft checkpoint is "
            "preset-tuned)."
        ),
    ),
    "soprano": EngineSpec(
        key="soprano",
        family="soprano",
        checkpoint="mlx-community/Soprano-80M-bf16",
        sampling=SOPRANO_SAMPLING,
        preset_voices=False,
        accepts_instruct=False,
        chunks_long_lines=False,
        sample_rate=32000,
        notes=(
            "RD-1 checkpoint, shared with the Swift port (condition E vs F). "
            "`voice` is accepted and discarded (soprano.py:387). Splits itself "
            "per sentence in `_preprocess_text`; 512-token context at 2048 "
            "samples/token is ~32 s of audio."
        ),
    ),
}

#: Engine keys in declaration order.
ENGINE_KEYS: Final[tuple[str, ...]] = tuple(ENGINE_SPECS)

#: The four engines the Sortie 5 smoke test must cover.
SMOKE_ENGINE_KEYS: Final[tuple[str, ...]] = (
    "qwen3-1.7b",
    "qwen3-0.6b",
    "chatterbox",
    "soprano",
)

#: The engines that can speak in a ``.vox`` reference voice (round 2).
CLONE_ENGINE_KEYS: Final[tuple[str, ...]] = tuple(
    key for key, s in ENGINE_SPECS.items() if s.clone_voices
)


def spec(key: str) -> EngineSpec:
    """Look up an :class:`EngineSpec`, raising ``KeyError`` if unknown."""
    try:
        return ENGINE_SPECS[key]
    except KeyError:
        raise KeyError(
            f"unknown engine {key!r}; known engines: {', '.join(ENGINE_KEYS)}"
        ) from None


# ---------------------------------------------------------------------------
# Requests and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CloneVoice:
    """A reference voice to clone (round 2), decoded and ready to condition on.

    Built from a ``.vox`` bundle by the episode planner
    (:mod:`comparativa.voices.cloned`); the engine layer only sees the decoded
    waveform plus the transcript, never the bundle itself. ``name`` is the
    stable cache key — one prepared conditioning per name per engine.
    """

    #: Stable identifier, e.g. ``"vox:ARCHER"``. Recorded as the line's voice.
    name: str
    #: 1-D float32 mono reference waveform.
    audio: np.ndarray
    sample_rate: int
    #: Transcript of the reference clip; required by qwen3 ICL cloning.
    ref_text: str | None = None
    #: Provenance string for the manifest (vox path + member).
    source: str = ""

    @property
    def seconds(self) -> float:
        return len(self.audio) / self.sample_rate if self.sample_rate else 0.0


@dataclass(frozen=True)
class LineRequest:
    """One unit of speech to generate.

    Maps directly onto Sortie 3's ``SpokenLine``: ``text`` is the whole line,
    ``segments`` are its inter-breath spans (generate each separately and splice
    :attr:`breath_gap_seconds` of silence between them), and ``direction`` is
    the non-spoken performance instruct derived from the parentheticals.
    """

    text: str
    #: Preset speaker id; ``None`` uses the engine's single/default voice.
    voice: str | None = None
    #: Reference voice to clone; requires an engine with ``clone_voices``.
    clone: CloneVoice | None = None
    #: Non-spoken performance instruct; ignored by engines without the capability.
    direction: str | None = None
    #: Inter-breath spans. Empty or single-element means "no breath splicing".
    segments: tuple[str, ...] = ()
    breath_gap_seconds: float = BREATH_GAP_SECONDS
    #: Base RNG seed. ``None`` leaves the RNG untouched (non-reproducible).
    seed: int | None = None
    #: Stable identifier echoed into the record (line index, character, ...).
    label: str = ""

    def units(self) -> tuple[str, ...]:
        """The spans to generate, in order."""
        spans = tuple(s for s in self.segments if s.strip())
        return spans if len(spans) > 1 else ((self.text,) if self.text.strip() else ())


@dataclass(frozen=True)
class ChunkRecord:
    """One model call: what was asked for, and what came back."""

    index: int
    text: str
    seed: int | None
    duration_seconds: float
    check: DurationCheck
    #: True when the first attempt failed the duration check and was retried.
    retried: bool = False
    #: The rejected first attempt's duration, when a retry happened.
    first_attempt_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "index": self.index,
            "chars": len(self.text),
            "seed": self.seed,
            "duration_seconds": round(self.duration_seconds, 3),
            "check": self.check.to_dict(),
        }
        if self.retried:
            d["retried"] = True
            d["first_attempt_seconds"] = round(self.first_attempt_seconds or 0.0, 3)
        return d


@dataclass
class LineResult:
    """Generated audio plus the full provenance of how it was made."""

    engine: str
    checkpoint: str
    voice: str | None
    #: 1-D float32 mono waveform.
    audio: np.ndarray
    sample_rate: int
    sampling: dict[str, Any]
    seed: int | None
    #: How the seed was applied, or :data:`SEEDING_UNAVAILABLE`.
    seeding: str
    chunks: list[ChunkRecord] = field(default_factory=list)
    #: Wall-clock seconds spent inside the model (excludes load).
    generate_seconds: float = 0.0
    direction: str | None = None
    label: str = ""

    @property
    def duration_seconds(self) -> float:
        return len(self.audio) / self.sample_rate if self.sample_rate else 0.0

    @property
    def real_time_factor(self) -> float:
        """Generation seconds per audio second. Lower is faster."""
        d = self.duration_seconds
        return self.generate_seconds / d if d > 0 else float("inf")

    @property
    def truncated(self) -> bool:
        """True when any chunk still failed its duration check after the retry."""
        return any(c.check.truncated for c in self.chunks)

    @property
    def overrun(self) -> bool:
        return any(c.check.overrun for c in self.chunks)

    @property
    def retried(self) -> bool:
        return any(c.retried for c in self.chunks)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "engine": self.engine,
            "checkpoint": self.checkpoint,
            "voice": self.voice,
            "sample_rate": self.sample_rate,
            "duration_seconds": round(self.duration_seconds, 3),
            "generate_seconds": round(self.generate_seconds, 3),
            "real_time_factor": round(self.real_time_factor, 3),
            "sampling": self.sampling,
            "seed": self.seed,
            "seeding": self.seeding,
            "chunks": [c.to_dict() for c in self.chunks],
        }
        if self.direction:
            d["direction"] = self.direction
        if self.label:
            d["label"] = self.label
        if self.truncated:
            d["truncated"] = True
        if self.overrun:
            d["overrun"] = True
        if self.retried:
            d["truncation_retry"] = True
        return d


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


def _silence(seconds: float, sample_rate: int) -> np.ndarray:
    return np.zeros(max(0, int(round(seconds * sample_rate))), dtype=np.float32)


def _as_mono(audio: Any) -> np.ndarray:
    """Flatten an mlx (or numpy) array to a 1-D float32 waveform."""
    arr = np.asarray(audio, dtype=np.float32)
    return arr.reshape(-1)


class Engine:
    """A loaded mlx-audio model, driven through one interface.

    Construct with :func:`load_engine`; the constructor takes an already-loaded
    model so tests can inject a fake.
    """

    def __init__(
        self,
        engine_spec: EngineSpec,
        model: Any,
        *,
        sampling: SamplingParams | None = None,
        load_seconds: float = 0.0,
    ) -> None:
        self.spec = engine_spec
        self.model = model
        self.sampling = sampling or engine_spec.sampling
        self.load_seconds = load_seconds
        self.sample_rate = int(getattr(model, "sample_rate", engine_spec.sample_rate))
        #: Prepared per-clone conditioning (chatterbox ``Conditionals``), keyed
        #: by :attr:`CloneVoice.name` so a character's reference is encoded once.
        self._conds_cache: dict[str, Any] = {}
        #: Reference waveforms converted once to ``mx.array``, same key.
        self._clone_audio_cache: dict[str, Any] = {}

    # -- properties ---------------------------------------------------------

    @property
    def key(self) -> str:
        return self.spec.key

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Engine {self.spec.key} @ {self.spec.checkpoint} sr={self.sample_rate}>"

    # -- voice resolution ---------------------------------------------------

    def resolve_clone(self, clone: CloneVoice | None) -> CloneVoice | None:
        """Validate a clone request against this engine's capabilities."""
        if clone is None:
            if self.spec.clone_voices and self.spec.clone_needs_ref_text:
                raise EngineError(
                    f"engine {self.spec.key!r} is a clone engine and requires a "
                    "CloneVoice on every request (Base checkpoints have no "
                    "preset voices)"
                )
            return None
        if not self.spec.clone_voices:
            raise EngineError(
                f"engine {self.spec.key!r} cannot clone voices; clone engines: "
                f"{', '.join(CLONE_ENGINE_KEYS)}"
            )
        if self.spec.clone_needs_ref_text and not (clone.ref_text or "").strip():
            raise EngineError(
                f"engine {self.spec.key!r} clones via ICL and needs the reference "
                f"transcript, but clone {clone.name!r} has no ref_text"
            )
        if len(clone.audio) == 0:
            raise EngineError(f"clone {clone.name!r} has an empty reference waveform")
        return clone

    def resolve_voice(self, voice: str | None) -> str | None:
        """Validate ``voice`` against the catalog, or supply the default."""
        if not self.spec.preset_voices:
            # Single-voice engines accept anything and ignore it; record the
            # canonical name so the manifest does not imply a choice was made.
            return catalog.DEFAULT_VOICE
        if voice is None:
            raise EngineError(
                f"engine {self.spec.key!r} has preset voices and requires one; "
                f"available: {', '.join(v.name for v in self.spec.voices)}"
            )
        known = {v.name.lower() for v in self.spec.voices}
        if voice.lower() not in known:
            raise EngineError(
                f"voice {voice!r} is not a preset of {self.spec.key!r}; "
                f"available: {', '.join(sorted(known))}"
            )
        return voice

    # -- chunking -----------------------------------------------------------

    def plan_chunks(self, text: str) -> list[str]:
        """Split ``text`` into model calls.

        Mirrors SwiftVoxAlta's ``enableAutoChunking`` behaviour for the engines
        that need it; engines that split themselves get the text whole.
        """
        stripped = text.strip()
        if not stripped:
            return []
        if not self.spec.chunks_long_lines:
            return [stripped]
        if estimate_duration(stripped) <= self.spec.chunk_target_seconds:
            return [stripped]
        return split_at_sentences(stripped, max_duration=self.spec.chunk_target_seconds)

    # -- generation ---------------------------------------------------------

    def _seed(self, seed: int | None) -> str:
        """Apply ``seed`` and report how (FR-10)."""
        if seed is None:
            return "unseeded"
        if not self.spec.seeding:
            return SEEDING_UNAVAILABLE
        import mlx.core as mx

        mx.random.seed(seed)
        return SEEDING_MLX_GLOBAL

    def _clone_ref_audio(self, clone: CloneVoice):
        """The clone's waveform as an ``mx.array`` at the engine's sample rate.

        Converted (and resampled, when the reference's rate differs — e.g. the
        24 kHz ``.vox`` samples on 44.1 kHz Dia) once per clone name.
        """
        cached = self._clone_audio_cache.get(clone.name)
        if cached is None:
            import mlx.core as mx

            audio = clone.audio
            if clone.sample_rate != self.sample_rate:
                from mlx_audio.utils import resample_audio

                audio = np.asarray(
                    resample_audio(audio, clone.sample_rate, self.sample_rate),
                    dtype=np.float32,
                )
            cached = mx.array(audio)
            self._clone_audio_cache[clone.name] = cached
        return cached

    def _chatterbox_conds(self, clone: CloneVoice, exaggeration: float):
        """Prepared ``Conditionals`` for a clone, encoded once per name.

        Both chatterbox families resample the reference internally, so the
        clone's native sample rate is passed through as-is.
        """
        conds = self._conds_cache.get(clone.name)
        if conds is not None:
            return conds
        if self.spec.family == "chatterbox":
            conds = self.model.prepare_conditionals(
                self._clone_ref_audio(clone), clone.sample_rate, exaggeration
            )
        else:  # chatterbox_turbo: prepare stores on the model, returns nothing
            self.model.prepare_conditionals(
                self._clone_ref_audio(clone),
                sample_rate=clone.sample_rate,
                exaggeration=exaggeration,
            )
            conds = self.model._conds
        self._conds_cache[clone.name] = conds
        return conds

    def _call_model(
        self,
        text: str,
        voice: str | None,
        direction: str | None,
        clone: CloneVoice | None = None,
    ):
        """Dispatch to the family-specific ``generate`` and return mono audio."""
        s = self.sampling
        family = self.spec.family

        if family == "qwen3_tts":
            if clone is not None:
                # Base-checkpoint ICL path: ref audio + transcript, no preset
                # voice, no instruct (qwen3_tts.py routes on ref_audio+ref_text).
                results = self.model.generate(
                    text=text,
                    ref_audio=self._clone_ref_audio(clone),
                    ref_text=clone.ref_text,
                    temperature=s.temperature,
                    top_p=s.top_p,
                    top_k=s.top_k if s.top_k is not None else 0,
                    repetition_penalty=s.repetition_penalty,
                    max_tokens=s.max_tokens,
                    lang_code=QWEN3_LANGUAGE,
                    verbose=False,
                )
            else:
                results = self.model.generate(
                    text=text,
                    voice=voice,
                    instruct=direction if self.spec.accepts_instruct else None,
                    temperature=s.temperature,
                    top_p=s.top_p,
                    top_k=s.top_k if s.top_k is not None else 0,
                    repetition_penalty=s.repetition_penalty,
                    max_tokens=s.max_tokens,
                    lang_code=QWEN3_LANGUAGE,
                    verbose=False,
                )
        elif family == "chatterbox":
            exaggeration = s.extra.get("exaggeration", 0.1)
            results = self.model.generate(
                text=text,
                conds=self._chatterbox_conds(clone, exaggeration) if clone else None,
                temperature=s.temperature,
                top_p=s.top_p,
                repetition_penalty=s.repetition_penalty,
                min_p=s.extra.get("min_p", 0.05),
                cfg_weight=s.extra.get("cfg_weight", 0.5),
                exaggeration=exaggeration,
                max_new_tokens=s.max_tokens,
                verbose=False,
            )
        elif family == "chatterbox_turbo":
            if clone is not None:
                # prepare_conditionals stores on the model; restore the cached
                # conditioning so interleaved characters do not re-encode.
                self.model._conds = self._chatterbox_conds(
                    clone, s.extra.get("exaggeration", 0.0)
                )
            results = self.model.generate(
                text=text,
                temperature=s.temperature,
                top_p=s.top_p,
                top_k=s.top_k if s.top_k is not None else 1000,
                repetition_penalty=s.repetition_penalty,
                min_p=s.extra.get("min_p", 0.0),
                max_tokens=s.max_tokens,
            )
        elif family == "dia":
            # Dialogue-native model: a single-character line is one [S1] turn,
            # and the reference transcript is tagged the same way.
            results = self.model.generate(
                text=f"[S1] {text}",
                ref_audio=self._clone_ref_audio(clone) if clone else None,
                ref_text=f"[S1] {clone.ref_text}" if clone else None,
                temperature=s.temperature,
                top_p=s.top_p,
                max_tokens=s.max_tokens or None,
                verbose=False,
            )
        elif family == "sesame":
            results = self.model.generate(
                text=text,
                ref_audio=self._clone_ref_audio(clone) if clone else None,
                ref_text=clone.ref_text if clone else None,
                max_audio_length_ms=s.extra.get("max_audio_length_ms", 90_000.0),
                verbose=False,
            )
        elif family == "higgs_audio":
            results = self.model.generate(
                text=text,
                ref_audio=self._clone_ref_audio(clone) if clone else None,
                ref_text=clone.ref_text if clone else None,
                temperature=s.temperature,
                top_p=s.top_p,
                max_new_frames=s.max_tokens,
                verbose=False,
            )
        elif family == "kokoro":
            # No sampling knobs; lang_code follows the voice prefix (af_->a).
            results = self.model.generate(
                text=text,
                voice=voice,
                lang_code=(voice or "a")[0],
            )
        elif family == "llama":
            results = self.model.generate(
                text=text,
                voice=voice,
                temperature=s.temperature,
                top_p=s.top_p,
                max_tokens=s.max_tokens,
                verbose=False,
            )
        elif family == "soprano":
            results = self.model.generate(
                text=text,
                temperature=s.temperature,
                top_p=s.top_p,
                max_tokens=s.max_tokens,
                verbose=False,
            )
        else:  # pragma: no cover - unreachable while ENGINE_SPECS is closed
            raise EngineError(f"no generation adapter for family {family!r}")

        pieces = [_as_mono(r.audio) for r in results if r is not None and r.audio is not None]
        if not pieces:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(pieces) if len(pieces) > 1 else pieces[0]

    def _generate_chunk(
        self,
        index: int,
        text: str,
        voice: str | None,
        direction: str | None,
        seed: int | None,
        clone: CloneVoice | None = None,
    ) -> tuple[np.ndarray, ChunkRecord, str]:
        """Generate one chunk, retrying once if it fails the duration check."""
        seeding = self._seed(seed)
        audio = self._call_model(text, voice, direction, clone)
        duration = len(audio) / self.sample_rate if self.sample_rate else 0.0
        check = check_duration(text, duration)

        if check.ok:
            return audio, ChunkRecord(index, text, seed, duration, check), seeding

        # One retry, on a different point of the sample stream (Risk §8.3).
        retry_seed = None if seed is None else seed + 1_000_003
        seeding = self._seed(retry_seed)
        retry_audio = self._call_model(text, voice, direction, clone)
        retry_duration = len(retry_audio) / self.sample_rate if self.sample_rate else 0.0
        retry_check = check_duration(text, retry_duration)

        # Keep whichever attempt is closer to the expected duration.
        if abs(retry_check.ratio - 1.0) <= abs(check.ratio - 1.0):
            kept, kept_check, kept_duration = retry_audio, retry_check, retry_duration
        else:
            kept, kept_check, kept_duration = audio, check, duration

        record = ChunkRecord(
            index=index,
            text=text,
            seed=retry_seed,
            duration_seconds=kept_duration,
            check=kept_check,
            retried=True,
            first_attempt_seconds=duration,
        )
        return kept, record, seeding

    def generate_line(self, request: LineRequest) -> LineResult:
        """Generate one line of speech with fixed, recorded parameters."""
        clone = self.resolve_clone(request.clone)
        voice = clone.name if clone is not None else self.resolve_voice(request.voice)
        direction = request.direction if self.spec.accepts_instruct else None

        result = LineResult(
            engine=self.spec.key,
            checkpoint=self.spec.checkpoint,
            voice=voice,
            audio=np.zeros(0, dtype=np.float32),
            sample_rate=self.sample_rate,
            sampling=self.sampling.to_dict(),
            seed=request.seed,
            seeding=SEEDING_UNAVAILABLE if not self.spec.seeding else "unseeded",
            direction=direction,
            label=request.label,
        )

        pieces: list[np.ndarray] = []
        chunk_index = 0
        started = time.perf_counter()

        units = request.units()
        for unit_index, unit in enumerate(units):
            if unit_index > 0 and request.breath_gap_seconds > 0:
                pieces.append(_silence(request.breath_gap_seconds, self.sample_rate))

            chunks = self.plan_chunks(unit)
            for chunk_number, chunk_text in enumerate(chunks):
                if chunk_number > 0 and self.spec.chunk_pause_seconds > 0:
                    pieces.append(
                        _silence(self.spec.chunk_pause_seconds, self.sample_rate)
                    )
                seed = None if request.seed is None else request.seed + chunk_index
                audio, record, seeding = self._generate_chunk(
                    chunk_index, chunk_text, voice, direction, seed, clone
                )
                result.seeding = seeding
                result.chunks.append(record)
                pieces.append(audio)
                chunk_index += 1

        result.generate_seconds = time.perf_counter() - started
        if pieces:
            result.audio = np.concatenate(pieces).astype(np.float32, copy=False)
        return result


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_engine(
    key: str,
    *,
    sampling: SamplingParams | None = None,
    checkpoint: str | None = None,
) -> Engine:
    """Load ``key``'s checkpoint from the local Hugging Face cache.

    Nothing is downloaded when ``HF_HUB_OFFLINE=1`` is set; a missing checkpoint
    surfaces as :class:`EngineError` naming the repo id rather than a network
    fetch.
    """
    engine_spec = spec(key)
    repo = checkpoint or engine_spec.checkpoint

    from mlx_audio.tts.utils import load as load_tts_model

    started = time.perf_counter()
    try:
        model = load_tts_model(repo)
    except Exception as exc:  # noqa: BLE001 - re-raised with the repo id attached
        raise EngineError(
            f"could not load engine {key!r} from checkpoint {repo!r}: {exc}"
        ) from exc
    load_seconds = time.perf_counter() - started

    return Engine(engine_spec, model, sampling=sampling, load_seconds=load_seconds)
