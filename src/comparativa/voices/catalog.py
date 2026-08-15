"""Built-in / default voice catalog for every engine comparativa can drive.

Round 1 is **defaults-only** (Resolved Decision RD-2): no cloning, no `.vox`
extraction. Every engine therefore contributes either a list of preset speakers
(qwen3 CustomVoice checkpoints) or a single built-in voice (chatterbox's
pre-computed conditionals, Soprano's one hardcoded voice).

Provenance of every list below — all determined by *static* inspection of the
installed ``mlx-audio==0.4.8`` package and of config files already present in
the local Hugging Face cache. No model was loaded and nothing was downloaded.

* qwen3 preset speakers — ``mlx_audio/tts/models/qwen3_tts/qwen3_tts.py:193``
  builds ``supported_speakers`` from ``config.talker_config.spk_id``; the
  cached ``Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16/config.json`` carries exactly
  nine ids. ``qwen3_tts/README.md:136-142`` documents their languages, and
  ``spk_is_dialect`` in the same config marks ``eric`` (Sichuan) and ``dylan``
  (Beijing) as dialect speakers.
* chatterbox / chatterbox-turbo default voice — ``chatterbox.py:608-647`` and
  ``chatterbox_turbo.py:317`` load ``conds.safetensors`` from the checkpoint as
  the "builtin voice"; it is the only voice available without a reference clip.
* soprano — ``soprano/soprano.py:365,387``: the ``voice`` argument is accepted
  and then discarded (``_ = voice``). Soprano has exactly one voice.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Voice id used by every engine that exposes exactly one built-in voice.
DEFAULT_VOICE = "default"


@dataclass(frozen=True)
class Voice:
    """One selectable built-in voice offered by an engine."""

    name: str
    #: ``male`` / ``female`` / ``neutral`` (unknown or not gendered).
    gender: str
    #: ``child`` / ``adult`` / ``elder`` / ``any``.
    age_band: str
    #: BCP-47-ish primary language of the preset, or ``mixed``.
    language: str
    #: False when the voice exists but must not be auto-assigned (see ``note``).
    assignable: bool = True
    note: str = ""


@dataclass(frozen=True)
class Engine:
    """An engine plus the built-in voices it can use without cloning."""

    key: str
    checkpoint: str
    voices: tuple[Voice, ...]
    #: Where the voice list came from, recorded verbatim in ``presets.yaml``.
    source: str
    #: True when the engine lets us choose between several built-in voices.
    multi_voice: bool = True
    #: False when the checkpoint was not present in the local HF cache and the
    #: voice list could therefore not be confirmed against a config.json.
    voices_verified_locally: bool = True
    #: Round-1 role of this engine ("condition C", "integration tests", ...).
    role: str = ""

    @property
    def assignable_voices(self) -> tuple[Voice, ...]:
        return tuple(v for v in self.voices if v.assignable)

    def voice(self, name: str) -> Voice | None:
        for v in self.voices:
            if v.name == name:
                return v
        return None


# --- qwen3 CustomVoice preset speakers ---------------------------------------
#
# Order is the order of the ``spk_id`` map in the checkpoint config, so the
# round-robin assignment below is reproducible from the checkpoint itself.
_QWEN3_SPEAKERS: tuple[Voice, ...] = (
    Voice("serena", "female", "adult", "zh"),
    Voice("vivian", "female", "adult", "zh"),
    Voice("uncle_fu", "male", "elder", "zh", note="Elder-presenting Chinese storyteller preset."),
    Voice("ryan", "male", "adult", "en"),
    Voice("aiden", "male", "adult", "en"),
    Voice(
        "ono_anna",
        "female",
        "adult",
        "ja",
        note="Present in spk_id but undocumented in the mlx-audio README; language inferred from the name.",
    ),
    Voice(
        "sohee",
        "female",
        "adult",
        "ko",
        note="Present in spk_id but undocumented in the mlx-audio README; language inferred from the name.",
    ),
    Voice(
        "eric",
        "male",
        "adult",
        "zh",
        assignable=False,
        note=(
            "spk_is_dialect='sichuan_dialect'. qwen3_tts.py:397-406 forces the Sichuan "
            "language id whenever language is chinese/auto, which would corrupt English "
            "dialogue. Excluded from auto-assignment."
        ),
    ),
    Voice(
        "dylan",
        "male",
        "adult",
        "zh",
        assignable=False,
        note=(
            "spk_is_dialect='beijing_dialect'. Same forced-dialect behaviour as eric. "
            "Excluded from auto-assignment."
        ),
    ),
)

_QWEN3_SOURCE = (
    "spk_id map in the checkpoint config.json (read from the local HF cache); "
    "consumed by mlx_audio/tts/models/qwen3_tts/qwen3_tts.py:193"
)

ENGINES: tuple[Engine, ...] = (
    Engine(
        key="qwen3-1.7b",
        checkpoint="mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
        voices=_QWEN3_SPEAKERS,
        source=_QWEN3_SOURCE,
        multi_voice=True,
        voices_verified_locally=True,
        role="condition C (Python qwen3 presets)",
    ),
    Engine(
        key="qwen3-0.6b",
        checkpoint="mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16",
        voices=_QWEN3_SPEAKERS,
        source=(
            _QWEN3_SOURCE
            + " -- for 0.6B the speaker set is assumed identical to 1.7B "
            "(qwen3_tts/README.md:136-142 documents one shared Base/CustomVoice speaker "
            "set); the 0.6B checkpoint is NOT in the local HF cache (only refs/main), so "
            "this list could not be confirmed against its own config.json"
        ),
        multi_voice=True,
        voices_verified_locally=False,
        role="size-degradation probe / integration tests",
    ),
    Engine(
        key="chatterbox",
        checkpoint="mlx-community/chatterbox-fp16",
        voices=(
            Voice(
                DEFAULT_VOICE,
                "neutral",
                "any",
                "en",
                note="Pre-computed conds.safetensors shipped with the checkpoint (chatterbox.py:608-647).",
            ),
        ),
        source="conds.safetensors in the checkpoint; mlx_audio/tts/models/chatterbox/chatterbox.py:608-647",
        multi_voice=False,
        role="condition D (Python chatterbox default)",
    ),
    Engine(
        key="chatterbox-turbo",
        checkpoint="mlx-community/chatterbox-turbo-fp16",
        voices=(
            Voice(
                DEFAULT_VOICE,
                "neutral",
                "any",
                "en",
                note="'builtin_voice' conds.safetensors (chatterbox_turbo.py:317).",
            ),
        ),
        source="conds.safetensors in the checkpoint; mlx_audio/tts/models/chatterbox_turbo/chatterbox_turbo.py:317",
        multi_voice=False,
        role="optional speed probe",
    ),
    Engine(
        key="soprano",
        checkpoint="mlx-community/Soprano-80M-bf16",
        voices=(
            Voice(
                DEFAULT_VOICE,
                "neutral",
                "any",
                "en",
                note="Soprano's only voice; the `voice` argument is discarded (soprano.py:387).",
            ),
        ),
        source="mlx_audio/tts/models/soprano/soprano.py:365,377,387 -- voice argument unused",
        multi_voice=False,
        role="condition E (Python Soprano, paired with Swift condition F)",
    ),
)

#: Engine keys in catalog order.
ENGINE_KEYS: tuple[str, ...] = tuple(e.key for e in ENGINES)


def engine(key: str) -> Engine:
    """Look up an engine by key, raising ``KeyError`` if it is unknown."""
    for e in ENGINES:
        if e.key == key:
            return e
    raise KeyError(f"unknown engine {key!r}; known engines: {', '.join(ENGINE_KEYS)}")
