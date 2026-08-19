"""The round-1 conditions matrix, as config (EXECUTION_PLAN.md § Conditions Matrix).

A *condition* is one (stack, engine, voice-source) triple that the listening
test and the performance tables compare. Eight ids exist; six of them are what
``comparativa bench`` can actually run:

======== ====== ================== =============================== ==========
id       stack  engine/ckpt        voices                          runner
======== ====== ================== =============================== ==========
A        swift  Qwen3 1.7B bf16    ``.vox`` production voices      external
B        python qwen3-1.7b-clone   ``.vox``-cloned per character   bench
C        python qwen3-1.7b         built-in presets, per character bench
D        python chatterbox fp16    engine default voice            bench
E        python Soprano-80M-bf16   single built-in voice           bench
F        swift  Soprano-80M-bf16   single built-in voice           external
G        python chatterbox fp16    ``.vox``-cloned per character   bench (opt)
T        python chatterbox-turbo   engine default voice            bench (opt)
======== ====== ================== =============================== ==========

**A and F belong to Sortie 8**, not to this runner: they are produced by the
signed ``produciesta`` binary and by mlx-audio-swift respectively. They are
declared here anyway so that both sorties agree on the condition ids, the
labels, and the ``metrics.json`` entries' ``condition``/``stack`` fields —
``bench`` refuses to run them rather than pretending they do not exist.

**T** (chatterbox-turbo) is the optional speed probe. It is not in
:data:`DEFAULT_CONDITIONS` and is not a listened round-1 condition; ask for it
explicitly with ``--conditions T``.

**B** (round 2) is the qwen3 ``.vox``-cloned condition RD-2 deferred: the
Python Base checkpoint ICL-clones every character's production voice from the
same ``.vox`` bundles condition A uses, so **A vs B** is the Swift-vs-Python
comparison with the voice-design confound removed. **G** (optional) is the
chatterbox counterpart, cloned from the same references. Both require a
``presets-cloned.yaml`` (``comparativa voices <project> --mode cloned
--write``); their :attr:`Condition.presets` carries that file to ``generate``.

Every condition run by this module uses ``--speech-policy produciesta-parity``
(:data:`BENCH_SPEECH_POLICY`) — a logged supervisor ruling: the Swift baseline
narrates action lines and sluglines, so A-vs-C is only valid if the Python side
does too. The policy is recorded in each run's manifest *and* in its metrics
entry, so a future run with a different policy can never be silently pooled
with these.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ..generation.engines import ENGINE_SPECS

#: Supervisor ruling: every bench condition narrates like the Swift stack.
BENCH_SPEECH_POLICY: Final = "produciesta-parity"

#: Where a condition's runs are written, relative to the bench root.
#: ``bench/<condition>/<episode>/`` — one run per directory.
BENCH_ROOT: Final = "bench"

#: Stack identifiers used in ``metrics.json`` entries.
STACK_PYTHON: Final = "python"
STACK_SWIFT: Final = "swift"

#: ``Condition.runner`` values.
RUNNER_BENCH: Final = "comparativa"
RUNNER_EXTERNAL: Final = "external"


class ConditionError(ValueError):
    """An unknown or unrunnable condition was requested."""


@dataclass(frozen=True)
class Condition:
    """One row of the round-1 matrix."""

    id: str
    stack: str
    label: str
    voices: str
    #: comparativa engine key; ``None`` for the Swift-side conditions.
    engine: str | None = None
    #: Checkpoint repo id. ``None`` where Sortie 8 must record what it actually
    #: loaded (condition A's Swift voice pipeline picks its own).
    checkpoint: str | None = None
    runner: str = RUNNER_BENCH
    #: True for probes that are not part of the listened round-1 set.
    optional: bool = False
    #: Presets file handed to ``generate --presets`` (repo-root relative);
    #: ``None`` uses the generate command's own default.
    presets: str | None = None
    notes: str = ""

    @property
    def runnable(self) -> bool:
        """True when ``comparativa bench`` itself can produce this condition."""
        return self.runner == RUNNER_BENCH

    def describe(self) -> str:
        engine = self.engine or "-"
        return f"{self.id}  {self.stack:<6} {engine:<17} {self.voices}"

    def to_dict(self) -> dict[str, object]:
        """The condition half of a ``metrics.json`` entry."""
        return {
            "id": self.id,
            "stack": self.stack,
            "label": self.label,
            "engine": self.engine,
            "checkpoint": self.checkpoint,
            "voices": self.voices,
            "runner": self.runner,
            "optional": self.optional,
        }


def _checkpoint(engine_key: str) -> str:
    return ENGINE_SPECS[engine_key].checkpoint


CONDITIONS: Final[dict[str, Condition]] = {
    "A": Condition(
        id="A",
        stack=STACK_SWIFT,
        label="Produciesta (Swift), Qwen3 1.7B, production .vox voices",
        voices=".vox (current production voices, as-is)",
        engine=None,
        checkpoint=None,
        runner=RUNNER_EXTERNAL,
        notes=(
            "Sortie 8 regenerates this with the signed ~/.local/bin/produciesta "
            "binary and records the checkpoint it actually loaded."
        ),
    ),
    "B": Condition(
        id="B",
        stack=STACK_PYTHON,
        label="comparativa, qwen3-1.7b Base, .vox-cloned voices (ICL)",
        voices=".vox-cloned per character (presets-cloned.yaml; same bundles as A)",
        engine="qwen3-1.7b-clone",
        checkpoint=_checkpoint("qwen3-1.7b-clone"),
        presets="presets-cloned.yaml",
        notes=(
            "Round 2. The clean Swift-vs-Python Qwen3 pair with condition A: "
            "same .vox voices, same 1.7B model family, different stack. "
            "Requires presets-cloned.yaml (voices --mode cloned --write)."
        ),
    ),
    "C": Condition(
        id="C",
        stack=STACK_PYTHON,
        label="comparativa, qwen3-1.7b, built-in preset voices",
        voices="built-in preset speakers, auto-assigned per character (presets.yaml)",
        engine="qwen3-1.7b",
        checkpoint=_checkpoint("qwen3-1.7b"),
    ),
    "D": Condition(
        id="D",
        stack=STACK_PYTHON,
        label="comparativa, chatterbox fp16, engine default voice",
        voices="engine default (the checkpoint's conds.safetensors)",
        engine="chatterbox",
        checkpoint=_checkpoint("chatterbox"),
    ),
    "E": Condition(
        id="E",
        stack=STACK_PYTHON,
        label="comparativa, Soprano-80M-bf16, single built-in voice",
        voices="single built-in voice (every character shares it)",
        engine="soprano",
        checkpoint=_checkpoint("soprano"),
        notes="RD-1 checkpoint; the clean Python-vs-Swift port pair with F.",
    ),
    "F": Condition(
        id="F",
        stack=STACK_SWIFT,
        label="mlx-audio-swift Soprano, single built-in voice",
        voices="single built-in voice (every character shares it)",
        engine=None,
        checkpoint=_checkpoint("soprano"),
        runner=RUNNER_EXTERNAL,
        optional=True,
        notes="Stretch condition; Sortie 8 writes outputs or bench/F/SKIPPED.md.",
    ),
    "G": Condition(
        id="G",
        stack=STACK_PYTHON,
        label="comparativa, chatterbox fp16, .vox-cloned voices",
        voices=".vox-cloned per character (presets-cloned.yaml, 1.7b reference clips)",
        engine="chatterbox",
        checkpoint=_checkpoint("chatterbox"),
        presets="presets-cloned.yaml",
        notes=(
            "Round 2/3 challenger: chatterbox conditioned on the production "
            ".vox references (Resemble AI, Canada)."
        ),
    ),
    "H": Condition(
        id="H",
        stack=STACK_PYTHON,
        label="comparativa, Higgs Audio v2 3B (q8), .vox-cloned voices",
        voices=".vox-cloned per character (presets-cloned.yaml, ref audio + transcript)",
        engine="higgs",
        checkpoint=_checkpoint("higgs"),
        presets="presets-cloned.yaml",
        notes=(
            "Round 3 challenger (Boson AI, US; Llama backbone). Only quantized "
            "MLX conversions exist — q8 is a stated quality caveat."
        ),
    ),
    "N": Condition(
        id="N",
        stack=STACK_PYTHON,
        label="comparativa, Dia-1.6B fp16, .vox-cloned voices",
        voices=".vox-cloned per character (presets-cloned.yaml, ref audio + transcript)",
        engine="dia",
        checkpoint=_checkpoint("dia"),
        presets="presets-cloned.yaml",
        notes="Round 3 challenger (Nari Labs, South Korea; dialogue-native, from scratch).",
    ),
    "S": Condition(
        id="S",
        stack=STACK_PYTHON,
        label="comparativa, Sesame CSM-1B fp16, .vox-cloned voices",
        voices=".vox-cloned per character (presets-cloned.yaml, ref audio + transcript)",
        engine="csm",
        checkpoint=_checkpoint("csm"),
        presets="presets-cloned.yaml",
        notes="Round 3 challenger (Sesame AI, US; Llama backbone).",
    ),
    "K": Condition(
        id="K",
        stack=STACK_PYTHON,
        label="comparativa, Kokoro-82M bf16, preset voices",
        voices="built-in preset speakers, auto-assigned per character (presets.yaml)",
        engine="kokoro",
        checkpoint=_checkpoint("kokoro"),
        notes=(
            "Round 3 presets control (indie/hexgrad, US; StyleTTS2 lineage). "
            "28 English presets; deterministic, no sampling knobs."
        ),
    ),
    "O": Condition(
        id="O",
        stack=STACK_PYTHON,
        label="comparativa, Orpheus 3B ft bf16, preset voices",
        voices="built-in preset speakers, auto-assigned per character (presets.yaml)",
        engine="orpheus",
        checkpoint=_checkpoint("orpheus"),
        notes="Round 3 presets probe (Canopy Labs, US; Llama backbone; 8 voices).",
    ),
    "T": Condition(
        id="T",
        stack=STACK_PYTHON,
        label="comparativa, chatterbox-turbo fp16, engine default voice (speed probe)",
        voices="engine default (the checkpoint's conds.safetensors)",
        engine="chatterbox-turbo",
        checkpoint=_checkpoint("chatterbox-turbo"),
        optional=True,
        notes="Optional speed probe; not a listened round-1 condition.",
    ),
}

#: Condition ids in matrix order.
CONDITION_IDS: Final[tuple[str, ...]] = tuple(CONDITIONS)

#: What ``bench`` runs when ``--conditions`` is not given: the round-3 active
#: roster — the .vox-cloned challengers plus the two presets controls. The
#: retired round-1/2 conditions (B/C/D/E) stay runnable by explicit id.
DEFAULT_CONDITIONS: Final[tuple[str, ...]] = ("G", "H", "N", "S", "K", "O")


def condition(name: str) -> Condition:
    """Look up a condition by id, case-insensitively."""
    try:
        return CONDITIONS[name.strip().upper()]
    except KeyError:
        raise ConditionError(
            f"unknown condition {name!r}; known conditions: "
            f"{', '.join(CONDITION_IDS)}"
        ) from None


def parse_conditions(spec: str | None) -> list[Condition]:
    """Resolve a ``--conditions C,D,E`` string, preserving matrix order.

    ``all`` selects every non-optional condition; ``None`` or an empty string
    selects :data:`DEFAULT_CONDITIONS`. Duplicates collapse.
    """
    if spec is None or not spec.strip():
        names: list[str] = list(DEFAULT_CONDITIONS)
    elif spec.strip().lower() == "all":
        names = [c.id for c in CONDITIONS.values() if not c.optional]
    else:
        names = [part for part in (p.strip() for p in spec.split(",")) if part]
    if not names:
        raise ConditionError("no conditions requested")

    seen: dict[str, Condition] = {}
    for name in names:
        found = condition(name)
        seen[found.id] = found
    return [CONDITIONS[cid] for cid in CONDITION_IDS if cid in seen]


def matrix_table() -> str:
    """The conditions matrix as a printable table."""
    rows = [
        f"{'ID':<3} {'STACK':<6} {'ENGINE':<17} {'RUNNER':<11} VOICES",
        f"{'-' * 3} {'-' * 6} {'-' * 17} {'-' * 11} {'-' * 40}",
    ]
    for cond in CONDITIONS.values():
        marker = " (optional)" if cond.optional else ""
        rows.append(
            f"{cond.id:<3} {cond.stack:<6} {(cond.engine or '-'):<17} "
            f"{cond.runner:<11} {cond.voices}{marker}"
        )
    return "\n".join(rows)


__all__ = [
    "BENCH_ROOT",
    "BENCH_SPEECH_POLICY",
    "CONDITIONS",
    "CONDITION_IDS",
    "DEFAULT_CONDITIONS",
    "RUNNER_BENCH",
    "RUNNER_EXTERNAL",
    "STACK_PYTHON",
    "STACK_SWIFT",
    "Condition",
    "ConditionError",
    "condition",
    "matrix_table",
    "parse_conditions",
]
