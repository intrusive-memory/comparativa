"""End-to-end episode generation: screenplay in, audio + ``manifest.json`` out.

Three phases, deliberately separable so ``--dry-run`` and Sortie 7's bench
runner can stop after the first:

1. :func:`build_plan` — parse the screenplay, prepare its speech (Sortie 3),
   resolve every character's voice from ``presets.yaml`` (Sortie 4), and assign
   a deterministic per-line seed. Loads no model and touches no checkpoint, so
   it is fast enough to run in a unit test.
2. :func:`generate_episode` — load the engine **once** and generate every line
   through :meth:`Engine.generate_line`, then hand the results to
   :func:`~comparativa.generation.assembly.assemble`.
3. :func:`write_outputs` — episode ``.wav``, ``.m4a`` (``afconvert``), and
   ``manifest.json``.

Manifest schema (``schema_version`` 1) — the contract Sorties 7/9/11 read:

``engine``/``voices``/``speech_policy``/``assembly``/``seed``
    Exactly what produced this run, including the active speech policy (the
    ``fr3`` vs ``produciesta-parity`` choice changes what an episode *contains*,
    so a manifest that does not record it is not comparable to anything).
``totals``
    Episode duration, gap total, wall-clock, RTF, and the truncation/loudness
    counters worth reporting per condition.
``lines[]``
    One record per **spoken line of the script**, in script order. Each is
    Sortie 5's ``LineResult.to_dict()`` (engine, checkpoint, voice, sampling,
    seed, seeding, per-chunk records, ``truncated``/``overrun``/
    ``truncation_retry``) merged with the planned-line metadata (character,
    text, element/scene indices, source lines) and the assembly fields
    (``offset_seconds``, ``assembled_duration_seconds``, ``gap_before_seconds``,
    ``trim``, ``loudness``).

Two duration fields per line, on purpose:

``duration_seconds``
    What the model produced, before trimming and normalization. This is the
    number to compare against the engine layer's own accounting.
``assembled_duration_seconds``
    What the line actually occupies on the episode timeline. ``Σ
    assembled_duration_seconds + Σ gap_before_seconds == totals.duration_seconds``
    exactly, to the sample.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from .. import __version__
from ..parsing import (
    PARSER_NAME,
    PARSER_VERSION,
    POLICIES,
    FR3_POLICY,
    SpeechPolicy,
    find_cast_file,
    load_cast_index,
    parse_file,
    prepare_script,
)
from ..voices.assign import PRESETS_FILENAME, load_presets, voice_for
from .assembly import AssembledEpisode, AssemblyOptions, LineAudio, assemble
from .encode import M4A_BITRATE, encode_m4a, write_wav
from .engines import Engine, EngineError, LineRequest, load_engine, spec

#: Manifest format version. Bump on any breaking schema change.
SCHEMA_VERSION: Final = 1

#: Default base seed. Fixed so two runs of the same condition are comparable.
DEFAULT_SEED: Final = 20260815

#: Per-line seed spacing. The engine layer derives each chunk's seed as
#: ``line_seed + chunk_index``, so the stride must exceed any plausible chunk
#: count for a single line or two lines would share a sample stream.
SEED_STRIDE: Final = 1000

#: Repository root, where the committed ``presets.yaml`` lives.
REPO_ROOT: Final = Path(__file__).resolve().parents[3]
DEFAULT_PRESETS_PATH: Final = REPO_ROOT / PRESETS_FILENAME


class GenerationError(RuntimeError):
    """An episode cannot be planned or generated."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannedLine:
    """One spoken line, resolved down to everything the engine needs."""

    index: int
    character: str
    voice: str | None
    text: str
    segments: tuple[str, ...]
    breath_gap_seconds: float
    direction: str | None
    scene_index: int
    element_index: int
    element_type: str
    start_line: int
    end_line: int
    raw_cue: str | None = None
    seed: int | None = None

    @property
    def label(self) -> str:
        return f"{self.index:04d}:{self.character}"

    def request(self) -> LineRequest:
        """The engine-layer request for this line."""
        return LineRequest(
            text=self.text,
            voice=self.voice,
            direction=self.direction,
            segments=self.segments,
            breath_gap_seconds=self.breath_gap_seconds,
            seed=self.seed,
            label=self.label,
        )

    def record(self) -> dict[str, Any]:
        """The script-side half of this line's manifest record."""
        d: dict[str, Any] = {
            "index": self.index,
            "character": self.character,
            "text": self.text,
            "element_index": self.element_index,
            "element_type": self.element_type,
            "scene_index": self.scene_index,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }
        if self.raw_cue is not None:
            d["raw_cue"] = self.raw_cue
        if len(self.segments) > 1:
            d["segments"] = list(self.segments)
            d["breath_gap_seconds"] = self.breath_gap_seconds
        return d


@dataclass
class EpisodePlan:
    """Everything decided before a model is loaded."""

    episode_path: Path
    episode_sha256: str
    engine_key: str
    speech_policy: str
    text_policy: str
    cast_source: str | None
    presets_path: Path | None
    presets_sha256: str | None
    lines: list[PlannedLine] = field(default_factory=list)
    #: character -> assigned voice, for the characters actually speaking here.
    voice_by_character: dict[str, str | None] = field(default_factory=dict)
    unresolved_cues: list[dict[str, Any]] = field(default_factory=list)
    #: Spoken lines in the script before ``--limit`` truncated the plan.
    script_line_count: int = 0
    seed_base: int | None = DEFAULT_SEED
    seed_stride: int = SEED_STRIDE

    @property
    def characters(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for line in self.lines:
            counts[line.character] = counts.get(line.character, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    @property
    def scene_count(self) -> int:
        return len({line.scene_index for line in self.lines})

    def summary(self) -> str:
        """A human-readable dry-run report."""
        rows = [
            f"episode:       {self.episode_path}",
            f"engine:        {self.engine_key}  ({spec(self.engine_key).checkpoint})",
            f"speech policy: {self.speech_policy}  (text: {self.text_policy})",
            f"cast:          {self.cast_source or 'not resolved'}",
            f"presets:       {self.presets_path or 'not used'}",
            f"lines:         {len(self.lines)} of {self.script_line_count} spoken",
            f"scenes:        {self.scene_count}",
            f"seed:          {self.seed_base} (stride {self.seed_stride})",
            "",
            f"{'CHARACTER':<18} {'LINES':>5}  VOICE",
            f"{'-' * 18} {'-' * 5}  {'-' * 20}",
        ]
        for character, count in self.characters.items():
            rows.append(
                f"{character:<18} {count:>5}  {self.voice_by_character.get(character) or '-'}"
            )
        return "\n".join(rows)


def build_plan(
    episode: str | Path,
    engine_key: str,
    *,
    presets_path: str | Path | None = DEFAULT_PRESETS_PATH,
    cast_path: str | Path | None = None,
    use_cast: bool = True,
    speech_policy: SpeechPolicy | str = FR3_POLICY,
    seed: int | None = DEFAULT_SEED,
    seed_stride: int = SEED_STRIDE,
    limit: int | None = None,
) -> EpisodePlan:
    """Parse, prepare, and voice-resolve an episode without loading a model."""
    engine_spec = spec(engine_key)

    policy = POLICIES[speech_policy] if isinstance(speech_policy, str) else speech_policy

    episode_file = Path(episode).expanduser()
    if not episode_file.is_file():
        raise GenerationError(f"episode not found: {episode_file}")

    stream = parse_file(episode_file)

    cast = None
    if use_cast:
        cast_file = Path(cast_path).expanduser() if cast_path else find_cast_file(episode_file)
        if cast_file is not None:
            cast = load_cast_index(cast_file)

    script = prepare_script(stream, cast, policy=policy)

    presets_file = Path(presets_path).expanduser() if presets_path else None
    presets: dict[str, Any] = {}
    if presets_file is not None:
        if not presets_file.is_file():
            raise GenerationError(
                f"presets file not found: {presets_file} "
                "(run `comparativa voices <project-dir> --write`)"
            )
        presets = load_presets(presets_file) or {}

    plan = EpisodePlan(
        episode_path=episode_file,
        episode_sha256=_sha256(episode_file),
        engine_key=engine_key,
        speech_policy=script.policy,
        text_policy=script.text_policy,
        cast_source=script.cast_source,
        presets_path=presets_file,
        presets_sha256=_sha256(presets_file) if presets_file is not None else None,
        unresolved_cues=[c.to_dict() for c in script.unresolved_cues],
        script_line_count=len(script.lines),
        seed_base=seed,
        seed_stride=seed_stride,
    )

    selected = script.lines if limit is None else script.lines[: max(0, limit)]

    missing: list[str] = []
    for line in selected:
        character = line.character
        if character not in plan.voice_by_character:
            assigned = voice_for(presets, character, engine_key) if presets else None
            if assigned is None and engine_spec.preset_voices:
                missing.append(character)
            plan.voice_by_character[character] = assigned

        plan.lines.append(
            PlannedLine(
                index=len(plan.lines),
                character=character,
                voice=plan.voice_by_character[character],
                text=line.text,
                segments=line.prepared.segments,
                breath_gap_seconds=line.prepared.breath_gap_seconds,
                direction=line.direction,
                scene_index=line.scene_index,
                element_index=line.element_index,
                element_type=line.element_type,
                start_line=line.start_line,
                end_line=line.end_line,
                raw_cue=line.raw_cue,
                seed=None if seed is None else seed + len(plan.lines) * seed_stride,
            )
        )

    if missing:
        names = ", ".join(sorted(set(missing)))
        raise GenerationError(
            f"engine {engine_key!r} needs a preset voice per character, but "
            f"{presets_file or 'presets.yaml'} assigns none to: {names}"
        )

    return plan


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    """A generated episode: audio, per-line records, and the manifest."""

    plan: EpisodePlan
    assembled: AssembledEpisode
    manifest: dict[str, Any]
    load_seconds: float = 0.0
    wall_seconds: float = 0.0

    @property
    def audio(self):
        return self.assembled.audio

    @property
    def sample_rate(self) -> int:
        return self.assembled.sample_rate

    @property
    def duration_seconds(self) -> float:
        return self.assembled.duration_seconds


#: Called after each line with ``(plan_line, line_result_record)``.
ProgressHook = Callable[[PlannedLine, dict[str, Any]], None]


def build_manifest(
    plan: EpisodePlan,
    assembled: AssembledEpisode,
    *,
    engine: Engine | None = None,
    load_seconds: float = 0.0,
    wall_seconds: float = 0.0,
    outputs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble the ``manifest.json`` document for one generated episode."""
    engine_spec = spec(plan.engine_key)
    totals = assembled.totals()
    totals["load_seconds"] = round(load_seconds, 3)
    totals["wall_seconds"] = round(wall_seconds, 3)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "comparativa generate",
        "comparativa_version": __version__,
        "episode": {
            "path": str(plan.episode_path),
            "name": plan.episode_path.stem,
            "sha256": plan.episode_sha256,
        },
        "parser": {"name": PARSER_NAME, "version": PARSER_VERSION},
        "speech_policy": plan.speech_policy,
        "text_policy": plan.text_policy,
        "engine": {
            "key": engine_spec.key,
            "family": engine_spec.family,
            "checkpoint": engine_spec.checkpoint,
            "sample_rate": assembled.sample_rate,
            "capabilities": engine_spec.capabilities(),
            "sampling": (engine.sampling if engine is not None else engine_spec.sampling).to_dict(),
        },
        "voices": {
            "source": "presets.yaml (RD-2 defaults-only)",
            "presets_path": str(plan.presets_path) if plan.presets_path else None,
            "presets_sha256": plan.presets_sha256,
            "cast_source": plan.cast_source,
            "by_character": dict(plan.voice_by_character),
        },
        "seed": {"base": plan.seed_base, "stride": plan.seed_stride},
        "assembly": assembled.options.to_dict(),
        "totals": totals,
        "script_line_count": plan.script_line_count,
        "unresolved_cues": plan.unresolved_cues,
        "outputs": outputs or {},
        "lines": assembled.lines,
    }


def generate_episode(
    plan: EpisodePlan,
    *,
    engine: Engine | None = None,
    options: AssemblyOptions = AssemblyOptions(),
    progress: ProgressHook | None = None,
) -> EpisodeResult:
    """Generate every planned line and assemble them into one episode.

    ``engine`` is loaded once when not supplied — never per line — because a
    checkpoint load is tens of seconds and would swamp the RTF numbers Sortie 7
    reports.
    """
    started = time.perf_counter()
    if engine is None:
        try:
            engine = load_engine(plan.engine_key)
        except EngineError as exc:
            raise GenerationError(str(exc)) from exc
    load_seconds = engine.load_seconds

    generated: list[LineAudio] = []
    for line in plan.lines:
        result = engine.generate_line(line.request())
        record = line.record()
        record.update(result.to_dict())
        record.pop("label", None)  # the index/character fields already say this
        generated.append(
            LineAudio(
                audio=result.audio,
                sample_rate=result.sample_rate,
                scene_index=line.scene_index,
                record=record,
            )
        )
        if progress is not None:
            progress(line, record)

    assembled = assemble(generated, options=options)
    wall_seconds = time.perf_counter() - started

    manifest = build_manifest(
        plan,
        assembled,
        engine=engine,
        load_seconds=load_seconds,
        wall_seconds=wall_seconds,
    )
    return EpisodeResult(
        plan=plan,
        assembled=assembled,
        manifest=manifest,
        load_seconds=load_seconds,
        wall_seconds=wall_seconds,
    )


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

#: The manifest's filename inside the output directory. Fixed, so a bench run
#: can point at ``bench/<cond>/<episode>/manifest.json`` without guessing.
MANIFEST_FILENAME: Final = "manifest.json"


def write_outputs(
    result: EpisodeResult,
    out_dir: str | Path,
    *,
    name: str | None = None,
    m4a: bool = True,
    m4a_bitrate: int = M4A_BITRATE,
) -> dict[str, str]:
    """Write the episode ``.wav``, ``.m4a``, and ``manifest.json``.

    One run per directory: the manifest filename is fixed, so pointing two runs
    at the same ``-o`` would overwrite the first.
    """
    directory = Path(out_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    stem = name or result.plan.episode_path.stem

    outputs: dict[str, str] = {}
    wav_path = write_wav(result.audio, result.sample_rate, directory / f"{stem}.wav")
    outputs["wav"] = str(wav_path)

    if m4a:
        outputs["m4a"] = str(
            encode_m4a(wav_path, directory / f"{stem}.m4a", bitrate=m4a_bitrate)
        )

    result.manifest["outputs"] = outputs
    manifest_path = directory / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(result.manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    outputs["manifest"] = str(manifest_path)
    return outputs


def manifest_line_count(manifest: dict[str, Any] | str | Path) -> int:
    """Number of line records in a manifest (accepts a path or a document)."""
    document = manifest
    if isinstance(manifest, (str, Path)):
        document = json.loads(Path(manifest).expanduser().read_text(encoding="utf-8"))
    return len(document.get("lines", []))  # type: ignore[union-attr]


def spoken_line_count(
    episode: str | Path,
    *,
    cast_path: str | Path | None = None,
    use_cast: bool = True,
    speech_policy: SpeechPolicy | str = FR3_POLICY,
) -> int:
    """Spoken-line count ``comparativa parse`` reports for ``episode``.

    Shares :func:`build_plan`'s parse path exactly, so the integration test's
    two sides cannot drift apart through a policy mismatch.
    """
    policy = POLICIES[speech_policy] if isinstance(speech_policy, str) else speech_policy
    episode_file = Path(episode).expanduser()
    stream = parse_file(episode_file)
    cast = None
    if use_cast:
        cast_file = Path(cast_path).expanduser() if cast_path else find_cast_file(episode_file)
        if cast_file is not None:
            cast = load_cast_index(cast_file)
    return len(prepare_script(stream, cast, policy=policy).lines)


__all__: Sequence[str] = (
    "DEFAULT_PRESETS_PATH",
    "DEFAULT_SEED",
    "MANIFEST_FILENAME",
    "SCHEMA_VERSION",
    "SEED_STRIDE",
    "EpisodePlan",
    "EpisodeResult",
    "GenerationError",
    "PlannedLine",
    "build_manifest",
    "build_plan",
    "generate_episode",
    "manifest_line_count",
    "spoken_line_count",
    "write_outputs",
)
