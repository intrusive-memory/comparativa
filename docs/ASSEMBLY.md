---
type: reference
state: current
updated: 2026-08-15
---

# ASSEMBLY.md — episode assembly, loudness, and the manifest schema

Deliverable of OPERATION BATTLING BARDS, Sortie 6 (FR-9, RD-3).

This document records what happens between "the engine produced 189 line
waveforms" and "there is an episode on disk": which silence is trimmed, what
loudness the lines are normalized to, where the gaps come from, and what every
field in `manifest.json` means. Sorties 7 (bench), 9 (listen/report), and 11
(verdict) all read that manifest, so its schema is a contract.

**Implementation**: `src/comparativa/generation/assembly.py` (timeline),
`src/comparativa/generation/episode.py` (plan, generation, manifest),
`src/comparativa/generation/encode.py` (WAV/`.m4a`),
`src/comparativa/generation/command.py` (CLI),
`src/comparativa/generation/audition.py` (`voices --audition`).
**Tests**: `tests/test_generate.py`.

---

## 1. Who owns which silence

Three layers insert silence, and they must not double up:

| Silence | Duration | Owner | Source |
|---|---|---|---|
| Chunk pause (inside a long line, at a sentence boundary) | 0.25 s | engine layer (Sortie 5) | `GenerationSettings.chunkPauseDuration` |
| Breath gap (at an authored `[[<breath/>]]`) | 0.15 s | engine layer (Sortie 5) | `InlineBreath.breathGapSeconds` |
| **Inter-line gap** | **0.25 s** (`--line-gap`) | **assembly (this doc)** | comparativa default |
| **Inter-scene gap** | **0.75 s** (`--scene-gap`) | **assembly (this doc)** | comparativa default |

`LineResult.audio` already contains the first two. Assembly adds only the last
two, and only *between placed lines* — a line that generated no audio occupies
no time and contributes no gap, so a silent failure cannot silently stretch the
episode.

A scene gap is used whenever `scene_index` differs between two consecutively
*placed* lines; otherwise the line gap is used.

## 2. Seam trimming (Produciesta parity)

Every TTS generation is padded with head and tail silence whose length is the
model's business, not the script's. Produciesta strips that padding at internal
seams with `WAVAssembler.trimSilence`
(`ProduciestaCore/Sources/ProduciestaCore/Pipeline/BreathRendering.swift:181-212`),
keeping the clip's outer envelope natural:

* silence threshold: amplitude `350` of int16 full scale ≈ **−39.4 dBFS**
  (`BreathRendering.swift:148`)
* guard kept either side of a trimmed edge: **8 ms** (`BreathRendering.swift:151`)
* `trimLeading` is off for the first audio part, `trimTrailing` off for the last
  (`SwiftVoxAltaAdapter.swift:150-162`)

`comparativa` ports this with the same two constants, applied at **episode**
seams: the first placed line keeps its natural head, the last keeps its natural
tail, every other edge is tightened. The audible pause between two lines is
therefore governed by `--line-gap` rather than by how much silence the model
felt like emitting. `--no-trim-seams` disables it.

Scope note: Produciesta applies the trim at *breath-span* seams inside a line;
`comparativa`'s engine layer does not trim its intra-line chunk and breath
seams (Sortie 5 concatenates them raw). See § 6, divergence **A-4**.

## 3. Loudness (RD-3)

Each line is normalized to **−16 LUFS integrated** with
`pyloudnorm.Meter` (ITU-R BS.1770-4, 400 ms block), measured *after* seam
trimming. Then a **sample-peak guard** at 0.99 scales the line back if the
loudness gain would have pushed it past full scale.

The guard matters, and it is the one place the target is not always met.
Speech at −16 LUFS with a natural, unlimited crest factor routinely needs more
headroom than 0 dBFS leaves: on the cold open at `qwen3-0.6b`, roughly two
thirds of lines are peak-guarded and land ~1–2 dB under target. The alternative
— hard clipping or a soft limiter — would inject exactly the kind of artifact
the listening test is supposed to score, so the honest choice is to take the
shortfall and **record it**:

* per line: `loudness.peak_limited` and `loudness.shortfall_db`
* per episode: `totals.peak_limited_lines`, `totals.mean_output_lufs`,
  `totals.max_loudness_shortfall_db`, `totals.mean_loudness_shortfall_db`

Because every condition is metered the same way, a loudness delta between two
conditions is a real difference in what the models produced, not a difference
in how they were measured. Condition A's Swift audio must be measured with this
same meter for the report's loudness row to mean anything (Sortie 8).

Lines that cannot be measured are left at their generated level with the reason
recorded rather than dropped: shorter than the 400 ms BS.1770 block, digital
silence, or gated to −∞ LUFS. `--no-normalize` disables normalization entirely;
`--peak-ceiling 0` disables the guard (and permits clipping).

## 4. Outputs

One run writes into one directory (`-o`), because `manifest.json` has a fixed
name:

| File | Content |
|---|---|
| `<episode>.wav` | 16-bit mono PCM at the engine's rate — the canonical deliverable, and what the listening set and any objective metric should read |
| `<episode>.m4a` | AAC-LC via `/usr/bin/afconvert` — what the podcast pipeline ships |
| `manifest.json` | § 5 |

**Sample rate is per engine, not per pipeline**: qwen3 and chatterbox emit
24 kHz, Soprano emits 32 kHz. Assembly refuses to lay lines of different rates
on one timeline (`AssemblyError`); one engine is loaded per episode, so this
only fires on a programming error.

**`.m4a` bitrate defaults to 64 kbps**, not the podcast-typical 128: AAC-LC at
24 kHz mono *rejects* anything above 64 kbps (`afconvert` fails with
`Couldn't set audio converter property ('!dat')`), and 64 kbps is already past
transparent for a 24 kHz mono speech signal. A rejected rate falls back to
`afconvert`'s own choice rather than failing the run. `--no-m4a` skips the
conversion.

## 5. `manifest.json` — schema version 1

Top level:

| Key | Meaning |
|---|---|
| `schema_version` | `1`. Bumped on any breaking change. |
| `episode` | `path`, `name`, `sha256` of the screenplay. |
| `parser` | jouvence name and version (Sortie 2). |
| `speech_policy` | `fr3` or `produciesta-parity`. **Load-bearing** — it changes what the episode *contains*, so two manifests with different policies are not comparable. |
| `text_policy` | The `TextPrepPolicy` name (`parity`). |
| `engine` | `key`, `family`, `checkpoint`, `sample_rate`, `capabilities` (preset voices, seeding, chunking), `sampling` (the full recorded parameter set). |
| `voices` | `presets_path` + `presets_sha256`, the `CAST.md` used, and `by_character` — the character → voice map actually used. |
| `seed` | `base` and `stride`; line *n*'s seed is `base + n * stride`, and the engine derives chunk seeds from it. |
| `assembly` | Every `AssemblyOptions` value, including the loudness meter's name. |
| `totals` | § 5.1 |
| `script_line_count` | Spoken lines in the script, before `--limit`. |
| `unresolved_cues` | Cues that matched no `CAST.md` character. Empty is healthy. |
| `outputs` | Absolute paths of the files written. |
| `lines[]` | § 5.2 |

### 5.1 `totals`

`line_count`, `placed_line_count`, `duration_seconds`, `audio_seconds`,
`gap_seconds`, `generate_seconds`, `real_time_factor`, `load_seconds`,
`wall_seconds`, `truncated_lines`, `overrun_lines`, `truncation_retry_lines`,
`peak_limited_lines`, `unnormalized_lines`, `mean_output_lufs`,
`max_loudness_shortfall_db`, `mean_loudness_shortfall_db`.

`duration_seconds` is derived from the assembled sample count, never by summing
the backend's reported durations, so it cannot drift from the bytes on disk.
`audio_seconds + gap_seconds == duration_seconds` exactly.

Sortie 7's `metrics.json` should take wall-clock, RTF, and model-load time from
here rather than re-deriving them; peak RSS is the only performance number
assembly does not know.

### 5.2 `lines[]`

One record per **spoken line of the script**, in script order — including any
line that generated no audio, so `len(lines)` always equals the parse's spoken
line count.

*Script side* (from Sortie 3): `index`, `character`, `text`, `raw_cue`,
`element_index`, `element_type`, `scene_index`, `start_line`, `end_line`, and
`segments` + `breath_gap_seconds` when the line has authored breaths.

*Engine side* (Sortie 5's `LineResult.to_dict()`): `engine`, `checkpoint`,
`voice`, `sample_rate`, `duration_seconds`, `generate_seconds`,
`real_time_factor`, `sampling`, `seed`, `seeding`, `chunks[]`, `direction`, and
the flags `truncated` / `overrun` / `truncation_retry` (present only when true).

*Assembly side*: `offset_seconds`, `assembled_duration_seconds`,
`gap_before_seconds`, `trim` (`head_seconds`, `tail_seconds`), `loudness`
(`normalized`, `input_lufs`, `output_lufs`, `requested_gain_db`,
`applied_gain_db`, `peak_before`, `peak_after`, `peak_limited`, `shortfall_db`,
`reason`). A line that produced no audio gets `placed: false` instead of
`trim` / `loudness`.

**Two duration fields, on purpose.** `duration_seconds` is what the model
produced, before trimming and normalization — compare it with the engine
layer's own accounting. `assembled_duration_seconds` is what the line occupies
on the episode timeline. The timeline invariant is:

```
Σ assembled_duration_seconds + Σ gap_before_seconds == totals.duration_seconds
```

exact to the sample. This is the invariant `test_full_cold_open_episode`
asserts (with the plan's ±0.1 s-per-line tolerance).

## 6. Divergences from the Swift stack

Numbered `A-*` to sit alongside `docs/TEXT_PREP.md`'s `D-*` register without
renumbering it.

| id | Topic | Swift (Produciesta) | Python (comparativa) | Reason |
|---|---|---|---|---|
| A-1 | Inter-line silence | None. Voiced clips are butted on the timeline; silence appears only for an authored `<pause>` (`TimelineComposer.swift:117-201`). | 0.25 s inter-line, 0.75 s inter-scene, both configurable. | An episode of butted clips reads as rushed, and FR-9 asks for configurable gaps. `--line-gap 0 --scene-gap 0` reproduces the Swift timing exactly. |
| A-2 | Loudness | Not normalized; whatever the model emitted is what ships. | −16 LUFS integrated per line, peak-guarded at 0.99 (RD-3). | RD-3. Condition A must be *measured* with the same meter (Sortie 8) so the report can state the delta. |
| A-3 | `<pause>` / `<include>` directives | Honoured — pauses become offset gaps, includes are placed on a second lane (`TimelineComposer`). | Not implemented. Neither corpus episode uses them. | Out of Sortie 6's scope; would matter for an episode that does use them. |
| A-4 | Intra-line seam trimming | Trimmed at every breath seam inside a line (`SwiftVoxAltaAdapter.swift:150-162`). | Not trimmed inside a line; trimmed only at inter-line seams. | The engine layer (Sortie 5) concatenates chunk and breath pieces raw. Effect is bounded by the model's own padding at each intra-line seam; corpus lines average well under one breath each. |
| A-5 | `.m4a` encode | AVFoundation `AVAssetExportSession` / `kAudioFormatMPEG4AAC`. | `/usr/bin/afconvert -f m4af -d aac -q 127 -b 64000`. | Same system AAC-LC encoder, driven from a different API. The WAV is the canonical artifact either way. |

## 7. Commands

```bash
# Plan only — no checkpoint loaded, prints the cast/voice table
uv run comparativa generate <episode.fountain> --engine qwen3-0.6b --dry-run

# Full episode (offline; all checkpoints are cached)
HF_HUB_OFFLINE=1 uv run comparativa generate \
    ~/Projects/podcasts/granville/episodes/episode_1_01_cold_open.fountain \
    --engine qwen3-0.6b -o out/

# Produciesta-parity contents (narrates action and sluglines) instead of FR-3
... --speech-policy produciesta-parity

# One fixed sentence per character x engine assignment
HF_HUB_OFFLINE=1 uv run comparativa voices ~/Projects/podcasts/granville \
    --audition --engines qwen3-0.6b --audition-characters HUNTER,JOANN

# Tests
uv run pytest tests/test_generate.py           # fast, fake model
uv run pytest tests/test_generate.py --smoke   # + the full cold-open episode
```

## 8. Verification status — 2026-08-15 (Sortie 6)

* Full cold open, `qwen3-0.6b`, `fr3` policy: 189 lines, manifest line count ==
  `parse` spoken-line count, timeline invariant exact.
* `.wav`, `.m4a`, and `manifest.json` all produced by one `generate` run.
* Audition verified on a two-character subset of the granville cast on
  `qwen3-0.6b`.
