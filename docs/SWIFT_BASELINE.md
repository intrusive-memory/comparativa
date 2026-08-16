---
type: reference
state: current
updated: 2026-08-15
---

# SWIFT_BASELINE.md — condition A (Produciesta), and why condition F is missing

Deliverable of OPERATION BATTLING BARDS, Sortie 8. Condition A is the Swift
production baseline the whole comparison is measured against; condition F (the
Swift Soprano port) is **skipped** — see `docs/CONDITION_F_RECON.md`.

The Swift stack is **measured, never modified**. Nothing here builds anything:
the only Swift artifact invoked is the pre-existing signed binary at
`~/.local/bin/produciesta` (→ `~/Projects/apps/Produciesta/bin/produciesta`).

## 1. Running it

```bash
# both corpus episodes (the default), fresh audio into bench/A/
scripts/bench_condition_a.sh

# one episode
scripts/bench_condition_a.sh episode_1_01a_bumper_donnie_and_arnie_1

# re-run a cell that already has audio
FORCE=1 scripts/bench_condition_a.sh episode_1_01_cold_open

# then, always: turn the run dirs into metrics.json
uv run python scripts/bench_condition_a_metrics.py
```

The wrapper is idempotent: a cell whose `<episode>.m4a` exists is skipped unless
`FORCE=1`. The metrics script is safe to re-run at any time — it only reads the
run dirs and rewrites `metrics.json`.

Environment overrides: `PRODUCIESTA`, `PROJECT_DIR`, `BENCH_DIR`, `SCRATCH_DIR`,
`FORCE`.

## 2. How the granville tree stays read-only

`produciesta export` writes exactly three things, and all three are redirected:

| what | where it would go | where it goes here |
|---|---|---|
| composed `.m4a` | `--out` | `bench/A/<episode>/<episode>.m4a` |
| sidecar `.vtt` | `<out>.vtt` → next to `--out` | same run dir |
| ephemeral store / render temp | `~/Library/Application Support/Produciesta` | `--cache-dir` = a fresh `mktemp -d`, deleted after the run |

`PROJECT.md` and `voices/` are passed explicitly (`--project-md`,
`--voices-dir`) and only ever read. No copy of the project is needed. Verified
by a full `stat`-based snapshot of all 606 files in the tree before and after a
run: byte-identical, and the tree's `git status` is unchanged (it carries two
pre-existing modifications to `episode_1_02` / `episode_1_03`, neither of which
is a corpus episode).

The 2026-08-09 `granville/audio/*.m4a` are reference artifacts and are never
read or written by this sortie (Decision §9.4).

## 3. What each number comes from

| metric | source |
|---|---|
| `wall_seconds` | `/usr/bin/time -l` → the `real` figure (includes model load) |
| `peak_rss_bytes` | `/usr/bin/time -l` → `maximum resident set size`, bytes on macOS |
| `peak_memory_footprint_bytes` | same block, recorded as an extra |
| `model_load_seconds` | **`null`** — the produciesta CLI does not report it separately |
| `line_count`, `audio_seconds` | the sidecar `.vtt`: one cue per Element; `audio_seconds` is the sum of cue durations |
| `duration_seconds`, `sample_rate` | `soundfile` on the `afconvert` decode |
| `gap_seconds` | `duration − Σ cue durations` (Produciesta butts clips together, so this is ~0) |
| `mean_output_lufs` | **our** meter — `pyloudnorm.Meter` (ITU-R BS.1770-4), per VTT cue, averaged |
| `episode_integrated_lufs` | the same meter over the whole episode |
| `mean_shortfall_db` / `max_shortfall_db` | **`null`** — Produciesta applies no LUFS target, so a shortfall is undefined rather than zero |
| `checkpoint` | read out of the run's own log: `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` (= `VoxAltaConfig.default.renderModel`) |
| `tool_versions` | `produciesta --version`, and `produciesta version` for the SwiftVoxAlta version and spec hash |

Two metrics files are written, both valid per `docs/BENCH.md` § 4:
`bench/A/metrics.json` (both episodes) and `bench/A/<episode>/metrics.json`
(one entry each). `metrics.load_entries` de-duplicates on
`(stack, condition, episode)`; the per-episode copies exist because
`eval.blind.discover_clips` and `eval.metrics` glob `*/*/metrics.json`.

The metrics script also **keeps** the `afconvert` decode as
`bench/A/<episode>/<episode>.wav`. `eval.blind._find_audio` only ever globs
`*.wav`, and Produciesta exports `.m4a` only — without the decode the Swift
baseline would silently vanish from the blinded listening set. The AAC
round-trip is therefore in the listened signal for condition A and not for the
Python conditions; the report should say so.

## 4. Results — 2026-08-15

Produciesta 1.0.0 / SwiftVoxAlta 0.14.2 (spec-hash `a0364c00…`), macOS 27.0,
arm64, checkpoint `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16`, voices = the
production `.vox` clone prompts, sampling `temp=0.7 topP=0.9 repPenalty=1.3
maxTokens=16384` (logged per line by `VoiceLockManager`).

| episode | lines | audio s | wall s | RTF | peak RSS | mean line LUFS | episode LUFS |
|---|---|---|---|---|---|---|---|
| `episode_1_01a_bumper_donnie_and_arnie_1` | 21 | 140.75 | 350.22 | 2.488 | 8.61 GiB | −19.73 | −17.59 |
| `episode_1_01_cold_open` | 208 | 1075.69 | 2860.76 † | 2.659 † | 8.61 GiB | −21.07 | −18.77 |

`model_load_seconds` is `null` for both. Peak memory *footprint* (the other
`/usr/bin/time -l` figure) was ~24.6 GB on both runs — the RSS number is the one
the schema carries, and it is the one comparable to `psutil` on the Python side.

† **The cold-open timing is contaminated** and should be treated as an upper
bound. From ~18:19 PDT a separate Claude Code session working in the granville
project ran an `echada` voice-design pass and then a second concurrent
`produciesta export`, overlapping roughly the last 23 of this run's 48 minutes.
The full caveat is in `bench/A/episode_1_01_cold_open/notes.txt` and in the
entry's `notes`. The bumper run was uncontended. `peak_rss_bytes` is per-process
and is unaffected. **Sortie 10 must re-measure condition A on the cold open when
the machine is quiet.**

Against Sortie 7's Python numbers on the *same* bumper (15 lines, 141.45 s audio
for C): condition A is ~1.8× slower per audio-second than condition C
(RTF 2.49 vs 1.35) on the same 1.7 B checkpoint family, uses ~2× the peak RSS
(8.61 GiB vs 4.40 GiB), and lands ~3 dB quieter (−19.7 vs −16.9 mean line LUFS,
because it applies no LUFS target at all).

### The compose-stage failure, and the cache-dir workaround

The cold open's **first** attempt generated all 208 elements successfully and
then died in compose:

```
error[adapter-error]: The operation could not be completed
```

exit 4, after 2721.73 s. `ProduciestaModelStore` documents this class of error:
the pipeline's per-run SwiftData store keeps generated audio in a
`.store_SUPPORT/_EXTERNAL_DATA` sidecar, and a failure to read those blobs back
at compose time surfaces as exactly this `adapter-error`. The attempt used
`--cache-dir` pointed at a scratch dir under `/private/tmp`; re-running with
`CACHE_MODE=default` (Produciesta's own Application Support store, still outside
the granville tree) succeeded on the first try.

One data point is not a diagnosis — it may be scale (208 elements vs the
bumper's 21) rather than the path. But the operational rule for Sortie 10 is
clear: **run full-length episodes with `CACHE_MODE=default`**, and expect the
bumper to work either way.

## 5. Parity findings the report must state

These are differences between what condition A *actually renders* and what the
Python conditions render from the same `.fountain`. They are not defects to fix
in this mission — they are confounds to declare.

### 5.1 The two stacks do not segment the script the same way

On `episode_1_01a_bumper_donnie_and_arnie_1`, Produciesta renders **21**
Elements where comparativa's `produciesta-parity` policy produces **15** lines.
The difference is Fountain lyrics (`~ …`): Produciesta makes each lyric line its
own Element, comparativa groups a consecutive run of them into one line. Same
words, different utterance boundaries — so per-line numbers (`line_count`,
mean line LUFS) are **not** directly comparable between A and C, while
per-episode numbers (`audio_seconds`, `wall_seconds`) are.

### 5.2 Condition A voices the bumper's singing as NARRATOR, condition C as KEVIN

`produciesta cast` resolves exactly one speaking character for that bumper —
`NARRATOR: voices/NARRATOR.vox` — and the run log confirms all 21 Elements were
rendered with the NARRATOR voice. Comparativa attributes the two lyric blocks to
KEVIN. So on this episode A-vs-C differs by *casting*, not only by stack and
voice design. The cold open does not have this problem: A distributes across
seven voices (NARRATOR 52, HUNTER 62, KEVIN 53, JOANN 17, RAY 16, TOBY 9,
GARETH 4 — 213 renders for 208 Elements, i.e. 5 regenerations).

### 5.3 Condition A applies no loudness normalization

Produciesta writes lines at whatever level the model produced: mean line
loudness −19.7 (bumper) / −21.1 LUFS (cold open) against comparativa's −16 LUFS
target. `mean_shortfall_db` / `max_shortfall_db` are `null`, not `0`. If the
listening test is not level-matched, the Swift baseline will simply sound
quieter, and quieter reliably scores worse. **Sortie 10 should level-match the
blinded clips**, or the loudness delta becomes an uncontrolled variable in every
naturalness score.

### 5.4 Condition A's listened audio has been through AAC

Produciesta exports `.m4a` only; the `.wav` in the run dir is our `afconvert`
decode of it. The Python conditions are native `.wav`. The AAC round-trip is in
the A signal and not in the C/D/E signal.

### 5.5 The granville corpus is not frozen, and its reference audio is being replaced

During this sortie a separate Claude Code session was actively editing the
granville project: `CAST.md` and `PROJECT.md` rewritten, `voices/MICKEY.vox`
added, several episode `.fountain` files rewritten and committed (`6b15f8e`),
and a `render_queue.sh` re-rendering seven pieces into `granville/audio/` —
overwriting the "2026-08-09 reference-only" artifacts Decision §9.4 refers to.

**The two corpus episodes were not among them.** `episode_1_01_cold_open.fountain`
and `episode_1_01a_bumper_donnie_and_arnie_1.fountain` are byte-identical
(size *and* mtime) to the snapshot taken at the start of this sortie, and their
SHA-256 is recorded in every metrics entry. But nothing guarantees that holds
for the rest of the mission. Sortie 10 should re-verify `episode_sha256` against
condition A's entries before pooling any A-vs-C comparison.
