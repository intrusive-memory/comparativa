---
type: reference
state: current
updated: 2026-08-16
---

# CORPUS_PIN.md — why `corpus/frozen/` exists, and the rule that comes with it

OPERATION BATTLING BARDS benchmarks comparativa's Python TTS stack against the
granville podcast project (`~/Projects/podcasts/granville`). That project is
**not** part of this repo, is not under this mission's control, and is a live
working tree with its own commit history.

## Why the pin exists

During the mission, a separate Claude Code session was actively rewriting the
granville project concurrently with our benchmark runs: a new character
(MICKEY) was added to `CAST.md` and `PROJECT.md`, `voices/MICKEY.vox` was
added, several episode `.fountain` files were rewritten and committed
(`6b15f8e`), and a `render_queue.sh` re-rendered reference audio into
`granville/audio/`.

The two corpus episodes benchmarked in this mission —
`episode_1_01_cold_open.fountain` and
`episode_1_01a_bumper_donnie_and_arnie_1.fountain` — were **not** among the
files that session touched, but nothing in the live tree guarantees that
stays true from one moment to the next. A test suite (or a bench run) that
reads `~/Projects/podcasts/granville` directly is reading a moving target: it
can pass one run and fail the next for reasons that have nothing to do with
comparativa's code, purely because CAST.md gained a character or a script was
rewritten out from under it.

So the mission supervisor extracted a **read-only, point-in-time snapshot**
of exactly the inputs this mission was benchmarked against, and pinned every
test and corpus-path default to resolve it from there instead of the live
tree.

## What is pinned, and at what revision

All extraction was **read-only** against the granville git history — `git
show <rev>:<path>` / `git archive <rev> <path>` piped straight to files under
`corpus/frozen/`. Nothing was written into `~/Projects/podcasts/granville` to
produce this snapshot, and the live tree's `git status` is unaffected by it.

| path | source rev | why this rev |
|---|---|---|
| `corpus/frozen/episodes/episode_1_01_cold_open.fountain` | `c909b1d5f5e4fd750945f1e8cfaf357c28b4ebb2` | the exact revision benchmarked in `bench/A/metrics.json` |
| `corpus/frozen/episodes/episode_1_01a_bumper_donnie_and_arnie_1.fountain` | `d7dd1ead26b2ff59bda9612dcdabf34c24201c3a` | the exact revision benchmarked in `bench/A/metrics.json` |
| `corpus/frozen/CAST.md` | `bfd6df500f75fb54b58dbb1ecd63b92c67311165` | last commit before MICKEY was added to the cast roster |
| `corpus/frozen/PROJECT.md` | `bfd6df500f75fb54b58dbb1ecd63b92c67311165` | same pre-MICKEY revision, kept in sync with CAST.md |
| `corpus/frozen/voices/*.vox` | `bfd6df500f75fb54b58dbb1ecd63b92c67311165` | same pre-MICKEY revision — matches the cast committed presets.yaml was generated from |

### SHA-256 of the frozen files

```
da8ff1a547038d7c0dc3a7bc13db812ed0509c4512c153e93bef3477494d9481  episodes/episode_1_01_cold_open.fountain
8365d42cfc4d9a3ba9d98901886ec783f78d0e6dcb5f1b5230f3cc8aa5c3e5b4  episodes/episode_1_01a_bumper_donnie_and_arnie_1.fountain
eb4665acd70d696c5f5894c83bb6492fe420086d6ae6b35a9ce318623c13daba  CAST.md
0433e215a3745ef76008a8d39ca9c553d313ba74e5fa0c74f49ee73bdb6350cd  PROJECT.md
```

The two episode hashes are the same `episode_sha256` values recorded in
`bench/A/metrics.json` — that is the verification that the frozen snapshot
and the benchmarked inputs are byte-identical.

`corpus/frozen/` is untracked (see `.gitignore`: `/corpus/`), the same way
`bench/` is untracked — it is a generated/extracted artifact, not source the
repo owns, and it is machine-local scratch that any contributor can
regenerate from the same revisions above.

## How resolution works

A single resolution point, `tests/conftest.py::resolve_corpus_root()`
(exposed as the `CORPUS_ROOT` constant), decides where "the corpus" is for
every test and CLI default that needs it:

1. `$COMPARATIVA_CORPUS_ROOT`, if set — explicit override, e.g. for a CI
   runner or a machine that keeps the snapshot somewhere else.
2. `<repo>/corpus/frozen`, if it exists — the pinned snapshot described above.
3. `~/Projects/podcasts/granville` — last-resort fallback on a machine
   without the frozen snapshot; individual tests still skip via their own
   `requires_corpus`-style marker if that path is also absent.

Every test file that previously hardcoded
`Path("~/Projects/podcasts/granville").expanduser()` now imports
`CORPUS_ROOT` from `conftest` and builds its paths from that instead. Nothing
about what the tests assert changed — only where they look for their input
files.

## The rule: `presets.yaml` is not regenerated mid-mission

`presets.yaml` (voice assignments per cast member per engine) was generated
once, from the pre-MICKEY cast roster, and is committed to the repo.
**It must not be regenerated against `corpus/frozen/CAST.md` — or against the
live granville tree — for the remainder of this mission.**

Regenerating it would shift condition-C voice assignments mid-benchmark:
every already-collected bench entry that references a character's assigned
voice would silently stop matching what a fresh `voices` run produces,
invalidating comparisons across sorties that ran before vs. after the
regeneration. The frozen `CAST.md` snapshot exists precisely so that
`uv run comparativa voices corpus/frozen` continues to reproduce the
already-committed `presets.yaml` byte-for-byte (this is asserted by
`tests/test_voices.py::test_presets_file_is_committed_and_current`) — it is a
verification fixture, not a signal that presets should be re-derived.

If a future mission adds cast members (MICKEY or otherwise) and needs new
voice presets, that is new, deliberate work with its own sortie — not a
side effect of pointing tests at a frozen snapshot.
