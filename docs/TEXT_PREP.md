---
type: reference
state: current
updated: 2026-08-15
---

# TEXT_PREP.md — TTS text preparation, and how it lines up with the Swift stack

Deliverable of OPERATION BATTLING BARDS, Sortie 3 (FR-2, FR-3, FR-4).

This document records, transform by transform, what `comparativa` does to
screenplay text before it reaches a TTS engine, which Swift source line each
behaviour was matched to, and every place the two stacks deliberately differ.
FR-4 requires the two stacks to speak *the same strings* wherever feasible, so
the comparison measures the port and the voices rather than a text-prep
mismatch.

**Implementation**: `src/comparativa/parsing/textprep.py` (transforms),
`src/comparativa/parsing/cues.py` (FR-2), `src/comparativa/parsing/speech.py`
(FR-3 and the walk). **Tests**: `tests/test_textprep.py`.

---

## 1. How this was verified

Two independent sources:

1. **Reading the Swift stack** (read-only): `~/Projects/apps/Produciesta`,
   `~/Projects/package-collection/pkg/SwiftVoxAlta`, and the vendored
   `SwiftCompartido` sources they parse with.
2. **The shipped transcripts**: `~/Projects/podcasts/granville/audio/*.vtt`.
   Produciesta writes each `<v SPEAKER>` cue from `VoicingUnit.speakableText` —
   the *exact* string it handed the TTS backend — so the `.vtt` is ground truth
   for what the Swift stack said, not a paraphrase of it.

The parity test
(`test_produciesta_parity_reproduces_the_shipped_transcript`) asserts that
under the `produciesta-parity` speech policy, comparativa reproduces the cold
open's transcript **exactly: 208 of 208 lines, speaker and text**. That is the
evidence behind every "matched" row below.

## 2. The Swift text path

```
GuionElementSnapshot.speakableText      strip {{stage directions}}; trim ends
  → VoicingPlanBuilder.build            parenthetical → non-spoken `instruct`
                                        scene heading → SluglineSpeech.expand
                                        action/heading/centered/… → NARRATOR
  → SwiftVoxAltaAdapter.generate        InlineBreath.mergedSegments splits on
                                        [[<breath/>]]; 0.15 s silence per seam
  → SwiftVoxAlta                        VoiceLockManager.splitAtSentences packs
                                        sentences into ≤ 12 s chunks
```

`comparativa` mirrors this order in `textprep.prepare()` and
`speech.prepare_script()`.

## 3. Matched behaviours

| # | Behaviour | Swift source | comparativa |
|---|-----------|--------------|-------------|
| M-1 | Cue extensions stripped, then trimmed and uppercased. All trailing parentheticals are removed, repeatedly, so `RAY (V.O.) (CONT'D)` → `RAY`. | `ProduciestaCore/Domain/CharacterNameNormalizer.swift:6-23` | `cues.normalize_cue` |
| M-2 | Dual-dialogue `^` and forced-cue `@` are not part of the name. | `SwiftCompartido/.../GuionParsedScreenplay+Characters.swift:100-112` (`cleanCharacterName`) | `cues.strip_cue_extensions` |
| M-3 | `{{stage directions}}` removed from spoken text; multi-line blocks included. | `SwiftCompartido/.../Sendable/GuionElementSnapshot.swift:198-213` | `textprep.strip_stage_directions` |
| M-4 | A parenthetical is **never spoken**. It becomes the backend's non-spoken `instruct`, split on commas, with blocking/addressing segments (`to Abbey`, `looking …`) dropped and delivery segments kept. | `ProduciestaCore/Parsing/ParentheticalDirection.swift:35-86` | `textprep.parenthetical_direction` (`BLOCKING_PREFIXES` copied verbatim) |
| M-5 | A parenthetical with no following dialogue is dropped, and a new cue or a narrated element clears the pending buffer. | `ProduciestaCore/Pipeline/VoicingPlan.swift:215-247` | `speech.prepare_script` walk |
| M-6 | A mid-block parenthetical splits one cue's dialogue into two spoken lines, both keeping the speaker. | `VoicingPlan.swift:235-272` | `speech.prepare_script` (speaker cursor) |
| M-7 | Scene headings are rewritten for the ear: marker expansion (`EST.` → `ESTABLISHING SHOT.`), ` - `/` — ` separators → sentence periods, scene-number trailers dropped, repeated periods collapsed. | `ProduciestaCore/Pipeline/SluglineSpeech.swift:37-138` | `textprep.expand_slugline` |
| M-8 | Inline `[[<breath/>]]` is a **split point**, never spoken: each span is generated separately and 0.15 s of silence is spliced at the seam. | `ProduciestaCore/Pipeline/BreathRendering.swift:25-63`, `SwiftVoxAltaAdapter.swift:64-85` | `textprep.breath_segments`, `PreparedText.segments` |
| M-9 | Breath spans shorter than 40 trimmed characters merge into a neighbour (forward first, backward at the line end) because the model clips very short generations. | `BreathRendering.swift:73-118` | `textprep.merge_runt_segments` |
| M-10 | Every other `[[note]]` is silent — production notes, `<pause>`, `<include>`. | `VoicingPlan.swift:274-317` | `speech.prepare_script` (NOTE branch) + `textprep.strip_inline_notes` |
| M-11 | Empty / whitespace-only text never reaches the backend; the unit is skipped. | `VoicingPlan.swift:247`, `SwiftVoxAltaAdapter.swift:59-61` | `speech.prepare_script` (`PreparedText.is_empty`) |
| M-12 | **Em-dashes, ellipses, and ALL-CAPS words are not transformed at all** — no expansion, no case folding, no spacing changes. They reach the model verbatim. | Verified absent across SwiftCompartido, ProduciestaCore, and SwiftVoxAlta; visible verbatim in the transcripts (e.g. `It's WHERE—`, `...It's on the corner of Ball and Sack.`) | `textprep.prepare` performs no such transform |
| M-13 | Line duration is estimated at 0.055 s/char and sentences are packed greedily into ≤ 12 s chunks with a 100-char floor; a trailing runt merges backward. | `SwiftVoxAlta/VoiceLockManager.swift:360-442`, `GenerationSettings.swift:97-113` | `textprep.estimate_duration`, `textprep.split_at_sentences` |
| M-14 | Inter-chunk silence is 0.25 s; breath silence is 0.15 s. | `GenerationSettings.swift:104`, `BreathRendering.swift:31` | `textprep.CHUNK_PAUSE_SECONDS`, `BREATH_GAP_SECONDS` |

### FR-2 — cue → character

Cues are normalised (M-1) and then resolved against the project's `CAST.md`
roster (`voices.roster.parse_cast_markdown`, shared with Sortie 4). Lookup is
case-, whitespace-, and punctuation-insensitive. `NARRATOR` resolves whether or
not the roster lists it, because narrated elements synthesise a NARRATOR
speaker — it is a first-class speaking character, not a fallback.

`comparativa parse` emits `unresolved_cues`; on both corpus episodes it is
empty. `--strict-cues` turns a non-empty list into a non-zero exit.

### FR-3 — speech classification

Every element keeps a `spoken` flag and stays in the stream, so the generation
layer retains full timing context. Under the default `fr3` policy only
character dialogue is `spoken: true`; action, scene headings, centered text,
transitions, `[[notes]]`, and `SHOT PROMPT` panels are `spoken: false`.

Neither corpus episode contains a `<shot prompt=…/>` note, so that rule is
implemented and unit-tested but never exercised against real data.

## 4. Divergences

Every intentional difference is registered in
`textprep.DIVERGENCES` (a test asserts this table and that register stay in
sync).

| ID | Topic | Swift behaviour | comparativa behaviour | Why |
|----|-------|-----------------|-----------------------|-----|
| D-1 | Action / scene heading / centered / transition narration | Voiced by NARRATOR (`VoicingPlan.swift:319-356`); headings rewritten by `SluglineSpeech` first | `spoken: false` under the default FR-3 policy; available via `--speech-policy produciesta-parity` | REQUIREMENTS.md FR-3 mandates it. **This is the one divergence that changes what an episode contains** — see § 5 |
| D-2 | Fountain lyric `~` prefix | Kept and spoken (`SwiftCompartido/.../FountainParser.swift:228-237`; `speakableText` does not strip it) | Stripped | A literal tilde is not speech. Affects only the bumper's sung lines |
| D-3 | Fountain/Markdown emphasis (`*x*`, `**x**`, `_x_`) | Kept verbatim — confirmed in the shipped transcript: `I'm the one that was **HERE**.` (`audio/episode_1_01_cold_open.vtt:182`) | Kept verbatim by default; `strip_emphasis=True` removes them | **Matched by default** so both stacks speak the identical string. Flagged because the Swift behaviour is an artefact source, not a design choice |
| D-4 | Internal whitespace | Only the outer ends are trimmed; removing a mid-string `{{…}}` leaves a double space (`GuionElementSnapshot.swift:198-213`), and multi-line dialogue keeps its newlines | All whitespace runs collapse to one space (so lyric lines join with a single space) | Cosmetic; keeps prepared text stable for hashing and diffing. `STRICT_PARITY` disables it |
| D-5 | Sentence segmentation for chunking | Foundation/ICU `enumerateSubstrings(.bySentences)` (`VoiceLockManager.swift:394-403`) | Regex approximation with an abbreviation/initial veto (`split_sentences`) | No ICU sentence segmenter in the Python stdlib. Chunk boundaries can differ on long lines with exotic abbreviations |
| D-6 | `<shot prompt=…>` attribute extraction | glosa-av `GlosaParser` (`VoicingPlan.swift:387-406`) | Regex (`shot_prompt`) | No Python glosa-av binding. Untested against real data — neither corpus episode has a `<shot>` note |
| D-7 | Empty breath spans | A leading/trailing/adjacent-breath empty span is preserved and contributes only its silence gap (`SwiftVoxAltaAdapter.swift:127-141`) | Empty spans are dropped, so their gap is lost | Simplification; no corpus line has a leading, trailing, or doubled breath marker |

### Not divergences, worth stating anyway

* `<SceneContext …>` glosa notes are voiced by Produciesta as
  `REFRAME. <time>. <location>. <ambience>` (`VoicingPlan.swift:394-404`).
  Neither corpus episode contains one, and FR-3 makes notes non-spoken, so
  comparativa does not implement it. If a future episode uses it, this becomes
  divergence D-8.
* Fountain forced-element prefixes (`.`, `!`, `@`, `>`, `>…<`) are handled by
  each stack's *parser*, not by text prep. jouvence strips them all; the Swift
  parser strips `.`, `>` and `>…<` but keeps `!`, `@`, `~`. Only `~` reaches
  spoken text, which is D-2.

## 5. Comparison validity — read this before running the benchmark

D-1 is not cosmetic. On the cold open:

| Speech policy | Spoken lines | What is missing vs the Swift baseline |
|---------------|--------------|----------------------------------------|
| `fr3` (default) | 189 | Episode/act titles, every slugline, every action line — 19 NARRATOR lines |
| `produciesta-parity` | 208 | nothing (exact transcript match) |

The granville scripts are audio drama: nearly all narration is already written
as `NARRATOR` character cues, which is why the gap is 19 lines rather than
hundreds. But those 19 lines include the two title cards and every scene
heading, so an `fr3` episode opens differently from the Swift baseline and
loses its scene transitions.

**Recommendation to the generation and benchmark sorties (5–7):** run the
Python conditions with `--speech-policy produciesta-parity` so condition A and
condition C contain the same utterances, and note the policy in
`manifest.json`. Running them under `fr3` is FR-3-compliant but makes A-vs-C a
comparison of two different episodes. This tension between FR-3 and FR-4 is a
requirements-level conflict; it is flagged here rather than silently resolved.

## 6. What the generation layer receives

`comparativa parse` adds to its JSON:

* `elements[].spoken` — boolean, on every element;
* `speech.policy`, `speech.text_policy`, `speech.cast_source`,
  `speech.line_count`, `speech.characters`;
* `speech.lines[]` — one entry per spoken line with `index`, `element_index`,
  `element_type`, `character` (canonical `CAST.md` name), `raw_cue`, `text`
  (the prepared string), optional `segments` + `breath_gap_seconds` when the
  line was breath-split, optional `direction` (the non-spoken `instruct`),
  `scene_index`, `start_line`/`end_line`, and `lyric`/`dual` flags;
* `unresolved_cues[]` — empty on a healthy parse.

`text` is always `" ".join(segments)`. An engine that cannot honour breaths
should speak `text`; one that can should generate each segment and splice
`breath_gap_seconds` of silence between them.
