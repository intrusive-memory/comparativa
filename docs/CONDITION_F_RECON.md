---
type: reference
state: current
updated: 2026-08-15
---

# CONDITION_F_RECON.md — is there a runnable Swift Soprano on this machine?

Deliverable of OPERATION BATTLING BARDS, Sortie 8, task 3. Verdict: **no —
condition F is skipped** (`bench/F/SKIPPED.md`).

Condition F was to run mlx-audio-swift's Soprano port on
`mlx-community/Soprano-80M-bf16` — the same checkpoint condition E runs in
Python (RD-1) — using **existing tooling only**. No `swift build`, no
`xcodebuild`, no `swift run`, no model downloads.

## What exists

The Swift Soprano port is real and complete:

| | |
|---|---|
| port source | `~/Projects/package-collection/pkg/mlx-audio-swift/Sources/MLXAudioTTS/Models/Soprano/` (`Soprano.swift`, `SopranoDecoder.swift`, `SopranoConfig.swift`, `TextUtils.swift`) |
| pinned repo id | `AudioModelManager.swift:51` — `case sopranoTTS = "mlx-community/Soprano-80M-bf16"` (RD-1 confirmed) |
| loader dispatch | `TTSModelUtils.swift:54-55` — model type `soprano_tts`/`soprano` → `SopranoModel.fromPretrained(modelRepo)`, inferred from the repo name |
| package checkout | `mlx-audio-swift` @ `d11da6c` |

So a Swift binary linking `MLXAudioTTS` *can* run Soprano. The question is
whether one that **exposes** it already exists as an artifact.

## What was searched, and what was found

| candidate | where looked | result |
|---|---|---|
| `mlx-audio-swift-tts` (the package's own CLI, `--model <hf-repo>`) | `find ~/Projects ~/Library/Developer/Xcode/DerivedData -type f -perm +111 -name mlx-audio-swift-tts` | **no built binary anywhere** — only the seven source checkouts |
| the package's `./bin` (where `make release` / `make install` deposit the CLI) | `~/Projects/package-collection/pkg/mlx-audio-swift/bin/` | contains only `check-local-only-suites.sh` |
| an SPM build product | `~/Projects/package-collection/pkg/mlx-audio-swift/.build/` | `checkouts/`, `artifacts/`, `index-build/`, `repositories/` — **no `debug/` or `release/`**; the package has never been built here |
| DerivedData products | `~/Library/Developer/Xcode/DerivedData` | nothing for mlx-audio-swift; the Produciesta DerivedData holds source checkouts only |
| `~/.local/bin` | full listing | `produciesta`, `produciesta-cli`, `diga`/`echada` (via the app), Python `mlx_audio.*` tools — no Soprano surface |
| `/Applications` | listing | `Produciesta.app` (and Audio Design Desk) |
| `produciesta` | `--help`, `help export`, `help generate`, `help cast` | screenplay-only; no `--engine` / `--model` option. Renders through SwiftVoxAlta, whose `VoxAltaModelManager` enumerates **Qwen3-TTS repos only** |
| `diga` (signed, in `Produciesta.app`) | `--help` | no options at all beyond `--version`/`--help`; Qwen3-TTS only |
| `echada` (signed, in `Produciesta.app`) | `--help` | voice *design* (`.vox` authoring), not episode rendering; Qwen3-TTS only |
| the Soprano symbols themselves | `strings ~/Projects/apps/Produciesta/bin/produciesta \| grep -i soprano` | `SopranoModel`, `SopranoDecoder`, `SopranoTTS`, … — **the code is linked in, but no CLI path reaches it** |
| Example apps | `Examples/SimpleChat`, `Examples/VoicesApp` | SwiftUI sources + an `.xcodeproj`; no built `.app`, and building is forbidden |

Cross-check on the Python side: `mlx_audio.tts.generate` is installed and *is*
runnable, but that is condition E, not F — it is the Python stack.

## The two blockers

1. **No built artifact.** `mlx-audio-swift-tts` is the only Swift CLI whose
   `--model` flag can reach `SopranoModel`. It is not built on this machine and
   producing it means `make release` → `swift build`, which the sortie's hard
   constraint forbids.
2. **The checkpoint is not in the Swift model store.** The Swift stack resolves
   models through SwiftAcervo's App Group container,
   `~/Library/Group Containers/group.intrusive-memory.models/SharedModels`,
   which currently holds FLUX.2, PixArt, T5, three Qwen3 LLMs and three
   Qwen3-TTS checkpoints — **no Soprano**. Python's `Soprano-80M-bf16` lives in
   the HuggingFace cache, which SwiftAcervo does not read. A built CLI would
   therefore still trigger a CDN download, also forbidden this sortie.

Either blocker alone is sufficient; both are outside a sortie's authority to
clear.

## Consequence for the report (must be stated)

Round 1 has **no clean Swift-vs-Python port comparison**. E-vs-F was designed to
be that pair — same checkpoint, same single built-in voice, only the stack
differs. Without F, the only cross-stack pair left is A-vs-C, which confounds
the port with voice design (`.vox` clones vs built-in presets), exactly the
confound § Conditions Matrix flags. The swift-port hypothesis (FR-15) should be
recorded as **insufficient evidence this round** unless F is later run.

## Unblocking it (supervisor work, a follow-on mission)

```bash
cd ~/Projects/package-collection/pkg/mlx-audio-swift
make release        # builds mlx-audio-swift-tts into ./bin
make codesign-cli   # App Group entitlement, needed to read SharedModels
```

then make the checkpoint visible to Acervo — either place
`mlx-community_Soprano-80M-bf16/` in the SharedModels container, or set
`ACERVO_MODELS_DIR` to a directory using that `<org>_<repo>` layout.

The CLI is per-utterance (`--text`, `--output`, `--max_tokens`, `--temperature`,
`--top_p`), so an F wrapper would drive it once per line from
`comparativa parse` output and then reuse the Python assembly — which keeps
gaps, trimming, and loudness identical to condition E and leaves the model as
the only difference.
