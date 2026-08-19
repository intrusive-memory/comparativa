# REPORT — OPERATION BATTLING BARDS round 1

_generated 2026-08-19T16:00:16 by `comparativa report`_

bench dir: `bench`

## Caveats

- Condition C has zero English female qwen3 presets — heavy voice reuse and accent artifacts are expected wherever it voices a female character.
- Loudness: many lines cannot reach -16 LUFS integrated without clipping; the peak-limited-line counts and shortfall stats are recorded per line and per episode in each condition's manifest.json / metrics.json (see docs/ASSEMBLY.md §3).

## Performance

| condition | episode | stack | engine | wall_seconds | rtf | peak_rss_bytes |
| --- | --- | --- | --- | --- | --- | --- |
| A | episode_1_01_cold_open | swift | produciesta | 2772.940 | 2.590 | 5270929408 |
| A | episode_1_01a_bumper_donnie_and_arnie_1 | swift | produciesta | 350.220 | 2.488 | 9241870336 |
| B | episode_1_01_cold_open | python | qwen3-1.7b-clone | 789.031 | 0.760 | 4899520512 |
| B | episode_1_01a_bumper_donnie_and_arnie_1 | python | qwen3-1.7b-clone | 164.509 | 1.241 | 4758913024 |
| C | episode_1_01_cold_open | python | qwen3-1.7b | 727.546 | 0.606 | 5166563328 |
| C | episode_1_01a_bumper_donnie_and_arnie_1 | python | qwen3-1.7b | 89.176 | 0.594 | 4841488384 |
| D | episode_1_01_cold_open | python | chatterbox | 617.921 | 0.816 | 3544875008 |
| D | episode_1_01a_bumper_donnie_and_arnie_1 | python | chatterbox | 75.005 | 0.738 | 3361669120 |
| E | episode_1_01_cold_open | python | soprano | 158.912 | 0.131 | 1040449536 |
| E | episode_1_01a_bumper_donnie_and_arnie_1 | python | soprano | 20.879 | 0.110 | 548421632 |
| G | episode_1_01_cold_open | python | chatterbox | 894.209 | 0.887 | 3659431936 |
| G | episode_1_01a_bumper_donnie_and_arnie_1 | python | chatterbox | 89.179 | 0.759 | 3371466752 |
| H | episode_1_01_cold_open | python | higgs | 2636.366 | 2.249 | 7547371520 |
| H | episode_1_01a_bumper_donnie_and_arnie_1 | python | higgs | 444.514 | 3.153 | 7545487360 |
| K | episode_1_01_cold_open | python | kokoro | 55.338 | 0.059 | 1176092672 |
| K | episode_1_01a_bumper_donnie_and_arnie_1 | python | kokoro | 9.080 | 0.071 | 923435008 |
| N | episode_1_01_cold_open | python | dia | 3968.864 | 5.110 | 4169170944 |
| N | episode_1_01a_bumper_donnie_and_arnie_1 | python | dia | 456.912 | 4.624 | 4124459008 |
| O | episode_1_01_cold_open | python | orpheus | 4094.934 | 3.883 | 7780696064 |
| O | episode_1_01a_bumper_donnie_and_arnie_1 | python | orpheus | 473.129 | 3.383 | 7492222976 |
| S | episode_1_01_cold_open | python | csm | 1728.972 | 1.293 | 4502405120 |
| S | episode_1_01a_bumper_donnie_and_arnie_1 | python | csm | 192.473 | 1.048 | 4467998720 |

## Listening scores

_empty until human scores arrive — run `comparativa listen`, fill in `scoring_sheet.csv`, then re-run `comparativa report`._

_empty until human scores arrive — fill in `scoring_sheet.csv` and re-run `comparativa report --scores ... --key ...`._

## Objective proxy — STT WER round-trip

objective proxy: dropped — mlx-whisper not installed in the pinned env

## Verdicts

_Skeleton only — Sortie 11 writes the actual verdicts once human listening scores exist (EXECUTION_PLAN.md Sortie 11). Nothing below this line is a finding._

### swift-port

E vs F is the clean pair — same checkpoint (Soprano-80M-bf16), same single built-in voice, only the stack differs. A vs C also bears on this hypothesis but **confounds** the port with voice design (A uses production `.vox` voices, C uses auto-assigned built-in presets) and must be read with that confound stated.

**Verdict:** _pending — insufficient evidence until Sortie 10 (execution) and Sortie 11 (verdict) run._

### model-ceiling

C vs D/E probes whether a larger/different checkpoint (qwen3-1.7b vs chatterbox vs Soprano-80M) actually sounds better, independent of stack.

**Verdict:** _pending — insufficient evidence until Sortie 10 (execution) and Sortie 11 (verdict) run._

### voice-design

Partial evidence only this round: condition B (qwen3 cloned from `.vox` reference audio) is deferred to the follow-on custom-voices mission, so a definitive voice-design verdict is not possible from round-1 defaults-only data alone.

**Verdict:** _pending — insufficient evidence until Sortie 10 (execution) and Sortie 11 (verdict) run._
