# v239 RESULT: MIXED by the frozen rule (k = 1 SHRINK, k = 2 TURN); the k = 1 label comes from one diverged run

**Provenance.**
- **Prereg:** `docs/prereg_v239_shrink_turn.md`. It was frozen in 9dea2a0 and pushed to origin as a freeze-only commit
  (490d532, branch `v239-freeze`) at 2026-11-10 22:27:33 UTC, before any run.
- **Kernels:** pushed 22:27:59–22:28:16 UTC, all four complete:
  - aurshalevmerin/astrocyte-v239-1 and -2;
  - aurmerin/astrocyte-v239-3 and -4.
- **Code:** every lane passed the v239 selftest on the real model, with combined driver sha ac4e7f8570b8, equal to the
  frozen files (the four files are unchanged since 9dea2a0).
- **Analysis:** `python v239_shrink_turn.py --analyze results_v239`; output in `results_v239/v239_analysis.txt`.

## Scored result

The floor gap was 0.166 nats (90% lower bound 0.165), so the gate passed. Seeds 2391101–2391105 were scored; none was
excluded.

| k | trunc | turn | shrink | margin | label |
|---|---|---|---|---|---|
| 1 | 0.226 ± 0.001 | 0.225 ± 0.002 | **22.9 ± 22.9** | 0.045 | SHRINK |
| 2 | 0.094 ± 0.002 | 0.093 ± 0.003 | −0.002 ± 0.002 | 0.019 | TURN |
| 4 (reported) | 0.016 ± 0.002 | 0.015 ± 0.002 | −0.004 ± 0.003 | 0.015 | TURN |

**Verdict (frozen rule): MIXED {1: SHRINK, 2: TURN}.**

## Why k = 1 reads SHRINK

The SHRINK label at k = 1 comes from a single run, `shrink1` with seed 2391105 (kernel 4, GPU 0), which diverged:
- **Before step 300:** its log is normal (|g_R| 1.4 and 1.26 at steps 100 and 200).
- **Around step 300:** the full gradient spiked to a norm of 291,229.
- **After the spike:** the fp16 loss scale halved 24 times to 0.004 (v174's manual scaling halves on every non-finite
  step and never grows back). The model collapsed: test loss 20.42, against 1.37 for full backprop on the same seed.
  Its cost is 114.5.
- **The other four `shrink1` runs** cost 0.007, 0.002, −0.006 and 0.001.

No other run of the 66 diverged. The largest non-finite count among the other scored runs was 7. One unscored run
crossed the 10-step limit: `shrink4` on the backup seed 2391106, with 11 non-finite steps and a normal test loss
(1.372). Over all 66 runs, 2 of 18 size-only runs crossed the limit, against 0 of 36 truncated and direction-only
runs and 0 of 12 full and floor runs. That is too few to show a difference, but v240 reports the comparison with its
own runs added (its amendment 1).

(This paragraph was corrected on 2026-11-11. It first said the largest non-finite count among the other runs was 7,
which held only for the scored seeds.)

The prereg's exclusion rule applies only to the full run (more than 10 non-finite steps excludes the seed). It has no
clause for an arm run that becomes unstable. So the diverged run counts, and the verdict is MIXED. This mirrors S3,
where one unstable run, with no exclusion rule yet, made the study inconclusive.

## Descriptive, after the fact (not a verdict)

- **The turn carries the cost at every k.** Turn-only cost 0.995, 0.995 and 0.97 of plain truncation's cost at k = 1, 2
  and 4. The per-seed values track each other closely (k = 1: 0.228 / 0.224 / 0.227 / 0.228 / 0.224 for trunc,
  0.223 / 0.219 / 0.229 / 0.228 / 0.225 for turn).
- **The shrink costs nothing except in the diverged run.**
  - The median of the five scored `shrink1` runs is 0.002.
  - Dropping runs with more than 10 non-finite steps (the full-run rule applied to arms) and taking the next seed in
    order gives 0.0008.
  - Seed 2391106, the unscored backup, gives `shrink1` 0.0005, with `trunc1` 0.216 and `turn1` 0.215.
  - On any of these readings, k = 1 would be TURN.
- **The truncated arm reproduces S2** within the seed spread at k = 1 and 2: 0.226 / 0.094 / 0.016 here, against
  0.224 / 0.100 / 0.024 in S2's pipeline (v163, R = 32). k = 4 is lower. (Corrected 2026-11-11: this line and the
  frozen prereg said S1, but S1 (v162) ran only at R = 8 and 16. The prereg is left as frozen.)
- **Adam-update agreement (Abar)** is the same for trunc and turn at every k (0.893 / 0.945 / 0.983). For shrink it is
  0.998–1.000, as expected from Adam's scale invariance.
- **Whether the divergence is caused by the shrink cannot be told** from one run. The spike was in g_R itself (the
  shrink arm follows g_R's direction). Rare spikes of this kind occur in this model's gradients (App. F of the paper:
  norms up to 24,865), and the loss-scale handling turned one into a collapse.

## Predictions vs outcome

| | predicted (prereg §4) | outcome |
|---|---|---|
| overall | TURN 0.35, PARTIAL 0.25, MIXED 0.30, SHRINK 0.03, INVALID/INCOMPLETE 0.07 | MIXED |
| k = 1 | TURN 0.45, PARTIAL 0.50, SHRINK 0.05 | SHRINK, from one diverged run; the turn share is 0.995, far closer to TURN than expected |
| k = 2 | TURN 0.60 | TURN |
| shrink ≤ 0.01 at every k | 0.85 | failed at k = 1 because of the diverged run; held for 14 of the 15 scored shrink runs |

## Paper change (prereg §2)

MIXED maps to: §5 reports the turn and shrink shares per k and says the reading holds only in part. The paper will
report:
- the pre-registered verdict, MIXED;
- that it comes from one diverged size-only run;
- the after-the-fact reading above, labelled as such: the turn alone gives 97–100% of the cost at every k, and the
  shrink alone about nothing in the other 14 runs.

App. G gains S22, and Table 4 gains an S22 row.
