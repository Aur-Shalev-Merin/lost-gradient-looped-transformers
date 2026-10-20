# v201 result: MEDIAN-OVERSTATES on Parcae and both Part II models, because fine-tuning shrinks the lost gradient, not because of averaging

**Plan and runs:**
- **Plan:** `docs/prereg_v201_batch_lost.md`. Frozen in 38b3203 and pushed before any measurement. Amendment 1
  (169d8cd, pushed before the Part II rerun) corrected a wrong parameter-count constant.
- **Runs:** Kaggle aurmerin, 2026-10-15, T4 × 2 each:
  - astrocyte-v201-parcae v1;
  - astrocyte-v201-retro v1;
  - astrocyte-v201-ifm v2. v1 stopped at selftest 3 before measuring anything; its log is in `results_v201/ifm-v1-failed/`.
- **Account move:** aurshalevmerin had hit its weekly 45-h GPU quota, so parcae and retro moved to aurmerin (cdf7ec1).
  Their run.py files are byte-identical to the frozen build.
- **Analysis:** `python v201_batch_lost.py --analyze <dirs>` (code 4fe2fff13a39 = frozen + amendment 1). Output in
  `results_v201/v201_analysis.txt`; it reproduces from `results_v201/` alone.

## Validity: all checks pass

- All 8 (model, stage) measurements ran.
  - Selftests passed: repeat pass identical (rel. diff ≤ 1e-6), k = 1 lost on batch 0 > 0.001.
  - Parameter counts: 144.3M, 163.5M, and 204,882,432 for both Part II models.
- B = 8 values agree between the two passes (max diff 2.8e-6).
- The fine-tunes had 0 non-finite steps.

## Results (medians; per-sequence = B = 1; "clip" = each batch clipped to norm 1 before averaging)

| model | k | stage 0: B=1 | stage 0: clip B=64 | stage 0: raw B=512 | stage 300: B=1 | stage 300: clip B=64 | stage 300: clip B=512 |
|---|---|---|---|---|---|---|---|
| Parcae-140m (R 32) | 1 | 0.800 | 0.968 | 1.000 | 0.598 | 0.575 | 0.569 |
| | 2 | 0.637 | 0.739 | 1.000 | 0.415 | 0.372 | 0.340 |
| | 4 | 0.383 | 0.446 | 1.000 | 0.200 | 0.170 | 0.122 |
| SmolLM2 (A) (R 8) | 1 | 0.359 | 0.355 | 0.362 | 0.325 | 0.314 | 0.311 |
| | 2 | 0.134 | 0.120 | 0.128 | 0.110 | 0.102 | 0.096 |
| | 4 | 0.018 | 0.013 | 0.014 | 0.012 | 0.013 | 0.010 |
| Part II, sampled loop count (R 5) | 1 | 0.758 | 0.766 | 0.771 | 0.615 | 0.604 | 0.567 |
| | 2 | 0.595 | 0.529 | 0.587 | 0.440 | 0.418 | 0.359 |
| | 4 | 0.288 | 0.222 | 0.261 | 0.217 | 0.229 | 0.175 |
| Part II, fixed 5 loops (R 5) | 1 | 0.830 | 0.877 | 0.823 | 0.705 | 0.689 | 0.663 |
| | 2 | 0.696 | 0.670 | 0.662 | 0.548 | 0.521 | 0.457 |
| | 4 | 0.380 | 0.291 | 0.299 | 0.290 | 0.273 | 0.218 |

Measurement batches clipped (|G_b| > 1):
- stage 0: 100% on every model (median norm 6–8);
- stage 300: 91–100% (median norm 1.05–1.99).

## Verdict (frozen rule)

**Primary gap:** clipped B = 64 at stage 300 minus the stage-0 per-sequence median.

| model | gap at k = 1 / 2 / 4 |
|---|---|
| Parcae | −0.23 / −0.27 / −0.21 |
| Part II, sampled loop count | −0.15 / −0.18 / −0.06 |
| Part II, fixed 5 loops | −0.14 / −0.18 / −0.11 |
| SmolLM2 | −0.04 / −0.03 / −0.01 |

**MEDIAN-OVERSTATES ['parcae', 'pln5', 'r5'].** My prediction at the freeze gave this outcome 0.03; I had expected
the gap to come from averaging and to go the other way.

## What the secondary numbers say (no verdict)

1. **Averaging barely matters once fine-tuning is under way.** At stage 300, clipped B = 64 minus B = 1 is between
   −0.04 and +0.01 on every model and k. The review's worry, that the per-sequence median is the wrong statistic for
   the update, holds only at stage 0.
2. **At stage 0, averaging matters on Parcae only.**
   - Clipped B = 64 sits 0.06–0.17 above the median there.
   - Unclipped, the average is dominated by rare large-gradient sequences: raw B = 512 = 1.000 at every k, against
     clipped 0.36–0.93 (RARE-SEQUENCES flag).
   - SmolLM2 shows no effect.
   - Part II shows −0.09 to +0.05.
3. **The gap is fine-tuning itself.** After 300 full-backprop steps, the per-sequence lost gradient falls by:
   - 0.18–0.22 on Parcae;
   - 0.07–0.16 on both Part II models;
   - at most 0.03 on SmolLM2.

   That is much larger than the paper's note that ρ falls 0.03–0.05 over fine-tuning.
4. **Plateau.** At stage 300, clipped B = 512 is below B = 64 by up to 0.06 (MOVING on Parcae k ≥ 2 and on Part II),
   so longer averaging windows lower it further.
5. **The two Part II models.** The fixed-5-loop model stays above the sampled one: +0.07 / +0.10 / +0.09 per sequence
   at stage 0, and +0.09 / +0.10 / +0.04 clipped at stage 300. The gap between them is about the same.

## Clipping during fine-tuning (correction for the paper)

The fine-tunes clipped 99.3% (Parcae), 100% (SmolLM2), 99.3% (sampled loop count) and 100% (fixed 5 loops) of steps.
- Parcae's pre-clip norm went 5.6 at step 0 → 1.1–1.6 from step 50 on; SmolLM2 stayed near 2.0.
- Appendix E's "only 14–30% of steps were clipped in a Parcae-140m check" comes from the v174 pilot
  (`docs/v174_pilot_result.md`). There, 14% / 30% is the share of steps where both the full and the k = 1 / 2
  truncated gradients exceeded the clip. The truncated gradient is smaller and often under the limit, but the full
  gradient is clipped on nearly every step.
- The paper sentence must be corrected. Its conclusion still holds: clipping does not remove the size difference,
  because the truncated gradient is often not clipped while the full one is.

## What this changes in the paper (per the frozen mapping, with the reading above)

- **§2:** on the three models whose loops fade slowly, the frozen-model per-sequence median overstates what truncation
  removes from the averaged update during most of fine-tuning, by up to 0.27. The cause is that fine-tuning itself
  lowers the lost gradient. Averaging changes it by at most 0.04 once fine-tuning is under way. On the untrained
  Parcae model, unclipped averages are dominated by rare sequences.
- **Table 1:** keeps the defined statistic. Appendix I gets the table above.
- **§4:** the cost curves were fitted on stage-0 per-sequence values. This stage dependence is one more reason the
  frozen-model number is a weak predictor of training cost.
- **Appendix E:** the clipping sentence is corrected.
- **Caveats:**
  - Stage 300 follows the full-backprop trajectory; truncated runs' own trajectories were not measured.
  - All measurements use training sequences; at stage 300 these have been seen about once.
