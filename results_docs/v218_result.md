# v218 result: robustness of the fine-tuning and measurement results

**Prereg:** `docs/prereg_v218_robustness.md`, frozen 00549f0 and pushed to origin before any run. No amendments.
**Code:** `v218_robustness.py` 6aeb0b6c25c5, `build_kaggle_v218.py` fba6da46528d, run unchanged.
**Runs:** Kaggle T4×2, astrocyte-v218-parcae and astrocyte-v218-ifm (aurmerin), 2026-10-24.
**Outputs:** `results_v218/`. `python v218_robustness.py --analyze results_v210 results_v218` reproduces
`results_v218/v218_analysis.txt`.

**Validity:**
- Step-0 GSM8K medians match v210's s0 exactly (max |diff| 0.0000 on every model).
- Every fine-tune used seed 2011. Non-finite steps: Parcae 1, the Part II models 0.
- Clipped share of steps: Parcae 0.87, pln5 0.97, r5 1.00.
- The step-300 values agree with v210's full300 run, which used the same seed (Parcae 0.605 / 0.430 / 0.215 here
  against 0.608 / 0.428 / 0.217).

## Verdicts (frozen rules)

**D1 (step 600): MIXED.** Fall (mean over k = 1, 2, 4, all-parameter, held-out GSM8K):

| model | fall at 300 | fall at 600 | change | share of the step-300 fall that reverses |
|---|---|---|---|---|
| Parcae-140m | 0.197 | 0.181 | −0.017 | 8% |
| Part II, sampled loop count (pln5) | 0.115 | 0.092 | −0.023 | 20% |
| Part II, fixed 5 loops (r5) | 0.115 | 0.085 | −0.030 | 26% |

On all three models part of the fall reverses as the learning rate decays to its floor. Parcae's −0.017 is just short
of the 0.02 bar for RECOVERS, so the verdict is MIXED. Most of the fall remains at step 600 (74–92%).

**D2 (WikiText): PARTIAL.** Ratio of the WikiText fall to the GSM8K fall at step 300:

| model | GSM8K fall | WikiText fall | ratio |
|---|---|---|---|
| Parcae-140m | 0.197 | 0.183 | 0.93 |
| pln5 | 0.115 | 0.006 | 0.05 |
| r5 | 0.115 | 0.038 | 0.33 |

On Parcae the fall is as large on unrelated text. On the Part II models it is mostly specific to the fine-tuning text;
their WikiText lost gradient also starts lower (step 0: 0.64 / 0.47 / 0.22 for pln5 against 0.76 / 0.59 / 0.29 on GSM8K).

**D3 (state draws): SMALL.** Median within-sequence SD over 8 draws, maximum over k = 1, 2, 4: Parcae 0.0001, pln5
0.0002, r5 0.0004. That is negligible next to the 0.02–0.05 sampling error across sequences.

Predictions at freeze: D1 MIXED 0.15 (HOLDS 0.45); D2 PARTIAL 0.40; D3 SMALL 0.50.

## Paper changes (v2), per the frozen mapping

- **D1 MIXED, stated per model.** §4: by the end of the 600-step run part of the fall reverses (8%, 20% and 26% of it),
  with most of it remaining.
- **D2 PARTIAL, stated per model.** §4 and §7: on Parcae the fall is as large on WikiText; on the two Part II models it is
  mostly specific to the fine-tuning text (0.05 and 0.33 of the GSM8K fall). This replaces "we did not test other text".
- **D3 SMALL.** §7: draw-to-draw variation is at most 0.0004, below the sampling error across sequences. This replaces
  "we did not measure how much the lost gradient varies across draws".
