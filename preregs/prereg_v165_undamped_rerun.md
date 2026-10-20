# Pre-registration v165: rerun of v164's undamped cells with fresh seeds and a control-instability exclusion

**Author:** Aur Shalev Merin
**Date:** 2026-10-03. **FROZEN at the commit that adds this file together with `v165_analyze.py` and the v165 mode of
`build_kaggle_v164.py`.** No v165 run had been made at the freeze.
**Trigger:** PI decision 2026-10-03 ("do the rerun of the undamped settings on kaggle").

## Why

v164 (`docs/v164_parcae_prediction.md`) was INCONCLUSIVE.
- The gradient-error curve beat "k alone" in 9/10 new cells and 5/5 discriminating cells (MAE ratio 0.66), but its MAE
  (0.034) missed the 0.03 bar.
- The miss came from v164's two undamped cells (R 32, a 1.0, k 3 and 6), whose φ was corrupted by one unstable
  full-backprop control: seed 1641102, 14 non-finite fp16 steps, test 1.417 vs ~1.37 for the other seeds.
- v164's frozen rules had no exclusion for such a run.
- v165 reruns only those cells, with fresh seeds and an exclusion rule fixed in advance.

## Design (LOCKED)

- **Cells:** R 32, a 1.0 (undamped), k ∈ {0 (floor), 3, 6, 32 (full)}.
- **Seeds:** 1651101–1651107 (7).
- **Driver:** the frozen v164 driver `v164_parcae_predict.py` (sha feaf9a0ba438, unchanged). Same recipe, fixed LRs
  (truncated and full 3e-4, floor 9e-4).
- **On-device check:** every job runs the R-64 selftest before training.
- **Platform:** Kaggle T4 ×2 (2 kernels, one per account).

**Exclusion rule (fixed before any run).**
- A seed is VALID iff its full-backprop control (k 32) has ≤ 10 non-finite steps and a finite test loss.
- The first 5 valid seeds in seed order are used.
- If fewer than 5 are valid, the outcome is UNINFORMATIVE.
- Rationale for the threshold: v164's healthy controls had 0–6 non-finite steps; the corrupted one had 14.
- The dry run of `v165_analyze.py` on v164's undamped runs (relabelled) excludes exactly that seed.

**Predictions.** v164's frozen predictors, applied to v164's locked measurement (`v164_analyze.py`, no refit):

| k | err | M_err | M_k |
|---|---|---|---|
| 3 | 0.5462 | 0.0466 | 0.0606 |
| 6 | 0.3071 | 0.0064 | 0.0162 |

## Criteria (LOCKED; `v165_analyze.py`)

**PRIMARY.** v164's outcome rule (ERR-PREDICTS / INCONCLUSIVE / K-ALONE-AS-GOOD, identical thresholds) over v164's 10
test cells, with v165's undamped cells replacing v164's. The a 0.7 and a 0.3 cells are taken as observed in v164.

**Disclosed weakness.** Those 8 cells are already known: their summed |residual| is 0.185 for M_err and 0.336 for
M_k. The primary criterion is therefore close to "do the two new cells not break the result". It mainly checks that
v164's miss was the bad control, not the curve.

**SECONDARY** (the new information). The two undamped cells alone: M_err closer than M_k **and** |observed − M_err| ≤
0.015, in both cells.
- Both cells are only weakly discriminating: the predictions differ by 0.014 and 0.010.
- So a PASS is evidence that the curve is accurate at low error. A FAIL with M_k closer would count against M_err.

## Predictions (mine)

- **PRIMARY:** ERR-PREDICTS 0.75 / INCONCLUSIVE 0.15 / UNINFORMATIVE 0.10.
- **SECONDARY:** PASS 0.5. The k = 6 cost is near zero, where both models are close; k = 3 decides it.
