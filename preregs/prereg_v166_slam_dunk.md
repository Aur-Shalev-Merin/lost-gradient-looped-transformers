# Pre-registration v166: fresh, discriminating out-of-sample test of "truncation cost is a function of the frozen-model gradient error" (Parcae-140m, heavy damping)

**Author:** Aur Shalev Merin
**Date:** 2026-10-03.
**Status:** FROZEN at the commit that adds this version together with `v166_analyze.py` and the `v166` mode of
`build_kaggle_v164.py`. No v166 treatment (truncated) cell had been trained at the freeze.
**Workstream:** misc (looped-transformer truncated-credit thread).

**Lineage.**
- Theory and adaptive method are Aicher, Foti & Fox 2019, for RNNs (`docs/aicher2019_overlap_2026-10-03.md`).
- v162 LAW-CONSISTENT; v163 MIXED; v164 INCONCLUSIVE (error curve closer in 9/10 out-of-sample cells); v165
  secondary PASS (fresh undamped cells predicted within 0.002).
- Pilots and pre-freeze measurements are in `docs/v166_pilots_2026-10-03.md`.

**What changed from the draft (all before any treatment run).**
- **Over-relaxation arm dropped.** Pilot A: a > 1 *raises* ρ (0.825 → 0.885 / 0.944), and a = 1.3 changes the
  function. So no cell in the "error rule predicts less cost" direction exists on this model.
- **Block B (Parcae-370m) not run** (PI decision 2026-10-03, cost). It was feasible in pilot B and remains a candidate
  follow-up.
- **Scoring threshold lowered from 0.04 to 0.025.** After the refit, only 2 cells reached 0.04. 0.025 is still ≥ 3×
  the typical per-cell φ standard error in damped variants (0.003–0.008 in v164).
- **The "≥ 1 cell in each direction" requirement is removed.** All scored cells have M_err > M_k. This is a stated
  limitation: the study tests whether the error rule correctly predicts *extra* cost at high-ρ loops that "k alone"
  misses, not the reverse.

## 0. Claim under test

**C:** on Parcae-140m, the fraction of the loop's training credit lost by backprop through only the last k of R loops
(φ) is predicted, out of sample, by one curve of the frozen-model gradient error err_k. It is predicted better than
by "k alone". This holds at loop counts / damping strengths never used in fitting (R 48, a 0.4) and at k never used
for the heavily damped variant (R 64, a 0.3, k 8 and 12).

The predictor is the measured gradient error, so the claim does not depend on separating ρ from "effective depth".

## 1. Predictors (refit on all 25 seen cells, LOCKED in `v166_analyze.py`)

- **Fit set:**
  - the 15 truncated cells of v162 + v163;
  - v164's 8 damped cells (a 0.7 and a 0.3);
  - v165's 2 undamped cells (they replace v164's, under v165's exclusion rule).
  - Evaluation-seed means.
- **M_err:** φ = 0.3390 · err^3.3791. Power law; AIC −197.2, vs −195.5 for the pre-specified saturating alternative
  c·errᵖ/(1 + d·errᵖ).
- **M_k:** φ = mean φ at k over the fit set: k = 1 / 2 / 3 / 4 / 6 / 8 → 0.207 / 0.1175 / 0.087 / 0.0468 / 0.0309 /
  0.0057. Unseen k: linear in log₂ k between neighbours; beyond k 8, the last segment's slope, floored at 0.

## 2. Cells (LOCKED)

- **Test-cell errors:** the pre-freeze frozen-model measurement.
  - Kernel `astrocyte-v166-pre-a`, `results_v166_pre/a/v164_measure_pre0.json`.
  - Frozen v164 driver procedure: median over 16 GSM8K-val sequences, fp32, all parameters.

| scored cell (R, a, k) | err | M_err | M_k | M_err − M_k |
|---|---|---|---|---|
| 64, 0.3, 8 | 0.6022 | 0.0611 | 0.0057 | +0.0554 |
| 64, 0.3, 12 | 0.4772 | 0.0278 | 0.0000 | +0.0278 |
| 48, 0.4, 4 | 0.6992 | 0.1012 | 0.0468 | +0.0543 |
| 48, 0.4, 8 | 0.5020 | 0.0330 | 0.0057 | +0.0274 |

- **Controls per variant:** floor (k = 0) and full backprop (k = R).
- (R 48, a 0.4) is a loop count and damping never trained before. Its frozen model has step-0 val 2.630, test 2.662.
  This is reported only; φ is a within-variant ratio.

## 3. Apparatus (LOCKED)

- **Driver:** the frozen v164 driver `v164_parcae_predict.py` (sha feaf9a0ba438; v163 module d5c2647126de),
  unchanged, run with cells 'R:a:k:lr:seed'.
- **Recipe:** as in v162–v165 (GSM8K, 600 steps, batch 8, AdamW, warm-up + cosine, clip 1.0, fp16 autocast with
  cache off, per-loop checkpointing).
- **On-device check:** every job runs the R-64 selftest before training.
- **LRs:** truncated and full 3e-4; floor 9e-4. Confirmed for R 48, a 0.4 by the pre-freeze 1-seed check:
  - floor 9e-4 best of {3e-4, 9e-4, 2.7e-3};
  - full 3e-4 best of {1e-4, 3e-4, 9e-4}.
  - R 64, a 0.3 used the same LRs in v164.
- **Seeds:** 1661101–1661107 (5 evaluation + 2 backups).
- **Platform:** Kaggle T4 ×2, 4 kernels (2 per account). 56 runs, est. 24 T4-h.

**Instability exclusion (v165's rule).**
- A seed is valid for a variant iff its full-backprop control has ≤ 10 non-finite steps, a finite test loss, and did
  not diverge.
- The first 5 valid seeds (in seed order) with all cells present are used.
- Fewer than 5 makes that variant UNINFORMATIVE.

## 4. Criteria (LOCKED; `v166_analyze.py`, selftested on err / k / midpoint synthetic worlds)

- **Floor gate per variant:** mean(L_floor − L_full) ≥ 0.03 with a 90% CI excluding 0. Variants failing it are not
  scored.
- **Per cell:** φ = mean over valid seeds of (L_k − L_full)/(L_floor − L_full); |res| = |φ − prediction|.
- **Calibrated** = |res_err| ≤ max(0.02, 0.3·φ).
- n = number of scored cells (4 if both variants pass); need = ⌈0.75·n⌉.

| outcome | rule |
|---|---|
| **ERR-PREDICTS** | M_err closer than M_k in ≥ need cells **and** MAE_err ≤ 0.5·MAE_k **and** calibrated in ≥ need cells |
| **K-ALONE-AS-GOOD** | MAE_err ≥ MAE_k |
| **ERR-ORDERS** | M_err closer in ≥ need cells and MAE_err ≤ 0.8·MAE_k (but not ERR-PREDICTS) |
| **INCONCLUSIVE** | otherwise |
| **UNINFORMATIVE** | fewer than 3 scored cells |

Rules are applied in the order listed.

**Reported, not scored:** per-cell residuals; ρ_end; M_kR (k/R alone).

## 5. Predictions (mine, at freeze)

- ERR-PREDICTS 0.45 / ERR-ORDERS 0.20 / INCONCLUSIVE 0.15 / K-ALONE-AS-GOOD 0.10 / UNINFORMATIVE 0.10.
- **Main risk:** v164 found the earlier power-law curve *over*predicting the heaviest-damping cells (by 0.025–0.054).
  The refit includes those cells, but the k = 12 / R 48 cells extrapolate along a curve that was too steep there
  before.
- Note: "k alone" predicts essentially zero cost at k ≥ 8. Any clearly positive φ there favours M_err.
