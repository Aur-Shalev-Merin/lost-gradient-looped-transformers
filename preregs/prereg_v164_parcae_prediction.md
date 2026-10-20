# Pre-registration v164: does the frozen-model gradient error predict the training cost of truncated loop backprop out of sample, better than "k alone"?

**Author:** Aur Shalev Merin
**Date:** 2026-10-03. **FROZEN at the commit that fills in §6.** No v164 training run had been made at the freeze.
The PI turned the laptop off for the night, so the test errors are locked **procedurally**, not by value: they are
the output of the frozen-model measurement kernel (frozen code, frozen model, fixed seeds), which runs concurrently and
cannot depend on any training run.

**Authorisation.** The PI asked for an overnight Kaggle run (2026-10-03, ~03:00: "draft v164 … something i can just
send to kaggle overnight"). Freeze and launch are automatic if the pre-freeze measurement passes its checks, as for
v163.

**Builds on:**
- v162 (LAW-CONSISTENT), v163 (MIXED);
- `docs/v163_parcae_damped_truncation.md`, whose exploratory reading was that φ looks like one convex function of the
  gradient error across both damping variants;
- `docs/aicher2019_overlap_2026-10-03.md` (the theory is Aicher et al. 2019's, for RNNs).

## 0. Question

v163's direction test was MIXED, and its strongest signal was exploratory: cost looked like a single curve of the
frozen-model gradient error. v164 turns that into an out-of-sample prediction test:
- **fit** two predictors on the 15 existing cells;
- **lock** their predictions for new cells before training;
- **score** which predicts the observed cost better.

**Predictors** (fit only on v162 + v163 evaluation means; `v164_analyze.py`):
- **M_err:** φ = c · errᵖ. Fitted: **c = 0.3759, p = 3.453**.
- **M_k ("k alone"):** φ = mean φ at that k in the fit set: k = 1: 0.207, 2: 0.102, 4: 0.031, 8: 0.0057. Unseen k
  are interpolated linearly in log₂ k: k = 3 → 0.0606, k = 6 → 0.0162.

**Test cells** (Parcae-140m, the same v162/v163 recipe):

| variant | loops R | k (truncated) | why |
|---|---|---|---|
| undamped a = 1.0 | 32 | 3, 6 | new k, same model: tests interpolation |
| damped a = 0.7 | 32 | 2, 3, 4, 6 | new ρ (~0.87) |
| damped a = 0.3 | 64 | 2, 3, 4, 6 | ρ ~0.95, outside the fit set's ρ range: tests extrapolation |

Each variant also gets its floor (k = 0) and full (k = R) controls, for φ.

## 1. Apparatus (LOCKED)

- **Code.** `v164_parcae_predict.py` reuses the frozen v163 driver (sha d5c2647126de, unchanged), adding only a per-cell
  R. Damped loop x ← (1−a)x + aF(x); backprop through the last k of R loops; grad loops checkpointed; fp16 autocast
  with cache off.
- **Recipe** (v162/v163): GSM8K packed 512, 600 steps, batch 8, AdamW β (0.9, 0.95), warm-up + cosine, clip 1.0.
- **LRs fixed, with no tuning wave:**
  - truncated and full cells: 3e-4;
  - floors: 9e-4.
  - Rationale: v161, v162 and v163 selected exactly these values for every cell (24/24 truncated/full, 6/6 floors,
    floor extension rejected each time).
  - Disclosed risk: the R = 64 / a = 0.3 variant was never tuned.
- **Evaluation seeds** 1641101–1641105 (5). 80 runs, ≈ 24 T4-h, Kaggle T4 ×2, both accounts.
- **Metric:** φ = (L_k − L_full)/(L_floor − L_full) per seed, mean over seeds.
- **Predictor input:** the frozen-model gradient error (median over 16 GSM8K-val sequences, fp32, all parameters) for
  each test cell, measured pre-freeze (kernel `astrocyte-v164-measure`) and locked in `v164_analyze.py`.

## 2. Pre-freeze checks (the measurement kernel)

1. v163's selftest at R = 64:
   - a = 1 damped forward = v162 forward;
   - checkpointed = plain gradient;
   - fp16 vs fp32 truncated gradient < 0.05 per group.
2. The measured errors are finite and increase as k decreases, for each variant.

If either fails, nothing is frozen.

## 3. Criteria (LOCKED)

**Floor check per variant** (evaluation seeds): L_floor − L_full ≥ 0.03 with a 90% CI excluding 0. Variants failing it
are dropped. With fewer than 6 test cells left, the outcome is UNINFORMATIVE.

**Scoring over the remaining test cells.**
- |residual| = |observed φ − predicted φ| for each model.
- MAE_err, MAE_k = mean |residual|.
- "Discriminating" cells: |pred_err − pred_k| ≥ 0.03.

**Outcome labels:**
- **ERR-PREDICTS** — all of:
  - MAE_err ≤ 0.7 · MAE_k;
  - MAE_err ≤ 0.03;
  - M_err closer in ≥ 70% of cells;
  - M_err closer in ≥ 70% of discriminating cells (if any).
- **K-ALONE-AS-GOOD:** MAE_err ≥ MAE_k.
- **INCONCLUSIVE:** otherwise.

The selftest checks that a synthetic world with an error law gives ERR-PREDICTS and a k-only world gives
K-ALONE-AS-GOOD.

**Reported only:**
- per-cell residuals and seed SEs;
- the end-of-run ρ per cell;
- a refit of M_err on all 25 cells.

## 4. Consequences

| outcome | consequence |
|---|---|
| ERR-PREDICTS | A frozen curve of the gradient error predicts truncation cost on new damping levels, a new loop count (R = 64) and new k, better than k alone. This is the strongest training-level claim of the thread ("measure the gradient error once; read off the cost"). It is still one model; ρ vs effective depth is no longer needed for the claim, since the predictor is the measured error itself. |
| K-ALONE-AS-GOOD | The error curve does not generalise beyond its fit cells. The training-level claim stays at v162/v163's level (direction only, MIXED). |
| INCONCLUSIVE | Reported as such, with per-cell residuals. |

## 5. No-post-hoc clause

No change to cells, LRs, seeds, recipe, fit set, model forms, thresholds or labels after the freeze.

## 6. Locked test errors and predictions (procedural lock)

- Test-cell errors = `err_median` from kernel `aurshalevmerin/astrocyte-v164-measure`, file `v164_measure_measure.json`
  (`v164_parcae_predict.py --measure`). They are read by `v164_analyze.py --measure_json` (`load_test_err`).
- Predictions = the frozen M_err (c = 0.3759, p = 3.453) and M_k above, applied to those errors.
- No other input.
- The pre-freeze checks of §2 run on the GPU at the start of every evaluation job (`--selftest`, abort on failure), and
  in the measurement kernel.
- §2 check 2 (errors increase as k decreases) is verified from the measurement JSON at scoring time. If it fails, the
  outcome is UNINFORMATIVE.

**Measurement kernel result** (version 2; version 1 was cancelled at ~4 min, before output). Recorded before any
training run had finished:
- Driver sha feaf9a0ba438 (v163 module d5c2647126de).
- Selftest at R 64 passed:
  - a = 1 vs v162 forward: 2.675924 at k ∈ {64, 32, 0};
  - checkpoint: 1.1e-6;
  - fp16 vs fp32 truncated gradient: 0.0022–0.0028 for all groups.
- §2 check 2 passed: errors fall with k for every variant.
- ρ = 0.8250 / 0.8732 / 0.9415 for (32, 1.0) / (32, 0.7) / (64, 0.3).

| test cell (R, a, k) | err | M_err prediction | M_k prediction |
|---|---|---|---|
| (32, 1.0, 3) | 0.5462 | 0.0466 | 0.0606 |
| (32, 1.0, 6) | 0.3071 | 0.0064 | 0.0162 |
| (32, 0.7, 2) | 0.7436 | 0.1352 | 0.1023 |
| (32, 0.7, 3) | 0.6456 | 0.0829 | 0.0606 |
| (32, 0.7, 4) | 0.5584 | 0.0503 | 0.0310 |
| (32, 0.7, 6) | 0.4229 | 0.0192 | 0.0162 |
| (64, 0.3, 2) | 0.8596 | 0.2229 | 0.1023 |
| (64, 0.3, 3) | 0.8092 | 0.1810 | 0.0606 |
| (64, 0.3, 4) | 0.7634 | 0.1480 | 0.0310 |
| (64, 0.3, 6) | 0.6789 | 0.0987 | 0.0162 |

Discriminating cells (|difference| ≥ 0.03): (32, 0.7, 2), (32, 0.7, 4), and all four a = 0.3 cells. The a = 0.3,
R = 64 cells carry most of the test.

**Layout.** 80 runs packed on 3 kernels (eval-a on aurshalevmerin; eval-c, eval-d on aurmerin; 2 T4 each), because the
measurement kernel holds one aurshalevmerin GPU slot.

## 7. Predictions (mine, on record at freeze)

| outcome | P |
|---|---|
| ERR-PREDICTS | 0.40 |
| INCONCLUSIVE | 0.30 |
| K-ALONE-AS-GOOD | 0.15 |
| UNINFORMATIVE (incl. untuned R 64 failing its floor check) | 0.15 |
