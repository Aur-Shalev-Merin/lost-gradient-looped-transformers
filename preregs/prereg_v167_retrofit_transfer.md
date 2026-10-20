# Pre-registration v167: does the Parcae truncation-cost curve transfer, unchanged, to a different looped model?

**Author:** Aur Shalev Merin
**Date:** 2026-10-04.
**Status:** FROZEN at the commit that adds this file together with `v167_analyze.py` and the `eval` mode of
`build_kaggle_v167.py`. No v167 treatment (truncated) cell had been trained at the freeze.
**Workstream:** misc.

**Lineage.**
- v162–v166 on Parcae-140m: v166 ERR-PREDICTS, `docs/v166_slam_dunk_result.md`.
- Pilots: `docs/v167_pilots_2026-10-04.md`.
- Theory: Aicher et al. 2019 (RNNs).

## 0. Why, and what is NOT the test

v166 confirmed one direction on one model: where the gradient-error rule predicts *more* cost than "k alone". The
opposite direction needs a loop that forgets fast. The healed SmolLM2 retrofit has ρ ≈ 0.35, vs Parcae's ≈ 0.8.

**Not the test:** beating a "k alone" rule borrowed from Parcae. The retrofit's loop settles within ~4 passes, so near-zero
cost at small k is expected on any account. Beating a borrowed per-k table would only show that the models differ.

**The test has two parts:**
- **(i) Transfer.** Do the *numbers* of the curve fitted on Parcae alone (φ = c·errᵖ) predict this model's costs?
  The test covers a range of costs, created by loop damping (a = 1, 0.5, 0.3), which raises err at the same k.
- **(ii) Fair rival.** On the damped cells, does that curve beat a "k alone" rule fitted on **this model's own**
  undamped cells (same seeds, same study)?

## 1. Model and apparatus (LOCKED)

- **Model:** healed looped SmolLM2-135M retrofit, healing run 1 (raven/Huginn architecture, split 11-8-11, adapter
  [0|I] at conversion; `docs/retrofit_heal_nodrop_2026-10-02.md`).
  - Private Kaggle dataset `retro-smollm2-healed` (both accounts, identical files).
  - transformers 4.51.0.
- **Driver:** `v167_retrofit_trunc.py`, sha b6fd416e48f2 (the pilots' driver, unchanged).
  - R = 8 loops.
  - Damped loop x ← (1 − a)x + a·F(x).
  - Backprop through the last k loops: k = 0 is the floor (the loop gets no gradient); k = 8 is full.
  - Per-loop checkpointing for k > 0.
  - fp16 autocast with cache off + GradScaler.
  - Every job runs the selftest (decomposition, checkpoint = plain, fp16 vs fp32, floor) before training.
- **Recipe** as v162–v166: GSM8K question+answer text (+ EOS), last 473 train examples = val, test split = test; 600
  steps, batch 8 × 512, AdamW (0.9, 0.95), 20-step warm-up + cosine to 10%, clip 1.0.
- **LRs (from pilots 1 + 2):**
  - floor 3e-4 (best of 1e-4 / 3e-4 / 9e-4 / 2.7e-3);
  - full and truncated 1e-4 (best of 3e-5 / 1e-4 / 3e-4 / 9e-4).
  - Val and test agree, at both a = 1 and a = 0.3.
- **Variants:** a ∈ {1.0, 0.5, 0.3}.
  - a = 0.2 was dropped in the pilot: it changes the function at R 8 (+0.040 loss).
  - a = 0.3 changes it by +0.007; a = 0.5 by +0.0004.
- **Cells per variant:** k ∈ {0, 1, 2, 8}.
- **Seeds:** 1671101–1671107 (5 evaluation + 2 backups). 84 runs on 4 Kaggle kernels × 2 T4, est. 12.5 T4-h.

**Instability exclusion (v165 rule).**
- A seed is valid for a variant iff its full control has ≤ 10 non-finite steps, a finite test loss, and did not
  diverge.
- The first 5 valid seeds are used.
- Fewer than 5 makes the variant UNINFORMATIVE.

## 2. Predictors (LOCKED in `v167_analyze.py`)

**M_err** = 0.3418 · err^3.4335.
- Power law refit on **all 29 Parcae truncated cells** (v162–v166). No retrofit data.
- Test-cell err = the pre-freeze frozen-model measurement (kernel astrocyte-v167-pilot, `v167_measure_floor.json`;
  16 GSM8K-val sequences, fp32, all parameters).

| cell (a, k) | err | M_err |
|---|---|---|
| 1.0, 1 | 0.3575 | 0.0100 |
| 1.0, 2 | 0.1372 | 0.0004 |
| 0.5, 1 | 0.5658 | 0.0484 |
| 0.5, 2 | 0.3436 | 0.0087 |
| 0.3, 1 | 0.6505 | 0.0781 |
| 0.3, 2 | 0.4418 | 0.0207 |

**M_k_own** (for damped cells only): the observed undamped φ at the same k in this study.
- Expected ≈ 0.01 at k 1 and ≈ 0 at k 2.
- It differs from M_err by ≈ 0.07 (a 0.3, k 1), 0.04 (a 0.5, k 1) and 0.02 (a 0.3, k 2).

## 3. Criteria (LOCKED; selftested on four synthetic worlds)

- **Floor gate per variant:** mean(L_floor − L_full) ≥ 0.03 with a 90% CI excluding 0.
- **Per cell:** φ = mean over valid seeds of (L_k − L_full)/(L_floor − L_full).
- **Calibrated** = |φ − M_err| ≤ max(0.02, 0.3·φ).

| part | rule |
|---|---|
| **(i) TRANSFER** | calibrated in ≥ n − 1 of the n scored cells **and** MAE(M_err) ≤ 0.015 over them |
| **(ii) RIVAL** | over damped cells: M_err closer than M_k_own in ≥ ⌈0.75·n_d⌉ **and** MAE(M_err) ≤ 0.5·MAE(M_k_own) |

| outcome | rule (applied in order) |
|---|---|
| **ERR-TRANSFERS** | (i) and (ii) pass |
| **K-ALONE-AS-GOOD** | MAE(M_err) ≥ MAE(M_k_own) on damped cells |
| **ERR-ORDERS-ONLY** | (ii) passes, (i) fails (right direction, wrong scale on this model) |
| **CALIBRATED-ONLY** | (i) passes, (ii) fails |
| **INCONCLUSIVE** | otherwise |
| **UNINFORMATIVE** | undamped variant not scored, or fewer than 4 cells / 2 damped cells scored |

**Disclosed limits.**
- **Weak cells:** the undamped cells predict ≈ 0 and are weakly discriminating. The informative cells are the damped
  ones, where the Parcae curve predicts clearly nonzero costs.
- **Noise:** the floor gap is ~0.08 here (Parcae ~0.18), so φ is noisier.
- **"k alone" sees part of the data:** M_k_own uses this study's undamped cells (same seeds), which gives it the
  benefit of in-study data.

## 4. Predictions (mine, at freeze)

- ERR-TRANSFERS 0.35 / ERR-ORDERS-ONLY 0.25 / CALIBRATED-ONLY 0.05 / INCONCLUSIVE 0.15 / K-ALONE-AS-GOOD 0.10 /
  UNINFORMATIVE 0.10.
- **Main risks:**
  - The curve's scale may not transfer between model families: different floor gap, different loss landscape.
  - On Parcae the curve ran slightly steep at high err.

## Amendment A1 (post-freeze, pre-result)

- **When:** 2026-10-04, after the freeze commit 880be7a and the kernel launch (12:38), while all four kernels were
  still RUNNING. No v167 output had been retrieved or seen.
- **Why:** a review critique. Three of the six cells predict ≈ 0, so they are near-automatic hits. Part (i) could
  therefore pass with a large miss on the single most informative cell. Example: a 0.3, k 1 predicted 0.078, observed
  0.03, others exact. That gives 5/6 calibrated and MAE 0.008, so it passes.
- **Change.** Part (i) TRANSFER additionally requires:
  - every damped k = 1 cell (a 0.5 and a 0.3, the high-cost cells) calibrated;
  - MAE(M_err) over the damped cells alone ≤ 0.015.
- **Unchanged:** everything else, including part (ii), the outcome table, cells, LRs and seeds.
- **Reporting:** the verdict under the rule as frozen is reported alongside, labelled "as frozen".
- **Selftest:** a fifth synthetic world exercises exactly this case (passes as frozen → INCONCLUSIVE under A1).
