# Pre-registration v170: is the retrofit's insensitivity to truncation error caused by Adam?

**Author:** Aur Shalev Merin
**Date:** 2026-10-04.
**Status:** FROZEN at the commit adding this file with `v170_analyze.py` and the eval/pilot2 modes of
`build_kaggle_v170.py`. No v170 treatment (truncated, k ≥ 1) run existed at the freeze; only control (k 0 / k R) LR
pilots.

## Why

- **v167:** on the SmolLM2 retrofit, damping raised the frozen-model gradient error at k 1 from 0.36 to 0.65, yet the
  Adam training cost stayed flat (φ 0.0087 → 0.0087).
- **v168 (exploratory)** measured, at those cells, the first-order progress deficit of an SGD step (1 − useful_sgd):
  0.21 → 0.62. The same deficit for an Adam-like step was flat: 0.024 → 0.016.
- **Derivation** (`docs/law1_derivation_formal.md`): preconditioned and sign-like updates can ignore the aligned part
  of truncation bias; plain SGD cannot.
- **Prediction:** under SGD, training cost on the retrofit should rise with damping.

## Design (LOCKED)

- **Optimiser:** SGD with momentum 0.9, via `v170_sgd.py`. It replaces AdamW inside the frozen drivers; everything
  else is unchanged:
  - data, 600 steps, batch, warm-up + cosine, clip 1.0, fp16 + GradScaler, damping, truncation, selftests;
  - drivers: `v164_parcae_predict.py` (feaf9a0ba438) for Parcae; `v167_retrofit_trunc.py` (b6fd416e48f2) for the
    retrofit.
- **Cells:**
  - Parcae-140m, R 32, a 1: k ∈ {0, 1, 2, 4, 32};
  - retrofit (healed run 1), R 8, a ∈ {1, 0.3}: k ∈ {0, 1, 2, 8}.
- **Seeds:** 1701101–1701107 (5 + 2 backups).
- **Exclusion:** v165's rule. A seed is valid iff its full control has ≤ 10 non-finite steps, a finite test loss, and
  did not diverge. The first 5 valid seeds are used.

**SGD learning rates.** Pilot 1 (`results_v170_pilot/`, 1 seed, LR grid 0.01 / 0.03 / 0.1 / 0.3 / 1.0, chosen by val
loss):
- **Parcae:** floor 0.3, full/truncated 0.1. Floor gap 0.195.
- **Retrofit:** floor 0.3. For full, 0.03 was best of the stable runs (0.1 and 1.0 diverged, 0.3 was bad); floor gap
  0.033.
- **Retrofit rule (locked):**
  - Pilot 2 adds full 0.02 / 0.05 at a 1, and a 0.3 controls (floor 0.3; full 0.03 / 0.05).
  - The full/truncated LR is the argmin of a = 1 full val over all stable pilot runs. "Stable" means finite and
    ≤ 10 non-finite steps.
  - Both damping levels use it. If the a 0.3 full run diverges at that LR, the next lower stable LR is used for both.
  - Floor stays 0.3.
- Truncated cells use the full-backprop LR (as v162–v167).

## Hypotheses and outcome (LOCKED; `v170_analyze.py`, selftested on 3 synthetic worlds)

- **Floor gate per variant:** mean(L_floor − L_full) ≥ 0.03 with a 90% CI > 0.
- **H1 (primary, within v170):** retrofit SGD at k 1: φ(a 0.3) − φ(a 1) ≥ 0.02 with a Welch 90% CI > 0.
- **H2 (secondary, cross-study):** retrofit a 0.3, k 1: φ_SGD − φ_Adam(v167 = 0.0087) ≥ 0.02.
- **Reported, not scored:**
  - the k 2 versions;
  - Parcae φ_SGD vs φ_Adam (v163: 0.224 / 0.100 / 0.024 at k 1 / 2 / 4);
  - SGD deficit vs φ_SGD across both models.

| outcome | rule |
|---|---|
| **OPTIMISER-EXPLAINS** | H1 and H2 pass |
| **NOT-OPTIMISER** | \|φ(a 0.3) − φ(a 1)\| ≤ 0.01 at k 1 under SGD (SGD is also flat) |
| **INCONCLUSIVE** | otherwise |
| **UNINFORMATIVE** | a retrofit variant fails its floor gate or has fewer than 5 valid seeds |

**Disclosed risks.**
- The retrofit's SGD floor gap is small (0.033 in pilot 1), so φ is noisier and the floor gate may fail.
- SGD full backprop on the retrofit is unstable above LR 0.03.
- H2 compares across studies (different seeds); H1 is within-study and is primary.

## Predictions (mine, at freeze)

- OPTIMISER-EXPLAINS 0.45 / INCONCLUSIVE 0.25 / NOT-OPTIMISER 0.15 / UNINFORMATIVE 0.15.

## Amendment A1 (before any retrofit treatment run)

- **When:** 2026-10-04, ~22:55. The Parcae arm's eval kernels were running (they don't enter H1) and the retrofit
  pilot 2 (controls only) was running. No retrofit truncated (k ≥ 1) SGD run existed.
- **Why:** pilot 1's retrofit SGD floor gap (0.033) sits at the 0.03 gate, so the study risked UNINFORMATIVE on a
  normalisation technicality.
- **Change:** if a retrofit variant fails its floor gate but both variants have ≥ 5 valid seeds, H1 is scored in raw
  loss, with no division by the gap:
  - excess = L_k1 − L_full per seed;
  - d = mean excess(a 0.3) − mean excess(a 1), with a Welch 90% CI;
  - CI > 0 → **OPTIMISER-DIRECTIONAL**, otherwise **NOT-OPTIMISER-DIRECTIONAL**;
  - both labelled underpowered.
- **Unchanged:** the primary path, used whenever both floor gates pass.
- **Selftest:** two small-gap worlds were added; all 5 worlds pass.

## Retrofit LR by the locked rule (pilot 2, controls only)

- **a = 1 full val** (all stable runs; 0.1 and 1.0 diverged): 0.01 → 1.3049, 0.02 → 1.2677, **0.03 → 1.2539**,
  0.05 → 1.2887, 0.3 → 2.4761.
- So full/truncated LR = 0.03. The a = 0.3 full run is stable at 0.03 (val 1.2609). Floor LR = 0.3.
- **Pilot-seed floor gaps (test):** a 1: 0.033; a 0.3: **0.023**, which is below the 0.03 gate. So amendment A1's
  raw-loss fallback is the likely scoring path for H1.
