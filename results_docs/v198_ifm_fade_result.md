# v198 result: both Part II looped LMs are TRANSFER-ONLY; their loops shrink although trained with full backprop

**Plan:** `docs/v198_plan.md`, frozen in 53d1bb8 and pushed before the run. **Runs:** Kaggle
aurmerin/astrocyte-v198-ifm-pln5 and aurshalevmerin/astrocyte-v198-ifm-r5, v1, 2026-10-14, T4 x2 each, driver
95b158362555. **Analysis:** `python v198_ifm_fade.py --analyze kaggle-output-v198-<variant>`. The decision statistics
were computed from the same JSONs with the frozen formulas.

## Validity: both pass

| check | huginn-s-fixed-pln5 | huginn-s-fixed-r5 |
|---|---|---|
| selftest (k = R = plain; checkpointing changes nothing; k = 0 gives the loop no gradient) | PASS | PASS |
| our port vs their reference code: fp32 logits, max diff | 4.3e-5 | 3.7e-5 |
| our port vs their reference code: gradients, relative error | ≤ 2.7e-6 | ≤ 3.5e-6 |
| noise floor (repeated full pass) | 0.0 | 0.0 |

Their reference code ran sandboxed on Kaggle, with its compiled extensions stubbed.

## Measurements

All values are medians over 16 GSM8K-val sequences of 512 tokens, over all parameters, unblended (a = 1).

| | pln5, R 5 | pln5, R 10 | r5, R 5 | r5, R 10 |
|---|---|---|---|---|
| ρ (per-sequence range) | 0.801 (0.78–0.95) | 0.831 | 0.846 (0.80–0.87) | 0.793 |
| lost gradient at k = 1 / 2 / 4 | 0.770 / 0.599 / 0.298 | 0.812 / 0.664 / 0.430 | 0.833 / 0.695 / 0.369 | 0.817 / 0.701 / 0.384 |
| lost gradient at k = 8 | | 0.156 | | 0.122 |
| rule ρᵢᵏ at k = 1 / 2 / 4 | 0.801 / 0.642 / 0.412 | 0.831 / 0.691 / 0.478 | 0.846 / 0.716 / 0.512 | 0.793 / 0.629 / 0.395 |
| rule ρᵢᵏ at k = 8 | | 0.228 | | 0.156 |

- No sequence has ρᵢ > 1 at R = 5 on either model.
- At R = 5 the rule's own assumption R ≫ k fails, so the exact finite-loop forms apply. pln5's k = 1 value lies between
  the geometric forms (0.703 / 0.773); r5's lies above both (0.728 / 0.806).

## Decision (frozen rule)

| | pln5 | r5 |
|---|---|---|
| GROWS (ρ > 1 at R = 5) | no (0.801) | no (0.846) |
| c(k) = 0.3418 · err^3.4335 at k = 1 / 2 | 0.139 / 0.059 | 0.182 / 0.098 |
| TRANSFER at k = 1: distance from 0.224 (Parcae) / 0.009 (SmolLM2), need ≥ 0.03 | 0.085 / 0.130, yes | 0.042 / 0.173, yes |
| TRANSFER at k = 2: distance from 0.100 / −0.001 | 0.041 / 0.060, yes | 0.002 / 0.099, no |
| CONTRAST: err₁ change a = 1 → 0.3 (need ≥ +0.10) | +0.011, no | −0.071, no |
| CONTRAST: Ā₁ change a = 1 → 0.3 (need ≤ 0.02) | +0.105, no | +0.127, no |
| **verdict** | **TRANSFER-ONLY** | **TRANSFER-ONLY** |

- **TRANSFER.** On both models the frozen cost curve predicts a cost at k = 1 between Parcae's and SmolLM2's own.
  Unblended fine-tuning settings would therefore separate the curve from rules that need no measurement. pln5 separates
  more (0.085 from Parcae's k = 1 cost, against r5's 0.042).
- **CONTRAST is not available on this family.** Blending at k = 1 does not shrink the truncated gradient while keeping
  its direction. The lost gradient barely moves, and the direction after Adam's rescaling improves: 0.756 → 0.861 on
  pln5, 0.671 → 0.798 on r5. On r5, blending also changes the model's function a lot (loss 2.35 → 3.38 at a = 0.3).

## Descriptive (no decision weight)

- **Full backprop does not make the signal grow.**
  - pln5 was trained through the last min(R, 10) loops at a mean R of 5, which is nearly full backprop.
  - r5 was trained with full backprop through all 5 loops.
  - Both have a shrinking backward signal (ρ 0.80 and 0.85).
  - So Ouro's growing signal is not explained by full-backprop training alone. Paper v15's Limitations sentence ("every
    model with ρ < 1 here was trained with truncated backprop, and the one apparently trained through all its loops
    (Ouro) has ρ > 1") stays literally true for the paper's models, but its implied link does not hold on this family.
- **Carry-over from R = 5 to R = 10 at k = 1 / 2:**
  - pln5: +0.04 / +0.07;
  - r5: −0.02 / +0.01.
  - R = 5 is short (k = 2 is close to R), so this is not the R ≫ k regime of S10.
- **ρᵏ at R = 10:**
  - pln5's lost gradient is 2–10% below ρᵏ at k = 1–4, and 32% below at k = 8;
  - r5's lies 3% above to 11% above at k = 1–2, and 3% below at k = 4.
- **The summed-signal stand-in does not hold on this family at short loop counts.** The hidden-state share against
  the loop's lost share in the weights:
  - pln5, R = 10: within 0.03 (mean |diff| 0.016);
  - pln5, R = 5: off by 0.08–0.11;
  - r5, R = 10: off by up to 0.09;
  - r5, R = 5: off by 0.13–0.18.
  - The likely reason is that in this architecture the input is re-injected inside every loop step (retention about
    0.35), so much of the gradient flows through the injection rather than the carried state. This is not checked.
- **Per-loop contributions (loop parameters, |c_j| / |c_0|):**
  - pln5, R = 5: 1, 0.74, 0.60, 0.57, 0.78;
  - r5, R = 5: 1, 0.86, 0.83, 0.86, 1.07.

  The first loop (on the initial state) is large again, as on Parcae.

## What follows

- **pln5 is the model for a follow-up cost test of transfer across models.** It needs its own pre-registration, with
  the lost-gradient curve and the direction-after-Adam curve both locked from Parcae, and unblended settings at
  several k.
- **The blended contrast between the two curves cannot be built on this family.** The two curves can be compared only
  through their unblended predictions.
- **A paper sentence would be useful either way.** Two looped models trained with nearly full backprop have shrinking
  backward signals, so the Limitations sentence linking full backprop to ρ > 1 should be softened.
