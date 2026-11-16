# v237 RESULT: Part A INVALID (Parcae failed the reproduction check; both Part II models FAST); Part B GROWS

**Provenance.**
- **Prereg:** `docs/prereg_v237_early_ft.md`, frozen and pushed in a51f044 before any run.
- **Kernels:** pushed 2026-11-09 at 21:02 UTC:
  - aurshalevmerin/astrocyte-v237-ifm and -ouro;
  - aurmerin/astrocyte-v237-parcae.
- **Analysis:** `python v237_early_ft.py --analyze results_v218 results_v237`; output in
  `results_v237/v237_analysis.txt`.

## Part A: INVALID by the frozen rule

| model | step 0 vs v218 | step 300 vs v218 | step 600 vs v218 | validity | label |
|---|---|---|---|---|---|
| Parcae-140m | 0.000 | **0.0138** (> 0.01) | 0.0048 | INVALID | (FAST) |
| Part II, sampled loop count | 0.000 | 0.0041 | 0.0043 | VALID | FAST |
| Part II, fixed 5 loops | 0.000 | 0.0018 | 0.0026 | VALID | FAST |

Parcae's step-300 values differed from v218's by up to 0.014. That is above the 0.01 set for GPU nondeterminism
across kernels, though step 0 matched exactly and step 600 within 0.005. By the rule, any failed check makes the
verdict INVALID, and §4 is unchanged.

**What the data show (descriptive).** Fall F(t) of the GSM8K all-parameter value, as a mean over k = 1, 2, 4:

| model | step 10 | 25 | 50 | 100 | 200 | 300 | 600 |
|---|---|---|---|---|---|---|---|
| Parcae | 0.161 | 0.208 | 0.205 | 0.212 | 0.221 | 0.190 | 0.177 |
| pln5 | 0.107 | 0.144 | 0.147 | 0.147 | 0.135 | 0.113 | 0.089 |
| r5 | 0.087 | 0.132 | 0.131 | 0.143 | 0.144 | 0.114 | 0.085 |

- **Timing.** On all three models the fall is essentially complete by step 25, just after the 20-step warm-up. It
  peaks between steps 50 and 200, at 1.1–1.3 times its step-300 size, and then partly reverses.
- **The text-mismatch reading** (the GSM8K value settling at the trained model's WikiText value) does not fit the Part
  II models. Their GSM8K value falls well below the step-0 WikiText value early, then climbs back toward it:
  - pln5: 0.401 at step 50 against 0.443;
  - r5: 0.498 at step 100 against 0.550.

  The step-300 coincidence the review pointed to is a point on the way back up.
- **Parcae's WikiText value falls with its GSM8K value** (0.19 by step 100), as v218 found at step 300.
- **The training loss also drops most in the first 50 steps** (from about 2.7–2.8 to about 1.5–1.7). As the prereg said,
  timing alone cannot separate a quick fix from an early real change.

## Part B: GROWS

On 16 GSM8K test chunks:
- **Fitted ρ, medians:** 1.493 for the raw backward signal (minimum 1.309), 1.353 scale-free on h (minimum 1.213), and
  1.153 scale-free on the pre-norm x (minimum 0.988, one chunk at or below 1).
- **Final-output lost gradient:** 0.926 / 0.846 / 0.730 at k = 1 / 2 / 3, against Table 1's 0.92 / 0.81 on validation
  text.
- **Exit-weighted objective:** 0.934 / 0.838 / 0.722.

## Predictions vs outcome

- **A:** I gave INVALID 0.05 and FAST 0.35. All three models would have been FAST, Parcae included, had it passed.
  I had expected Parcae to be slower.
- **B:** GROWS 0.85, as predicted.

## Paper change (prereg §3)

- **A, INVALID:** §4 is unchanged in its claims. Its sentence "we measured only steps 0, 300 and 600" now points to
  S21. App. I reports the data descriptively, with the verdict, and Table 4 gains S21.
- **B, GROWS:** §3 says Ouro's growth holds on test text too, with the numbers.
