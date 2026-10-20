# v210 result: review-6 follow-ups (loop-only fall, truncated fine-tune, Ouro scale-free and exit objective)

**Prereg:** `docs/prereg_v210_followups.md`, frozen 5cb861e and pushed to origin before any run. No amendments.
**Code:** `v210_followups.py` 619235c6b191, `build_kaggle_v210.py` 35cd86b26463, run unchanged.
**Runs:** Kaggle T4×2, kernels astrocyte-v210-parcae and astrocyte-v210-ouro (aurmerin) and astrocyte-v210-ifm (aurshalevmerin), 2026-10-19.
**Outputs:** `results_v210/` (JSONs, stdout, logs). `python v210_followups.py --analyze results_v210` reproduces
`results_v210/v210_analysis.txt` exactly.

## Verdicts (frozen rules)

| part | verdict | numbers | prediction at freeze |
|---|---|---|---|
| A1 | **IN-LOOP** | fall_loop / fall_all = 0.80 (Parcae), 0.72 (pln5), 0.75 (r5); bar ≥ 0.5 on all three | 0.5 |
| A2 | **TRUNC-FALLS** | fall_trunc / fall_all = 0.98, 0.85, 0.61; bar ≥ 0.5 on all three | 0.4 |
| B1 | **GROWS-SCALE-FREE** | median ρ̃ = 1.656 on h, 1.454 on pre-norm x; both > 1 | 0.55 |
| B2 | **MOST-LOST** | exit-objective lost gradient 0.849 at k = 2; bar ≥ 0.5 | 0.6 |

## Part A (held-out GSM8K test, first 64 sequences; medians over sequences at k = 1 / 2 / 4)

| model | state | all = \|G − T_k\| / \|G\| | loop = \|G − T_k\| / \|G − T_0\| |
|---|---|---|---|
| Parcae-140m (R 32) | s0 | 0.798 / 0.636 / 0.408 | 0.818 / 0.657 / 0.417 |
| | full300 | 0.608 / 0.428 / 0.217 | 0.689 / 0.484 / 0.246 |
| | trunc300 | 0.617 / 0.434 / 0.211 | 0.693 / 0.486 / 0.238 |
| pln5 (R 5) | s0 | 0.764 / 0.591 / 0.291 | 0.805 / 0.626 / 0.308 |
| | full300 | 0.633 / 0.456 / 0.221 | 0.722 / 0.519 / 0.254 |
| | trunc300 | 0.646 / 0.470 / 0.246 | 0.730 / 0.533 / 0.279 |
| r5 (R 5) | s0 | 0.833 / 0.702 / 0.387 | 0.867 / 0.730 / 0.400 |
| | full300 | 0.718 / 0.564 / 0.298 | 0.791 / 0.620 / 0.328 |
| | trunc300 | 0.743 / 0.604 / 0.366 | 0.809 / 0.658 / 0.399 |

Mean falls over k = 1, 2, 4:

| model | fall_all | fall_loop (ratio) | fall_trunc (ratio to fall_all) |
|---|---|---|---|
| Parcae-140m | 0.196 | 0.157 (0.80) | 0.193 (0.98) |
| pln5 | 0.112 | 0.081 (0.72) | 0.095 (0.85) |
| r5 | 0.114 | 0.086 (0.75) | 0.070 (0.61) |

**Reading.**
- Most of the fall is in the loop itself: 72–80% of it remains once the no-loop-credit gradient is removed from the
  denominator. The read-out's growing share accounts for the other 20–28%; at s0 the two statistics differ by
  0.01–0.04, at full300 by 0.03–0.09.
- A fine-tune that itself backpropagates through only the last 2 loops lowers the lost gradient too, by 61–98% as much
  as full backprop. On Parcae it is practically the same (0.193 vs 0.196); on r5 it is the smallest (0.070 vs 0.114),
  and at k = 4 r5's truncated run barely moves (0.387 → 0.366).
- The full300 falls here match S13's held-out falls (Parcae 0.20 / 0.22 / 0.21 at k = 1 / 2 / 4 vs 0.19 / 0.21 / 0.19
  here, with a different seed; see below).

**Validity.**
- Guard (≤ 10 non-finite steps, finite last loss): every run used the first pre-listed seed, 2011. Parcae full300 had
  exactly 10 non-finite steps, at the guard's limit (in S13 the same seed had 12 and failed; we did not investigate the
  difference). All other runs had 0.
- Clipped fraction of steps: full300 0.97 (Parcae), 0.99 (pln5), 1.00 (r5); trunc300 0.60, 0.13, 0.46. The truncated
  gradient is smaller, so it is clipped less often. The two arms share recipe, data order and LR, but not their
  effective update sizes under the clip.
- 64 rows per (model, state).

## Part B (Ouro-1.4B, R = 4, the 16 GSM8K-validation sequences of Table 1)

- Median norms per loop t = 1..4: pre-norm |x_t| 678 / 595 / 552 / 496 (falls); loop output |h_t| = |norm(x_t)| 722 /
  873 / 943 / 997 (rises).
- Median fitted ρ, final-output loss, k = R: raw |dL/dh_t| 1.877 (Table 1's 1.88 reproduced); scale-free
  |dL/dh_t|·|h_t| 1.656; scale-free on pre-norm |dL/dx_t|·|x_t| 1.454.
- **B1:** the backward signal grows from loop to loop even after normalising for state scale: ρ̃ is 1.45–1.66, less than
  1.88 but well above 1, on both the normalised and the pre-norm state.
- **B2:** lost gradient of the exit-weighted objective Σ_t p_t L_t (no entropy term), all parameters, every loop's exit
  read-out kept differentiable: 0.941 / 0.849 / 0.733 at k = 1 / 2 / 3. That is slightly above the final-output numbers
  on the same sequences (0.92 / 0.81 / 0.71). Exits before the kept loops add their own read-out gradient, but they do
  not make up for the loops cut off.
- Repeated full pass of the exit objective: relative difference 0.0 on both GPUs.

## Paper changes (per the frozen mapping) → draft v16.7

- A1 IN-LOOP: §1 item 4 and §3 drop the "could come from the layers after the loop" caveat and report the loop-only
  fall (72–80% of the fall).
- A2 TRUNC-FALLS: §3 says the lost gradient also falls along a truncated (k = 2) fine-tune; the "nor measure a truncated
  fine-tune" / "truncated runs follow their own paths, which we did not measure" caveats are replaced.
- B1 GROWS-SCALE-FREE: the Ouro paragraph says the growth survives normalising for state scale (ρ̃ 1.45–1.66).
- B2 MOST-LOST: under its training objective too, keeping two of four loops loses most of the gradient (0.85).
- Table 3 gets an S15 row; "fourteen studies, S1–S14" becomes fifteen.
