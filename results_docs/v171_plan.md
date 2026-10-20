# v171 plan (EXPLORATORY; fixed before any v171 data, 2026-10-05)

**Question.** Does an optimiser-aware deficit, the first-order loss decrease lost by Adam's actual step, put Parcae-140m
and the SmolLM2 retrofit on ONE truncation-cost curve under Adam?

v168's Adam metric hinted at this, but used only 4 single-batch gradients. v170 established that the optimiser matters.

**Measurement.** `v171_adam_state_probe.py`, on frozen models, fp32, all parameters.
- Data: 24 batches × 8 training sequences per variant.
- Per arm: Adam direction u = mean g / √(mean g²), each arm using its own gradient statistics.
- **Primary metric:** adam_def = 1 − (ḡ_R · u_k)/(ḡ_R · u_R).
- Also reported: sgd_def, err_mean, adam_cos.
- Cells: the same 29 Parcae + 6 retrofit cells as v168, with observed Adam φ from v162–v167.

**Analysis.** `v168_analyze.py`'s procedure on the new primary metric (d = adam_def):
- Fit φ = c·dᵖ on the 29 Parcae cells only.
- Predict the 6 retrofit cells unchanged.

**Candidate rule.** All four must hold:
- (a) Parcae R² ≥ 0.9;
- (b) retrofit damped MAE ≤ 0.015;
- (c) all 4 damped retrofit cells calibrated (|res| ≤ max(0.02, 0.3·φ));
- (d) NEW: Spearman ρ ≥ 0.85 between predicted and observed φ over all 35 cells. Because the retrofit costs are tiny,
  (b) and (c) are lenient; (d) requires the ordering to hold too.

**Outcomes.**
- **Candidate:** a pre-registered test on a third model is earned. No claim yet, since these costs were already seen.
- **Not a candidate:** Rule 2 is closed as "model- and optimiser-dependent, not predictable from a frozen-model
  measurement". The paper then reports this as a negative.

## Result of the first run: APPARATUS FLAW, not scorable (2026-10-05)

`results_v171/analysis.txt`: every metric fits Parcae at R² ≈ 0.06–0.09.

**The cause is in the measurement, not the hypothesis.**
- err_mean = sgd_def = 1.000 at EVERY k for Parcae R 16 and R 32 (undamped), and 0.986–0.999 at R 32, a 0.5, even
  at k = 8 of 16. That is impossible for a real truncation effect. v168 measured those cells at err 0.32–0.85
  (per-batch).
- **Likely cause:** v171 averages RAW, UNCLIPPED batch gradients. In long undamped loops an occasional batch's full
  gradient is huge, and it dominates the mean. Training clips each batch to norm 1.0 before AdamW, so v171 measured a
  quantity training never sees.
- **Supporting pattern:** the R 8, retrofit and heavily damped cells look sane, and the retrofit values match v168.

**Fix (v171b):**
- clip each batch gradient to global norm ≤ 1.0 exactly as `clip_grad_norm_(…, 1.0)` does in training;
- log each batch's pre-clip norm to confirm the outlier hypothesis.
- Primary metric, cells and candidate rule are unchanged.

## v171b RESULT (2026-10-05): NOT A CANDIDATE on the primary metric → Rule 2 closes

The full output is in `results_v171b/analysis.txt`; 35/35 cells were measured.

**Apparatus checks.**
- The outlier hypothesis is confirmed. Full-arm pre-clip batch norms reach 24,865 (R 16), 20,465 (R 32) and 1,666
  (R 32, a 0.5), against medians of ~6.5.
- The sanity gate passes, and the retrofit values agree with v168.

| metric | Parcae R² | retrofit damped MAE | damped calibrated | Spearman (35) | candidate |
|---|---|---|---|---|---|
| **adam_def (PRIMARY)** | 0.775 | 0.028 | no | 0.737 | **no** |
| sgd_def | 0.770 | 0.064 | no | 0.734 | no |
| err_mean | 0.946 | 0.0148 | no | 0.959 | no (calibration) |
| 1 − adam_cos | 0.953 | 0.0027 | yes | 0.964 | passes all four (secondary) |

**Verdict under the plan (primary metric):** NOT A CANDIDATE. As pre-stated, Rule 2 is closed as "model- and
optimiser-dependent; not predictable from a frozen-model measurement by the pre-specified metric".

**Recorded, not claimed:** 1 − adam_cos (the angle between the truncated and full Adam step directions) passes every
criterion. It is a secondary metric (1 of 4 here, about 11 tried across v168 + v171). It is also measurement-fragile:
the same metric gave Parcae R² 0.73 in v168, which used 4 unclipped single batches. A real claim would need fresh,
pre-registered cells on a third, loop-load-bearing model. Per the stopping commitment, no further Rule 2 runs on these
cells.

**Side observation (clipping).** On the retrofit, clipping erases much of the norm part of the truncation error. For
example, at a 0.3, k 1 the SGD deficit falls from 0.62 unclipped to 0.14 clipped, and every training step is clipped.
With clipping, err_mean also fits Parcae much better (R² 0.95 vs 0.89).
