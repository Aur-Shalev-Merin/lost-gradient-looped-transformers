# v168 result (EXPLORATORY): no metric passes the fixed candidate rule (2026-10-04)

**Sources.**
- Script `v168_direction_probe.py`; analysis `v168_analyze.py`, written before the data returned.
- Primary metric fixed by derivation before any result was seen: `docs/v168_primary_metric_derivation.md`.
- Outputs: `results_v168/`, full table in `results_v168/analysis.txt`.
- Coverage: 29/29 Parcae cells and 6/6 retrofit cells measured. Frozen models, fp32, 4 batches × 8 sequences.

## Fixed rule

A metric is a candidate if, after fitting φ = c·dᵖ on Parcae only:
- Parcae R² ≥ 0.9;
- retrofit damped MAE ≤ 0.015;
- all 4 damped retrofit cells calibrated.

| metric (distance d) | Parcae R² | retrofit MAE (damped) | damped calibrated | candidate |
|---|---|---|---|---|
| err (norm) | 0.886 | 0.028 | 2/4 | no |
| **1 − useful_adam (PRIMARY)** | **0.806** | **0.0047** | **4/4** | **no (fails R²)** |
| 1 − useful_sgd (secondary) | 0.831 | 0.045 | 1/4 | no |
| 1 − cos | 0.859 | 0.0045 | 4/4 | no (fails R²) |
| adam_err | 0.736 | 0.0037 | 4/4 | no |
| 1 − adam_cos | 0.727 | 0.0064 | 4/4 | no |
| 1 − sign_agree | 0.888 | 0.0056 | 4/4 | no |

- **No candidate.** Under the stopping rule stated before the run, the training-cost line closes: the gradient law
  stands, and err → cost is Parcae-specific.
- **The retrofit passes are weak evidence.** Every direction metric "passes" the retrofit only by predicting
  near-zero costs (0.001–0.005), and the observed costs are also small (−0.001 to 0.011). That is the same leniency
  amendment A1 targeted in v167. Most of them underpredict the retrofit by 2–5×.

## Observation worth recording (not a result)

**At matched norm error, the Adam-step deficit differs ~10× between the models.**
- Retrofit a 0.3, k 1: err 0.66, 1 − useful_adam **0.016**.
- Parcae R 32, a 0.7, k 3: err 0.63, 1 − useful_adam **0.163**.
- The plain-SGD deficit at matched err is similar across the models (0.62 vs 0.53).
- So the difference appears only in Adam's per-coordinate view. That is consistent with the retrofit's insensitivity
  to damping and with Parcae's costs.
- On the retrofit, damping does not raise the Adam deficit at all (0.024 undamped, 0.016–0.022 damped, at k 1), just
  as the measured cost did not rise.

**Derivation bracket check** (d_sgd = 1 − useful_sgd vs err; prediction recorded before the data):
- **Parcae: near the aligned end.** d_sgd/err = 0.77–0.91. ✓
- **Retrofit undamped: intermediate.** k 1: d_sgd 0.21 vs err 0.37 and err² 0.13.
- **Retrofit damped: near the aligned end** (0.94), not the orthogonal end. ✗ The predicted retrofit ≈ orthogonal
  fails for the damped cells.
- So the plain-SGD angle does not explain the retrofit's damping insensitivity; the Adam preconditioning might.

**Caveats.**
- Adam's second moment v is estimated from only 4 batches of the frozen model, not from a training run's Adam state.
- The Parcae R 8 cells look noisy (k 2: 0.025 vs R 16's 0.22), which is a likely contributor to the lower R² of the
  Adam metrics.
- 7 metrics were tried; this observation is post hoc.

**If pursued**, it needs a fresh pre-registered test:
- v taken from actual Adam training states;
- the Adam-deficit curve fitted on Parcae and frozen;
- new cells on a third model, or on held-out retrofit settings with larger predicted costs.
