# v167 result: K-ALONE-AS-GOOD; the Parcae cost curve does NOT transfer to the retrofit (2026-10-04)

- **Prereg:** `docs/prereg_v167_retrofit_transfer.md` (frozen 880be7a; amendment A1 33fc028, made while all kernels
  were running and before any output was seen).
- **Analysis:** `v167_analyze.py`, locked.
- **Runs:** `results_v167/{a,b,c,d}/gpu*/v167_runs_*.jsonl`. Kernels astrocyte-v167-eval-{a,b} (aurshalevmerin) and
  -{c,d} (aurmerin).
- 84/84 runs completed; 8/8 on-device selftests OK.

## Validity

- All 21 full-backprop controls are valid (0–1 non-finite steps). Seeds 1671101–1671105 used.
- Floor gates:
  - a 1.0: 0.0808 [0.0801, 0.0816];
  - a 0.5: 0.0806 [0.0800, 0.0813];
  - a 0.3: 0.0797 [0.0791, 0.0803].
  - All pass.

## Cells

| a | k | frozen err | observed φ (se) | M_err (Parcae curve) | M_k_own |
|---|---|---|---|---|---|
| 1.0 | 1 | 0.357 | **0.0087** (0.0005) | 0.0100 | — |
| 1.0 | 2 | 0.137 | **−0.0012** (0.0005) | 0.0004 | — |
| 0.5 | 1 | 0.566 | **0.0113** (0.0006) | 0.0484 | 0.0087 |
| 0.5 | 2 | 0.344 | **0.0019** (0.0006) | 0.0087 | −0.0012 |
| 0.3 | 1 | 0.650 | **0.0087** (0.0009) | 0.0781 | 0.0087 |
| 0.3 | 2 | 0.442 | **0.0042** (0.0008) | 0.0207 | −0.0012 |

- **(i) TRANSFER: FAIL.** Calibrated 4/6 (both high-cost cells missed); MAE 0.022; damped MAE 0.032. It also fails
  under the rule as frozen.
- **(ii) RIVAL: FAIL.** M_err closer in 0/4 damped cells; MAE 0.032 vs 0.0028 for "k alone" (ratio 11.6).
- **OUTCOME: K-ALONE-AS-GOOD** (pre-registered prediction for this outcome: 0.10).

## What it shows

- **Undamped, the curve was right** (0.010 vs 0.009 at k 1; ≈ 0 at k 2). But that is the trivial case the prereg
  warned about.
- **Damping broke the link between gradient error and cost.**
  - Damping raised the frozen-model gradient error at k = 1 from 0.36 to 0.65. On Parcae, an error that large meant
    ~8% of the loop's credit lost.
  - Here the cost stayed at ~0.9–1.1%.
  - At k = 2 there is a small rise with damping (−0.001 → 0.002 → 0.004), in the predicted direction but 5× smaller
    than predicted.
- **"Cost is one function of the frozen gradient error" is not model-general.** It held out of sample on Parcae-140m
  (v164–v166) and fails on a looped model of a different family and training history.
- **What survives:**
  - the gradient law (err ≈ ρᵏ, a property of the frozen model, which held on the retrofit again);
  - the Parcae-specific training result.
  - The claim that gradient error is a *universal* predictor of truncation cost is refuted.

## Candidate explanations (untested; for the next step, not conclusions)

1. **Error size ≠ error usefulness.**
   - err is a norm of the gradient difference. What hurts training is losing the gradient components that reduce the
     loss.
   - In the retrofit, the last loop alone may already carry the useful direction, even when most of the norm comes
     from earlier loops.
   - Related: Jacobian-free / one-step gradients train fixed-point models well (Fung et al. 2022). Not checked here.
2. **The retrofit's loop barely matters for capacity.**
   - The never-loop control matched the looped model (2.907 vs 2.905; `docs/retrofit_heal_drop6_2026-10-02.md`).
   - The 0.08 floor gap may come mostly from adapting the shared block's weights as ordinary layers, which the last
     loop's gradient already reaches.
   - Parcae's loop is native and load-bearing.
3. **Damping and the retrofit interact differently.** With a < 1 the first loop's output is mixed with the state, and
   the training signal through the last loop changes in ways the norm-error metric does not capture.

A cheap discriminating diagnostic for (1): the cosine between the truncated and full gradient, and the
loss-reduction of one step along each (a "useful error"), measured on Parcae vs the retrofit at matched err.
