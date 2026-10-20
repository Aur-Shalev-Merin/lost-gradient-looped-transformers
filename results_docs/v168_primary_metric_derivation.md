# v168: primary metric fixed by derivation, before results (2026-10-04, 17:38)

**Status at writing:**
- astrocyte-v168-parcae RUNNING.
- astrocyte-v168-retro COMPLETE, but its output had not been downloaded or viewed.
- No v168 measurement had been seen.

## Derivation

Write the truncated gradient as g_k = g − b, where b is the dropped earlier-loop contribution. The gradient law gives
|b| = err·|g| with err ≈ ρᵏ.

To first order, a step along direction s lowers the loss by η·(g · s). Relative to a full-gradient step:

- **SGD:** (g · g_k)/|g|² = 1 − err·cos(g, b).
  - The progress deficit per step is err × cos(g, b).
  - Norm error alone (the gradient law) fixes only the first factor.
  - Within one model, cos(g, b) may be roughly constant. φ then looks like a function of err alone, which is what
    v162–v166 saw on Parcae. Across models the angle can differ, which is what v167 saw.
- **Adam (the optimiser actually used):** the step is u = g/√v. The analogous deficit is
  1 − (g · u_k)/(g · u_R) = **1 − useful_adam** in `v168_direction_probe.py`.

**Not derived:**
- the end-of-training effect (truncated training converges where g_k = 0, not where g = 0);
- the curve's shape (Parcae's exponent ≈ 3.4).

## Pre-specified for the v168 exploratory analysis

- **PRIMARY metric: 1 − useful_adam.** It is scored by `v168_analyze.py`'s fixed candidate rule:
  - fit φ = c·dᵖ on the 29 Parcae cells;
  - Parcae R² ≥ 0.9;
  - retrofit damped MAE ≤ 0.015;
  - all 4 damped retrofit cells calibrated.
- **Secondary (SGD analogue):** 1 − useful_sgd.
- The other five metrics are reported as exploratory only.
- A hit on the primary metric still requires a pre-registered confirmation on fresh cells before any claim.

## Addendum: deriving the gradient law, and a bracket linking it to cost (recorded before any v168 result was seen)

**Derivation.**
- g = Σ_j c_j, the per-loop contributions counted back from the output.
- At a settled loop, the adjoint passes through the same Jacobian each loop, so |c_j| ∝ ρʲ, with ρ = the dominant
  adjoint eigenvalue (the asymptotic rate, not the transient norm).
- **Aligned contributions:** err_k = (ρᵏ − ρᴿ)/(1 − ρᴿ).
- **Mutually orthogonal contributions:** err_k = √((ρ²ᵏ − ρ²ᴿ)/(1 − ρ²ᴿ)).
- Both are ≈ ρᵏ, so the law is insensitive to the direction structure. This is Aicher et al. 2019's argument (RNNs).
- **Failure mode:** a slow oscillating adjoint mode that contributes little to the parameter gradient (over-relaxation,
  v166 pilot A). In general err is a weighted mixture of geometric modes.

**Bracket** (SGD first-order progress deficit per step, d_sgd = 1 − useful_sgd = err·cos(g, b)):
- aligned contributions: cos(g, b) = 1, so d_sgd = err;
- orthogonal contributions: g·b = |b|², so cos(g, b) = err and d_sgd = err².
- Hence err² ≲ d_sgd ≲ err.

**Prediction (before v168 data).**
- Parcae-140m sits near the aligned end: d_sgd/err closer to 1.
- The SmolLM2 retrofit sits near the orthogonal end: d_sgd ≈ err².
- (A tiny local pipeline check on the laptop GPU with random tokens, 1 batch, gave the retrofit err 0.32 and d_sgd 0.08. That is
  pipeline-test data, not the measurement.)
