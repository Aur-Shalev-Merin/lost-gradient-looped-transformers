# Loop-gradient law on the two healed looped SmolLM2 retrofits (2026-10-02)

`retrofit_rho_probe.py` via `colab_retrofit_rho_probe.ipynb` (Colab; fp32; 16 held-out sequences × 512 tokens from the
healing run's held-out set; all parameters).

**Checks.** The step-by-step forward reproduces each model's own loss exactly at R = 1 and R = 4:
- run 1: 3.51298 / 3.46909;
- plan B: 3.58341 / 3.56124.

**Results** (medians over sequences; ρ = exp(−slope) of a log-linear fit of the adjoint norms over all R loops):

| model | R | ρ | k = 1 err / ρᵏ | k = 2 | k = 4 | k = 8 |
|---|---|---|---|---|---|---|
| run 1 (11-8-11, no drop) | 8 | 0.347 | 0.304 / 0.347 | 0.108 / 0.120 | 0.015 / 0.015 | — |
| run 1 | 16 | 0.424 | 0.304 / 0.424 | 0.109 / 0.180 | 0.016 / 0.032 | 0.001 / 0.001 |
| plan B (drop 12–17) | 8 | 0.384 | 0.316 / 0.384 | 0.122 / 0.147 | 0.021 / 0.022 | — |
| plan B | 16 | 0.465 | 0.317 / 0.465 | 0.124 / 0.216 | 0.023 / 0.047 | 0.001 / 0.002 |

Median |err − ρᵏ| at R = 8:
- run 1: 0.024 / 0.018 / 0.005;
- plan B: 0.068 / 0.026 / 0.002.

The error of the loop parameters alone ≈ the all-parameter error (0.32–0.33 at k = 1).

**Reading.**
1. **err_k ≈ ρᵏ holds on both retrofits at R = 8** (closely at k ≥ 2; ρᵏ overestimates by 0.04–0.07 at k = 1).
   That is five models in total: Huginn-3.5B, Parcae-1.3B, Parcae-140m, and these two. They are the first
   retrofitted, rather than natively looped, models.
2. **ρ is much smaller here (0.35–0.38) than in the natively trained truncated models (0.77–0.86).** This fits the
   forward behaviour: loss at R = 4, 8 and 16 is identical, so the loop converges within ~4 passes.
3. **The R = 16 fit is biased, not the law.** The truncation error at R = 16 equals R = 8's (only the last loops
   matter), but the adjoint in the early loops is ≤ 1e-4. A log-linear fit through that near-floor region flattens the
   slope (ρ 0.42–0.47). The last loops decay at ~0.27–0.35 per loop. Fix: fit ρ only over loops where the adjoint is
   measurable (e.g. ≥ 1e-3 of the last), or over the last k_max + 1 loops.
4. **Implications.**
   - With ρ ≈ 0.35, backprop through 4 loops gives a gradient within 2% of the full one. By v162's convex
     cost-vs-error relation, truncation at k ≥ 2 should cost almost nothing in these models.
   - The two models have nearly equal ρ, so they cannot act as the "different ρ at fixed k, R" contrast. That needs
     deliberately built variants with ρ moved towards 0.7–0.9 (e.g. damping the loop update during healing).

Predictions (recorded before the run):
- ρ smaller than Huginn/Parcae ✓.
- Law within 0.05 at R = 8: ✓ run 1; plan B ✓ at k = 2, 4 and ✗ at k = 1 (0.068).
