# v162 result: LAW-CONSISTENT — the cost of truncated loop backprop follows the ρᵏ ordering, but this design cannot separate the law from "more backprop loops is better"

**Date:** 2026-10-01. **Pre-registered outcome: LAW-CONSISTENT.**
- Prereg `docs/prereg_v162_parcae_truncation_training_rerun.md` (frozen 89d8758).
- Re-run of v161, which was invalidated by an autocast weight-cast-cache bug (version-dependent: present on PyTorch
  2.5.1 and Kaggle's image, absent on 2.11).
- Driver sha 0ae4891d7abc on all runs.
- Analysis: `v162_analyze.py --verdict` (v161's analysis with v162 seeds, unchanged since the freeze).

## Runs
- **Tuning:** 66 runs, plus 4 edge-rule runs on the floor at 2.7e-3. 0 diverged.
- **Selected LR:** 3e-4 for every truncated cell and full backprop; 9e-4 for the floor at both R.
- **Evaluation:** 55 runs (11 cells × 5 seeds, 1621101–1621105) on Kaggle T4 ×2, 8 GPUs, both accounts.
- Same logistics as v161: evaluation runs for cells whose LR was already final started during the floor extension.

## Results (evaluation seeds, test loss; 90% CIs over 5 paired seeds)

**Floor re-check passes:** L(k = 0) − L(full) = 0.200 [0.198, 0.201] at R 8 and 0.174 [0.171, 0.178] at R 16.

| cell | law ρᵏ | measured err | φ (fraction of loop credit lost) | raw ΔL vs full |
|---|---|---|---|---|
| R 8, k 1 | 0.773 | 0.805 | 0.183 [0.172, 0.193] | 0.0365 |
| R 8, k 2 | 0.597 | 0.648 | 0.077 [0.070, 0.084] | 0.0154 |
| R 8, k 4 | 0.356 | 0.386 | 0.012 [0.004, 0.020] | 0.0023 |
| R 16, k 1 | 0.802 | 0.813 | 0.209 [0.198, 0.221] | 0.0365 |
| R 16, k 2 | 0.643 | 0.663 | 0.090 [0.080, 0.099] | 0.0156 |
| R 16, k 4 | 0.414 | 0.421 | 0.016 [0.005, 0.027] | 0.0028 |
| R 16, k 8 | 0.171 | 0.180 | −0.005 [−0.013, 0.004] | −0.0008 |

| test | result |
|---|---|
| H0: truncation costs something | **PASS** (φ at k = 1: 0.18 and 0.21) |
| H1: Spearman(law, φ) ≥ 0.75 | **PASS** (1.000) |
| H2 (16, 4) − (8, 4), the one scored pair | φ +0.005 [−0.008, 0.017] → **TIE** |
| H3 (8, 2) − (16, 4) | φ +0.061 [0.048, 0.074] → **LAW direction** |
| H3 (8, 4) − (16, 8) | φ +0.016 [0.004, 0.028] → **TIE** (CI excludes 0 but below the 0.02 minimum) |

**Outcome: LAW-CONSISTENT.** H1 passes and nothing goes against the law, but the H2 family has no directional pair,
so the frozen rule for LAW (each scored family has ≥ 1 directional pair in the law's direction) is not met.

## Reading

1. **With correct gradients, truncation is cheap and its cost falls steeply with k.**
   - Backprop through the last loop only keeps ~80% of the loop's benefit (φ 0.18–0.21).
   - 2 loops lose 8–9%; 4 loops 1–2%; 8 of 16 nothing measurable.
   - Parcae's own training depth (4 of 8) costs ~1%.
2. **The ordering follows ρᵏ exactly** (Spearman 1.0 over 7 cells), and the law beats "same fraction of loops" (k/R):
   at a quarter of the loops, R 16 k 4 loses far less than R 8 k 2 (0.016 vs 0.077). On raw ΔL both H3 pairs are in
   the law's direction (+0.0126 [0.010, 0.015]; +0.0031 [0.001, 0.005]).
3. **It is not separated from "k alone".** On raw loss, R 8 and R 16 agree at every k within ~0.0005 (0.0365/0.0365,
   0.0154/0.0156, 0.0023/0.0028). The law predicts slightly higher cost at R 16, since ρ 0.80 > 0.77; that difference
   is too small to resolve here. This weakness was disclosed before the run (H2 had one barely-scored pair). Both H3
   "law" results are also predicted by "k alone" (k 2 < k 4, k 4 < k 8).
4. **The cost is a steep, convex function of the gradient error, not proportional to it.**
   - φ ≈ c·ρᵏ fits poorly (R² 0.65, c ≈ 0.18).
   - From k 1 → 2 → 4, the error goes 0.81 → 0.65 → 0.39 and φ goes 0.18 → 0.08 → 0.01: roughly φ ∝ err^3.5–4
     (exploratory; three points per R).
   - A small gradient error costs almost nothing; only large errors (k ≤ 2) cost real loss.

## Predictions (prereg §7)

| prediction | P | outcome |
|---|---|---|
| Outcome LAW / LAW-CONSISTENT / AGAINST / UNINF. | 0.30 / 0.40 / 0.20 / 0.10 | **LAW-CONSISTENT** (modal ✓) |
| φ_{8,4} < 0.10 | 0.7 | ✓ (0.012) |
| φ_{8,1} < 0.5 | 0.7 | ✓ (0.18) |
| (16, 4) cheaper than (8, 2), directional | 0.45 | ✓ |
| (16, 8) cheaper than (8, 4), directional | 0.5 | ✗ on φ (TIE; below 0.02 minimum), ✓ on raw ΔL |
| H2 pair ties | 0.6 | ✓ |

## Consequences (prereg §4)

- **The write-up reports:**
  - the frozen-model law (err_k ≈ ρᵏ on Huginn-3.5B, Parcae-1.3B and Parcae-140m);
  - its persistence during training (drift probe);
  - a training result: the cost ordering is consistent with ρᵏ and beats the k/R rule, with steeply convex cost
    in the error.
- **No claim over "k alone".** Separating the two needs models that differ in ρ at the same k and R.
- **The retrofit testbed** (`docs/retrofit_gpt2_looped_sketch_2026-10-01.md`: SmolLM2-135M, variants with different
  ρ) is exactly that test. The sketch's build condition (LAW or LAW-CONSISTENT) is met.
- **Methods lesson** (v161 → v162): selftest the truncated gradient in the training precision. The autocast-cache bug
  is silent and version-dependent.
