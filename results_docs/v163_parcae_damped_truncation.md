# v163 result: MIXED — the damped (higher-ρ) variant pays more for truncation at k = 2 and 4; ties at k = 1 and 8; nothing opposite

**Date:** 2026-10-03. **Pre-registered outcome: MIXED** (2 of 4 scored pairs in the law's direction, 0 opposite; LAW
needed ≥ 3).
- Prereg `docs/prereg_v163_parcae_damped_truncation.md` (frozen b9bf35c).
- Study driver sha d5c2647126de, on all 84 study runs (36 tuning, 2 extension, 48 evaluation, incl. floors).
- Analysis `v163_analyze.py --verdict`, unchanged since the freeze apart from the reported-only ρ_end block added
  before the freeze.

## Runs
- **Tuning** (seed 1631001):
  - every truncated cell and full backprop: LR 3e-4;
  - floors: 9e-4, confirmed by the edge extension (2.7e-3 worse).
- **Evaluation:** seeds 1631101–1631104, Kaggle T4 ×2, 8 GPUs, both accounts.
- **Logistics:** evaluation runs of the 10 cells with final LRs started during the floor extension, as in v161/v162.
- No divergence among the selected runs. One tuning run (a = 1.0, k = 32, LR 9e-4) went unstable (val 4.05) and was not
  selected.

## Results (test loss; φ = fraction of the loop's credit lost; 90% CIs over 4 paired seeds)

**Floor re-check passes:** L(k = 0) − L(full) = 0.166 [0.165, 0.167] (a = 1.0) and 0.173 [0.169, 0.177] (a = 0.5).

| cell | law ρᵏ | grad err (pilot) | φ | raw ΔL |
|---|---|---|---|---|
| a 1.0, k 1 | 0.825 | 0.820 | 0.224 [0.215, 0.232] | 0.0370 |
| a 1.0, k 2 | 0.681 | 0.675 | 0.100 [0.088, 0.112] | 0.0166 |
| a 1.0, k 4 | 0.463 | 0.443 | 0.024 [0.012, 0.036] | 0.0039 |
| a 1.0, k 8 | 0.215 | 0.205 | 0.002 [−0.012, 0.015] | 0.0003 |
| a 0.5, k 1 | 0.902 | 0.877 | 0.212 [0.196, 0.228] | 0.0368 |
| a 0.5, k 2 | 0.815 | 0.791 | 0.142 [0.128, 0.155] | 0.0246 |
| a 0.5, k 4 | 0.663 | 0.645 | 0.072 [0.060, 0.083] | 0.0125 |
| a 0.5, k 8 | 0.440 | 0.418 | 0.020 [0.010, 0.030] | 0.0035 |

| test | result |
|---|---|
| H0 (truncation costs at k = 1, both a) | PASS |
| H2 Spearman(law, φ), 8 cells | **0.976** PASS |
| H1 k = 1: φ(0.5) − φ(1.0) | −0.011 [−0.027, +0.004] → TIE |
| H1 k = 2 | **+0.042 [+0.023, +0.061] → law direction** |
| H1 k = 4 | **+0.048 [+0.028, +0.068] → law direction** |
| H1 k = 8 | +0.018 [−0.004, +0.041] → TIE (law sign, CI crosses 0; below the 0.02 minimum) |

Raw ΔL gives the same pattern: k = 2 +0.0081 [0.0043, 0.0118] LAW; k = 4 +0.0085 [0.0049, 0.0122] LAW; k = 1 and
k = 8 TIE.

**Outcome: MIXED.**

## Reading

1. **"k alone" fails wherever the costs are measurable.**
   - At k = 2 and 4, the damped variant (same weights, same converged function, higher ρ) loses 1.4× and 3× as much of
     the loop's credit as the undamped one, at the same k and R.
   - At k = 1 both errors are ≥ 0.82, the top of the convex cost curve, and the costs tie, as predicted in advance.
   - At k = 8 both costs are near zero (0.002 vs 0.020). The difference has the law's sign but is not resolved with
     4 seeds.
   - Under the frozen rule this is MIXED, not LAW.
2. **Cost behaves as one function of the gradient error across both variants.**
   - Ordering all 8 cells by pilot gradient error gives an almost monotone φ:
     0.877 → 0.212, 0.820 → 0.224, 0.791 → 0.142, 0.675 → 0.100, 0.645 → 0.072, 0.443 → 0.024, 0.418 → 0.020,
     0.205 → 0.002.
   - The only inversion is at the top, within noise.
   - The pre-stated cross-variant match holds: a = 0.5 at k = 4 (err 0.645, φ 0.072) sits next to a = 1.0 at k = 2
     (err 0.675, φ 0.100).
   - Exploratory (not a pre-registered test): this is the clearest evidence so far that **truncation cost tracks the
     gradient error, not the loop count**.
3. **Scope** (pre-stated):
   - damping cannot separate ρ from effective depth a·k;
   - one model, GSM8K fine-tune, 600 steps.

## Reported only: does truncated training reshape the loop? Prediction FAILED

End-of-run ρ (mean over evaluation seeds); starting ρ 0.825 / 0.903:

| a | k 0 (floor) | k 1 | k 2 | k 4 | k 8 | k 32 |
|---|---|---|---|---|---|---|
| 1.0 | 0.825 | 0.784 | 0.800 | 0.791 | 0.773 | 0.780 |
| 0.5 | 0.885 | 0.860 | 0.858 | 0.859 | 0.856 | 0.858 |

- Spearman(k, ρ_end) over k > 0: −0.60 (a = 1.0) and −0.50 (a = 0.5). The prediction "smaller k → smaller ρ_end" fails.
- What happens instead:
  - any training that gives the loop credit (k ≥ 1) lowers ρ by ~0.03–0.05, roughly independent of k;
  - the floor (k = 0, loop weights receive no gradient) leaves ρ near its start.
- So the "truncation-trained models decay, full-backprop-trained Ouro grows" pattern is not explained by truncation
  depth reshaping the loop, at least in 600-step fine-tunes.

## Predictions scored (prereg §7)

| prediction | P | outcome |
|---|---|---|
| Outcome LAW/ED / K-ALONE / MIXED / AGAINST / UNINF. | 0.45 / 0.20 / 0.20 / 0.05 / 0.10 | **MIXED** (my 0.20) |
| φ(1.0, 4) < 0.05 and φ(0.5, 4) > 0.05 | 0.55 | ✓ (0.024, 0.072) |
| k = 1 pair TIE | 0.5 | ✓ |
| Spearman(k, ρ_end) > 0 within each a | 0.6 | ✗ (−0.60, −0.50) |

## Consequences (prereg §4, MIXED)

- **Reported as MIXED:** the law's direction is supported at k = 2 and 4 (where costs are measurable and the law
  predicts the largest differences), with ties at k = 1 (ceiling) and k = 8 (floor), and no contrary pair. No method
  claim from this test alone.
- **For the write-up:**
  - "k alone" is not sufficient: at matched k and R, the variant with the slower-decaying adjoint pays more where the
    costs are measurable;
  - cost is close to a single convex function of the gradient error;
  - ρ vs effective depth is unresolved;
  - the theory and adaptive method are Aicher et al. 2019's (RNNs) — see `docs/aicher2019_overlap_2026-10-03.md`.
- **Natural next tests:**
  - more seeds, or a larger damping gap (a = 0.3 at R = 64), to resolve k = 8;
  - a non-damping ρ knob, to separate ρ from effective depth;
  - a second model.
