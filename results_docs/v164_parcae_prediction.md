# v164 result: INCONCLUSIVE (frozen rule) — the gradient-error curve beats "k alone" on 9/10 new cells and 5/5 discriminating cells, but its MAE (0.034) misses the 0.03 bar, driven by one unstable full-backprop control run

**Date:** 2026-10-03. **Pre-registered outcome: INCONCLUSIVE.**
- Prereg `docs/prereg_v164_parcae_prediction.md` (frozen 1c5380c, procedural lock; locked predictions recorded in
  74f07cd before any training result existed).
- Driver sha feaf9a0ba438 on all 80 runs (v163 module d5c2647126de).
- All 6 GPU jobs passed the on-device selftest. The measurement kernel was re-run once (v1 cancelled at ~4 min, before
  output).

## Results (test-loss φ, evaluation seeds 1641101–1641105)

**Floor checks pass:** 0.158 [0.137, 0.178] (a 1.0); 0.170 [0.166, 0.174] (a 0.7); 0.177 [0.173, 0.180] (a 0.3, R 64).

| cell (R, a, k) | err | observed φ (se) | M_err pred (\|res\|) | M_k pred (\|res\|) |
|---|---|---|---|---|
| 32, 1.0, 3 | 0.546 | −0.029 (0.071) | 0.047 (0.075) | 0.061 (0.089) |
| 32, 1.0, 6 | 0.307 | −0.074 (0.075) | 0.006 (0.080) | 0.016 (0.090) |
| 32, 0.7, 2 | 0.744 | 0.127 (0.004) | 0.135 (0.008) | 0.102 (0.025) |
| 32, 0.7, 3 | 0.646 | 0.076 (0.003) | 0.083 (0.007) | 0.061 (0.015) |
| 32, 0.7, 4 | 0.558 | 0.046 (0.003) | 0.050 (0.005) | 0.031 (0.015) |
| 32, 0.7, 6 | 0.423 | 0.015 (0.003) | 0.019 (0.005) | 0.016 (0.001) |
| 64, 0.3, 2 | 0.860 | 0.169 (0.005) | 0.223 (0.054) | 0.102 (0.067) |
| 64, 0.3, 3 | 0.809 | 0.137 (0.004) | 0.181 (0.044) | 0.061 (0.076) |
| 64, 0.3, 4 | 0.763 | 0.111 (0.004) | 0.148 (0.037) | 0.031 (0.080) |
| 64, 0.3, 6 | 0.679 | 0.073 (0.004) | 0.099 (0.025) | 0.016 (0.057) |

| criterion | needed | observed | |
|---|---|---|---|
| MAE_err ≤ 0.7 · MAE_k | ratio ≤ 0.70 | 0.0339 / 0.0516 = **0.66** | ✓ |
| M_err closer in ≥ 70% of cells | ≥ 7/10 | **9/10** | ✓ |
| M_err closer in ≥ 70% of discriminating cells | ≥ 4/5 | **5/5** | ✓ |
| MAE_err ≤ 0.03 | ≤ 0.030 | **0.0339** | ✗ |

**Outcome: INCONCLUSIVE** (the absolute-accuracy bar alone fails).

## Why the absolute bar failed: one unstable control run (post-hoc diagnosis; does NOT change the verdict)

- In the a = 1.0 variant, seed 1641102's **full-backprop control** hit 14 non-finite (fp16-overflow) steps (others:
  0–6) and ended at test **1.4167** vs 1.369–1.375 for the other four seeds.
- φ normalises by that seed's full run, so its k = 3 and k = 6 runs (1.3795, 1.3721) look better than full backprop.
  That gives the negative φ and a seed SE ~20× larger than in every other cell (0.07 vs 0.003–0.005).
- The frozen rules drop only non-finite runs, so this run counts.

**Sensitivity (post-hoc, exploratory).** Dropping only seed 1641102 from the two a = 1.0 cells:
- φ(32, 1.0, 3) ≈ 0.042 (M_err 0.047, M_k 0.061);
- φ(32, 1.0, 6) ≈ 0.001 (M_err 0.006, M_k 0.016);
- MAE_err ≈ 0.020 vs MAE_k ≈ 0.037 (ratio ≈ 0.53), M_err closer in 10/10.

That would meet every ERR-PREDICTS criterion. It is reported as exploratory only; the pre-registered label stands.

## Reading

1. **On the cells unaffected by the bad control, the frozen gradient-error curve predicts truncation cost at new
   damping levels and a new loop count far better than "k alone".**
   - a = 0.7: residuals 0.005–0.008 (except 0.005 at k = 6, where both models agree).
   - a = 0.3, R = 64: the observed costs (0.07–0.17) are **2–5× what k alone predicts** (0.016–0.10).
   - At a = 0.3, M_err is closer than M_k in every cell.
   - This is the strongest training-level evidence in the thread that cost tracks the gradient error, not the loop
     count.
2. **M_err systematically overpredicts at the high-ρ extreme** (a = 0.3: by 0.025–0.054). The fitted curve
   (φ = 0.376·err^3.45) is too steep at errors above ~0.75 when extrapolated to R = 64 / ρ 0.94. The functional form
   needs a gentler top (or ρ-specific saturation), so the claim is "ordering and magnitude roughly right; exact curve
   not universal".
3. **Instability risk of fp16 full backprop at R = 32:**
   - 1 of 5 full-backprop runs degraded here;
   - v163's tuning had an unstable full run at LR 9e-4;
   - future designs should either use bf16-capable hardware or treat the full-backprop control's instability as a
     pre-registered exclusion (e.g. > 10 non-finite steps → rerun the seed).

## Predictions scored (prereg §7)

| outcome | P | |
|---|---|---|
| ERR-PREDICTS | 0.40 | |
| **INCONCLUSIVE** | 0.30 | ✓ |
| K-ALONE-AS-GOOD | 0.15 | |
| UNINFORMATIVE | 0.15 | |

## Consequences

- Per prereg: reported as INCONCLUSIVE, with per-cell residuals.
- For the write-up, the defensible statements are:
  - "k alone" fails badly out of sample at high ρ (a = 0.3, R = 64: costs 2–5× its predictions);
  - the gradient-error curve gets the ordering right in 9/10 new cells and is accurate to ≤ 0.008 at a = 0.7;
  - it overpredicts at the extreme;
  - the absolute-accuracy criterion was missed because of one unstable control run (disclosed, with the post-hoc
    sensitivity).
- **A clean confirmation would need:** a pre-registered rerun of only the a = 1.0 variant (5 fresh seeds, plus an
  instability exclusion rule for controls), and optionally a refit of the curve's top end before predicting further
  extreme cells.
