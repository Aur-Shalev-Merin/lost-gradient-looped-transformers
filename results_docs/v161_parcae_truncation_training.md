# v161 result: INVALID (apparatus bug) — originally reported as AGAINST

> **INVALIDATED 2026-10-01.** Every truncated arm (0 < k < R) and the floor trained with **almost no gradient to the
> looped core's weights**. Cause: PyTorch autocast's weight-cast cache. Parcae runs the no-grad loops first, so the
> fp16 copy of each core weight is cached inside `torch.no_grad()` without a grad connection, and the later grad loops
> reuse it. Measured on Kaggle (`v161_gradcheck.py`, kernel astrocyte-v161-gradcheck, frozen Parcae-140m, batch 8):
>
> | path | |g| at k = R | |g| at k = R−1 | rel. error | core_block |Δ| / |g| |
> |---|---|---|---|---|
> | fp32 (any path, batch 1 or 8) | 6.10 | 5.76 | 0.14 | 0.68 / 5.40 |
> | **fp16 autocast (the v161 training path)** | 6.10 | **2.82** | **0.87** | **5.30 / 5.41** |
>
> (R = 8; R = 16 the same: 0.90, core 5.75 / 5.81.) Minimal repro: a Linear layer first called under `no_grad` inside an
> autocast region gives no gradient on a later grad-enabled call in the same region. So the truncated arms were
> effectively "loop weights frozen; prelude, injection and readout train". That explains the flat-in-k cost, the
> k = R − 1 result (same 0.040 as k = 4) and the 27–35% lower training gradient norm. The full arm (k = R) has no
> no-grad loop and was unaffected. The frozen-model gradient law (fp32 throughout) is unaffected.
>
> **Version dependence (2026-10-01, Colab A100, `colab_parcae_autocast_check.ipynb`):** on PyTorch **2.11.0** the
> cache bug does **not** occur, in bf16 or fp16, eager or compiled: the looped-core gradient with the cache on equals
> the cache-off gradient. It does occur on PyTorch 2.5.1 (local repro) and on Kaggle's image (v161, gradcheck). So it is
> a PyTorch-version issue, fixed by 2.11 (the exact version not yet bisected). Parcae's published setup
> (`torch.compile`) is fine on 2.11.
>
> **Status:** the AGAINST verdict and its readings 1–4 below are void. The question is untested at training level.
> The pre-freeze selftest checked fp16 loss and the k = R gradient but never a truncated gradient in fp16. That gap is
> what let it through. Re-run: a new version with `cache_enabled=False` (or fp32) and a selftest comparing the
> truncated gradient under fp16 against fp32.
> The exploratory k = R−1, drift-probe truncated arms and single-pass runs share the bug and are void as well
> (drift-probe *measurements* were fp32 and stand).

**Date:** 2026-09-30. **Pre-registered outcome as originally scored: AGAINST — VOID (see box above).**
Prereg `docs/prereg_v161_parcae_truncation_training.md` (frozen e922ed4). Driver sha ddb4e99b660b (all 125 study
runs). Analysis `v161_analyze.py --verdict`, unchanged since the freeze.

## Runs
- **Tuning:** 66 runs (11 cells × LR {1e-4, 3e-4, 9e-4} × 2 tuning seeds) + 4 edge-rule runs (floor k = 0 at 2.7e-3).
  0 diverged.
- **Selected LR:** 3e-4 for every truncated cell and for full backprop (grid centre); 9e-4 for the floor at both R
  (2.7e-3 worse).
- **Evaluation:** 55 runs (11 cells × 5 seeds) at the selected LRs. Kaggle T4 ×2, 8 GPUs, both accounts.
- **Deviation (logistics only):** evaluation runs for the 9 cells whose LR was already final were started while the
  floor extension ran. The floor's evaluation ran after re-selection. Same runs as the frozen "eval" wave. Two
  kernels crashed at start-up from a builder bug (NameError, no run executed) and were re-pushed.

## Results (evaluation seeds, test loss; 90% CIs over 5 paired seeds)

**Gates.** The floor re-check passes: L(k = 0) − L(full) = 0.197 [0.195, 0.200] at R 8 and 0.171 [0.168, 0.174] at
R 16.

| cell | law ρᵏ | measured err | φ (fraction of loop credit lost) | raw ΔL vs full |
|---|---|---|---|---|
| R 8, k 1 | 0.773 | 0.805 | 0.314 [0.311, 0.317] | 0.0619 |
| R 8, k 2 | 0.597 | 0.648 | 0.250 [0.245, 0.255] | 0.0493 |
| R 8, k 4 | 0.356 | 0.386 | 0.214 [0.209, 0.220] | 0.0423 |
| R 16, k 1 | 0.802 | 0.813 | 0.362 [0.353, 0.371] | 0.0618 |
| R 16, k 2 | 0.643 | 0.663 | 0.288 [0.281, 0.295] | 0.0492 |
| R 16, k 4 | 0.414 | 0.421 | 0.249 [0.240, 0.257] | 0.0425 |
| R 16, k 8 | 0.171 | 0.180 | 0.241 [0.231, 0.250] | 0.0412 |

Full backprop: 1.3747 (R 8), 1.3745 (R 16). Seed SD of raw ΔL ≈ 0.0015.

| test | result |
|---|---|
| H0: truncation costs something | **PASS** (φ at k = 1: 0.31 and 0.36) |
| H1: Spearman(law, φ) ≥ 0.75 | **PASS** (0.964) |
| H2 (16, 4) − (8, 4), law predicts + | φ: +0.034 [0.025, 0.043] → LAW direction |
| H3 (8, 2) − (16, 4), law predicts + | φ: +0.001 [−0.006, 0.009] → TIE |
| H3 (8, 4) − (16, 8), law predicts + | φ: **−0.026 [−0.036, −0.016] → OPPOSITE** |

**Outcome: AGAINST**, by the frozen rule "any discriminating pair directional opposite to the law". At R 16, k 8 has
less than half the gradient error of k 4 at R 8 (0.18 vs 0.39), yet loses *more* of the loop credit.

## Reading

1. **The training cost depends on k alone, not on R and not on ρᵏ.** In raw loss, R 8 and R 16 agree at every
   matched k to within 0.0002: 0.0619/0.0618, 0.0493/0.0492, 0.0423/0.0425. The two H2/H3 "law" signals on φ come from
   the normalisation: the floor gap is smaller at R 16 (0.171 vs 0.197), which inflates every R 16 φ by about 15%. On
   raw ΔL, H2 is a tie (+0.0002), and the one H3 pair in the law's direction ((8, 2) vs (16, 4), +0.007) is also
   predicted by "k alone" (k 2 vs k 4).
2. **The cost plateaus at about 0.04 nats for k ≥ 4, then falls to zero only at k = R.** The gradient error falls
   from 0.39 (R 8, k 4) to 0.18 (R 16, k 8) with no change in cost (0.042 vs 0.041). Any truncation carries a fixed
   penalty of ~0.04 nats (~21–24% of the loop credit) that the step-0 gradient error does not see.
3. **Gradient error overstates the cost.** A last-loop-only gradient that is 80% wrong at step 0 loses 31–36% of
   the loop credit. That is the program's recurring pattern (Papers 1–2): crude credit trains much better than its
   gradient error suggests.
4. **Extra loops add nothing after fine-tuning.** Full backprop reaches the same loss at R 8 and R 16 (1.3747 vs
   1.3745), although R 16 started lower (2.62 vs 2.77).

## Scored predictions (§7 of the prereg)

| prediction | P | outcome |
|---|---|---|
| Outcome LAW / LAW-CONSISTENT / AGAINST / UNINFORMATIVE | 0.25 / 0.40 / 0.15 / 0.20 | AGAINST (my 0.15) |
| φ_{8,1} < 0.5 while err 0.81 | 0.7 | ✓ (0.31) |
| φ_{8,4} < 0.15 | 0.65 | ✗ (0.21) |
| The H2 pair ties | 0.6 | ✗ on φ (normalisation), ✓ on raw ΔL |

## Consequences (per the prereg §4)

- The write-up reports err_k ≈ ρᵏ as a **gradient-level** fact only (now three checkpoints: Huginn-3.5B,
  Parcae-1.3B, Parcae-140m). It states that ρᵏ did not predict training cost here. The "measure ρ, then choose k"
  method claim is **dropped**.
- The positive training-level finding is the plateau: the cost depends on k with a fixed ~0.04 penalty for any
  truncation. One model, one task, 600 steps; exploratory as a mechanism.
- **Open (exploratory, not part of this verdict):** where the fixed penalty comes from.
  `v161_drift_probe.py` re-measures ρ and err during training (full arm and truncated arms). If training moves ρ
  towards 1, the step-0 measurement is simply the wrong time to measure.
