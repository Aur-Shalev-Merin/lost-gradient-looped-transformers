# v166 result: ERR-PREDICTS (2026-10-04)

- **Prereg:** `docs/prereg_v166_slam_dunk.md`, frozen d0a0715 before any treatment run.
- **Analysis:** `v166_analyze.py` (locked, unchanged).
- **Runs:** `results_v166/{a,b,c,d}/v164_runs_*.jsonl`. Kaggle kernels astrocyte-v166-eval-{a,b} (aurshalevmerin) and
  -{c,d} (aurmerin).
- 56/56 runs completed; 8/8 on-device selftests OK.

## Validity

- All 14 full-backprop controls are valid (0–4 non-finite steps; the threshold is 10).
- Evaluation seeds = the first 5 (1661101–1661105). The backups were not needed.
- Floor gates:
  - R 64, a 0.3: 0.1755 [0.1740, 0.1770];
  - R 48, a 0.4: 0.1838 [0.1795, 0.1881].
  - Both pass.

## Scored cells (out of sample; predictions locked at freeze)

| cell (R, a, k) | err | observed φ (se) | M_err (\|res\|) | M_k "k alone" (\|res\|) |
|---|---|---|---|---|
| 64, 0.3, 8 | 0.602 | **0.0578** (0.0041) | 0.0611 (**0.0033**) | 0.0057 (0.0521) |
| 64, 0.3, 12 | 0.477 | **0.0268** (0.0027) | 0.0278 (**0.0011**) | 0.0000 (0.0268) |
| 48, 0.4, 4 | 0.699 | **0.0879** (0.0054) | 0.1012 (**0.0133**) | 0.0468 (0.0411) |
| 48, 0.4, 8 | 0.502 | **0.0316** (0.0034) | 0.0330 (**0.0015**) | 0.0057 (0.0259) |

- **MAE:** M_err 0.0048 vs M_k 0.0365 (ratio **0.13**; the bar was ≤ 0.5).
- M_err closer in **4/4** cells; calibrated in **4/4**.
- **OUTCOME: ERR-PREDICTS** (the top pre-registered outcome).

## What it means

- "k alone" says keeping 8 or 12 loops costs essentially nothing (≤ 0.006). In these heavily damped loops it costs 3–6%
  of the loop's credit, 8–14 standard errors above zero.
- One curve of the frozen-model gradient error, fitted on earlier experiments and measured in minutes before training,
  predicted those costs to within 0.001–0.003.
- That holds at a loop count (R 48) and damping (a 0.4) never trained before.
- The largest miss (R 48, k 4: 0.013 over) repeats v164's pattern: the power law runs slightly steep at the highest
  errors. It is still inside the calibration band.

## Limits (as pre-registered)

- One model (Parcae-140m); 600-step fine-tunes on GSM8K.
- **One direction only.** Every scored cell has the error rule predicting *more* cost than "k alone". No cell where it
  predicts *less* was tested: over-relaxation failed in the pilot, and Parcae-370m was deferred.
- The predictor is the measured gradient error. Damping still cannot separate ρ from "effective depth" (a·k).
- The theory and adaptive method are Aicher et al. 2019 (RNNs). What is new here is the looped-transformer measurement
  and this training-level test.

## Reported, not scored

- **ρ_end** (after training):
  - R 64, a 0.3: floor 0.94, truncated 0.913–0.923, full 0.91–0.92;
  - R 48, a 0.4: floor 0.92, truncated 0.88–0.90, full 0.88–0.90.
  - Same pattern as v163: training that gives the loop credit lowers ρ by ~0.02–0.03, regardless of k.
