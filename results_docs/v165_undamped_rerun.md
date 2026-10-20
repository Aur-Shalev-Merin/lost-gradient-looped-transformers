# v165 result: undamped cells rerun — the frozen gradient-error curve predicts both cells to within 0.002 (secondary PASS); primary ERR-PREDICTS (a clean-up, not independent confirmation)

**Date:** 2026-10-03.
- Prereg `docs/prereg_v165_undamped_rerun.md` (frozen 6cec94e).
- Frozen v164 driver (sha feaf9a0ba438) on all 28 runs; all 4 GPU jobs passed the on-device selftest.
- Predictions: v164's frozen M_err (φ = 0.3759·err^3.453) and M_k, applied to v164's locked measurement. No refit.

## Exclusion rule (pre-stated) in action

| seed | full-control non-finite steps | full test loss | status |
|---|---|---|---|
| 1651101 | 1 | 1.3725 | valid |
| **1651102** | **13** | **1.3942** | **excluded** |
| 1651103 | 2 | 1.3749 | valid |
| 1651104 | 1 | 1.3743 | valid |
| 1651105 | 1 | 1.3723 | valid |
| 1651106 | 2 | 1.3747 | valid (backup, used) |
| 1651107 | 2 | 1.3746 | valid (backup, not needed) |

- Used: the first 5 valid seeds.
- Floor gap 0.164 (min 0.162).
- Note: the unstable full-backprop control was again the second seed of its batch (v164: 1641102), with different seed
  values. It is probably a coincidence, but fp16 full backprop at R 32 is fragile (2 of 12 such controls across
  v164/v165).

## Results

| cell | observed φ (se) | M_err (\|res\|) | M_k (\|res\|) |
|---|---|---|---|
| R 32, a 1.0, k 3 | 0.0479 (0.0039) | 0.0466 (**0.0014**) | 0.0606 (0.0126) |
| R 32, a 1.0, k 6 | 0.0046 (0.0017) | 0.0064 (**0.0018**) | 0.0162 (0.0116) |

- **SECONDARY** (new information; M_err closer and |res| ≤ 0.015 in both cells): **PASS**.
- **PRIMARY** (v164's criteria over its 10 cells, with these replacing v164's corrupted undamped cells):
  MAE_err 0.0188 vs MAE_k 0.0361 (ratio 0.52), M_err closer in 9/10, 5/5 discriminating → **ERR-PREDICTS**.

## How to read it (pre-stated framing)

- **The primary result is a clean-up, not independent confirmation.** 8 of its 10 cells are v164 results already seen.
  It confirms that v164's miss of the absolute bar was the unstable control, not the curve.
- **The new evidence is the secondary check.** On two fresh cells, a curve frozen before v164 ran predicts the cost to
  within 0.002, versus 0.012 for "k alone". These cells only weakly discriminate between the rules (predictions differ
  by 0.010–0.014). So this shows the curve's calibration at low error more than it shows a large contrast with k
  alone.
- **Combined with v164:**
  - the error curve is accurate at moderate error (a 0.7: ≤ 0.008; undamped: ≤ 0.002);
  - it gets the ordering right everywhere;
  - it overpredicts at the high-ρ extreme (a 0.3, R 64: by 0.025–0.054), where "k alone" fails far worse (2–5×
    underprediction).
- **What a slam dunk still needs:** fresh, strongly discriminating cells in both directions, and ideally a second model.
  That is v166 (draft: `docs/prereg_v166_slam_dunk.md`; pilots running).

## Predictions scored

| prediction | P | outcome |
|---|---|---|
| primary ERR-PREDICTS | 0.75 | ✓ |
| secondary PASS | 0.5 | ✓ |
