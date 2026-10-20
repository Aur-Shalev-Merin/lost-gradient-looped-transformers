# v204 result: PREDICTED. The two-mode window account predicted ρ at 96 and 128 loops on Parcae-140m

**Plan and code.**
- Plan: `docs/prereg_v204_drift_prediction.md`, frozen e35d92f, amendment 1 f767b13; both pushed before the run.
- Scorer: `v204_score.py`, committed 73655d9 before any data existed.
- Run: aurmerin/astrocyte-v204-drift v1, 2026-10-18, T4 × 2. The S10 driver `v196_band.py` was unchanged; same 16
  sequences and protocol as S10.
- Derivation and predictions: `docs/drift_derivation_2026-10-15.md`.
- Raw data: `results_v204/`.

**Parcae-140m (decides).**

| | predicted | measured |
|---|---|---|
| ρ_fit at R = 96 | 0.891 [0.875, 0.904] | 0.898 (inside) |
| ρ_fit at R = 128 | 0.900 [0.875, 0.910] | 0.908 (inside) |
| paired rise, ρ(128) − ρ(64) | about +0.03, bootstrap interval > 0 | +0.039 [+0.009, +0.047] |
| lost gradient, k = 1/2/4/8, change from 64 loops | within 0.03 | 0.000 at every k (to 3 decimals) |

**Verdict: PREDICTED.** Both measured values sit 0.007–0.008 above the predicted centre; the 128-loop one is close to
the top of its interval. The two-mode fit slightly underestimates how far the fit keeps rising.

Window check (reported, no verdict): the 128-loop profile fitted over its first 96 loops matches the 96-loop run
(median difference +0.001, max 0.030). The window account holds at the new loop counts too.

**Parcae-370m (descriptive only, per amendment 1).**
- ρ_fit 0.931 at 96 and 0.932 at 128 loops, inside its wide intervals.
- Paired rise +0.025 [+0.013, +0.042].
- The lost gradient moved little at 96 loops (at most 0.006). At 128 loops it fell by 0.010–0.032, and k = 8 crossed
  the 0.03 bar (−0.032). On this model the "lost gradient carries over" statement holds to 96 loops and starts to slip
  at 128 for k = 8.
- Window check: median +0.000, max 0.100.

**What this changes in the paper (per the frozen mapping).**
- §3 gains one sentence: the drift is a fitting-window effect, and a closed form fitted on 64-loop profiles predicted ρ
  at 96 and 128 loops on Parcae-140m before they were measured.
- Appendix J gains the closed form, presented that way.
- The S10 sentence on carry-over gets the Parcae-370m caveat: −0.032 at k = 8 at 128 loops.
