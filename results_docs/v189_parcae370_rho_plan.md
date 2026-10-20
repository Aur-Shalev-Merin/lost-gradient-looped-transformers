# v189: Parcae-370m lost gradient vs ρᵏ at 16 and 32 loops (plan, written before data)

**Type.** Exploratory measurement on frozen models, no training. Asked for by the 2026-10-13 outside review of the
looped-truncation paper (M1): Parcae-370m was measured in the v166 pilot (`docs/v166_pilots_2026-10-03.md`) at R = 32,
a = 1: ρ 0.868, lost gradient 0.674 / 0.451 / 0.238 at k = 2 / 4 / 8 against ρᵏ 0.75 / 0.57 / 0.32 (0.08–0.12 high).
The paper left it out of Table 1. That run saved only medians.

**This run.** `v189_parcae_rho.py` calls the unchanged v164 `measure()` (the Table 1 protocol for Parcae-140m: 16 GSM8K
validation sequences, batch 1, fp32, all parameters, a = 1), saving per-sequence rows. One Kaggle kernel, two T4s:
- GPU 0: Parcae-370m at R = 16 (Parcae's other Table 1 rows are measured at 16), k = 1 / 2 / 4 / 8; then Parcae-140m at
  R = 16 as a reproduction check.
- GPU 1: Parcae-370m at R = 32, k = 1 / 2 / 4 / 8 / 16 (re-measures the pilot with rows saved).

**Checks.** Parcae-140m at R = 16 must reproduce Table 1 (0.81 / 0.66 / 0.42 at k = 1 / 2 / 4) within 0.02, and
370m at R = 32 the pilot's 0.674 / 0.451 / 0.238 within 0.01. Otherwise the protocol has drifted; report that and do not
use the run for Table 1.

**Expectation.** Huginn at 64 loops (v187) showed ρ fitted over many loops overstating the loss, because the backward
signal fades more slowly far from the output. If the 370m miss is the same effect, then at R = 16 the lost gradient
at k = 1 / 2 / 4 / 8 lies within the paper's spread for the other models (within about 0.07 of the median ρᵢᵏ, between
18% below and 12% above it). This is uncertain: Parcae-140m at R = 32 sits on the line (0.44 against 0.46 at k = 4).

**What each outcome changes in the paper.** Both loop counts are reported whatever happens.
- R = 16 within that spread → Table 1 gets a Parcae-370m row at R = 16. The text says ρᵏ overestimates it at R = 32 by
  0.08–0.12, like Huginn at 64 loops, and the agreement range quoted in §3 and the abstract covers every point measured.
- R = 16 also outside → Table 1 gets the row anyway, and the abstract and §3 say the rule overestimates the lost
  gradient on Parcae-370m (by the measured amount), so it holds closely on five of six models with shrinking signals.
- Reproduction check fails → no Table 1 row from this run; the pilot's R = 32 numbers go into the text as they are.

**Compute.** About 1 T4-hour in one kernel (the pilot's model load, data and R = 32 measurement took well under an hour).
