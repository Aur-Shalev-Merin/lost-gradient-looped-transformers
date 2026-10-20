# v187 result: Huginn's lost gradient carries over across loop counts, its fitted ρ does not

**Plan:** `docs/v187_huginn_rho_transfer_plan.md` (committed 1543e21, before data). **Run:** Kaggle
aurshalevmerin/astrocyte-v187-huginn-rho v1, 2026-10-12. R = 8 on one T4 (52 min), R = 64 on the other (2.6 h); R = 64
fit, so the R = 48 fallback did not run. 32 / 32 sequences at each R. **Analysis:** `python v187_analyze.py` (outputs in
`kaggle-output-v187/`, gitignored like the other Kaggle outputs). R = 16 / 32 are from the 2026-09-30 probe.

| R | ρ | lost gradient k = 1 / 2 / 4 / 8 | ρᵏ (fit over all R loops) | residual at the last loop |
|---|---|---|---|---|
| 8 | 0.85 | 0.88 / 0.75 / 0.51 | 0.85 / 0.73 / 0.53 | 0.27 |
| 16 | 0.86 | 0.90 / 0.80 / 0.61 / 0.32 | 0.86 / 0.74 / 0.54 / 0.29 | 0.073 |
| 32 | 0.88 | 0.90 / 0.80 / 0.62 / 0.36 | 0.88 / 0.77 / 0.59 / 0.35 | 0.017 |
| 64 | 0.91 | 0.91 / 0.81 / 0.63 / 0.39 | 0.91 / 0.82 / 0.68 / 0.46 | 0.014 |

R = 64 minus R = 32 (same 32 sequences, paired bootstrap 95%): k = 1 +0.005 [−0.006, +0.022], k = 2 +0.013 [−0.012,
+0.037], k = 4 +0.010 [−0.022, +0.047], k = 8 +0.030 [−0.018, +0.074].

**Against the plan.** The expectation was: R = 64 within 0.03 of R = 32 at every k, and ρ in 0.85–0.89. The lost
gradient is within 0.03 at k = 1, 2, 4 and at the edge at k = 8 (+0.0304). ρ = 0.907 is outside the range. So by the
plan's own rule this is the second outcome: the claim "ρ carries over to other loop counts" is narrowed, and the shift is
reported.

**Why ρ moves (exploratory, after the fact).** The backward signal is not quite geometric over 64 loops. It shrinks by
about 0.88 per loop over the 32 loops nearest the output and about 0.92 over the 32 farthest from it (at R = 32: 0.86
nearest half, 0.88 farthest half). Truncation at small k drops mostly loops just beyond k, so the lost gradient barely
moves, while one fit over all 64 loops picks up the slower far decay and overstates the loss at k = 4 and 8 (0.68 and 0.46
against 0.63 and 0.39).

**R = 8.** The plan said to compare k = 4 against the aligned finite-loop form, about 0.37. With the measured per-sequence
ρ that form gives 0.34, and the orthogonal form 0.47. Measured: 0.51, above both, and at k = 1 / 2 too (0.88 / 0.75
against at most 0.84 / 0.70). Plain ρᵏ (0.85 / 0.73 / 0.53) is closer, which the finite-loop argument does not explain.
Huginn at 8 loops is far from settled (residual 0.27 against 0.014 at 64) and outside its training range, so we report
it without an explanation.

**Follow-up (same day, after a review note).** The two forms are not a permitted range. They assume one shared
angle between all contributions. Without cancellation, the proven range (`docs/law1_derivation_formal.md`, Prop. 3b) is
0.17–0.91 here. Huginn sits above both forms at R = 8, 16 and 32 (orthogonal form +0.03–0.04 at R = 32, +0.04–0.07 at R = 16,
+0.04–0.05 at R = 8), so R = 8 is not special. [Corrected 2026-10-13, outside review M3: at R = 64 it falls below
both forms for k ≥ 2.] The paper's §3 had read "below both forms" (blended SmolLM2) as implying
cancellation. Uneven angles can do it too: kept loops parallel and dropped loops mutually orthogonal give 0.225 against
the measured 0.23. Draft §2, §3, the Table 1 footnote and Appendix A were reworded to treat both directions alike; no
numbers or conclusions changed.

**Paper change (draft §2, "Why ρ and not just the lost gradient?").** It now says the lost gradient at small k carries
over (Huginn 16 / 32 / 64, Parcae 8 / 16), and the fitted ρ drifts (0.86 / 0.88 / 0.91). ρ is best fitted near the loop
count where it will be used. R = 8 is reported with the finite-loop mismatch. The Table 1 footnote points to these loop
counts.
