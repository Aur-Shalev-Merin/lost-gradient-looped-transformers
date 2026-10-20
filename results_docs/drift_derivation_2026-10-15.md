# Why the fitted ρ drifts up with the loop count while the lost gradient stays put

Written 2026-10-15. Checked after the fact on the stored S10 data (v196, `kaggle-output-v196`); the code is
`paper_drift_derivation.py`, which runs on a CPU. An out-of-sample test at 96 and 128 loops is frozen separately
(`docs/prereg_v204_drift_prediction.md`).

## 1. Setup

Let s_j be the size of the backward signal after the loop j steps before the output (j = 0 is the last loop), on one
sequence. ρ_fit(R) is exp of the least-squares slope of log s_j over j = 0, …, R − 1, the paper's definition.

**Fact used.** Once the forward pass has settled near the output, s_j for small j does not depend on R. Adding loops
adds them at the far end, next to the input; it does not change the near-output ones.

## 2. The drift is a fitting-window effect

If s_j does not depend on R, then ρ_fit(R) is the slope of one fixed curve, log s_j, fitted over a window that grows
with R. A curve that is steeper near the output and shallower further back gives a fitted slope that rises with R.
Nothing about the model changes.

**Check.** On each sequence, take the 64-loop profile and fit its first 32 or 48 loops. Compare with the ρ fitted in
the separate 32- and 48-loop runs:

| model | R | ρ fitted in its own run | 64-loop profile, first R loops | median per-sequence difference |
|---|---|---|---|---|
| Parcae-140m | 32 | 0.825 | 0.831 | −0.005 |
| Parcae-140m | 48 | 0.852 | 0.861 | 0.000 |
| Parcae-370m | 32 | 0.868 | 0.861 | +0.001 |
| Parcae-370m | 48 | 0.890 | 0.886 | +0.002 |

The near-output profiles agree across runs: the median absolute difference of the log profiles over the shared loops
is 0.01–0.05. The far end differs more (90th percentile 0.45–1.0), because those loops sit next to the random initial
state, which is at a different distance from the output in each run.

## 3. A closed form: two modes and a hockey stick

**Model.** s_j = A μ_f^j + B μ_s^j, a fast near-output mode and a slow far-fade mode, with μ_f < μ_s < 1. Its log is
close to a hockey stick with:
- slope a_f = log μ_f up to the crossover j* = log(A/B) / log(μ_s/μ_f);
- slope a_s = log μ_s after it.

**Proposition.** For the hockey stick, with j continuous and u = j*/R in [0, 1], the least-squares slope over
[0, R] is

  log ρ_fit(R) = a_s + (a_f − a_s)(3u² − 2u³).

**Proof.** Write x = j/R, uniform on [0, 1]. The curve is a_s·x·R + (a_f − a_s)·R·(x − u)·1[x < u] plus a constant.
The slope per unit of j is a_s + (a_f − a_s)·Cov(x, −(u − x)₊)/Var(x). With E[(u − x)₊] = u²/2 and
E[x(u − x)₊] = u³/6:

  Cov(x, −(u − x)₊) = −(u³/6 − u²/4) = u²/4 − u³/6, and Var(x) = 1/12.

The ratio is 3u² − 2u³. ∎

**Consequences.**
- At u = 1 (window inside the fast part) the fit gives μ_f.
- As R grows, u → 0 and the fit tends to μ_s. For small u it approaches as μ_s (μ_f/μ_s)^{3u²}.
- The overshoot of ρ_fit^k over the lost gradient therefore grows with R and levels off once R ≫ j*.

**Fits.** A two-mode curve was fitted to each sequence's 64-loop profile (log-space least squares; rms 0.10–0.12):

| model | μ_f | μ_s | j* | ρ_fit at R = 32 / 48 / 64: closed form | same, exact slope of fitted curve | measured |
|---|---|---|---|---|---|---|
| Parcae-140m | 0.75 | 0.91 | 17 | 0.830 / 0.866 / 0.876 | 0.833 / 0.861 / 0.871 | 0.825 / 0.852 / 0.871 |
| Parcae-370m | 0.71 | 0.93 | 10 | 0.864 / 0.896 / 0.913 | 0.858 / 0.888 / 0.907 | 0.868 / 0.890 / 0.907 |

These are medians over 16 sequences. The fit uses the same 64-loop data that the R = 32 and 48 values come from, so
this shows the form fits; it does not test it.

μ_f (0.71–0.75) is faster than the rate at which the dropped sum shrinks (0.81–0.84 on Parcae-140m, Appendix J). The
near-output contributions are inflated by turning, so the lost gradient is not simply μ_f^k.

## 4. Why the lost gradient stays put: a bound

Let g and g_k be the full and truncated gradients at R loops, and g′, g′_k at R′ > R. If the contributions of the
R loops nearest the output are the same in both runs, the extra loops add the same vector E to both g′ and to the
dropped part, so:
- |D′_k − D_k| = |E|;
- |g′ − g| = |E|.

Hence

  |lost′(k) − lost(k)| ≤ 2|E| / (|g| − |E|).

If each contribution is at most κ times its loop's signal size (the paper finds 0.9–1.3 per unit of signal, relative
to the last loop), |E| is at most κ times the summed signal beyond loop R, scaled to the contributions. The share of
the summed 64-loop signal lying beyond loop 32 is:

| model | median | max |
|---|---|---|
| Parcae-140m | 0.6% | 3.6% |
| Parcae-370m | 2.0% | 97% (the sequence whose signal grows) |

This is consistent with the observed changes from 32 to 64 loops, of at most 0.013 at k ≤ 8 (Table 2). The bound is
loose by the factor |g| / Σ|c_j|, which depends on how much the contributions cancel. We did not compute it here; the
v195 per-loop data at R = 32 would give it.

## 5. What this adds, and what it does not

**Adds:**
- a quantitative account of S10. S10 confirmed only the direction of the drift. This account says the drift is a
  window effect on a fixed near-output profile, with a closed form in three numbers.
- out-of-sample predictions for 96 and 128 loops (§6).

**Does not add:**
- why the signal has two modes;
- why the contributions turn near the output;
- anything about training cost.

Aicher et al. (2019, Sec. 5.4) describe the far-fade effect qualitatively; the closed form and the window check are
the new part. The far-end mismatch in §2 means the account holds for the near-output part, where the lost gradient
lives.

## 6. Predictions for 96 and 128 loops (frozen in v204)

Median over the same 16 sequences, from the two-mode fits to the 64-loop profiles, with bootstrap 95% intervals:

| model | R = 96 | R = 128 |
|---|---|---|
| Parcae-140m | 0.891 [0.875, 0.904] | 0.900 [0.875, 0.910] |
| Parcae-370m | 0.922 [0.858, 0.951] | 0.924 [0.866, 0.954] |

The lost gradient at k ≤ 8 should stay within 0.03 of its 64-loop value (the S10 bar).
