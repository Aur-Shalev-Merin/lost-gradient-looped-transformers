# v192 result: on Huginn too the shortfall comes from angles, not from contribution sizes

**Plan:** `docs/v192_huginn_contrib_plan.md` (frozen c24a147; amendment c722d7d made before launch and before any data:
direct per-loop vector-Jacobian products instead of bf16 subtraction). **Run:** Kaggle
aurmerin/astrocyte-v192-huginn-contrib v1, 2026-10-13, 16 sequences (probe WikiText 0–7 on GPU 0, GSM8K 16–23 on GPU 1),
R = 32, bf16, adapter + core_block[0]. About 15 minutes per sequence. **Analysis:** `python v191_analyze.py
kaggle-output-v192` (as frozen, extended to read v192 files before data).

**Checks (all pass).**
- The lost gradient reproduces the Sep-30 probe on the same sequences: largest per-sequence difference 0.0049 over
  k = 1, 2, 4, 8, 16. The bar was 0.01.
- The lost gradients from the direct per-loop contributions agree with the probe's own truncated passes, which are
  the subtraction values, to within 0.0003 at k = 1, 2, 4, 8 and 16 in every sequence.
- Sketch against exact: norms within 0.23%, neighbouring cosines within 0.0023, the lost gradient rebuilt from the Gram
  within 0.0008. The bars were 1%, 0.01 and 0.01.
- Recorded, no bar: the sum of the per-loop contributions differs from one bf16 full backward by 0.75–0.89% of its
  norm. That is the accumulation noise that the subtraction method would have carried, roughly twice the reviewer's
  estimate. It confirms that the switch to direct contributions was needed.

**Q1: ANGLES**, at all four k. That is the same verdict as all four Parcae cells of v191, so per the plan §3 states
the cause as measured on two model families.

| k | err_loop | P0 (signal sizes) | P1 (measured sizes) | gap closed by sizes |
|---|---|---|---|---|
| 1 | 0.908 | 0.808–0.832 | 0.850–0.853 | 27% |
| 2 | 0.815 | 0.677–0.705 | 0.720–0.729 | 22% |
| 4 | 0.642 | 0.482–0.516 | 0.519–0.545 | 23% |
| 8 | 0.383 | 0.264–0.304 | 0.294–0.318 | 18% |

Huginn's subset holds loop parameters only, so err_loop is the paper's err (g_0 = 0).

**Sizes nearly follow the signal.** Gain |cⱼ| / sⱼ, relative to the last loop, rises from 1.00 at j = 0 to 1.14 at
j = 1, and stays at 1.18–1.26 from j = 2 on, including the first loop (j = 31, 1.22). There is no first-loop spike as on
Parcae (1.8–1.9).

**Angles: the pattern differs from Parcae.**
- Neighbouring contributions (exact cosine): 0.24 near the output (j = 0–3), 0.47 at j = 3–7, 0.64 at j = 7–15,
  0.66 further back. As on Parcae, directions change fastest near the output, but far back they line up much less
  (Parcae 0.86–0.96).
- Mean pairwise cosine: 0.156 among the kept loops (j < 4) and 0.148 among the dropped ones, against Parcae's
  0.27–0.32 and 0.53–0.66. Kept vs dropped: 0.027 (Parcae 0.09–0.17). Only 7 of 16 sequences have kept loops less
  aligned than dropped (Parcae 13–14 of 16).
- At k = 4 the kept sum reaches 0.61 of the sum of its members' norms and the dropped sum 0.47 (Parcae 0.67–0.70 and
  0.76–0.82). So on Huginn the dropped loops are *less* coherent than the kept ones.
- The cosine between the kept and the dropped sums is 0.16 (Parcae 0.24–0.35).
- Opposing pairs: 22% of pairs have negative cosines (minimum −0.69), against Parcae's 5–14%. The mean stays positive.

So "the dropped loops line up more than the kept ones" is a Parcae finding, not a shared one. What the two families
share is set out in the identity below.

**Post hoc, both families: what makes the lost gradient exceed the size-based prediction.** Descriptive, computed after
v191 and v192 were seen, from the stored quantities; nothing was refitted. Define the coherence of a set of
contributions as |Σ cⱼ| / Σ |cⱼ| over that set. Then exactly, for every sequence and k,

  err_loop = A1 × (coherence of the dropped sum) / (coherence of the whole sum),

where A1 = Σ_{j≥k} |cⱼ| / Σⱼ |cⱼ| is the aligned form on the measured sizes. The identity was checked to 1e-9 on every
sequence. So the lost gradient exceeds the aligned form exactly when the dropped loops add up more coherently than all
the loops together. That can happen in two ways: the dropped loops line up better than the kept ones (Parcae), or the
kept and dropped groups point in different directions (both families).

| cell | coherence of whole | dropped / whole at k = 1 / 2 / 4 / 8 | cos(kept sum, dropped sum) at k = 1 / 2 / 4 / 8 |
|---|---|---|---|
| Huginn R 32 | 0.41 | 1.05 / 1.09 / 1.15 / 1.20 | 0.09 / 0.12 / 0.16 / 0.17 |
| Parcae-140m R 32 | 0.62 | 1.11 / 1.18 / 1.29 / 1.41 | 0.20 / 0.25 / 0.35 / 0.42 |
| Parcae-370m R 32 | 0.58 | 1.11 / 1.18 / 1.31 / 1.47 | 0.13 / 0.19 / 0.24 / 0.24 |
| Parcae-140m R 16 | 0.61 | 1.11 / 1.19 / 1.29 / 1.36 | 0.18 / 0.24 / 0.31 / 0.33 |
| Parcae-370m R 16 | 0.59 | 1.11 / 1.19 / 1.31 / 1.45 | 0.14 / 0.19 / 0.26 / 0.25 |

**Post hoc: why ρᵏ lands close anyway.** This is an accounting over the loop's part of the gradient, using medians
over sequences, with ρ from the paper's per-sequence straight-line fit over all loops. Let A0 be the aligned form on
the backward-signal profile, what signal sizes and parallel pushes would give. Then:
- "sizes" = A1 / A0, the effect of the measured contribution sizes;
- "angles" = err_loop / A1, the coherence ratio above;
- "fit" = ρᵏ / A0, how far the single fitted ρ overstates the signal-size form, because the fit follows the slower far
  fade.

| cell | k = 1 / 2 / 4 / 8: sizes | angles | lost / A0 | fit = ρᵏ / A0 | ρᵏ / lost |
|---|---|---|---|---|---|
| Huginn R 32 (ρ 0.880) | 1.03 / 1.03 / 1.04 / 1.05 | 1.05 / 1.09 / 1.15 / 1.20 | 1.08 / 1.13 / 1.19 / 1.24 | 1.05 / 1.08 / 1.13 / 1.21 | 0.97 / 0.95 / 0.93 / 0.94 |
| Parcae-140m R 32 (0.825) | 1.02 / 1.01 / 0.95 / 0.91 | 1.11 / 1.18 / 1.29 / 1.41 | 1.12 / 1.20 / 1.20 / 1.27 | 1.14 / 1.23 / 1.38 / 1.56 | 0.99 / 0.99 / 1.02 / 1.03 |
| Parcae-370m R 32 (0.868) | 1.01 / 1.00 / 0.97 / 0.98 | 1.11 / 1.18 / 1.31 / 1.47 | 1.12 / 1.17 / 1.25 / 1.45 | 1.14 / 1.24 / 1.42 / 1.77 | 1.03 / 1.10 / 1.23 / 1.33 |
| Parcae-140m R 16 (0.802) | 1.04 / 1.05 / 1.00 / 1.04 | 1.11 / 1.19 / 1.29 / 1.36 | 1.13 / 1.23 / 1.28 / 1.47 | 1.11 / 1.17 / 1.26 / 1.40 | 0.97 / 0.95 / 0.96 / 0.93 |
| Parcae-370m R 16 (0.810) | 1.02 / 1.01 / 0.97 / 1.03 | 1.11 / 1.19 / 1.31 / 1.45 | 1.12 / 1.17 / 1.26 / 1.45 | 1.10 / 1.16 / 1.27 / 1.38 | 0.97 / 0.96 / 0.95 / 0.87 |

(Medians of ratios, so the columns do not multiply exactly.)

So ρᵏ's agreement rests on two effects of similar size that offset each other:
- angles raise the lost gradient above the signal-size form by 1.05–1.47;
- the single fitted ρ raises ρᵏ above the same form by 1.05–1.77.

On four of the five cells they match to within 13% (ρᵏ / lost 0.87–1.03). On Parcae-370m at 32 loops the fit's factor
is larger, and ρᵏ overstates the loss by up to 33% over the loop part. That is the miss of v189 / v190. v191's Q2
locates it: 370m's farthest loops keep more signal (0.009 against 0.003 of the last loop's), which steepens the fit
without changing the lost gradient.

**What drives the angle factor in both families (post hoc).** The kept-vs-dropped angle does. If the kept and dropped
sums were parallel, keeping their sizes and coherences, the coherence ratio would be 0.85 / 0.84 / 0.89 / 0.99 on
Huginn at k = 1 / 2 / 4 / 8, and 0.87–0.89 / 0.94–0.96 / 1.07–1.09 / 1.26–1.28 on Parcae. The actual ratios are
1.05–1.20 and 1.11–1.47. So at k ≤ 4 nearly all of the excess comes from the two groups pointing different ways. The
dropped loops being more coherent than the kept ones (Parcae only) adds a little, and more at k = 8. At matched
separation (loops 1–3 apart), the turning is shared:
- Huginn: cosine 0.16 near the output (j < 4), 0.34 at j = 4–11, 0.43 in the far half;
- Parcae: 0.27–0.32, 0.72–0.83 and 0.77–0.92.

Directions turn fastest near the output in both families. Huginn's far loops drift over longer ranges, so as a group
they are not coherent.

**Why are the two factors similar? Open.**
- **Parcae's account does not transfer.** v191's account was that the coherent far tail fades at the slow rate, which
  the fit tracks. It does not fit Huginn, whose dropped loops are less coherent as a group than its kept ones (review
  note, 2026-10-13, correct).
- **A candidate common cause also gets no support.** The backward signal may mix fast- and slow-fading parts, with the
  fast parts both steepening the near fade and turning the kept loops away from the dropped ones. We checked it after
  the fact (`link_check.py` in the session scratchpad; reading rule fixed before running: LINK-CONSISTENT only if
  r(fit, angle) ≥ 0.5 and r(near-output turning, fit) ≥ 0.3 in both families, otherwise UNRESOLVED, with low power
  stated). The result is **UNRESOLVED**:

  | k = 4, Spearman across 16 sequences | Parcae R 32 (140m, 370m) | Parcae R 16 (140m, 370m) | Parcae pooled | Huginn | Huginn within-source |
  |---|---|---|---|---|---|
  | r(fit factor, angle factor) | +0.78, +0.57 | +0.00, +0.33 | +0.47 | −0.14 | +0.04 |
  | r(near-output turning, fit factor) | +0.05, −0.26 | −0.01, +0.12 | −0.02 | −0.27 | +0.16 |

  The pattern at k = 2 and 8 is similar.
- **What the data allow.** The two factors go together across sequences on Parcae at 32 loops, hardly at 16, and not on
  Huginn. Near-output turning does not track the fit factor anywhere.

The fast/slow-parts account gets no support from these data, and the match on Huginn may be a coincidence of two
separate effects. With 16 sequences only a strong association could show, so this does not rule out a link either. A
direct test would need the loop Jacobian's slow and fast directions, and the backward signal projected onto them, per
loop. That is a GPU measurement we have not made.

On Huginn the far contributions also line up less than on Parcae (neighbouring cosine 0.66 against 0.86–0.96). One
possible reason, unmeasured: Huginn's state is still moving there (forward step 0.03–0.08 at j = 8–16, up to 1.8 at the
first loop), so each loop's parameter Jacobian differs. Parcae's state is settled within 1–2 loops.

**Q3 (descriptive).** Across loops, gain and forward step both rise with distance from the output: the median Spearman
correlation is 0.64, against 0.03–0.18 on Parcae. This shared trend is confounded with loop position. v193's frozen
test, which compares profile slopes across sequences, is the controlled version (`docs/v193_work_profile_result.md`).

**Other.** The per-sequence lost gradient varies with the text, as in the Sep-30 probe: WikiText 0.48–0.68 and GSM8K
0.62–0.82 at k = 4.

**Paper change (draft v13).**
- **§3** gets a new paragraph, "Why ρᵏ lands close anyway: two effects that offset", measured on two families: sizes
  follow the signal; directions raise the lost gradient, through the identity and mainly the kept-vs-dropped angle; a
  single fitted ρ raises ρᵏ by a similar factor; the factors agree within 13% on 4 of 5 pairs; the 370m miss is
  explained. It states plainly that why the two factors are similar is not known, and that the match may be partly
  coincidence. That uses the link check's numbers. No common-cause mechanism is claimed.
- **§2** gets one descriptive sentence on 370m (v191 Q2). Its Huginn sentence now points to the measurement instead of
  "not tested".
- **Appendix A**'s uneven-angles bullet gets the identity.
- **Limitations** says the account rests on three frozen models and that the reason for the match is unknown.

These are still diagnostics on frozen models, with no new prediction rule.
