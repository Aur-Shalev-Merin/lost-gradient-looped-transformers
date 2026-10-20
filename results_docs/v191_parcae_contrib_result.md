# v191 result: on Parcae the shortfall comes from angles, not from contribution sizes

**Plan:** `docs/v191_parcae_contrib_plan.md` (frozen and pushed 0d47047, before data). **Run:** Kaggle
aurshalevmerin/astrocyte-v191-parcae-contrib v1, 2026-10-13. **Analysis:** `python v191_analyze.py kaggle-output-v191`
(as committed with the plan). Outputs were downloaded after the v193 prereg freeze (c6e7a52) and opened only after its
amendment A1 (76fed8a).

**Checks (all pass).**
- The lost gradient reproduces v190 / v189 exactly: max difference 0.0000 in every cell.
- The sketch matches the exact quantities in every sequence: norms within 0.33%, neighbouring cosines within 0.0026, the
  lost gradient rebuilt from the Gram within 0.0012. Bars were 1%, 0.01 and 0.01.

**Q1: ANGLES in all four cells.** The lost gradient over the loop (err_loop) is above the measured-size forms (P1) at
every k, by a wide margin. Using the measured contribution sizes instead of the backward signal closes only part of the
gap: 3–26% at k = 1–4 and 6–35% at k = 8.

| cell | err_loop k = 1 / 2 / 4 / 8 | P0 (signal sizes) | P1 (measured sizes) |
|---|---|---|---|
| 140m R 32 | 0.84 / 0.69 / 0.46 / 0.21 | ≤ 0.72 / 0.55 / 0.34 / 0.16 | ≤ 0.75 / 0.56 / 0.34 / 0.16 |
| 370m R 32 | 0.84 / 0.69 / 0.46 / 0.24 | ≤ 0.73 / 0.56 / 0.34 / 0.17 | ≤ 0.75 / 0.58 / 0.37 / 0.20 |
| 140m R 16 | 0.83 / 0.68 / 0.43 / 0.19 | ≤ 0.71 / 0.53 / 0.32 / 0.12 | ≤ 0.74 / 0.56 / 0.32 / 0.13 |
| 370m R 16 | 0.84 / 0.68 / 0.45 / 0.21 | ≤ 0.71 / 0.53 / 0.31 / 0.12 | ≤ 0.74 / 0.56 / 0.34 / 0.15 |

**Sizes nearly follow the signal.** Gain is |cⱼ| / sⱼ relative to the last loop. It is 1.0, 1.1, 1.1–1.15, 1.0–1.1 and
0.9–1.0 at j = 0, 1, 2, 4 and 8–16. The exception is the very first loop (j = R − 1) at 1.8–1.9. That loop acts on the
random initial state; its forward step is 70–200 times the state size.

**Angles do not.**
- **Neighbouring contributions** (cosine, exact): 0.41–0.50 near the output (j = 0–3), 0.75–0.86 at j = 3–7, and
  0.86–0.96 further back.
- **Within the kept loops** (j < 4), the mean pairwise cosine is 0.27–0.32. Among the dropped loops (j ≥ 4) it is
  0.53–0.66. Between kept and dropped it is 0.09–0.17. 13–14 of 16 sequences per cell show kept loops less aligned
  than dropped.
- **At k = 4:** the kept sum reaches 0.67–0.70 of the sum of its members' norms, the dropped sum 0.76–0.82, and the
  cosine between the two sums is 0.24–0.35.
- **Cancellation is minor:** 5–14% of pairs have negative cosines (minimum −0.40 to −0.89), but the means are clearly
  positive.

So truncation keeps a set of loops whose contributions point in different directions, and drops a set whose
contributions line up. It therefore loses more than the signal profile predicts. This is the "uneven angles" case of
Appendix A, now measured.

**Likely mechanism (inferred, not measured).** Parcae's state is settled: the forward step is 0.1–1% of the state size
for every loop after the first one or two. So the parameter Jacobian is nearly the same at every loop, and contribution
directions are set by the backward signal's direction. Near the output that signal is a mixture of fast-fading and
slow-fading parts, and its direction changes from loop to loop. Far back only the slowest part is left, so successive
contributions are nearly parallel (cosine 0.94–0.96). This matches the mode picture of the derivation (Prop. 4,
Remark 1). Confirming it would need an eigen-decomposition of the loop Jacobian.

It also explains why a single fitted ρ does reasonably well. The lost gradient is dominated by the coherent far tail,
which fades at the slow rate. A log-linear fit over all loops is pulled toward that slow rate rather than the faster
fade near the output.

**[Update after v192, 2026-10-13.]** The two paragraphs above are Parcae-specific and do not transfer to Huginn.
Huginn also gives ANGLES, but its dropped loops are *less* coherent as a group than its kept ones (0.47 against 0.61
at k = 4). Across both families, most of the excess comes from the kept and the dropped sums pointing different ways
(at k <= 4 the coherence ratio would be 0.84-1.09 if they were parallel). The single fit's overshoot offsets that by a
similar factor on 4 of 5 cells, but an after-the-fact check finds no across-sequence link on Huginn. Why the two
factors are similar is open. See `docs/v192_huginn_contrib_result.md`.

**Q2: Parcae-370m vs 140m at R = 32 (descriptive).**
- The lost gradient over the loop is nearly the same: 0.84 / 0.69 / 0.46 / 0.24 for 370m, 0.84 / 0.69 / 0.46 / 0.21 for
  140m. Near-output angles and gains are also alike.
- 370m's far loops keep more signal: 0.009 of the last loop's at j = 31, against 0.003. Their contributions are larger
  too: |c_31| is 0.016 of |c_0| against 0.005.
- 370m has more opposing pairs: 14% against 8%.
- The far loops pull the all-loop fit up (ρ 0.87 against 0.83) but carry little of the gradient at k ≤ 8. So ρᵏ
  overshoots on 370m without its lost gradient differing from 140m's. The miss is a property of the single fit, driven
  by the far loops, not of what truncation loses.

**Q3:** gain does not track the forward step across loops. The Spearman correlation is 0.03–0.18, median over
sequences. The first loop is the only one with both a large step and a large gain.

**Other.** The gradient of the layers after the loop is 20–23% of the full gradient's norm (|g_0| / |g_R|).

**Paper change.** Held until v192 (Huginn, direct per-loop VJPs) lands. The plan says §3 states the cause as measured on
two families only if Huginn gives the same verdict.
