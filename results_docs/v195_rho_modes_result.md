# v195 result: the turning is in the backward signal; the mode test is uninformative (the map is strongly non-normal)

**Prereg:** `docs/prereg_v195_rho_modes.md`, frozen and pushed in c1d8de0 before the run. **Run:** Kaggle
aurmerin/astrocyte-v195-rho-modes v1, 2026-10-13/14. All 48 sequences finished:
- Parcae-140m: 16 sequences, 30 s each;
- Parcae-370m: 16 sequences, 76 s each;
- Huginn: 16 sequences, about 160 s each.

**Analysis:** `python v195_analyze.py`, as frozen.

**Bug in the frozen analysis code, fixed after the run.** The D block was guarded by `blend is not None`, so it also ran
for Huginn (flag False) and compared Huginn's predictions with Parcae's blended measurements. That output is void. The
fix (`if blend and ...`) changes no pre-registered verdict: D was scheduled only for Parcae-140m, and only if B passed
there, which it did not.

## Validity: all three models pass every check
- **Reproduction against v191 / v192.** Signal sizes agree within 1.7e-6 (Parcae) and 3e-8 (Huginn). The lost
  gradient from the per-loop contributions agrees within 0.0006, 0.0005 and 0.0008.
- **Run-time checks:**
  - read-out loss = model loss (Parcae);
  - the first Krylov step equals the backward pass (cos 1.000000 in every sequence);
  - Arnoldi orthogonality ≤ 4e-15;
  - sketch norms within 0.33%.

## Pre-registered verdicts

| | Parcae-140m | Parcae-370m | Huginn-3.5B |
|---|---|---|---|
| **A** near-output turning | **SIGNAL** (T_c 0.50, T_p 0.000) | **SIGNAL** (0.58, 0.000) | **SIGNAL** (0.76, 0.000) |
| **A** far band (j = 8–14) | **SIGNAL** (0.034, 0.000) | **SIGNAL** (0.048, 0.000) | **SIGNAL** (0.35, 0.001) |
| **B** one loop's maps reproduce err and ρ | **FAIL** | **FAIL** | **PASS** |
| **C** turning parts carry both effects | not run (B failed) | not run | **UNINFORMATIVE** |
| **D** blended out of sample | not run (B failed) | – | – |

**A, where the turning comes from.** Neighbouring loops map one fixed vector to the same direction: 1 − cos ≤ 0.001.
So the loop-to-loop turning of the contributions comes from the backward signal itself, in both families and in both
bands. That rules out the candidate raised for Huginn, that the map to the weights changes as the state still moves.
Two descriptive qualifications:
- **The signal turns as much as the contributions do.** Neighbouring-loop cosine near the output: signal 0.59 / 0.52 /
  0.16 against contributions 0.50 / 0.43 / 0.24 (140m / 370m / Huginn).
- **Over many loops the map does drift.** A fixed vector's image at the far loops has, at worst, cosine 0.92 (140m),
  0.94 (370m) and 0.48 (Huginn) with the last loop's (median over sequences). It drifts slowly, not from loop to loop.

**B, does one loop's backward map, applied repeatedly, reproduce the dynamics?**

| model | err_k actual (k = 1 / 2 / 4 / 8) | err_k constant-map | largest diff | ρ actual / constant-map |
|---|---|---|---|---|
| Parcae-140m | 0.836 / 0.688 / 0.456 / 0.208 | 0.838 / 0.691 / 0.442 / 0.208 | 0.014 | 0.8250 / 0.8453 (+0.0203) |
| Parcae-370m | 0.844 / 0.686 / 0.462 / 0.243 | 0.855 / 0.701 / 0.494 / 0.244 | 0.032 (k = 4) | 0.8681 / 0.8834 |
| Huginn | 0.909 / 0.815 / 0.642 / 0.383 | 0.916 / 0.818 / 0.630 / 0.370 | 0.013 | 0.8799 / 0.8831 |

- **Parcae-140m fails narrowly,** on ρ: 0.0203 against the 0.02 bar. Its lost gradient matches within 0.014.
- **Parcae-370m fails on the lost gradient,** at k = 4.
- **Huginn passes.**

The constant-map signal follows the actual one closely for the first 8 loops (cos ≥ 0.995) and drifts further back:
- at j = 16, cos 0.99 / 0.96 / 0.92;
- at the first loop (j = 31), which acts on the random initial state, 0.49 / 0.62 / 0.13.

**C (Huginn only).** The modal model reproduces the constant-map model (sizes within 1.7%, err within 0.002). But the
non-normality ratio Σ|b_m| / |λ_out| is 141 (85–203), so the mode amplitudes cancel by two orders of magnitude, and the
guard (≤ 10) returns UNINFORMATIVE. The descriptive dial confirms it: scaling the turning modes by anything other than 1
breaks the cancellation (ρ⁴/err 5.9 / 4.0 / 0.97 / 4.0 / 6.5 at s = 0 / 0.5 / 1 / 1.5 / 2).

## Post hoc (descriptive, outside the pre-registered decisions)
- **C would have been uninformative on Parcae too.** The non-normality ratio is 163 (99–883) for 140m and 148 (67–1532)
  for 370m. On all three models the backward map is strongly non-normal. Its turning is a transient of a single map,
  not a sum of independent fading parts, so a mode-by-mode knock-out cannot answer the question.
- **D on Parcae-140m.** It was not scheduled because B failed by 0.0003, so it is reported only as a description. The
  one-loop model is built from unblended data only (the last loop's maps, the output signal, g_0). It predicts all 20
  blended lost-gradient points (four settings, a = 0.3–0.7, R = 32–64) within 0.030, and within 0.006 at k ≤ 8:
  - R 32, a 0.5: 0.878 / 0.791 / 0.645 against 0.877 / 0.791 / 0.645 measured at k = 1 / 2 / 4;
  - errors grow at k = 12–16 (up to 0.031), where the far loops matter.

  The unblended all-parameter prediction is 0.821 / 0.677 / 0.430 / 0.204, against 0.820 / 0.675 / 0.443 / 0.205 in v190.
- **Turning without bending.** This fits the exploration's finding that contributions turn 10–400 times more than a mix
  of fade rates allows: a non-normal map can turn the signal strongly while barely changing how fast it shrinks.

## What this answers, and what it does not
- **Answered: where the turning comes from.** It comes from the backward signal itself, generated by the loop's own
  backward map, which is nearly the same at every loop and strongly non-normal. Iterating that one map reproduces the
  lost gradient within 0.013–0.032 and both offsetting effects:

  | factors at k = 4 (angle / fit) | actual | constant-map |
  |---|---|---|
  | Parcae-140m | 1.29 / 1.38 | 1.29 / 1.41 |
  | Parcae-370m | 1.31 / 1.42 | 1.26 / 1.44 |
  | Huginn | 1.15 / 1.13 | 1.15 / 1.17 |

  So both effects come from the same linear dynamics, a single map's transient near the output. They are not two
  unrelated features of the model.
- **Not answered: why the two effects are similar in size.** The planned mode decomposition cannot run on a map this
  non-normal. That they match to within about 10% is still a measured fact without a derivation.

## Paper change (per the prereg's mapping)
§3 gets one or two sentences:
- the turning is in the backward signal, not in how each loop maps it onto the weights;
- one loop's backward map applied repeatedly reproduces the lost gradient within 0.013–0.032;
- the map is strongly non-normal, so it cannot be split into independent fading parts, and why the two effects match
  stays open.

D stays out of the paper, because it was not scheduled.
