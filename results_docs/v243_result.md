# v243 RESULT: SIZE-UNCLEAR (S25): k = 1 clean-up of S24 under SGD

**Provenance.**
- **Prereg:** `docs/prereg_v243_sgd_k1_cleanup.md`. Frozen in b690e13 and pushed as freeze-only commit 3820270 (branch
  `v243-freeze`, parent v241's freeze c9d8c82) at 2026-11-13 20:13:07 UTC, before any run.
- **Kernels:** the run was split into two halves that ran at different times.
  - aurshalevmerin/astrocyte-v243-1 and -2 were pushed at 20:13:41 UTC on 2026-11-13 and finished by 02:42 UTC on
    2026-11-14.
  - aurmerin/astrocyte-v243-3 and -4 were blocked by that account's weekly GPU quota. They were pushed unchanged from
    the same frozen build at 02:44 UTC on 2026-11-14, once the quota had reset.
  - No result was looked at before the second half was pushed. The first half's rows were checked only for integrity
    (hashes, selftests, row counts).
- **Code:** every lane (8 of 8) passed v241's selftest on the real model, with combined driver sha 3b446a33effc on every
  row.
  - Each kernel's copy of `v241_sgd_split.py` hashes to the frozen e505057ac886 (LF).
  - Each kernel's copy of `v243_sgd_k1_cleanup.py` hashes to the frozen 063ad397fe61.
- **Analysis:** `OUT_DIR=results_v243 python v243_sgd_k1_cleanup.py --analyze results_v243`. The output is reproduced
  below and is also in `results_v243/v243_analysis.txt`.
  - The pooled and clean-seed figures come from a scratch script over the same rows. They are not scored and add no
    rule.

## Scored result

All 60 runs completed (12 seeds × 5 cells).

**Valid seeds:** 8 of 12, exactly the minimum (N_MIN = 8). So all 8 were scored, not 10. Four seeds were invalid
under S23's rule:

| seed | invalid run | non-finite steps | test loss |
|---|---|---|---|
| 2431102 | full | 583 | NaN |
| 2431103 | full | 19 | 5.78 |
| 2431103 | shrink1 | 525 | NaN |
| 2431104 | turn1 | 546 | NaN |
| 2431105 | full | 14 | 1.469 |

**Gate:** the floor gap was 0.219 nats (90% lower bound 0.204), so the gate passed.

| arm | cost φ (mean ± s.e., 8 seeds) | per seed | non-finite steps | Mbar | step vs full | clipped |
|---|---|---|---|---|---|---|
| trunc1 | 0.199 ± 0.054 | 0.457 0.147 0.197 0.229 0.200 −0.093 0.267 0.187 | 4 1 1 2 1 0 1 0 | 0.786 | 0.686 | 0.02 |
| turn1 | 0.186 ± 0.040 | 0.200 0.196 0.202 0.213 0.176 −0.052 0.193 0.362 | 1 1 1 1 0 1 0 4 | 0.805 | 1.000 | 0.05 |
| shrink1 | −0.020 ± 0.039 | 0.024 −0.024 0.081 0.013 −0.050 −0.277 0.032 0.043 | 1 1 2 1 1 0 2 0 | 0.996 | 0.694 | 0.01 |

Scored seeds, in order: 2431101, 2431106, 2431107, 2431108, 2431109, 2431110, 2431111, 2431112.

**Primary:** D = trunc1 − turn1 = **0.013 ± 0.043**. The one-sided 90% bounds are −0.049 / 0.074. Per seed: 0.257
−0.049 −0.005 0.016 0.024 −0.041 0.074 −0.175. The mean is below 0.015 and the lower bound is below 0, so the size is not
shown to cost. The upper bound is above 0.015, so it is not shown to be free either.

**Verdict: SIZE-UNCLEAR.**

The full length recovered 0.06 of truncation's cost (D / φ_trunc1).

## Secondary (labelled, not deciding)

- **v239 rule at k = 1:** truncation 0.199, turn 0.186, shrink −0.020, margin 0.040. The turn share was 0.94 and the
  shrink share −0.10, so the label is **TURN**. (S24 gave PARTIAL; its turn arm missed the bar by 0.005.)
- **Shrink-only cost:** −0.020 ± 0.039 (bounds −0.076 / 0.036), so **UNCLEAR**.

## Reported, not scored

**Why the spread is wide.** The per-seed spread of D was 0.12, against 0.039 in S24, and the operating
characteristics in the prereg assumed S24's spread. Most of the extra spread comes from three runs that stayed inside
the ≤10 non-finite-step limit but were visibly hurt by spikes:
- **seed 2431110, full** (7 non-finite steps, test 1.481 against 1.414–1.428 for the other valid full runs). Its
  inflated baseline makes every cost on that seed negative: trunc −0.093, turn −0.052, shrink −0.277.
- **seed 2431101, trunc1** (4 non-finite steps, test 1.527). Its cost was 0.457 and its D 0.257.
- **seed 2431112, turn1** (4 non-finite steps, test 1.499). Its cost was 0.362 and its D −0.175.

**Clean seeds (after the fact, unscored).** These are the five seeds where every run had at most 2 non-finite steps
(2431106–09 and 2431111):

| quantity | mean ± s.e. | one-sided 90% bounds |
|---|---|---|
| D | 0.012 ± 0.020 | −0.019 / 0.043 |
| trunc1 | 0.208 | |
| turn1 | 0.196 | |
| shrink1 | 0.010 | |

That is still UNCLEAR by the same rule, and the mean D matches the scored 0.013. The spike-hurt runs widen the
interval but do not move the centre.

**Pooled with S24's 5 scored seeds (13 seeds, after the fact).** The prereg said 15 seeds; only 8 were valid here.

| quantity | mean ± s.e. | one-sided 90% bounds | median |
|---|---|---|---|
| D | 0.030 ± 0.028 | −0.008 / 0.067 | 0.024 |
| trunc1 | 0.221 ± 0.033 | | |
| turn1 | 0.192 ± 0.024 | | |
| shrink1 | 0.003 ± 0.029 | | |

**Comparison with S24.** S24's after-the-fact D was 0.057 ± 0.018. On new seeds it came out at 0.013 ± 0.043 (median
0.006). This is the shrinkage the prereg warned of, and more. S24's lead did not replicate, but the test cannot rule out
an effect of about S24's size either: the upper bound is 0.074.

**Stability** (all 60 runs; unstable means more than 10 non-finite steps or a non-finite test loss):

| cell | unstable runs |
|---|---|
| full backprop | 3 of 12 (one diverged, two with 14 and 19 non-finite steps) |
| floor | 0 of 12 |
| trunc1 | 0 of 12 |
| turn1 | 1 of 12 (diverged) |
| shrink1 | 1 of 12 (diverged) |

- Together with S24, full backprop at learning rate 0.1 was unstable in 4 of 18 runs, against 2 of 72 two-gradient runs
  (S24's 36 and these 36).
- One more full run (seed 2431110, 7 steps) was valid but visibly hurt.
- Learning rate 0.1 is near the edge of stability for full backprop under SGD with clipping at norm 5, as S24's pilot
  suggested: 0.3 diverged there.

## Predictions (from the prereg, §4)

| prediction | stated | outcome |
|---|---|---|
| Verdict | SIZE-COSTS 0.55, **SIZE-UNCLEAR 0.27**, SIZE-FREE 0.08, INVALID/INCOMPLETE 0.10 | **SIZE-UNCLEAR**, with exactly the minimum number of valid seeds (one short of INCOMPLETE) |
| Mean D | about 0.04 | 0.013, lower than predicted |
| v239 rule | PARTIAL 0.55, **TURN 0.40** | **TURN** |
| Shrink-only | **UNCLEAR 0.6** | **UNCLEAR** |
| At least one unstable full run among 12 | 0.6 | **held** (3 of 12) |

## What it means

On 8 new seeds the pre-registered test of S24's lead was inconclusive. Giving the truncated direction its full length
recovered 0.013 ± 0.043 of the loop's benefit at k = 1 (bounds −0.049 / 0.074): far below S24's after-the-fact 0.057,
but too noisy to call the size free. The turn arm alone again carried nearly all of truncation's cost (0.186 of 0.199),
and the shrink arm alone again cost nothing measurable (−0.020 ± 0.039).

Taken together, S24 and S25 say that under SGD the direction carries most or all of truncation's cost at k = 1. Any cost
of the lost size is too small to detect against the noise from spikes at this learning rate. The study's design limit
stands: under SGD, "the size costs" and "the shared learning rate is too small for the truncated direction" would be the
same statement, and no per-arm learning-rate sweep was run.

## Paper changes (prereg §2, SIZE-UNCLEAR row)

- **Abstract and §1:** unchanged in substance. Whether size costs under SGD is unclear.
- **§5:** one sentence. A pre-registered test on new seeds was inconclusive, with D and its bounds. Only 8 seeds were
  valid, not the 10 the mapping's wording assumed.
- **In every case:**
  - App. G gains S25 (design, results, secondaries, stability, and the pooled figures labelled after the fact);
  - Table 4 gains an S25 row;
  - §1 counts twenty-five studies;
  - §5's SGD paragraph is shortened to its result, with the step-matched controls' caveats moved to App. G (already
    there in full).
