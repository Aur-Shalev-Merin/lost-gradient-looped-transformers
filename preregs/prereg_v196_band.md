# PREREG v196: band-of-R test of the ρᵏ account, and a head-to-head of lost-gradient estimators (frozen before any v196 data)

**Status.** Pre-registration, requested by the user ("1", 2026-10-14) after
`docs/rho_match_first_principles_2026-10-14.md`. Frozen by the commit that adds this file, together with:
- `v196_band.py` (measurement);
- `v196_analyze.py` (analysis);
- `build_kaggle_v196.py` (kernel).

Pushed to origin before the kernel is launched.

## What had been seen at freeze time

**Seen:**
- **Earlier runs:** everything through v195, and the post hoc first-principles analyses: tail sums, lag profiles,
  counterfactuals, and the one-loop model's sweep over R.
- **Measured ρᵏ / lost ratios** at other loop counts:
  - Huginn: R 8, 16, 32, 64;
  - Parcae: R 16, 32.
- **The R = 32 scores of the three estimators below** (calibration), as mean absolute error over k = 1, 2, 4, 8:

  | | rule | share | law |
  |---|---|---|---|
  | Parcae-140m | 0.008 | 0.030 | 0.017 |
  | Parcae-370m | 0.069 | 0.011 | 0.013 |
  | Huginn | 0.034 | 0.013 | 0.016 |

**Not seen:** any lost gradient or signal profile at R = 48 or 64 for Parcae without blending, or at R = 48 for Huginn.

**Two quantitative prediction methods were tried and rejected before freezing,** so no number below comes from them:
- **The one-loop model iterated past 32 steps diverges** on 3 of 16 Parcae-140m and 4 of 16 Parcae-370m sequences. The
  last loop's backward map has slightly growing directions that the real far loops do not show.
- **Extrapolating the far fade at its R = 32 rate fails calibration.** It misses the measured ρ by 0.025–0.06 for Huginn
  32 → 64 and for Parcae 16 → 32.

So only the account's qualitative predictions (P1–P3) are tested, plus the estimator head-to-head.

## Questions

1. **The account.** Is the lost gradient set by the steadily shrinking tail sum, while the fitted ρ follows the far
   loops? If so, as R grows past the measured range:
   - the lost gradient barely changes;
   - the fitted ρ rises;
   - ρᵏ overshoots more and more.
2. **A better estimator?** Two estimators come from the same single backward pass that measures ρ:
   - the state-space tail share |T_k| / |T_0|;
   - the one-number tail rate r_tailᵏ.

   Do they predict the lost gradient better than ρᵏ at loop counts never measured?
3. **What makes the signal turn** (descriptive): redistribution across token positions, or turning within positions?

## Measurements (`v196_band.py`)

**Settings.** Frozen models, no training.
- Parcae-140m and Parcae-370m at R = 32, 48 and 64: the Table 1 protocol (16 GSM8K validation chunks, seed 777 + i, fp32,
  all parameters, a = 1; per-loop checkpointing, which gives the same values).
- Huginn-3.5B at R = 48: the 16 probe sequences of v192 / v195, bf16, adapter + core_block[0].

**Per sequence:**
- **The backward signal at every loop** (exact float64 Gram, giving the tail sums).
- **Per-token statistics:** the neighbouring-loop cosine factored exactly as BC × W. BC is the overlap of the two loops'
  position profiles; W is the norm-weighted within-position cosine. Also the position centre of mass, the participation
  ratio and the effective rank.
- **Lost gradients** at k = 1, 2, 4, 8, 16: over all parameters (the paper's err) and over the loop part (err_loop,
  using g_0). Huginn uses the probe's own truncated passes.

**TINY tests, all passed:**

| check | result |
|---|---|
| lost gradients vs v164's frozen measure() on the same tiny model | within 4e-6 |
| ρ | identical |
| BC × W vs the plain cosine | 1e-16 |
| tail share from the Gram vs direct vector sums | exact |
| analysis code, Parcae and Huginn branches | runs end to end |

## Decisions (`v196_analyze.py`)

**Statistic.** As in the paper: medians over sequences; the rule value is the median of ρ_iᵏ.

**Validity.**
- Parcae at R = 32 reproduces v190 within 0.002 in err at k = 1, 2, 4, 8, and within 0.001 in ρ.
- BC × W = cos within 1e-6 everywhere.

**Account (per Parcae model).**

| check | rule |
|---|---|
| P1 | \|median err_k(R) − median err_k(32)\| ≤ 0.03 at k = 1, 2, 4, 8, for R = 48 and 64 |
| P2 | median ρ(32) < median ρ(48) < median ρ(64) |
| P3 | q_k(R) = median ρ_iᵏ / median err_k rises, q(32) < q(48) < q(64), at k = 4 and 8; and q_4(64) ≥ 1.10 |

A model passes if P1, P2 and P3 all hold.

| verdict | condition |
|---|---|
| CONFIRMED | both models pass |
| PARTIAL | one model passes |
| REFUTED | P3 fails on both |
| MIXED | otherwise |

Huginn at R = 48 is compared with its measured R = 32 (v195) and R = 64 (v187) on the same sequences (P1′–P3′). This is
supportive only and carries no verdict weight.

**Head-to-head.** Five new cells: Parcae-140m R 48 and 64, Parcae-370m R 48 and 64, Huginn R 48.
- **Target:** the loop's lost share, err_loop.
- **Predictors:**
  - rule = ρ_iᵏ;
  - share = |T_k| / |T_0|;
  - law = r_iᵏ, where r_i is the straight-line rate of log(|T_k| / |T_0|) over k = 0 … 8.
- **Score:** mean over k = 1, 2, 4, 8 of |median predictor − median target|.

| check | rule |
|---|---|
| H1 | share score ≤ 0.04 in all 5 cells |
| H2 | law score < rule score in ≥ 4 of 5 cells |
| H3 | law score ≤ 0.04 in all 5 cells |

TAIL LAW BETTER if H2 and H3 hold; TAIL SHARE ACCURATE if H1 holds.

**Per-token reading** (descriptive, rule fixed). Take the near-output pairs j = 0, 1, 2 and compute
s = log BC / log(BC × W), the share of the log-cosine due to redistribution across positions. Using the median of s:
- TRANSPORT if s ≥ 2/3;
- WITHIN-POSITION if s ≤ 1/3;
- MIXED otherwise.

This is read per model: Parcae at R = 32, Huginn at R = 48.

## What each outcome changes in the paper (decided before data, including the reviewer's point)

- **Every measured point is reported, whatever the verdicts.** The R = 48 and 64 points join §3's ranges and Figure 1's
  data.
- **ACCOUNT CONFIRMED.**
  - §3 states the first-principles account as confirmed: the lost part shrinks at a steady rate, and the fitted ρ follows
    the far loops, so ρᵏ undershoots at short and overshoots at long loop counts.
  - The abstract's and takeaways' "within about 20%" is restricted to the loop counts these models are used at
    (16–32), and the larger misses at 48 and 64 loops are reported with their sizes.
- **PARTIAL / MIXED / REFUTED:** the account stays labelled after the fact, or is withdrawn if REFUTED. The new points are
  reported, and the headline is narrowed in the same way if they fall outside "about 20%".
- **TAIL LAW BETTER:** §2 and §7 add the tail-sum rate as the recommended one-number summary. It costs one backward pass
  with a running sum of the backward signal. ρᵏ is kept as the simple approximation that works near typical loop counts.
- **TAIL SHARE ACCURATE (but not the law):** §2 gives the tail share as the cheap direct estimate.
- **Per-token TRANSPORT or WITHIN-POSITION:** one sentence on what turns the signal. MIXED gives no claim.

## Compute

One Kaggle kernel (aurshalevmerin):
- **GPU 0:** Parcae-140m and -370m at three loop counts (about 50 min), then Huginn sequences 20–23;
- **GPU 1:** Huginn sequences 0–7 and 16–19.

About 1.5–2 h of wall time, 3 T4-hours.
