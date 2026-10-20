# v196 result: account CONFIRMED; the tail-sum law beats ρᵏ; the turning is within each token position

> **Corrections after outside review 5 (2026-10-14; paper v15). These change wording, not verdicts.**
> - **P1's largest change is 0.013** (0.0133) on the pre-registered all-parameter measure. The 0.014 below came from
>   subtracting rounded medians; the loop share's is 0.0144.
> - **Per-token: "not the gradient moving between positions" overreads the BC × W split.** The split shows each
>   position's share of the signal staying put while its direction at each position changes. Attention mixing across
>   positions could produce exactly that, so the split does not separate the two.
> - **P3 used point estimates.** With paired bootstrap 95% intervals over the 16 sequences (`paper_v15_numbers.py`):
>   - q₄(64) = 1.34 [1.19, 1.47] on 140m and 1.46 [1.14, 1.65] on 370m;
>   - the growth from 32 loops is clear on 140m (+0.29 [+0.10, +0.35]);
>   - it is not resolved on 370m (+0.20 [−0.13, +0.36]), which already overshoots at 32 (1.26 [1.03, 1.47]).
> - **The far-fade effect behind P2/P3 was described before**, by Aicher et al. 2019, §5.4, for RNNs over time.
> - **Paper changes as made (v15).** The paper recommends the exact one-pass measurement of the lost gradient, not the
>   tail-sum rate as a one-number summary. The k ≤ 4 extrapolation is labelled after the fact and reported in relative
>   terms (about half the measured value at k = 16 on 370m).

**Prereg:** `docs/prereg_v196_band.md`, frozen and pushed in 24a61a8 before the run. **Run:** Kaggle
aurshalevmerin/astrocyte-v196-band v1, 2026-10-14. All 48 Parcae sequence-runs and all 16 Huginn sequences finished.
**Analysis:** `python v196_analyze.py`.

**Bug fix to the frozen analysis, after the run (disclosed).** The file pattern `v196_%s_R*.json` missed the kernel's
Huginn files, which are named `v196_huginn_part0_R48.json` and `_part1_`. The first run of the analysis therefore scored
the head-to-head on 4 cells instead of the pre-registered 5 and skipped Huginn's supportive checks. The loader now merges
the parts. No decision rule changed. On the 4 Parcae cells alone, all three head-to-head checks also pass, so the
verdicts are the same either way. The TINY test had not caught this because the test file had been named by hand.

## Validity
- At R = 32, both Parcae models reproduce v190 exactly: err 0.0000, ρ 0.0000.
- The per-token identity BC × W = cos holds to 1e-16 everywhere.

## Account (per Parcae model): CONFIRMED (both pass P1, P2, P3)

| | R | median err at k = 1 / 2 / 4 / 8 | median ρ | q = ρᵏ / err at k = 1 / 2 / 4 / 8 |
|---|---|---|---|---|
| Parcae-140m | 32 | 0.820 / 0.675 / 0.443 / 0.205 | 0.825 | 1.01 / 1.01 / 1.05 / 1.05 |
| | 48 | 0.818 / 0.678 / 0.434 / 0.206 | 0.852 | 1.04 / 1.07 / 1.21 / 1.35 |
| | 64 | 0.818 / 0.677 / 0.429 / 0.206 | 0.871 | 1.06 / 1.12 / 1.34 / 1.61 |
| Parcae-370m | 32 | 0.827 / 0.674 / 0.451 / 0.238 | 0.868 | 1.05 / 1.12 / 1.26 / 1.35 |
| | 48 | 0.834 / 0.680 / 0.462 / 0.237 | 0.890 | 1.07 / 1.16 / 1.36 / 1.66 |
| | 64 | 0.834 / 0.679 / 0.462 / 0.237 | 0.907 | 1.09 / 1.21 / 1.46 / 1.93 |

- **P1, the lost gradient settles.** The largest change from R = 32 is 0.014 (140m) and 0.011 (370m) at k ≤ 8.
- **P2, the fitted ρ rises with R.**
- **P3, ρᵏ overshoots more and more.** q₄(64) is 1.34 and 1.46, both above the 1.10 bar.

At 64 loops the lost gradient falls 25–32% below ρᵏ at k = 4 and 38–48% below at k = 8.

**Huginn at R = 48** (supportive only, same 16 sequences as v195 at R = 32 and v187 at R = 64):
- **ρ:** 0.880 / 0.902 / 0.907 at 32 / 48 / 64, so P2′ holds.
- **q:** 0.93 / 1.01 / 1.07 at k = 4 and 0.94 / 1.05 / 1.11 at k = 8, so P3′ holds.
- **P1′ misses narrowly.** Huginn's lost gradient rose by 0.034 at k = 8 (0.383 → 0.417), against the 0.03 bar.

**Two Parcae-370m sequences (12 and 15) have a backward signal that grows going back** (per-sequence ρ 0.98–1.12), so
they lose nearly all of the gradient (err ≈ 0.99–1.00). The medians are robust to them. They are the extreme form of
"370m's far loops keep more signal".

## Head-to-head: TAIL LAW BETTER, TAIL SHARE ACCURATE

Mean |median predictor − median err_loop| over k = 1, 2, 4, 8:

| new cell | ρᵏ (rule) | tail share | tail law r_tailᵏ |
|---|---|---|---|
| Parcae-140m R 48 | 0.050 | 0.027 | 0.014 |
| Parcae-140m R 64 | 0.090 | 0.024 | 0.014 |
| Parcae-370m R 48 | 0.110 | 0.020 | 0.019 |
| Parcae-370m R 64 | 0.150 | 0.018 | 0.016 |
| Huginn R 48 | 0.016 | 0.010 | 0.021 |

- **H1:** the share is ≤ 0.04 everywhere.
- **H2:** the law beats the rule in 4 of 5 cells. Huginn at 48 is the exception, where ρᵏ is inside its band.
- **H3:** the law is ≤ 0.04 everywhere.

At R = 32 (reference): rule 0.008 / 0.069, share 0.030 / 0.011, law 0.017 / 0.012 for 140m / 370m.

The tail law is the decay rate of the tail sums of the backward signal, from the same single backward pass that gives
ρ. Its error stays at 0.012–0.021 at every loop count. ρᵏ's error grows with the loop count beyond about 32.

**Interpretation caveat (review note, 2026-10-14; this changes the wording, not the verdicts).** The tail share is
the lost gradient measured in the hidden state instead of in the weights. It is a cheaper direct measurement, not a
prediction from a different quantity. The tail law is a single rate fitted to that same curve over k = 0–8, and it was
scored at k = 1, 2, 4, 8, inside its fitting range. Its win over ρᵏ is therefore largely built in. ρᵏ predicts from
per-loop sizes.

What the head-to-head does show:
- One backward pass gives the whole lost-gradient curve, within 0.01–0.03 of the weight-space value.
- That curve shrinks at a near-constant rate near the output.

**A fairer extrapolation check (post hoc).** Fit the rate on k = 0–4 only, then score it at k = 8 and 16:

| | k = 8: rate vs measured | k = 16: rate vs measured | ρᵏ at k = 8 / k = 16 |
|---|---|---|---|
| Parcae-140m | 0.19 vs 0.21 | 0.038 vs 0.044 | 0.22–0.33 / 0.05–0.11 |
| Parcae-370m | 0.20 vs 0.24 | 0.042 vs 0.08 (low, because its tail slows further back) | 0.32–0.46 / 0.10–0.21 |
| Huginn R48 | 0.44 vs 0.42 | 0.19 vs 0.17 | 0.44 / 0.19 |

Unlike ρᵏ, the rate does not drift with the loop count.

The paper calls the tail sum a cheaper direct measurement, not a better law.

## Per-token: WITHIN-POSITION on all three models (s = 0.01–0.05)

The loop-to-loop turning near the output is almost all turning within each token position's features. It is not the
gradient moving between positions.

| | overlap of neighbouring loops' position profiles (BC), j = 0 / 1 / 2 / far | within-position cosine (W), j = 0 / 1 / 2 / far |
|---|---|---|
| Parcae-140m | 0.967 / 0.983 / 0.988 / 0.998 | 0.44 / 0.62 / 0.74 / 0.98 |
| Parcae-370m | 0.944 / 0.978 / 0.985 / 0.998 | 0.33 / 0.56 / 0.68 / 0.98 |
| Huginn | 0.960 / 0.991 / 0.993 / 0.998 | 0.11 / 0.16 / 0.25 / 0.70 |

Descriptive:
- **Mass drifts slowly toward earlier tokens across many loops.** The position centre of mass goes from token ~237–241
  to ~72 of 512 by j = 16 on Parcae, and from 52 to 42 of 128 on Huginn. So transport exists, but per loop it is small
  next to the within-position turning.
- **The signal collapses into fewer directions further back.** Effective rank: 104–106 → 135–138 (j = 2) → 18–31
  (j = 16) on Parcae; 42 → 56 → 30 on Huginn.

**Reading.** Near the output, each loop's backward map rotates the gradient within each position's feature space. This
fits the outside consult's "read/write asymmetry": the backward pass reads the gradient through different feature
directions than the forward pass wrote into. Further back, the gradient funnels into a few directions that the loop
passes along without turning them. That slow, low-rank part is what truncation loses.

## Paper changes (as pre-registered)
- **§3:** the first-principles account is stated as confirmed (S10):
  - the lost part shrinks at a steady rate;
  - near-output turning within each position's features inflates the per-loop sizes;
  - the fitted ρ follows the far loops, so ρᵏ undershoots at short and overshoots at long loop counts.
- **Abstract, §1, takeaways:** "within about 20%" is restricted to the loop counts these models are used at (16–32),
  and the 48 and 64 loop overshoots are reported with their sizes.
- **§2 and §7:** the tail-sum rate is recommended as the one-number summary (one backward pass, running sum of the
  backward signal). ρᵏ is kept as the simple approximation near typical loop counts.
- **Per-token:** one sentence, saying the turning is within positions.
- **Table 4:** S10 is added.
