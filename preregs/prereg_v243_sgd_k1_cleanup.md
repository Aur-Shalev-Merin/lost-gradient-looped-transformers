# PREREG v243: clean-up of v241 (S24) at k = 1: does giving the truncated direction its full length help under SGD?

**Status: FROZEN** (this commit), before any run. The freeze is pushed to origin (branch `v243-freeze`, whose parent is
v241's freeze commit c9d8c82) before the first kernel. Approved by the author ("do it", 2026-11-13, on the plan below).

**In the paper:** S25, labelled a clean-up of S24, as S23 was for S22.

## 0. Why, and what is known before this freeze

v241 (S24) ran S22's size-versus-direction split under SGD with momentum 0.9 and clipping only as a spike guard
(norm 5). Its frozen verdict was MIXED: PARTIAL at k = 1, TURN at k = 2 (`docs/v241_result.md`). At k = 1:
- truncation cost 0.257 ± 0.012, the turn-only arm 0.201 ± 0.007 and the shrink-only arm 0.039 ± 0.038 (5 seeds);
- the turn-only arm missed the "turn" bar by 0.005;
- the shrink-only arm, the pre-set test of the size, was inside its margin, with per-seed costs from −0.070 to 0.167.

**The quantity this study tests was chosen after S24's results.** In S24, paired by seed, truncation minus turn only
was 0.057 ± 0.018 at k = 1, with per-seed values 0.060, 0.116, 0.060, 0.007 and 0.040. The two arms share the
truncated direction and differ only in length: truncation steps at 0.69 of full backprop's length, the turn-only arm at
full length. Under SGD that difference is the cleanest available measure of whether the lost size costs. Draft v18.2
reports it as a lead, not a finding. This study tests it on new seeds with the rule fixed in advance.

**Everything about S24 is known at this freeze**, including its pilot, the step-matched controls and the stability
counts. This study is not independent of S24's choice of quantity. It is independent of S24's data: new seeds, the same
frozen code.

**Selection effect, disclosed.** A difference picked because it looked large in 5 seeds will tend to come out
smaller on new seeds. The predictions in §4 allow for that.

## 1. Design (LOCKED)

**Runs:** v241's driver, unchanged (`v241_sgd_split.py` e505057ac886, combined driver sha 3b446a33effc), with its
frozen setting G:
- SGD with momentum 0.9 and no weight decay;
- learning rate 0.1 for every arm, 0.1 for the floor;
- global clipping at norm 5;
- S8's recipe on Parcae-140m at R = 32 (600 steps, GSM8K, fp16 autocast with manual loss scaling).

Only k = 1 is run: `full`, `floor`, `trunc1`, `turn1` and `shrink1` on each seed. The step-matched controls are
dropped. Answering their question needs a learning-rate curve for full backprop, which this study does not buy. Every
GPU lane runs v241's selftest on the real model first.

**Seeds:** 2431101–2431112, all run. The first 10 valid seeds in that order are scored. Fewer than 8 valid seeds:
INCOMPLETE.

**Validity (S23's rule):** a seed is valid only if every one of its five runs has at most 10 non-finite steps and a
finite test loss.

**Cost:** φ = (L_arm − L_full) / (L_floor − L_full), on GSM8K test loss, per seed.

**Gate:** the floor gap is at least 0.03 nats, with its one-sided 90% t lower bound above 0. Otherwise INVALID.

**Primary quantity:** D = φ_trunc1 − φ_turn1, paired by seed. It is how much of the loop's benefit the full length
gives back to the truncated direction.

**Verdict (on the scored seeds; one-sided 90% t bounds of the mean of D):**
- **SIZE-COSTS:** mean D ≥ 0.015 and the lower bound is above 0;
- **SIZE-FREE:** the upper bound is below 0.015;
- **SIZE-UNCLEAR:** otherwise.

0.015 is the floor of the margin S22–S24 use for "costs nothing" (m = max(0.015, 0.2 φ_trunc)).

**Operating characteristics** (simulated with 10 seeds and S24's per-seed spread of D, 0.039):

| true D | SIZE-COSTS | SIZE-FREE | SIZE-UNCLEAR |
|---|---|---|---|
| 0 | 0.08 | 0.45 | 0.47 |
| 0.015 | 0.42 | 0.10 | 0.48 |
| 0.030 | 0.84 | 0.01 | 0.16 |
| 0.045 | 0.98 | 0.00 | 0.02 |
| 0.057 | 1.00 | 0.00 | 0.00 |

The test detects an effect of about half S24's estimate most of the time. It can call the size free only if the true
effect is near zero, and even then only about half the time.

**Secondary (reported with labels; they do not decide the verdict):**
- v239's k = 1 label rule (TURN / PARTIAL / SHRINK), unchanged;
- the shrink-only cost under the same three-way rule (COSTS / FREE / UNCLEAR).

**Reported, not scored:**
- per-seed costs; Mbar, step length relative to full backprop, share of steps clipped;
- unstable runs per cell over all runs made;
- the share of truncation's cost the full length recovers, D / φ_trunc1;
- pooled with S24's 5 scored seeds (15 seeds), D and the three costs, labelled as pooled after the fact.

**What the design can and cannot show.**
- **Can:** whether, under SGD with a spike guard on Parcae-140m at k = 1, restoring the truncated direction's length
  recovers part of truncation's cost.
- **Cannot:**
  - separate "the size costs" from "the shared learning rate is too low for the truncated direction". Under SGD these
    are the same statement. A per-arm learning-rate sweep would separate them and is not run;
  - whether the full gradient's direction loses anything when shrunk (the shrink-only arm stays noisy, secondary);
  - k = 2 (S24 labelled it TURN), other models, other optimisers.

## 2. What each outcome changes in the paper

| outcome | paper change |
|---|---|
| SIZE-COSTS | **Abstract:** the SGD sentence becomes "Under SGD the direction still carried most of the cost, but the lost size carried some: with one loop kept, giving the truncated direction its full length recovered [D/φ_trunc as a percentage] of truncation's cost." **§5** reports S25 as the test of S24's lead, with the caveat that under SGD a shorter gradient is a smaller step, so this says the shared learning rate was too small for the truncated direction. **§1 finding 3, §7 and §8** say the size costs a little under SGD and nothing under AdamW. |
| SIZE-FREE | **Abstract:** "Under SGD too, giving the truncated direction its full length recovered no measurable cost (S25), though there truncation shortened the step only to 0.70 of its length." **§5, §1, §7, §8:** S24's after-the-fact difference did not replicate; the split holds under both optimisers on this model. §8's takeaway drops "Under Adam". |
| SIZE-UNCLEAR | **Abstract and §1** unchanged (whether size costs under SGD is unclear). **§5** says one sentence: a pre-registered test on 10 new seeds was inconclusive (with D and its bounds). |
| INVALID or INCOMPLETE | Appendix only; the main text is unchanged. |

In every case:
- App. G gains S25 (design, results, secondaries, stability, the pooled figures labelled after the fact);
- Table 4 gains an S25 row;
- §1 counts twenty-five studies;
- §5's SGD paragraph is shortened to its result, with the step-matched controls' caveats moved to App. G.

## 3. Checked before the freeze

- **Code path on real rows:** S24's 60 rows, mapped onto this study's seeds, run through the analysis without error and
  reproduce S24's numbers (D = 0.0567, φ_trunc1 0.2573, φ_turn1 0.2007, v239 label PARTIAL).
- **Synthetic worlds** come out as built (`--synthetic`; noise alternates so that each world's mean is exact):
  - SIZE-COSTS (D 0.057); SIZE-FREE (D 0, spread 0.01);
  - SIZE-UNCLEAR (D 0.01, spread 0.04; D 0.03, spread 0.10);
  - SIZE-COSTS with two seeds broken (10 of the remaining used);
  - INCOMPLETE (5 seeds broken); INVALID (floor gap 0.01).
- **Tiny-model run on CPU** (`TINY=1`, random tiny Parcae): v241's driver runs `turn1` and `shrink1` on seed 2431101
  and writes rows with every field the analysis reads.
- **The kernel build** embeds the frozen v241 files (hashes below), runs v241's selftest on every lane, and splits the
  60 runs into eight lanes of about 228 minutes each.

## 4. Predictions (mine, at the freeze, knowing S24)

- **Verdict:** SIZE-COSTS 0.55, SIZE-UNCLEAR 0.27, SIZE-FREE 0.08, INVALID or INCOMPLETE 0.10.
- **Mean D:** about 0.04. I expect shrinkage from S24's 0.057 because the quantity was picked after the fact.
- **Secondary:** v239's rule gives PARTIAL 0.55 and TURN 0.40; shrink-only UNCLEAR 0.6.
- **Stability:** at least one full-backprop run unstable among the 12, 0.6 (S24: 1 of 6).

## 5. Cost and runs

60 runs (12 seeds × 5 cells).
- Estimated minutes on a T4: two-gradient arms 38, full backprop 29, floor 9; 1,824 in total.
- Four kernels × 2 T4s, about 3.8 h per lane plus setup and the selftests, so about 16 session-hours of Kaggle quota.

## Hashes

sha256 prefixes (LF).
- `v243_sgd_k1_cleanup.py`: 063ad397fe61
- `build_kaggle_v243.py`: 374ce30ab58a
- Unchanged from v241's freeze: `v241_sgd_split.py` e505057ac886, v239 7f5262e51b75, v170_sgd 4ce6128b21ed,
  v174 f3cdba92e749, v163 db639c411892, v162 0ae4891d7abc.
