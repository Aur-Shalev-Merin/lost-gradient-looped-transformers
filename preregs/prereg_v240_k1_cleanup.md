# PREREG v240: clean-up of v239 (S22) at k = 1, with a validity rule for every run

**Status: FROZEN** (this commit), before any run. The freeze is pushed to origin (branch `v240-freeze`) before the
first kernel. Approved by the author.

**In the paper:** S23, labelled a clean-up of S22, not an independent test. S4 played the same role for S3.

## 0. Why, and what is known before this freeze

v239 (S22) asked whether truncation's training cost on Parcae-140m is carried by the truncated gradient's turn
(direction) or its shrink (size). Its frozen verdict was MIXED: k = 1 SHRINK, k = 2 TURN. The k = 1 label comes from
one diverged run (`docs/v239_result.md`):
- the size-only arm, seed 2391105;
- the full gradient spiked to a norm of 291,229 near step 300, the fp16 loss scale halved 24 times, and test loss
  ended at 20.4 (cost 114.5);
- the other four size-only runs cost 0.007, 0.002, −0.006 and 0.001.

v239's exclusion rule looked only at each seed's full-backprop run, so the diverged arm run counted.

**Everything about v239's results is known at this freeze.** In particular, at k = 1:
- turn-only 0.225 against truncation 0.226;
- size-only about 0 in the other four runs, and 0.0005 on the unscored backup seed.

This study cannot be an independent test of the reading. Its job is narrower: to give the k = 1 label under a
validity rule that was fixed before the runs, on new seeds. The paper keeps S22's MIXED verdict and reports S23 next
to it.

## 1. Design (LOCKED)

**Runs:** v239's driver, unchanged (`v239_shrink_turn.py` 7f5262e51b75, combined driver sha ac4e7f8570b8), with its
recipe, arms and learning rates. Only k = 1 is run: `full`, `floor`, `trunc1`, `turn1` and `shrink1` on each seed.
Every GPU lane runs v239's selftest on the real model first.

**Seeds:** 2401101–2401107, all run. The first 5 valid seeds in that order are scored. If fewer than 5 are valid,
2401108 and 2401109 are run. Fewer than 4 valid seeds: INCOMPLETE.

**Validity (new, the point of this study):** a seed is valid only if every one of its five runs has at most 10
non-finite steps and a finite test loss. The threshold is the paper's existing exclusion rule, now applied to every
run instead of the full run alone. Applied to v239's data, this rule would have replaced seed 2391105 (shrink1: 24
non-finite steps) with 2391106. We state that here; it does not change v239's verdict.

**Quantity, gate and label:** v239's, unchanged.
- Cost φ = (L_arm − L_full) / (L_floor − L_full) on GSM8K test, paired by seed, as the mean over the scored seeds.
- Gate: the floor gap is at least 0.03 nats, with its 90% t lower bound above 0; otherwise INVALID.
- Label at k = 1, with m = max(0.015, 0.2 · φ_trunc):
  - TURN if φ_turn ≥ φ_trunc − m and φ_shrink ≤ m;
  - SHRINK if φ_shrink ≥ 0.5 · φ_trunc;
  - PARTIAL otherwise.

**Verdict:** the k = 1 label (or INVALID / INCOMPLETE).

**Reported, not scored:** per-seed costs; non-finite steps per run; which seeds were invalid and why; Abar per arm.
Also, as one combined descriptive figure, v239's and v240's valid seeds pooled.

**Amendment 1 (2026-11-11, after the kernels started and before any v240 output was read; report-only, no rule or
label changes).** The validity rule discards a seed when any of its runs becomes unstable, so it cannot show whether
one arm makes training unstable more often than another. The paper will therefore report, for each arm, the number
of runs with more than 10 non-finite steps or a non-finite test loss, out of all runs made. This counts every seed
run, backups included, pooled over v239 (k = 1, 2, 4) and v240 (k = 1), and separately for k = 1. It also reports
each arm's mean non-finite steps. v239 alone: 2 of 18 shrink runs crossed the limit, against 0 of 36 trunc and turn
runs and 0 of 12 full and floor runs. One is k = 1, seed 2391105, which diverged. The other is k = 4, seed 2391106,
the unscored backup: 11 non-finite steps with a normal test loss. Too few to show a difference, but the comparison
will be reported with v240's runs added.
All three two-gradient arms compute g_R and g_k on every step, so a spike in either gradient can hit any of them; the
arms differ only in which update they take.

## 2. What each outcome changes in the paper

| outcome | paper change |
|---|---|
| TURN | §5: on Parcae-140m under the paper's recipe, the turn carries truncation's cost at k = 1 (S23, a clean-up of S22) and at k = 2 (S22). S22's MIXED verdict and its diverged run are reported, and k = 4 is reported descriptively. The abstract gains one sentence: what costs is the change of direction, not the lost size, which is why the lost gradient's size is a poor guide to cost. The limits stay explicit: one model, AdamW with clipping, and the reason the shrink is cheap (Adam or clipping) untested. |
| SHRINK | §5: at k = 1 the size-only arm costs at least half of truncation's cost even with diverged runs excluded, so the shrink matters there; the reading holds only at k = 2. |
| PARTIAL | §5 gives the turn and shrink shares at k = 1 and says the reading holds in part. |
| INVALID or INCOMPLETE | Reported; §5 reports S22 as MIXED with its descriptive reading only. |

## 3. Checked before the freeze

- **Synthetic worlds** come out as built: TURN, SHRINK, PARTIAL, INVALID (floor gap 0.01), INCOMPLETE (3 seeds), and
  a world where one size-only run diverges (24 non-finite steps, test 20.4). In that world the seed is replaced by the
  next in order and the verdict stays TURN.
- **A tiny CPU run** of the kernels' exact command line (v239 driver, v240 seeds, tag `c10`) wrote
  `v239_train_c10.jsonl`, and the v240 analysis read it.
- **The kernels embed v239's frozen driver unchanged,** with its imports matching their frozen hashes.

## 4. Predictions (mine, at the freeze)

Knowing v239: TURN 0.85, PARTIAL 0.08, SHRINK 0.02, INVALID or INCOMPLETE 0.05. A diverged run, if one occurs, is
replaced by a backup seed. Under the old rule it would have decided the label again; this rule is what this study
adds.

## 5. Cost and runs

35 runs (7 seeds × 5 cells). Four kernels × 2 T4s, about 2.3 h per lane plus setup and the selftests, so about 10
session-hours of Kaggle quota.

## Hashes

sha256 prefixes (LF).
- `v240_k1_cleanup.py`: 37ad1840140b
- `build_kaggle_v240.py`: 1b5c665d4504
- `v239_shrink_turn.py`: 7f5262e51b75 (unchanged since its freeze, 9dea2a0)
- `v174_mix_driver.py`: f3cdba92e749
- `v163_parcae_damped_trunc.py`: db639c411892
- `v162_parcae_trunc_train.py`: 0ae4891d7abc
