# PREREG v225: does truncation's cost survive learning rates chosen per arm, on data it is not scored on?

**Status: FROZEN** (this commit), before any v225 run. The freeze commit is pushed to origin before the first kernel.

**Code (sha256 prefixes, LF):**
- `v225_lr_per_arm.py` (Ouro driver, selection rules and locked analyses): 5f356467decd;
- `build_kaggle_v225.py`: 4d53cb1aa1e7;
- imported unchanged: `v219_ouro_cost.py` f8543506d909, `loop_grad_probe.py` 27b80fa0b999;
- Part P runs through `v200_ifm_trunc.py` bd9ee68769e0 and `build_kaggle_v200.py` 57dc9f9ccda5 (cells mode), both
  unchanged since v200's freeze (4849a02).

Kaggle accounts are assigned at push, by free slots (another thread shares them). The result doc records which ran
where.

## 0. Why

A cold TMLR-style review (2026-10-27) made two requests about the cost tests:
1. **S18 chose Ouro's learning rates on GSM8K test, the split it was scored on** (App. K, a disclosed deviation). The
   reviewer asks for selection on one part of test and scoring on the other.
2. **Truncated arms reused full backprop's LR.** Their gradients are smaller and were clipped on fewer steps, so part
   of their cost may be a step-size mismatch.

Checked before this freeze:
- **Parcae-140m (v163) already tuned every truncated cell separately.** The grid was {1e-4, 3e-4, 9e-4} on val, and
  every cell chose full backprop's 3e-4. So the paper's "we did not tune it per k" is wrong for Parcae, and this study
  does not rerun Parcae.
- **The two Part II models (v200), SmolLM2 (v167) and Ouro (v219) reused full backprop's LR.** SmolLM2's costs are about
  1%, so it is left out.

## 1. Part O: Ouro-1.4B (LOCKED)

**Recipe.** S18's recipe, unchanged (`v219_ouro_cost.train_cell` imported):
- model revision 574fa66, fp32 split over two T4s, transformers 4.54.1;
- 600 steps of batch 8 × 512;
- AdamW (0.9, 0.95), weight decay 0, warm-up and cosine, clip 1.0, fp16 autocast with the cache off;
- final-output training loss;
- the token embedding frozen in every arm;
- cells k = 4 (full), 2, 1 and 0 (the floor; only the output head trains).

**Split.** GSM8K test, 1,319 problems in dataset order:
- **SEL** is the even-indexed problems (0, 2, …) and **SCORE** the odd-indexed ones;
- each half is packed separately into 512-token chunks, so no problem is in both;
- val (the last 473 train problems) is not used, because Ouro has seen GSM8K's train split (S18, App. K).

**Selection.** Selection seed 2251001.
- **Grid:** every arm runs LR 3e-5, 1e-4 and 3e-4.
- **Choice:** each arm's LR is the one with the lowest SEL final-output loss among valid runs (not diverged, finite
  losses; full backprop at most 10 non-finite steps). For S18's recipe the grid is centred on S18's 1e-4.
- **Edge rule:** if the best LR is at an edge of the grid, the one step beyond it (1e-5 or 1e-3) is run once on the
  selection seed, and the choice is then made over all the LRs run. If the best is still at the new edge, it is taken
  and reported as such.
- **Where it is coded:** `--select-ouro` prints either the extension cells to run or the scored cells.

**Scoring.**
- **Seeds:** 2251011–2251015, with 2251016–2251017 as backups. Each seed runs all four arms, each at its selected LR.
- **Exclusion:** the v165 rule, as S18: a seed is valid if its full-backprop run has at most 10 non-finite steps and no
  run of that seed diverged. The first 5 valid seeds are used. There is no seed rule: N = 5 is fixed, since S18 showed
  the spread is large.
- **Cost:** φ_k = mean over seeds of (L_k − L_full) / (L_floor − L_full), on SCORE, final-output loss, with a 90%
  t-interval. The added nats, L_k − L_full, are reported with intervals too.
- **Floor gate:** S18's. The mean SCORE gap must be at least 0.03 nats with its 90% lower bound above 0; otherwise the
  study is UNINFORMATIVE.

**Verdicts** (`--analyze-ouro`):
- **V1:** S18's verdict rule, re-applied on SCORE with per-arm LRs and the same predictions (L = 0.258 / 0.187 at
  k = 1 / 2; KP = 0.224 / 0.100). The first rule that applies:
  - BELOW-BOTH;
  - ABOVE-BOTH;
  - L-CLOSER (MAE(L) ≤ 0.5 · MAE(KP));
  - KP-CLOSER;
  - TIE.
- **V2, k = 1 against the floor:**
  - FLOOR-LEVEL if the 90% upper bound of φ1 is at least 1 (keeping one loop is not shown to beat not training the
    loop);
  - BETTER-THAN-FLOOR otherwise.
- **Also reported, not scored:**
  - the selection tables (SEL and SCORE loss per LR and arm);
  - the exit-weighted objective;
  - per-seed values;
  - S18's costs alongside (different seeds and a different eval set: descriptive only).

## 2. Part P: the two Part II models (LOCKED)

**Recipe.** S11's recipe (`v200_ifm_trunc.py`, unchanged; embedding frozen; R = 5).

**Selection.**
- **Grid:** for k = 1, 2 and 4 on both models, LR 3e-5, 1e-4 and 3e-4 on v200's pilot seed 9001. v200's full-backprop
  LR is 1e-4.
- **Choice:** the lowest val loss, v200's rule for its controls. Val is held out from these models' fine-tune, as in
  v200.
- **Edge rule:** the same as Part O (1e-5 or 1e-3, run once).
- **Where it is coded:** `--select-ifm`.

**Rescoring.**
- **If every cell selects 1e-4:** no further runs, and the verdict is SAME-LR.
- **Otherwise,** for each model with a moved cell, the 5 scored seeds of v200 (2001101–2001105) run:
  - full backprop (1e-4);
  - the floor (3e-4, v200's);
  - each moved cell at its selected LR and at 1e-4.

  The tuned and untuned costs then come from the same batch of runs and are paired by seed. v200's own runs are not
  reused, and their values serve as a reproduction check (reported).
- **Verdict:** SAME-LR, or LR-MOVED with:
  - each moved cell's tuned and untuned cost and their paired difference (90% interval);
  - S11's transfer verdict re-scored with the tuned costs in place of the moved cells' v200 costs (`--analyze-ifm`;
    MAE of L against KP and KS, as in prereg v200).

## 3. What each outcome changes in the paper

**Part O:**

| outcome | paper change |
|---|---|
| V1 ABOVE-BOTH and V2 FLOOR-LEVEL | §5, the abstract and App. K: S18's result holds with each arm's learning rate chosen on separate problems. The qualifier "at full backprop's learning rate" is replaced, and the test-selection deviation is marked as resolved by S19. |
| V2 BETTER-THAN-FLOOR | The abstract and §5 drop "no better than not training the loop" and report the tuned φ1. §5 says that much of S18's k = 1 cost came from the learning rate. |
| V1 other than ABOVE-BOTH | §5 states the new verdict, and S18's verdict is reported as not robust to per-arm learning rates. The abstract's Ouro sentence is rewritten to match. |
| UNINFORMATIVE | Reported. S18 stands with its deviation. |

**Part P:**

| outcome | paper change |
|---|---|
| SAME-LR | §5: on Parcae (v163) and both Part II models, tuning the truncated arms' learning rate per k chose full backprop's rate. The step-size caveat applies only to Ouro (and SmolLM2). |
| LR-MOVED | §5 and App. H report the tuned costs, and S11's verdict is re-stated as re-scored. If the re-scored verdict differs from K-AS-GOOD, the abstract and §1 change to match. |

**Paper fix independent of the outcome:** §5's "we did not tune it per k" is corrected for Parcae (v163 tuned every
cell, and all chose 3e-4).

## 4. Checked before the freeze

- **Tiny-Ouro CPU run (TINY=1, transformers 4.54.1):**
  - the selftest passes through v225;
  - SEL and SCORE are disjoint by problem index;
  - one run per arm trains and records SEL and SCORE.
- **Synthetic worlds (no training).** These come out as built:
  - interior selection;
  - an edge that asks for the extension, and selection after it;
  - ABOVE-BOTH with FLOOR-LEVEL;
  - BELOW-BOTH with BETTER-THAN-FLOOR;
  - a failed gate (UNINFORMATIVE);
  - Part P SAME-LR;
  - Part P LR-MOVED, with its rescoring cells and the re-scored S11 verdict.

## 5. Predictions (mine, at the freeze)

**Part O:**
- **selected LRs:** full 1e-4 (0.75); floor 1e-4 (0.6). Each truncated arm 1e-4 (0.45), 3e-4 (0.4), 3e-5 (0.15).
- **V2:** FLOOR-LEVEL 0.6, BETTER-THAN-FLOOR 0.4.
- **V1:** ABOVE-BOTH 0.5, TIE 0.2, KP-CLOSER 0.1, L-CLOSER 0.05, BELOW-BOTH 0.05, UNINFORMATIVE 0.1.

**Part P:** SAME-LR 0.55, LR-MOVED 0.45. If a cell moves, I expect it to move up (3e-4), at k = 1.

## 6. Cost and runs

**Part O selection** (three kernels, one model across both T4s each; per run: full 2.9 h, k = 2 1.8 h, k = 1 1.25 h,
floor 0.65 h, plus about 6 minutes of evaluation):

| kernel | account | cells (seed 2251001) | hours |
|---|---|---|---|
| astrocyte-v225-sel-a | aurshalevmerin | full at 1e-4, 3e-5, 3e-4 | about 9.3 |
| astrocyte-v225-sel-b | aurmerin | k = 2 at 1e-4, 3e-5, 3e-4; floor at 1e-4 | about 6.6 |
| astrocyte-v225-sel-c | aurmerin | k = 1 at 1e-4, 3e-5, 3e-4; floor at 3e-5, 3e-4 | about 5.7 |

**Part O extensions:** if needed, `build_kaggle_v225.py cells`.

**Part O scoring:** 5 kernels, one per seed, about 7 hours each.

**Part O total:** about 57 kernel-hours.

**Part P selection:** two kernels (`build_kaggle_v200.py cells --tag v225sel`), 9 runs each over two GPUs, about
1.5 hours. Rescoring, if needed, takes about 2 hours per model.
