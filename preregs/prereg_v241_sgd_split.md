# PREREG v241: does the lost size cost under SGD? S22's shrink-versus-turn split without Adam's normalisation

**Status: FROZEN** (this commit), after the controls-only learning-rate pilot and before any treatment run.
- **Pilot rule pushed first:** §1a's rule was pushed to origin (branch `v241-freeze`, 2e01de8, 2026-11-11 21:15:39 UTC)
  before the pilot ran.
- **The pilot trained only full backprop and the floor.** Its results are in §1a.
- **This freeze** is pushed to the same branch before the first scored kernel.
- **Approved by the author:** the smaller design (setting C1 dropped) was chosen after the pilot.

**In the paper:** S24.

## 0. Why

S22 and S23 split the truncated gradient into its size and its direction on Parcae-140m under AdamW with global
clipping at norm 1. The turn alone cost as much as truncation, and the shrink alone cost nothing (k = 1: 0.224 / 0.224
/ 0.001). That outcome was close to built in:
- **Adam ignores a constant rescale of the gradient.** The shrink-only arm's update direction after Adam agreed with
  full backprop's at 0.998–1.000, and v239's prereg said it should cost little by construction.
- **Draft v18.1** therefore states the result as holding under AdamW, with the split untested without Adam.

Under SGD, a shorter gradient is a shorter step. This study runs the split under SGD with clipping only as a spike
guard (norm 5), so that nothing normalises the size away. It asks whether truncation's loss of size costs there.

**A shrink under SGD is close to a lower learning rate.** If the shrink costs, it may cost only as a smaller step
would. Two reported controls separate the two readings: full backprop at a constant learning rate scaled by the
measured shrink (`fullm1`, `fullm2`).

**What the pilot showed about this test's power (disclosed; it changed the design).** In training, the truncated
gradient is 0.70 (k = 1) and 0.84 (k = 2) of the full gradient's length (median over the pilot's full-backprop run,
steps ≥ 100). The draft had assumed about half, from one sequence on the untrained model. And under SGD here, a full
backprop step cut to 30% (learning rate 0.03 instead of 0.1) cost only 0.017 nats on test, about 0.08 of the loop's
benefit. So a shrink to 0.70 is expected to cost little. A TURN outcome would then say that, under SGD as well,
truncation's loss of size was too small a change in step to cost anything, not that size can never matter under
SGD. The step-matched controls test exactly that.

**Known before this freeze (disclosed).**
- All of S22's and S23's results (`docs/v239_result.md`, `docs/v240_result.md`).
- S7's Parcae SGD runs (v170, clip 1, learning rate 0.1): costs 0.263 / 0.104 / 0.013 at k = 1 / 2 / 4, and its
  one-seed learning-rate pilot.
- This study's pilot (§1a), controls only.

## 1a. Learning-rate pilot (rule fixed and pushed before it ran)

**Runs:** clip at 5, seed 2419901 (not a scored seed), this study's pipeline.
- Round 1: full backprop at 0.03, 0.1 and 0.3; the floor at 0.1, 0.3 and 1.0.
- Round 2 (the rule's grid-end extension for the floor): the floor at 1/30.
- Every lane passed the v241 selftest on the real model (driver 87e4cdb665dc).

**Rule** (`v241_sgd_split.py --pick`):
- Per control, take the grid rate with the lowest validation loss among stable runs (finite losses, at most 10
  non-finite steps).
- If that rate is a grid end, run one more 3× step on that side and apply the rule again, once.
- LR_G is full backprop's pick; every truncated, turn and shrink arm uses it. LR_G_FLOOR is the floor's pick.
- R_MATCH[k] is the median, over probes at step ≥ 100, of the clipped |g_k| over the clipped |g_R| in the chosen
  full-backprop run.

**Results:**

| control | rate | val | test | non-finite steps |
|---|---|---|---|---|
| full | 0.03 | 1.4589 | 1.4422 | 1 |
| full | **0.1** | **1.4445** | **1.4249** | 0 |
| full | 0.3 | diverged | diverged | 587 |
| floor | 1/30 | 1.7472 | 1.7332 | 0 |
| floor | **0.1** | **1.6689** | **1.6522** | 1 |
| floor | 0.3 | diverged | diverged | 554 |
| floor | 1.0 | diverged | diverged | 588 |

- **Picks:** LR_G = 0.1 and LR_G_FLOOR = 0.1.
- **R_MATCH:** {1: 0.702, 2: 0.836}.
- **Floor gap** at the picks: 0.227 nats.
- **Full-backprop run at 0.1:** median |g_R| 0.90, with 25 steps above the spike guard of 5.

## 1. Design (LOCKED)

**Model and pipeline:** S8's (v174, imported unchanged) on Parcae-140m at R = 32.
- GSM8K question-and-answer text, 600 steps of batch 8 × 512, 20-step warm-up then cosine decay to 10%.
- fp16 autocast with manual loss scaling: a step with a non-finite gradient is skipped and the scale halved.
- The optimiser is SGD with momentum 0.9 and no weight decay (S7's), in place of AdamW.
- Clipping at global norm 5, as a spike guard.
- Each step of a two-gradient arm computes g_R and g_k on the same batch and state draw, builds the update gradient
  with v239's constructions (imported unchanged), clips it and steps SGD at learning rate 0.1:
  - `trunc`: g = g_k;
  - `turn`: g = g_k · |g_R| / |g_k|;
  - `shrink`: g = g_R · |g_k| / |g_R|.
- `full` is g = g_R through the same loop, at 0.1.
- `floor` is v163 train_cell with k = 0, with AdamW replaced by v170's SGD and v162's clip set to 5 for that call
  only, at 0.1.

**Cells:**
- Scored: `full`, `floor`, and `trunc`, `turn`, `shrink` at k = 1 and k = 2.
- Reported, not scored: `fullm1` (full backprop at 0.0702) and `fullm2` (at 0.0836), a constant step shrink matched to
  R_MATCH.
- Setting C1 (the k = 1 trio at clip 1) was dropped at the freeze. The pilot showed that a shrink this mild is
  unlikely to cost even without clipping at 1, so C1 could not add information worth its quota. Its code stays in
  the driver but is not run.

**Seeds:** 2411101–2411106, all run. The first 5 valid seeds in that order are scored. If fewer than 5 are valid,
2411107 and 2411108 are run. Fewer than 4 valid seeds: INCOMPLETE.

**Validity (S23's rule):** a seed is valid only if every one of its eight scored cells has at most 10 non-finite steps
and a finite test loss.

**Quantity:** cost φ = (L_arm − L_full) / (L_floor − L_full), on GSM8K test loss, paired by seed, as the mean over the
scored seeds.

**Gate:** the floor gap is at least 0.03 nats, with its 90% t lower bound above 0. Otherwise INVALID.

**Label per k (v239's rule, unchanged),** with m_k = max(0.015, 0.2 · φ_trunc,k):
- TURN if φ_turn ≥ φ_trunc − m_k and φ_shrink ≤ m_k;
- SHRINK if φ_shrink ≥ 0.5 · φ_trunc;
- PARTIAL otherwise.

**Verdict:** the common label of k = 1 and k = 2, MIXED (with the per-k labels) otherwise.

**Step reading (reported, pre-set):** compare φ_shrink with the matched-step control φ_fullm,k:
- AS-STEP if |φ_shrink − φ_fullm,k| ≤ m_k;
- BEYOND-STEP if φ_shrink is larger by more than m_k;
- BELOW-STEP if it is smaller by more than m_k.

It is interpreted at SHRINK and PARTIAL ks. At a TURN k it is reported together with φ_fullm,k, which says whether
a step that much shorter costs anything at all.

**Reported, not scored:**
- per-seed costs; turn and shrink shares;
- per run: non-finite steps, Mbar (agreement of the momentum buffer with a shadow full-backprop one), mean step
  length relative to full backprop, share of steps clipped, spike-guard hits;
- the truncated arm against S7's 0.263 / 0.104, which used clip 1, so the comparison is indicative only;
- unstable runs per cell over all runs made, backups included (as S23's amendment 1).

**What the design can and cannot show.**
- **Can:** whether, on Parcae-140m with SGD and only a spike guard, truncation's actual loss of size costs, and
  whether a constant step shrink of the same size costs.
- **Cannot:**
  - whether size would matter for a larger shrink, or for an optimiser more sensitive to step size;
  - whether per-arm learning-rate tuning would change anything (every arm uses full backprop's rate, as in S2 and S7);
  - other models;
  - k = 4.

## 2. What each outcome changes in the paper

| outcome | paper change |
|---|---|
| TURN | §5 adds that under SGD with only a spike guard the turn also carries the cost (S24). It says why: in training, truncation shortened the step only to 0.70 (k = 1) and 0.84 (k = 2), and a constant step that much shorter cost about φ_fullm. The abstract's "as Adam's insensitivity to the gradient's scale predicts" becomes "under AdamW, and under SGD, where truncation shortened the step only to 0.70 of its length". §8's takeaway drops "under Adam" and adds the same qualifier. If φ_fullm,1 > m_1 (a shorter step does cost, yet the shrink did not), §5 reports that tension instead of the reason. |
| SHRINK or PARTIAL, AS-STEP at the deciding k | §5: under SGD the lost size costs, about as much as an equally smaller learning rate, so whether the size matters depends on whether the optimiser turns size into step length. The abstract keeps the AdamW qualifier and adds that under SGD the size costs like a smaller learning rate; §8 says the same. |
| SHRINK or PARTIAL, BEYOND-STEP | as above, but the shrink costs more than its step length explains, so its step-to-step variation matters too. |
| MIXED | §5 reports per-k labels, shares and step readings; the abstract keeps the AdamW qualifier and says the SGD split is mixed. |
| INVALID or INCOMPLETE | appendix only; §5 and the abstract unchanged (the SGD split stays untested). |

In every case, App. G gains S24 with the pilot's shrink sizes, Table 4 gains an S24 row, and §1 counts twenty-four
studies.

## 3. Checked before the freeze

- **Tiny-model run on CPU** (`TINY=1`, random tiny Parcae) with the frozen rates:
  - the selftest passes:
    - v174's own selftest;
    - the three constructions at k = 1 and 2;
    - below each clip norm, trunc and shrink keep |g_k| and turn keeps |g_R|;
    - full backprop through the SGD loop gives Mbar = 1 and step 1;
    - each arm's applied update has its construction's direction and length on every step;
    - the floor runs through v170's SGD patch, and the patch and the clip are restored after;
  - one run each of `full`, `floor`, `shrink1`, `fullm1` and `fullm2`.
- **The real-model selftest** passed on all three pilot lanes.
- **Synthetic worlds** come out as built:
  - TURN, SHRINK with AS-STEP, SHRINK with BEYOND-STEP, PARTIAL, MIXED;
  - a diverged shrink run (seed replaced, verdict TURN);
  - INVALID (floor gap 0.01), INCOMPLETE (3 seeds);
  - a world without C1 rows, as this study will run ("C1: not run").
- **Imported drivers unchanged:** v239 7f5262e51b75, v170_sgd 4ce6128b21ed, v174 f3cdba92e749, v163 db639c411892,
  v162 0ae4891d7abc (LF), all already on origin.

## 4. Predictions (mine, at the freeze, knowing the pilot)

- **k = 1:** TURN 0.75, PARTIAL 0.20, SHRINK 0.05.
- **k = 2:** TURN 0.80, PARTIAL 0.15, SHRINK 0.05.
- **Overall:** TURN 0.65, MIXED 0.12, PARTIAL 0.10, SHRINK 0.03, INVALID or INCOMPLETE 0.10. The last is mostly the
  risk that spike-only clipping makes some arm unstable; the turn arm takes larger steps than truncation.
- **The matched-step controls:** φ_fullm,1 ≤ 0.02 with probability 0.8.

## 5. Cost and runs

60 runs (6 seeds × 10 cells).
- Estimated minutes on a T4: two-gradient arms 38, full backprop and the matched-step controls 29, floor 9.
- Four kernels × 2 T4s, about 4.1 h per lane plus setup and the selftests, so about 18 session-hours of Kaggle quota.
- The pilot used about 2 session-hours.

## Hashes

sha256 prefixes (LF).
- `v241_sgd_split.py`: e505057ac886
- `build_kaggle_v241.py`: 4312eeda1d44
- Imports as in §3.
