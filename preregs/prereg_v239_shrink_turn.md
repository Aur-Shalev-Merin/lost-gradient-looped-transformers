# PREREG v239: which part of a truncated gradient costs training, its shrink or its turn?

**Status: FROZEN** (this commit), before any scored run. The freeze is pushed to origin (branch `v239-freeze`) before
the first kernel. Approved by the author.

**Code:** sha256 prefixes in the "Hashes" section at the end. In the paper this is study S22.

## 0. Why

The looped-truncation paper (draft v18.0, §5) ends on an untested reading. The lost gradient mixes two things: how much
smaller the truncated gradient is (size ratio r = |g_k| / |g_R|) and how far its direction is off (cosine); for one
batch, lost² = 1 + r² − 2r·cos. On Parcae-140m at R = 32 (Table 6, batch-averaged):
- at k = 4, r = 0.79 and cos = 0.93: lost gradient 0.40, cost 0.024 of the loop's benefit;
- at k = 1, r = 0.46 and cos = 0.61: lost gradient 0.81, cost 0.22.

The reading is that a shrink costs less than a turn. Under the paper's recipe, two things would make a shrink cheap:
Adam's per-weight rescaling, and global clipping at norm 1, which is active on nearly every full-backprop step. Neither
was tested. S8 (v174) was uninformative, and S11's direction-after-Adam curve tied the lost-gradient curve. This study
removes one component at a time and measures the cost of each.

**Known before this freeze (disclosed).**
- **S8's arms nearest a pure turn.** S8 trained mixes g = g_R + λ(g_k − g_R), rescaled to |g_R|, under the same
  pipeline. Its arms closest to λ = 1 cost 0.166 (k = 1, λ 0.924; per-seed SD 0.046 over 9 seeds) and 0.0215 (k = 4,
  λ 0.94; SD 0.0054). Plain truncation cost 0.224 and 0.024 in S1, which used a different pipeline (v163's train_cell
  with a GradScaler). So a turn alone may carry most of the cost, but S8 had no λ = 1 arm, no shrink-only arm, no
  k = 2 arm, and no truncated arm in its own pipeline.
- **The shrink arm under Adam.** On a tiny random model, the shrink arm's Adam step direction agreed with full
  backprop's at 1.0000 over 3 steps. Adam is invariant to a constant rescale of the gradient, so the shrink arm differs
  from full backprop only through the step-to-step variation of r, and through clipping where |g_k| < 1 < |g_R|. It is
  expected to cost little by construction. It is a control for whether a varying shrink matters under this recipe,
  not a test that could surprise easily.

## 1. Design (LOCKED)

**Model and recipe:** S8's (v174, imported unchanged). Parcae-140m at R = 32, GSM8K question-and-answer text, 600
steps of batch 8 × 512, AdamW (0.9, 0.95), eps 1e-8, no weight decay, 20-step warm-up then cosine decay to 10%,
global-norm clip 1.0, fp16 autocast with manual loss scaling (a step with a non-finite gradient is skipped and the
scale halved).

**Arms** (for k = 1, 2 and 4, each step computes g_R and g_k on the same batch and state draw, builds g, clips it to
norm 1 and steps AdamW at lr 3e-4):
- `trunc`: g = g_k, plain truncation through this pipeline;
- `turn`: g = g_k · |g_R| / |g_k|, truncation's direction at the full gradient's length (S8's mix at λ = 1);
- `shrink`: g = g_R · |g_k| / |g_R|, the full gradient's direction at the truncated gradient's length.

Plus `full` (S8's mix arm at λ = 0, g = g_R, lr 3e-4) and `floor` (the S2/S5 no-loop-gradient baseline, v163
train_cell with k = 0, lr 9e-4), both through v174.run_scored unchanged. 3e-4 is full backprop's rate; in S2 every
truncated cell on this model chose it too.

**Seeds:** 2391101–2391106, all run. The first 5 valid seeds in that order are scored. A seed is valid if its full
run has at most 10 non-finite steps and a finite test loss (the paper's exclusion rule). If fewer than 5 are valid,
2391107 and 2391108 are run. Fewer than 4 valid seeds: INCOMPLETE.

**Quantity:** cost φ = (L_arm − L_full) / (L_floor − L_full), on GSM8K test loss, paired by seed, as the mean over the
scored seeds.

**Gate:** the floor gap L_floor − L_full is at least 0.03 nats, with its 90% t-interval lower bound above 0 (as S11
and S18). If it fails: INVALID.

**Per-k label**, with margin m_k = max(0.015, 0.2 · φ_trunc,k), checked in this order:
- TURN if φ_turn ≥ φ_trunc − m_k and φ_shrink ≤ m_k: the turn alone reproduces the cost, and the shrink alone adds
  almost none;
- SHRINK if φ_shrink ≥ 0.5 · φ_trunc: the shrink alone gives at least half the cost;
- PARTIAL otherwise.

**Verdict:** decided at k = 1 and k = 2. The label is TURN, SHRINK or PARTIAL if both k get it, and MIXED (with the
per-k labels) otherwise. k = 4 is reported but not scored, because its cost (about 0.02) is too small for the margins.

**Margins.** In S8, the per-seed SD of the cost of the near-turn arms was 0.012–0.046 at k = 1 and 0.005 at k = 4.
Over 5 paired seeds that gives a standard error of about 0.005–0.02. With φ_trunc near 0.22 and 0.10, m_1 ≈ 0.044
and m_2 ≈ 0.02.

**Reported, not scored:**
- the k = 4 labels;
- each arm's per-seed costs and the turn and shrink shares (φ_turn / φ_trunc, φ_shrink / φ_trunc);
- the truncated arm's costs against S1's 0.224 / 0.100 / 0.024;
- each arm's mean Adam-update agreement with a full-backprop shadow (Abar, logged by v174);
- non-finite steps per run.

**What the design can and cannot show.** The arms separate size from direction under the paper's recipe (AdamW with
clipping). TURN would say that under this recipe the cost of truncation is carried by the direction. It would not say
whether Adam or clipping makes the shrink cheap, since this study changes neither. It covers one model, one task,
and k = 1, 2, 4.

## 2. What each outcome changes in the paper

| outcome | paper change |
|---|---|
| TURN | §5's untested reading becomes S22. On Parcae-140m under the paper's recipe, the truncated gradient's direction at full length costs as much as truncation, and the full gradient at the truncated length costs almost nothing, with the numbers. The abstract gains one sentence: what costs is the change of direction, not the lost size, which is why the lost gradient's size is a poor guide to cost. The limits stay explicit: one model, AdamW with clipping. The "least certain" takeaway becomes a qualified finding. |
| SHRINK | §5 says the reading fails on Parcae: the full gradient at the truncated length costs at least half of truncation's cost, so size matters under this recipe. The "least certain" takeaway is removed. |
| PARTIAL or MIXED | §5 reports the turn and shrink shares per k and says the reading holds only in part. |
| INVALID or INCOMPLETE | Reported in the appendix; §5 unchanged. |

In every case, App. G (S8) is extended to hold S22, and Table 4 gains an S22 row.

## 3. Checked before the freeze

- **Tiny-model run on CPU** (`TINY=1`, random tiny Parcae; seconds per run):
  - the selftest passes: S8's own selftest, the three constructions at k = 1, 2 and 4 (trunc equals g_k; turn has
    g_k's direction and |g_R|'s length; shrink has g_R's direction and |g_k|'s length), the full arm's agreement of
    exactly 1, and each arm's construction used on every step of a 3-step run;
  - scored cells of every arm type ran and wrote v174's output format.
- **Synthetic worlds** for the analysis, built with chosen costs, come out as built: TURN, SHRINK, MIXED (k = 1
  TURN, k = 2 PARTIAL), PARTIAL, INVALID (floor gap 0.01), INCOMPLETE (3 seeds), and an excluded seed (full run with
  15 non-finite steps, replaced by the next seed in order).
- **Imported drivers unchanged:** v174 is unchanged since its freeze commit 095bd38; v162 and v163 match the hashes
  their own preregs record.
- **In the kernels,** every GPU lane runs the selftest on the real model before its scored runs, and an assertion
  stops that lane.

## 4. Predictions (mine, at the freeze)

- **k = 1:** TURN 0.45, PARTIAL 0.50, SHRINK 0.05. S8's λ 0.924 arm (0.166 against S1's 0.224) suggests that the turn
  alone gives about 75–85% of the cost, close to the margin.
- **k = 2:** TURN 0.60, PARTIAL 0.35, SHRINK 0.05.
- **Overall:** TURN 0.35, PARTIAL 0.25, MIXED 0.30, SHRINK 0.03, INVALID or INCOMPLETE 0.07.
- **The shrink arm** costs at most 0.01 at every k: 0.85.

## 5. Cost and runs

66 scored runs (6 seeds × 11 arms). Estimated from S8's scored runs: two-gradient arms 38 min, full 29, floor 9 on a
T4. Four kernels × 2 T4s, about 5 h each including the selftests, so about 20 session-hours of Kaggle quota in total.
Kaggle accounts are assigned at push by free slots, and the result doc records them.

## Hashes

sha256 prefixes (LF). The first two are new; the others are imported unchanged.
- `v239_shrink_turn.py`: 7f5262e51b75
- `build_kaggle_v239.py`: f1e114420164
- `v174_mix_driver.py`: f3cdba92e749 (unchanged since 095bd38)
- `v163_parcae_damped_trunc.py`: db639c411892
- `v162_parcae_trunc_train.py`: 0ae4891d7abc
