# PREREG v201: the lost gradient of averaged gradients, against batch size, at the start and middle of fine-tuning

**Status: FROZEN** (this commit), before any measurement. The freeze is pushed to origin before the first run.

**Code (sha256 prefixes, LF):**
- `v201_batch_lost.py` 671468417237 (measurement, selftests, analysis);
- `build_kaggle_v201.py` 8a8773c78e1a;
- study drivers imported unchanged: v162 0ae4891d7abc and v163 db639c411892 (S5), v167 b6fd416e48f2 (S6),
  `v200_ifm_trunc.py` bd9ee68769e0 and `ifm_looped.py` 502739cdccc8 (S11, as frozen).

**Why.** Point 2 of the outside review of 2026-10-15, plus a review note on training stage.

The paper reports the median over held-out sequences of the per-sequence lost gradient, |g − g_k| / |g|. Two other
quantities are the same ratio for averaged gradients:
- Aicher et al.'s relative bias, which is what their convergence bound prices;
- what an optimiser's update sees.

Data already on disk says these can differ:
- §2 / Table 5: averages over batches of 8 read 0.05–0.07 higher than the medians in S5's settings.
- v171 / v171b, frozen Parcae-140m, 24 batches of 8 training sequences:
  - unclipped, the average loses about 100% at R = 16 and 32, driven by one batch whose gradient norm is about
    25,000;
  - with training's clip it loses 0.96 / 0.72 / 0.41 at k = 1 / 2 / 4 (R = 32).
  - On SmolLM2 (A) the figure is 0.35 at k = 1 either way.

Clipping depends on the training stage. On the frozen models nearly every batch gradient is 3–7 times the clip
limit. During fine-tuning, only 14–30% of steps were clipped in a Parcae check. So we measure at two stages.

## 1. Models, stages, measurement (LOCKED; `v201_batch_lost.py`)

**Models.** The four whose truncation cost was trained, at their cost settings, unblended (a = 1), with k ∈ {1, 2, 4}:

| model | R | driver | full-backprop LR (study) | parameters measured |
|---|---|---|---|---|
| Parcae-140m | 32 | v163 (S5) | 3e-4 (v166) | all, 144.3M |
| SmolLM2 retrofit (A) | 8 | v167 (S6) | 1e-4 (v167) | all, 163.5M |
| Part II, sampled loop count (pln5) | 5 | v200 (S11) | 1e-4 (v200) | embedding frozen, 205.3M |
| Part II, fixed 5 loops (r5) | 5 | v200 (S11) | 1e-4 (v200) | embedding frozen, 205.3M |

**Stages.**
- **Stage 0:** the trained model as released or converted. Every measurement in the paper describes this stage, which
  is the first steps of fine-tuning.
- **Stage 300:** the same model after 300 of 600 steps of full-backprop fine-tuning with its study's recipe:
  - AdamW (0.9, 0.95), weight decay 0;
  - 20-step warm-up, then cosine to 10% over 600 steps;
  - clip 1.0; fp16 autocast with the cache off, plus GradScaler;
  - batch 8 × 512, the study's LR, data order and state draws from seed 2011.

  This follows the full-backprop trajectory; the truncated arms' own trajectories are not followed. Most of
  fine-tuning looks like this stage.

**Sequences.** The first 512 packed GSM8K training sequences under permutation seed 201, as 64 batches of 8. These are
training data, what the optimiser sees, as in v171. At stage 300 most of them have been trained on once, as the
batches in the second half of a 600-step run (about two epochs) have.

**Per batch b** (weights frozen, fp32, train mode as in training):
- G_b is the gradient of the batch-mean loss with backprop through all R loops.
- T_{k,b} uses the same batch and the same state-init draw (seed 201000 + b), with backprop through the last k loops.
- **Raw lost gradient** of a group S of batches: |Σ_S (G_b − T_{k,b})| / |Σ_S G_b|.
- **Clipped:** each G_b and T_{k,b} is first scaled to norm ≤ 1.0, as training's `clip_grad_norm_` does, then the same
  ratio.

**Batch sizes.**
- B = 8: each of the 64 batches.
- B = 64: 8 groups of 8 consecutive batches. Adam with β₁ = 0.9 at batch 8 averages about 80 sequences.
- B = 512: all 64 batches.
- B = 1: the first 64 sequences, one pass each with its own state-init draw (seed 301000 + s); raw only.

The summary statistic is the median over groups; B = 512 is a single value. Every batch's |G_b| and |T_{k,b}| is
logged, so the fraction clipped is known.

**Selftests** (each process and stage, before measuring; a failure stops that measurement):
1. A repeated full-backprop pass on batch 0 with the same seed reproduces itself (relative difference < 1e-3). This
   catches draws that are not shared between arms.
2. The raw lost gradient at k = 1 on batch 0 exceeds 0.001, so truncation acts.
3. The parameter count matches the table.
4. Stage 300 only: the fine-tune had ≤ 10 non-finite steps and a finite last loss.
5. B = 64 and B = 512 come from two passes over the same batches, and their B = 8 values must agree. This is checked in
   the analysis.

The local TINY test (`TINY=1`) checks the running sums against direct sums of stored vectors.
`--selftest-analysis` checks the outcome mapping on synthetic worlds.

## 2. Outcomes and what each changes in the paper (LOCKED; `v201_batch_lost.py --analyze`)

**Primary.** For each model and k, the gap is the clipped B = 64 median at stage 300 minus the per-sequence (B = 1)
median at stage 0. It measures how far the paper's kind of number (frozen model, per sequence) is from the share of
the averaged update that truncation removes during most of fine-tuning.

- **MEDIAN-FAIR:** |gap| ≤ 0.05 at every k on all four models. §2 gets one sentence and Appendix I a table: the
  per-sequence median of the frozen model matches the lost share of the averaged training update within 0.05.
- **MEDIAN-UNDERSTATES (models listed):** gap > +0.05 at some k, and none below −0.05.
  - §2 states that the per-sequence median understates what truncation removes from the averaged update on those
    models, by up to the largest gap.
  - Table 1 keeps the defined statistic, and Appendix I gives the update-level values.
  - §4 notes that the cost curves were fitted on the per-sequence statistic.
- **MEDIAN-OVERSTATES:** gap < −0.05 at some k, and none above +0.05. The same changes, with the sign reversed.
- **MIXED:** both. The same changes, model by model.

0.05 is about twice the sampling error of a median over 16–64 sequences (0.02–0.05, §2).

**Secondary** (reported, no verdict):
- The aggregation gap at each stage: clipped B = 64 minus B = 1, at stage 0 (the first steps) and at stage 300.
- Plateau at stage 300: |clipped B = 512 − clipped B = 64| ≤ 0.03 means it is settled within Adam's window;
  otherwise the window matters.
- Raw vs clipped at B = 512 (Aicher's expected-gradient quantity vs the clipped update). If raw exceeds clipped by
  more than 0.10 at some k, the paper says that the expected-gradient relative bias is dominated by rare
  large-gradient sequences on that model, and that training's clip limits their pull.
- Training effect on the per-sequence statistic: the stage-300 B = 1 median minus the stage-0 one. The paper's
  "ρ changes with training" note covers ρ only.
- After the fact relative to S11, descriptive only: whether the two Part II models differ more at clipped B = 64
  (stage 300) than in their per-sequence medians. No cost curve is refitted.

**Not the test.** S5, S6 and S11 were scored on the per-sequence statistic as pre-registered, and are not re-scored
with these numbers.

## 3. Predictions (mine, at freeze)

- **Parcae:** understates at stage 0 (v171b). At stage 300, understates 0.55, fair 0.45.
- **SmolLM2:** fair at both stages, 0.8.
- **Part II:** fair at both stages, 0.6. No sequence's signal grows on them at R = 5, and growing sequences drive
  v171's outliers.
- **Primary:** MEDIAN-FAIR 0.40, MEDIAN-UNDERSTATES 0.50, MEDIAN-OVERSTATES 0.03, MIXED 0.07.
- **Raw B = 512 exceeds clipped by more than 0.10:** Parcae at stage 0, 0.8; at stage 300, 0.4.

## Amendment 1 (2026-10-16): Part II parameter check

Made after both Part II jobs stopped at selftest 3, before measuring anything; no Part II data exist. The Parcae
and SmolLM2 outputs were checked for validity only, not read for results.

**What happened:** the frozen code expected 205.3M trainable Part II parameters. That figure came from the v200 plan's
rounded subtraction, 303.6M − 98.3M. S11's own run records give 204,882,432 with the embedding frozen; the embedding
is 98,697,216.

**Change:**
- The expected value is now 204.9M (same 0.2M tolerance).
- The actual count is now logged before the check.

**Unchanged:** the parameter set itself (`v200_ifm_trunc.load_model`, embedding frozen, as S11), and nothing else in the
code or this plan. The Part II kernel is re-run on the amended code.

## 4. Cost and runs

There are three kernels, each on T4 × 2. Kernels are built by `build_kaggle_v201.py`.

| kernel | account | GPU 0 | GPU 1 |
|---|---|---|---|
| parcae | aurshalevmerin | stage 0 | stage 300 |
| retro | aurshalevmerin (dataset retro-smollm2-healed) | stage 0 | stage 300 |
| ifm | aurmerin | pln5, both stages | r5, both stages |

Estimates from v171's timings (fp32 batch passes of 5.6–5.8 s at R = 32 / 8) and v200's (Part II):
- about 55 min per stage on Parcae and SmolLM2, plus the fine-tunes;
- about 35 min per GPU on Part II;
- about 5 T4-hours in all.
