# v200 plan notes: truncated-loop fine-tuning of the Part II looped LMs (apparatus and pilots)

*2026-10-14. Status: driver and builder written and tested (TINY end to end on CPU; real-model selftest on the laptop
GPU). Pilot kernels built for both models, **not pushed**. Nothing committed. The locked predictions and the
pre-registration are separate (coordinator).*

Files: `v200_ifm_trunc.py` (driver; sha 0618b3417d74 with the port), `build_kaggle_v200.py` (builder), the port
`ifm_looped.py` unchanged since v198 (file sha 502739cdccc8, the version v198 checked against their reference code on
both models; the selftest refuses any other version).

## 1. Recipe as implemented

The v162-v167 recipe, with the model's own text format:

- **Models:** IFM/LoopedLM-P2-huginn-s-fixed-pln5 (revision c581f369) and -fixed-r5 (17cdf78e), the v198 revisions.
  `--variant` picks the model, and the driver checks the downloaded manifest against it.
- **Data:** GSM8K question + "\n" + answer. Each example is BOS + text + EOS, as the model was trained, and attention
  stays within an example (their `multi_segments`).
  - val: the last 473 train examples (150 chunks);
  - test: the test split (434 chunks);
  - train: the other 7,000 examples, 4 epochs shuffled by `rng(seed)`, packed into 512-token chunks.
- **Optimisation:** 600 steps of 8 x 512 tokens. AdamW with betas (0.9, 0.95), eps 1e-8 and weight decay 0, as
  v162-v167; the model's own pretraining used (0.9, 0.98) and 0.1. 20-step linear warm-up, then cosine to 10%; global
  gradient clip 1.0.
- **Precision:** fp16 autocast with the weight-cast cache OFF, plus GradScaler. The forward is one autocast region,
  so the cache-off setting is what keeps the v161 bug away; the selftest checks it.
- **Loop:** R = 5 (the trained mean and their evaluation depth), a = 1. For backprop through the last k loops, the
  first 5 − k loops run under `no_grad` and the state is detached, as in their training. Kept loops are checkpointed
  per block.
- **Initial state:** during training, a fresh N(0, 1/d) draw (as their training) seeded by (seed, step). Every arm of
  a seed therefore sees the same data and the same states, which makes the arms paired. Evaluation uses the model's
  deterministic eval-mode state, identical for every arm.
- **Evaluation:** val and test loss after training, all 5 loops, no gradient, fp16. Mean next-token CE per 512-token
  chunk, averaged over chunks, as v167. Also reported: `rho_end` (median over 4 val chunks, fp32), the step count of
  non-finite gradient norms, peak memory and time.
- **Cells:** k = 0 (floor), 1, 2, 4, 5 (full). Cost is φ_k = (L_k − L_full) / (L_floor − L_full), per seed, on val
  and test.
- **Exclusion rule (v165):** a seed is valid iff its full run has ≤ 10 non-finite steps, a finite test loss and did
  not diverge. The first 5 valid seeds are used.
- **Analysis:** `--analyze` prints every run, the LR-pilot table (flagging an optimum at the grid edge), the floor gap
  L_floor − L_full with its 90% t-interval (v167's gate: ≥ 0.03 with the interval above 0), and per-cell φ with its
  mean, s.e., per-seed values and the s.e. projected for 5 seeds.

One implementation change from the port: the loss uses shifted labels with `ignore_index`, as v162/v167, instead of
`ifm_looped.ce`. It has the same value (the selftest checks this), but avoids a 1 GB copy of the fp32 logits at
batch 8.

## 2. What the floor means on IFM

At k = 0 no loop is kept, so gradient reaches only what sits after the loop's output:
- **Trains (124.7M of 303.6M parameters):** the coda block, the final norm and the head.
- **Frozen by lack of gradient:**
  - the two loop blocks and the input injection;
  - the prelude and the embedding. Their only path to the loss is the injected input inside the loops; the prelude
    output feeds nothing else.

This is the same floor as v167's SmolLM2 retrofit (untied head; prelude, adapter, core and embedding get no gradient).
It differs from Parcae's floor, where the embedding still trains through the tied head. The floor is therefore "no
credit to anything that acts through the loop", and on this model that includes the input side.

## 3. Selftest

Every Kaggle process runs it before training and stops before any training if a check fails. It checks:

1. the training forward at k = R equals the model's own forward (fp32);
2. per-block checkpointing changes no gradient (fp32, k = 2 and 5);
3. fp16-autocast (cache off) truncated gradients match fp32 per parameter group (embed, prelude, injection, core,
   coda, head) at k = 1, 2, 4, 5, using v167's criterion (core < 0.5 and ≤ 3 x coda + 0.02), plus the injection and
   the loss;
4. **the negative control (v162):** with the cache on, the loop's gradient must be lost (core > 0.5), or the check
   proves nothing;
5. the floor: k = 0 gives embed, prelude, injection and core exactly zero gradient, and coda and head nonzero; k = 1
   reaches every group;
6. the training state draw is reproducible;
7. the port file is the v198-validated one.

**Real model, laptop GPU (pln5, batch 1, seconds):** PASS.

| check | result |
|---|---|
| decomposition | 2.684189 = 2.684189 |
| checkpointed vs plain | 0 in every group |
| fp16 vs fp32 (k = 1 / 2 / 4 / 5), largest group difference | 0.0018 / 0.0018 / 0.0017 / 0.0017 (loss 2.68403 vs 2.68402) |
| negative control, cache on (k = 2): core / injection | 0.995 / 0.996 (bug reproduced); other groups ≤ 0.0018 |
| floor k = 0 gradient norms: embed, prelude, injection, core / coda, head | 0, 0, 0, 0 / 2.22, 1.43 |

fp16 noise here is far below v167's retrofit (0.22-0.60 relative), so the check has a lot of margin. r5 has the same
architecture; its selftest runs first in every r5 kernel.

**TINY=1 on CPU:** selftest, the LR pilot in two parts, spread cells, both variant names and `--analyze` all ran. The
spread kernel's LR auto-selection was tested on fake pilot outputs: it stops on a grid-edge optimum and otherwise fills
in the LRs and reuses the controls.

## 4. Timings

The laptop GPU cannot time a training step: batch 2 already needs 4.4 GB on its 4 GB card, and Windows then spills
into system memory. The estimates are a FLOP count calibrated on real T4 runs:
- **FLOPs per step at batch 8 x 512:** k = 0 5.7 TFLOP, k = 1 7.5, k = 2 9.0, k = 4 11.8, k = 5 13.2. The count
  covers the forward pass, the per-block recompute and the backward pass through the kept part, with attention scores
  included.
- **Calibration:** v167's T4 pilot (fp16, batch 8 x 512, the same recipe, a model of similar compute) ran its floor in
  350 s and full backprop in about 890 s. Both imply 8.3-8.5 TFLOPS effective.
- **Cross-check:** v198's T4 fp32 batch-1 passes ran at about 2.7 TFLOPS, about a third of that, consistent with
  fp16 tensor cores at batch 8.

The two models have identical architecture and cost, so these per-run times apply to both.

| cell | T4 time per 600-step run (incl. load, val + test eval, rho_end) |
|---|---|
| k = 0 (floor) | 7.5 min |
| k = 1 | 9.7 min |
| k = 2 | 11.4 min |
| k = 4 | 14.8 min |
| k = 5 (full) | 16.4 min |

These carry roughly ±30% uncertainty; IFM's wider matrices may run faster than v167's. Expected peak memory is about
10-11 GB of the T4's 15 GB at batch 8, mostly AdamW state (2.4 GB) and the fp32 logits and softmax of the 64k
vocabulary. Each run records its true peak.

| plan | runs | T4-hours | wall time |
|---|---|---|---|
| LR pilot, per model | 6 (k = 0 and 5 at 3 LRs, seed 9001) | 1.2 | about 45 min (T4 x2) |
| spread pilot, per model | 7 (full and floor at seeds 9002-9003; k = 1 at 9001-9003) | 1.3 | about 47 min |
| **pilots, both models** | 26 | **5.0** | 4 kernels, about 1.6 h in all if two run at a time |
| scored study, per model | 35 (k = 0, 1, 2, 4, 5 x 7 seeds) | 7.0 | 2 kernels of about 1.9 h |
| **scored study, both models** | 70 | **14.0** | 4 kernels in parallel (2 per account), about 1.9 h |

## 5. Pilot kernels to launch (built in the repo, gitignored like the other kernel directories)

| order | kernel | account | what it runs |
|---|---|---|---|
| 1 | `astrocyte-v200-pln5-pilot-lr` | aurmerin | selftest on each GPU, the frozen model's val/test loss, k = 5 and k = 0 at LR 1e-4 / 3e-4 / 9e-4 (seed 9001) |
| 1 | `astrocyte-v200-r5-pilot-lr` | aurshalevmerin | the same on r5 |
| 2 | `astrocyte-v200-pln5-spread` | aurmerin | full and floor at seeds 9002-9003, k = 1 at 9001-9003 |
| 2 | `astrocyte-v200-r5-spread` | aurshalevmerin | the same on r5 |

Push each with `kaggle kernels push -p kaggle-kernel-v200-<model>-<tag> --accelerator NvidiaTeslaT4`. The two LR
pilots can run at once (one per account). Each spread kernel needs its model's LR pilot finished:
- it reads that output (`kernel_sources`);
- it takes, per arm, the LR with the lowest val loss among valid runs (the v167 rule);
- it reuses the seed-9001 controls at those LRs.

If either choice sits at the edge of the grid, it stops without training. Either extend the grid first (a second LR
pilot, as v167 did) or rebuild with explicit LRs:
`python build_kaggle_v200.py spread --variant V --lr-floor F --lr-trunc T` (or `--allow-edge`).

Analysis after download: `python v200_ifm_trunc.py --analyze <pilot-lr output> <spread output>`. Each spread kernel
also analyses itself, since the reused controls are copied into its output.

The scored study is built after the prereg, with the locked LRs and seeds, for example:
`python build_kaggle_v200.py study --variant huginn-s-fixed-pln5 --lr-floor F --lr-trunc T --seeds 2001101,...,2001107
--kernels 2 --accounts aurmerin,aurshalevmerin` (the same for r5; `--dry-run` prints the plan). The seeds 2001101-07
are only a default.

## 6. For you to decide

- **LRs:** separate floor and truncated/full LRs, chosen by val loss (v167's rule, built in). Only k = 5 and k = 0
  are piloted; truncated cells use the full arm's LR, as v167. If an optimum lands at the grid edge, decide whether to
  extend the grid.
- **The floor gap is unknown.** On IFM the floor still trains the head and the coda (124.7M parameters), and the
  frozen model's GSM8K val loss is about 2.68. If L_floor − L_full is small, φ will be noisy (v167's gap was 0.08,
  Parcae's 0.18). The LR pilot shows the gap at seed 9001, and the spread pilot shows its seed spread.
- **Seed count:** before committing 14 T4-hours, use the spread pilot's s.d. of φ at k = 1 and its projected 5-seed
  s.e. to check that the study can resolve the smallest gap between the locked predictions at k = 1. Per your
  message, pln5's direction-after-Adam curve and Parcae's "k alone" differ by only about 0.04 there.
- **Optimiser settings:** the recipe keeps v162-v167's (0.9, 0.95) and weight decay 0, not the model's own
  pretraining settings. Comparable with Parcae and SmolLM2, as intended; stated here so it is a choice, not an
  accident.
- **Quota:** 14 T4-hours for the study plus about 5 for the pilots, across both accounts.
