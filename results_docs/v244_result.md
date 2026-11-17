# v244 RESULT (unregistered, descriptive): where truncation's shrink falls under SGD

**What it is.** A check made after the fact for a caveat in the paper's §5 (draft v18.4). No threshold, no verdict.

**Why.** A review of the repo's earlier Adam-versus-SGD work raised a point about S24 and S25 (v241, v243). Their
turn/shrink split rescales the whole gradient, but truncation changes only part of it. So under SGD the turn-only arm
may also take a larger step on the parameters truncation leaves unchanged.

**Run.**
- Commit a8be9b2 (before the run). Script `v244_group_shrink.py` e277f204fc46, builder `build_kaggle_v244.py`, kernel
  aurmerin/astrocyte-v244, pushed 2026-11-16 20:31 UTC.
- v241's setting G for full backprop, unchanged:
  - SGD with momentum 0.9 and learning rate 0.1, with warm-up and cosine decay;
  - global clipping at norm 5;
  - S8's recipe on Parcae-140m at R = 32 (600 steps).
- Two new seeds, 2441101 and 2441102, one per T4. Combined v241 driver sha 3b446a33effc.
- Every 10 steps the script also computes g_1 and g_2 on the same batch (not used for training).
- **Parameter split:** a tensor counts as loop-affected if any element of g_1 differs from g_R on the first 3 training
  batches. Every probe then checks that the other tensors' gradients really are equal.
- Summary: `OUT_DIR=results_v244 python v244_group_shrink.py --summary results_v244` (`results_v244/v244_summary.txt`).

## Result

**Seed 2441101 diverged** after a few steps (583 non-finite steps, NaN test loss), so it has only 2 probes. This is
the same instability at learning rate 0.1 that S24 and S25 saw. All figures below come from **seed 2441102**: 59
probes, 4 non-finite steps, test loss 1.421. Medians are over steps 100–600.

**Parameters truncation leaves unchanged:** 40.0M of 144.3M (20 of 61 tensors). They are:
- `transformer.C`, which maps the loop's output into the coda;
- both coda blocks;
- `ln_f`;
- `value_embeds.5`, the coda's value embedding.

Their gradient under truncation equals full backprop's. r_rest is 1.0000, and the largest difference is 1.1e-4 of the
largest gradient entry, at the level of fp16 noise. They carry **25%** of |g_R|².

**Shrink ratio |g_k| / |g_R| and cosine (interquartile range in brackets):**

| k | all parameters | loop-affected (104.4M) | cos, all | cos, loop-affected |
|---|---|---|---|---|
| 1 | 0.699 [0.684, 0.717] | 0.560 [0.542, 0.573] | 0.837 | 0.789 |
| 2 | 0.833 [0.819, 0.847] | 0.764 [0.750, 0.783] | 0.931 | 0.909 |

The all-parameter ratios match S24's pilot (0.702 and 0.836).

**Step length relative to full backprop:**

| arm | loop-affected, k = 1 / 2 | after the loop, k = 1 / 2 |
|---|---|---|
| truncation | 0.56 / 0.76 | 1.00 / 1.00 |
| turn only | 0.80 / 0.92 | **1.43 / 1.20** |
| shrink only | 0.70 / 0.83 | 0.70 / 0.83 |

**By module, at k = 1** (shrink ratio, with share of |g_R|² in brackets):
- core block 0.564 (0.55)
- prelude 0.525 (0.17)
- coda 1.000 (0.21)
- `ln_f` 1.000 (0.03)
- `C` 1.000 (0.01)
- token embedding 0.973 (0.003)

**A correction to an earlier count.** S8's pilot (`docs/v174_pilot_result.md`) reported 29.3M "loop-affected"
parameters. That came from an element-wise mask on one validation sequence, which misses embedding rows for tokens not
in that sequence. The per-tensor count here, 104.4M, is the one to use.

## What it means

Under SGD, S24 and S25's "turn" is more than a change of direction. The turn-only arm also moves step size away from
the loop and toward the layers after it: 1.43 times full backprop's step after the loop against 0.80 on the loop, at
k = 1. Adam's per-weight scaling would mostly undo this, but SGD does not. So the SGD turn-only cost mixes the cost of
the direction with this reallocation. The paper's §5 and §7 now say so (v18.4). The AdamW results (S22, S23) are
unaffected.
