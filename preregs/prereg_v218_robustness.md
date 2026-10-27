# PREREG v218: robustness of the fine-tuning and measurement results (step 600, unrelated text, state draws)

**Status: FROZEN** (this commit), before any v218 measurement. Pushed to origin before the first run.

**Code (sha256 prefixes, LF):**
- `v218_robustness.py` 6aeb0b6c25c5 (measurement and analysis);
- `build_kaggle_v218.py` fba6da46528d;
- imported unchanged: `v201_batch_lost.py` 4fe2fff13a39, `v162_parcae_trunc_train.py` 0ae4891d7abc,
  `v163_parcae_damped_trunc.py` db639c411892, `v200_ifm_trunc.py` bd9ee68769e0, `ifm_looped.py` 502739cdccc8.

## Why

Three questions from outside reviews 7 and 8 that the paper (arXiv:2610.28871) leaves open:

1. **The fall during fine-tuning rests on one checkpoint** (step 300 of 600). Does it continue, hold or reverse by the end
   of the schedule? (review 7, Q3)
2. **The held-out problems are GSM8K text, like the fine-tuning data.** Does the lost gradient fall on unrelated text?
   (§4 and §7: "we did not test other text"; review 8, minor 8)
3. **Parcae and the Part II models start the loop from a random state,** drawn once per sequence. How much does the lost
   gradient vary across draws? (§7: "we did not measure how much the lost gradient varies across draws"; review 7, Q6)

## Design

- **Models:** Parcae-140m (R = 32) and the Part II models pln5 ("sampled loop count") and r5 ("fixed 5 loops")
  (R = 5), with v201/v210's setups.
- **Fine-tune (D1, D2):** v201's full-backprop recipe (AdamW (0.9, 0.95), wd 0, warm-up and cosine over 600 steps, clip
  1.0, the model's full-backprop LR, the same data order), run to step 600 instead of stopping at 300.
  - Measured at steps 0, 300 and 600.
  - The global RNG state is saved before and restored after each checkpoint measurement, so measuring does not change
    the trajectory (checked on CPU: bit-identical final weights with and without).
  - Seeds are pre-listed as 2011, 2012, 2013. The first that passes the guard is used: ≤ 10 non-finite steps in the
    first 300, ≤ 20 over all 600, and a finite last loss.
- **Measurement:** v210 Part A, unchanged: batch 1, fp32, train mode, arms k = R, 0, 1, 2, 4.
  - Statistics: all = |g − g_k| / |g| and loop = |g − g_k| / |g − g_0|.
  - GSM8K: the first 64 test sequences, state seed 301000 + s, the same sequences and draws as v210.
  - WikiText-103: the test split, one document per article, tokenised as each model's GSM8K text is, packed into
    512-token chunks; the first 64 chunks, state seed 501000 + s.
- **State draws (D3):** at step 0, the first 16 GSM8K test sequences, each measured with 8 state draws (seeds
  401000 + 1000 d + s, d = 0..7). Statistic: per sequence, the SD of all-parameter lost gradient across draws; the
  median over sequences; the maximum over k = 1, 2, 4.

## Validity checks (LOCKED)

- Step-0 GSM8K medians agree with v210's s0 within 0.005 at every k, on both statistics.
- The fine-tune passes the guard.

If either fails for a model, D1/D2 are INVALID for it and are reported as such.

## Decisions (LOCKED; `--analyze results_v210 results_v218`)

All falls are means over k = 1, 2, 4 of (median at step 0 − median at the checkpoint), all-parameter statistic.

| | verdict | rule | paper change (v2) |
|---|---|---|---|
| D1 | FALLS-FURTHER | fall(600) − fall(300) ≥ +0.02 on all three models | §4: "it falls further by the end of the 600-step run", with numbers |
| D1 | HOLDS | \|fall(600) − fall(300)\| < 0.02 on all three | §4: "the fall holds to the end of the 600-step run" |
| D1 | RECOVERS | fall(600) − fall(300) ≤ −0.02 on all three | §4 and abstract: part of the fall reverses by step 600, with numbers; the "overstates what truncation removes later" sentence is qualified |
| D1 | MIXED | otherwise | stated per model |
| D2 | GENERAL | WikiText fall(300) / GSM8K fall(300) ≥ 0.5 on all three | §4: it falls on unrelated text (WikiText) too; "we did not test other text" removed |
| D2 | DOMAIN | ratio < 0.25 on all three (including negative) | §4 and §7: the fall is specific to the fine-tuning domain, with numbers |
| D2 | PARTIAL | otherwise | stated per model |
| D3 | SMALL | max over k of the median within-sequence SD ≤ 0.02 on all three | §7: draw-to-draw variation is at most 0.02, below the sampling error across sequences |
| D3 | LARGE | > 0.05 on any model | §2/§7: the state draw is a noise source of that size, with numbers |
| D3 | MODERATE | otherwise | the number is reported in §7 |

## Predictions (mine, at freeze)

- D1: HOLDS 0.45, FALLS-FURTHER 0.35, MIXED 0.15, RECOVERS 0.05.
- D2: PARTIAL 0.40, GENERAL 0.35, DOMAIN 0.25.
- D3: SMALL 0.50, MODERATE 0.35, LARGE 0.15.

## Cost and runs

| kernel | account | GPU 0 | GPU 1 |
|---|---|---|---|
| parcae | aurmerin | fine-tune to 600 with checkpoints (D1, D2) | state draws (D3) |
| ifm | aurmerin | pln5: fine-tune and checkpoints, then draws | r5: the same |

About 40 minutes per kernel, about 2.5 T4-hours in all. The TINY pipeline test passed on CPU (tiny Part II model):
checkpoints at steps 0/2/4, WikiText-style chunks, draws, and the analysis on synthetic inputs.
