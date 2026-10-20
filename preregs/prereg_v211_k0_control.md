# PREREG v211: does the fall need the loop's weights to change? (k = 0 fine-tune control)

**Status: FROZEN** (this commit), before any v211 measurement. Pushed to origin before the first run.

**Code (sha256 prefixes, LF):**
- `v211_k0_control.py` c516ee2fd492 (measurement and analysis);
- `build_kaggle_v211.py` af7b6fd7bbf9;
- imported unchanged: `v201_batch_lost.py` 4fe2fff13a39 (setup and fine-tune), `v162_parcae_trunc_train.py`
  0ae4891d7abc, `v163_parcae_damped_trunc.py` db639c411892, `v200_ifm_trunc.py` bd9ee68769e0, `ifm_looped.py`
  502739cdccc8.

## Why

S15 (v210, `docs/v210_result.md`) found that the fall of the lost gradient during full-backprop fine-tuning mostly
survives leaving the no-loop-credit gradient out of the denominator (loop = |g − g_k| / |g − g_0|: 0.72–0.80 of the
all-parameter fall). Paper v16.7 reads this as "mostly a change in the loop itself". Outside review 7 (M1,
`docs/looped_truncation_review7_2026-10-19.md`) points out that this locates the fall in the loop's part of the gradient
but does not show that the loop's weights cause it. A change in the layers after the loop or the read-out could produce
it by changing the backward signal that enters the loop.

Control: fine-tune with no loop credit (k = 0). The loop's own weights then get no gradient and stay fixed, while
everything that gets gradient without loop credit trains (the layers after the loop, the read-out, and on Parcae any
weights tied to them). If the lost gradient falls as much, the loop's weights are not needed for the fall.

## Design

- **Models:** Parcae-140m (R = 32) and the Part II models pln5 ("sampled loop count") and r5 ("fixed 5 loops")
  (R = 5), with v201/v210's setups.
- **Fine-tune:** v201's recipe (AdamW (0.9, 0.95), wd 0, warm-up and cosine over 600 steps, clip 1.0, the
  model's full-backprop LR), same data order, stopped after 300 steps, but at k = 0.
  - Seeds are pre-listed as 2011, 2012, 2013. The first that passes the guard (≤ 10 non-finite steps, finite last
    loss) is used.
  - On Parcae a second GPU runs the same fine-tune from seed 2012 (zero300b). It is a reported seed check, not used
    by any rule.
- **Measurement:** v210 Part A, unchanged.
  - The first 64 GSM8K test sequences, batch 1, fp32, train mode, state draw seed 301000 + s.
  - Arms: k = R, 0, 1, 2, 4.
  - Statistics: all = |g − g_k| / |g| and loop = |g − g_k| / |g − g_0|.
  - States: s0 (re-measured) and zero300.
- **Reference:** v210's s0 and full300 (results_v210/, seed 2011), not re-run.

## Validity checks (LOCKED)

A model's result is valid only if:
1. **Loop fixed:** every parameter changed by the k = 0 fine-tune has a nonzero k = 0 gradient on the first measured
   sequence. The loop's weights get none, so this shows they stayed exactly fixed.
2. **s0 agrees with v210:** v211's s0 medians agree with v210's within 0.005 at every k, on both statistics.

If any check fails, the verdict is INVALID. The paper keeps v16.7's wording, toned down per review 7, and reports the
failure.

## Decision (LOCKED; `--analyze results_v210 results_v211`)

Per model, with means over k = 1, 2, 4 of the median-over-sequences values:
- fall_full = loop(s0, v210) − loop(full300, v210);
- fall_zero = loop(s0, v211) − loop(zero300, v211);
- ratio r = fall_zero / fall_full.

The all-parameter version is reported alongside but decides nothing.

| verdict | rule | paper change |
|---|---|---|
| LOOP-NEEDED | r < 0.25 on all three models | Keeps "a change in the loop itself", adding: a fine-tune that leaves the loop's weights fixed lowers it by less than a quarter as much (S16). |
| OUTSIDE-SUFFICES | r ≥ 0.75 on all three | "Change in the loop itself" is removed from the abstract, §1 and §3. The paper says the fall shows in the loop's part of the gradient but needs no change to the loop's weights: training only outside the loop gives r of it. |
| PARTIAL | otherwise | Stated per model. The abstract and §1 drop "a change in the loop itself" and say the fall is not just the read-out's share of the gradient; training only outside the loop gives r of it. |

Whatever the verdict, v16.8 also reports the truncated arm's loop-only fall from v210 (review 7, M1): 1.01 / 0.81 / 0.51
of the full-backprop fall on Parcae / pln5 / r5, flat at k = 4 on r5. This was computed from saved v210 data before
this freeze.

## Predictions (mine, at freeze)

PARTIAL 0.45, LOOP-NEEDED 0.35, OUTSIDE-SUFFICES 0.20. Most likely: a modest r of 0.2–0.5, since training the read-out
alone changes the backward signal entering the loop, but less than training the loop does.

## Cost and runs

| kernel | account | GPU 0 | GPU 1 |
|---|---|---|---|
| parcae | aurmerin | s0 + zero300 | s0 + zero300b (seed 2012) |
| ifm | aurshalevmerin | pln5 s0 + zero300 | r5 s0 + zero300 |

About 20 minutes each, about 1.5 T4-hours in all. The TINY pipeline test passed on CPU (tiny Part II model): the k = 0
fine-tune changed only the post-loop layer and the read-out, and the loop-fixed check passed.
