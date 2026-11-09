# PREREG v226: do S15–S17's results on the Part II models hold on other fine-tune seeds?

**Status: FROZEN** (this commit), before any measurement on seeds 2012 or 2013. The freeze commit is pushed to origin
before the kernel is pushed.

**Code (sha256 prefixes, LF):**
- `v226_ft_seeds.py` (wrapper and locked analysis): 6903930422ea;
- `build_kaggle_v226.py`: c5c8cb30342c;
- imported unchanged, each matching the hash its own prereg records: `v218_robustness.py` 6aeb0b6c25c5,
  `v210_followups.py` 619235c6b191, `v211_k0_control.py` c516ee2fd492, `v201_batch_lost.py` 4fe2fff13a39,
  `v200_ifm_trunc.py` bd9ee68769e0, `ifm_looped.py` 502739cdccc8.

The Kaggle account is assigned at push, by free slots; the result doc records it.

## 0. Why

S15 (v210), S16 (v211) and S17 (v218) each followed one fine-tune per model: seed 2011, the first pre-listed seed that
passed the guard. Their bootstrap intervals cover sequences, not fine-tune seeds. A cold TMLR-style review (2026-10-27)
asked for a second seed for at least one Part II model. The Part II fine-tunes are cheap (about 20 minutes each), so
v226 runs two more seeds on both Part II models.

## 1. Design (LOCKED)

- **Models:** the two Part II models of S11–S17, `huginn-s-fixed-pln5` and `huginn-s-fixed-r5`, at their pinned
  revisions, R = 5, embedding frozen, as in v200 / v210 / v211 / v218.
- **Seeds:** 2012 and 2013, the next pre-listed seeds of all three drivers (FT_SEEDS = 2011, 2012, 2013). Each seed runs
  on its own. If a seed fails a driver's guard (more than 10 non-finite steps, or a non-finite last loss), it is
  reported as excluded and not replaced.
- **Per (model, seed),** with the drivers imported unchanged and FT_SEEDS set to that one seed:
  1. `v218 --do ft`: the full-backprop fine-tune run to step 600, measured at steps 0, 300 and 600 on the first 64
     GSM8K test sequences (state seed 301000 + s) and on the first 64 WikiText-103 test chunks (501000 + s).
  2. `v210 A trunc300`: the fine-tune with backprop through the last 2 loops, stopped at step 300 of 600, measured on the
     same GSM8K sequences.
  3. `v211 s0,zero300`: the k = 0 fine-tune to step 300 of 600, measured, with v211's check that the loop's weights
     stayed exactly fixed.
- **Statistics:** v210 Part A's `all` = |g − g_k| / |g| and `loop` = |g − g_k| / |g − g_0|, batch 1, fp32, train mode,
  per-sequence medians, falls averaged over k = 1, 2, 4.
- **One computation for all seeds.** Every seed's step-0 and full-backprop step-300 values come from its v218 fine-tune
  (for seed 2011 too, from `results_v218`), so the full-backprop fall is computed the same way for every seed. v218
  checked that its step-300 values agree with v210's full300 within 0.003. Recomputed this way, seed 2011 gives the
  same labels as the paper (this was checked before the freeze on seen data; the numbers are in §4).

## 2. Quantities and labels (LOCKED; `python v226_ft_seeds.py --analyze results_v210 results_v211 results_v218 <v226 outputs>`)

Per (model, seed), with each study's own thresholds:

| label | quantity | classes |
|---|---|---|
| falls (S12/S13) | all-parameter fall at step 300 | FALLS if ≥ 0.05, else NO-FALL |
| a1 (S15) | loop-only fall / all-parameter fall | IN-LOOP if ≥ 0.5, else READOUT-SHARE |
| a2 (S15) | truncated fall (all) vs full fall | FALLS if ≥ 0.5 × full; FLAT if \|·\| < 0.03; RISES if ≤ −0.03; else PARTIAL |
| s16 (S16) | k = 0 loop-only fall / full loop-only fall | LOOP-NEEDED if < 0.25; OUTSIDE-SUFFICES if ≥ 0.75; else PARTIAL |
| d1 (S17) | fall at 600 − fall at 300 | FALLS-FURTHER if ≥ +0.02; HOLDS if \|·\| < 0.02; RECOVERS if ≤ −0.02 |
| d2 (S17) | WikiText fall / GSM8K fall at 300 | GENERAL if ≥ 0.5; DOMAIN if < 0.25; else PARTIAL |

A ratio whose denominator is not positive is UNDEFINED.

**Per model and label:** REPLICATED if both new seeds give seed 2011's label; NOT-REPLICATED otherwise; INCOMPLETE if
a seed is missing or excluded.

**Overall verdict:** ALL-REPLICATED (12 of 12 model-labels), PARTLY-REPLICATED, or INCOMPLETE.

**Also reported:** per-seed values; for each quantity, the mean, SD and range over the three seeds; any failed
loop-fixed check (which makes that seed's s16 label INVALID).

## 3. What each outcome changes in the paper

| outcome | paper change |
|---|---|
| ALL-REPLICATED | §4, App. I and §7: the Part II results "held on three fine-tune seeds", with the 3-seed means and SDs; the "one fine-tune seed per model" limitation drops for the Part II models (it stays for Parcae). |
| PARTLY-REPLICATED | Each non-replicated label is restated per seed in §4 and App. I, and any sentence that relies on it is softened to what held on all three seeds. The abstract changes only if `falls` or `a1` fails to replicate. |
| INCOMPLETE | Reported as such; the one-seed limitation stays. |

## 4. Seed-2011 values recomputed (seen data, before the freeze)

| model | fall (all) | a1 ratio | a2 ratio | s16 ratio | d1 change | d2 ratio |
|---|---|---|---|---|---|---|
| pln5 | 0.115 | 0.727 IN-LOOP | 0.825 FALLS | 0.490 PARTIAL | −0.023 RECOVERS | 0.052 DOMAIN |
| r5 | 0.115 | 0.742 IN-LOOP | 0.608 FALLS | 0.142 LOOP-NEEDED | −0.030 RECOVERS | 0.330 PARTIAL |

These match the paper's values within 0.03 (the paper's a1 and a2 use v210's full300 run).

## 5. Predictions (mine, at the freeze)

- falls, a1 replicate on both models: 0.90.
- a2 (FALLS): pln5 0.80; r5 0.55 (its ratio, 0.61, is close to the 0.5 bar).
- s16: pln5 PARTIAL 0.65; r5 LOOP-NEEDED 0.55 (0.14 against a 0.25 bar).
- d1 (RECOVERS): pln5 0.45, r5 0.55. Both are within 0.01 of the ±0.02 bar.
- d2: pln5 DOMAIN 0.80; r5 PARTIAL 0.60.
- ALL-REPLICATED overall: 0.10; PARTLY-REPLICATED 0.85; INCOMPLETE 0.05.

## 6. Cost and runs

| kernel | account | GPU 0 | GPU 1 |
|---|---|---|---|
| astrocyte-v226-ifm | (set at build) | pln5 seeds 2012, 2013 | r5 seeds 2012, 2013 |

About 50 minutes per (model, seed): fine-tune to 600 (about 21 minutes), three checkpoint measurements on two texts (about
14 minutes), the truncated and k = 0 fine-tunes (about 9 minutes) and their measurements (about 6 minutes). About 2 hours in
all on one kernel.
