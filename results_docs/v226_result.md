# v226 RESULT: fine-tune seeds 2012 and 2013 for S15–S17 on the Part II models — PARTLY-REPLICATED (11 of 12)

**Provenance.**
- **Prereg:** `docs/prereg_v226_ft_seeds.md`, frozen and pushed in 6735ddb before any run.
- **Kernel:** aurshalevmerin/astrocyte-v226-ifm, version 3, run 2026-11-09 01:09–02:27 UTC.
  - Version 1 was cancelled on 2026-10-27 before any result (`docs/v225_v226_cancelled.md`). No output of it was read.
  - The build is unchanged (`v226_ft_seeds.py` 6903930422ea; every imported driver matches its prereg hash, re-checked
    before the push).
- **Analysis:** `python v226_ft_seeds.py --analyze results_v210 results_v211 results_v218 results_v226`; output in
  `results_v226/v226_analysis.txt`.

**Validity.**
- All four (model, seed) runs passed the drivers' guards; no seed was excluded.
- v211's loop-weights-fixed check passed on every seed.

## Verdict: PARTLY-REPLICATED

11 of 12 model-labels gave seed 2011's label on both new seeds. The exception is pln5's d1:
- the change from step 300 to step 600 was −0.023 and −0.023 on seeds 2011 and 2012 (RECOVERS);
- on seed 2013 it was −0.018 (HOLDS, just inside the ±0.02 band);
- the mean over 3 seeds is −0.021.

| model | quantity | seed 2011 | 2012 | 2013 | mean (SD) over 3 | labels |
|---|---|---|---|---|---|---|
| pln5 | fall, all parameters | 0.115 | 0.117 | 0.109 | 0.114 (0.004) | FALLS ×3 |
| pln5 | loop-only / all (A1) | 0.727 | 0.731 | 0.734 | 0.731 (0.004) | IN-LOOP ×3 |
| pln5 | truncated / full (A2) | 0.825 | 0.858 | 0.868 | 0.850 (0.023) | FALLS ×3 |
| pln5 | k = 0 / full, loop only (S16) | 0.490 | 0.464 | 0.504 | 0.486 (0.020) | PARTIAL ×3 |
| pln5 | 600 − 300 (D1) | −0.023 | −0.023 | −0.018 | −0.021 (0.003) | RECOVERS, RECOVERS, HOLDS |
| pln5 | WikiText / GSM8K (D2) | 0.052 | 0.076 | 0.086 | 0.071 (0.018) | DOMAIN ×3 |
| r5 | fall, all parameters | 0.115 | 0.110 | 0.109 | 0.111 (0.003) | FALLS ×3 |
| r5 | loop-only / all (A1) | 0.742 | 0.749 | 0.755 | 0.749 (0.006) | IN-LOOP ×3 |
| r5 | truncated / full (A2) | 0.608 | 0.524 | 0.565 | 0.566 (0.042) | FALLS ×3 |
| r5 | k = 0 / full, loop only (S16) | 0.142 | 0.148 | 0.149 | 0.146 (0.004) | LOOP-NEEDED ×3 |
| r5 | 600 − 300 (D1) | −0.030 | −0.026 | −0.027 | −0.028 (0.002) | RECOVERS ×3 |
| r5 | WikiText / GSM8K (D2) | 0.330 | 0.380 | 0.389 | 0.366 (0.031) | PARTIAL ×3 |

Seed-to-seed spread is small next to every effect the paper reports:
- the SD is at most 0.006 for the falls and A1;
- it is at most 0.042 for the ratios.

## Predictions vs outcome

I gave PARTLY-REPLICATED 0.85 and ALL-REPLICATED 0.10. My guesses at which labels would fail were:
- r5 A2 (0.55 to replicate): it held, at 0.52–0.61;
- the two D1 labels (0.45 / 0.55): pln5 D1 is the one that failed, r5 D1 held.

## Paper change (prereg §3, PARTLY-REPLICATED)

- §4, App. I and §7: the Part II results are restated as holding on three fine-tune seeds, with the 3-seed means and
  SDs.
- The non-replicated label is restated per seed: on pln5 the step-600 reversal is 0.018–0.023, at the 0.02 bar.
- The "one fine-tune seed per model" limitation drops for the Part II models; it stays for Parcae.
- The abstract changes only if `falls` or `a1` failed to replicate. Both replicated, so the abstract is unchanged.
