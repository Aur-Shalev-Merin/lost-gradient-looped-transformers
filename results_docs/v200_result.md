# v200 result: K-AS-GOOD and TIE. On two new looped models, the lost-gradient curve fitted on Parcae predicts truncation cost no better than Parcae's own cost at the same k

**Prereg and runs:**
- **Prereg:** `docs/prereg_v200_ifm_transfer.md`. It was frozen in 4849a02 and pushed before any scored run. The seed
  rule was applied in 2a86c33 (s = 0.0106, so N = 5) and also pushed.
- **Runs:** Kaggle, 2026-10-14, two T4s each:
  - aurmerin/astrocyte-v200-{pln5,r5}-study0, version 2;
  - aurshalevmerin/astrocyte-v200-{pln5,r5}-study1, version 1.
- **Analysis:** `V200_N_SEEDS=5 python v200_analyze.py <the four outputs>` (sha 8fc6df477ab6, no diff since the
  freeze). Its output is `results_v200/v200_analysis.txt`. It reproduces exactly from the run files copied into
  `results_v200/`.

**Run log.** The protocol was not changed.
- The two aurmerin kernels were cancelled by accident about 10 minutes after launch (version 1, around 19:48). They were
  re-pushed unchanged from the same 19:37 build as version 2 at 21:07. No version 1 output was read.
- During one download Kaggle returned 502 errors, and the CLI saved the error pages as if they were the output files.
  That copy was deleted. The copy used here was checked and contains no error pages.
- Each kernel's own end-of-run `v200_analysis.txt` covers only its subset of runs. None of them was read.

## Validity: all checks pass

| check | result |
|---|---|
| runs present | 70 of 70 (2 models × 5 cells × 7 seeds), each exactly once |
| selftest, all 8 GPU jobs | PASS (details below) |
| cache-on negative control | does not reproduce on Kaggle's PyTorch; informational, as frozen |
| controls | 0 non-finite steps, no divergence, finite val and test losses in all 70 runs |
| code | driver bd9ee68769e0 (driver_sha 22a56eba93e9) and port 502739cdccc8 in every kernel |
| setup | embedding frozen; LR 1e-4 for full and truncated, 3e-4 for the floor; 600 steps, batch 8 × 512 |
| seeds scored | 2001101–2001105 on both models: the first 5, all valid. Backups 2001106–07 were not needed. |
| floor gate (≥ 0.03 nats, 90% lower bound > 0) | pln5 0.1285 (lower 0.1276), r5 0.1407 (lower 0.1391); both scored. The pilot had 0.128 and 0.141. |

The selftest checks that:
- the training forward pass equals the model's own;
- checkpointing changes nothing;
- fp16 and fp32 truncated gradients agree within 0.25%;
- the floor gives the loop exactly zero gradient;
- the embedding is frozen;
- the synthetic negative control is flagged at 1.0.

## Measured cost against the four locked predictions

φ = (L_k − L_full) / (L_floor − L_full) on test, mean ± s.e. over the 5 seeds. Per-seed values are in
`results_v200/v200_analysis.txt`. An asterisk marks a calibrated prediction, |φ − prediction| ≤ max(0.02, 0.3·φ).

| cell | φ measured | L (lost gradient) | A (after Adam) | KP (Parcae's cost at the same k) | KS (SmolLM2's cost at the same k) |
|---|---|---|---|---|---|
| pln5 k = 1 | 0.162 ± 0.002 | 0.138* | 0.258 | 0.224 | 0.009 |
| pln5 k = 2 | 0.071 ± 0.001 | 0.058* | 0.102 | 0.100 | −0.001 |
| pln5 k = 4 | 0.015 ± 0.001 | 0.005* | 0.008* | 0.024* | 0.000* |
| r5 k = 1 | 0.291 ± 0.002 | 0.182 | 0.412 | 0.224* | 0.009 |
| r5 k = 2 | 0.149 ± 0.002 | 0.097 | 0.197 | 0.100 | −0.001 |
| r5 k = 4 | 0.023 ± 0.003 | 0.011* | 0.017* | 0.024* | 0.000 |
| MAE, all 6 cells | | 0.0367 | 0.0515 | 0.0360 | 0.1157 |
| calibrated cells | | 4 of 6 | 2 of 6 | 3 of 6 | 1 of 6 |
| MAE, 3 in-range cells (pln5 k = 2, r5 k = 2, r5 k = 4) | | 0.0256 | 0.0284 | | |

## Verdicts (frozen rules)

- **Transfer: K-AS-GOOD.** The better measurement-free rule had MAE 0.0360 (KP), at or below L's 0.0367.
  - L-TRANSFERS and L-BEATS-K both needed MAE(L) ≤ 0.018, half of KP's.
  - L-TRANSFERS also needed L calibrated in 5 of 6 cells; it was calibrated in 4.
- **Which curve, primary (3 in-range cells): TIE.** L 0.0256 against A 0.0284. Neither is half the other.
- **Which curve, secondary (all 6 cells, A extrapolated in 3): TIE.** L 0.0367 against A 0.0515.
- **Best predictor: KP**, by 0.0007 over L.

My predictions at the freeze were K-AS-GOOD 0.40 (the most likely of the five) and TIE 0.35.

## After the fact (not pre-registered)

1. **K-AS-GOOD is a near-tie between L and k alone, not a win for k alone.**
   - Method: a seed bootstrap (`v200_posthoc_boot.py`, `results_v200/v200_posthoc_boot.txt`), with 20,000 resamples
     and seeds resampled within each model.
   - MAE(L) − MAE(KP) = +0.0006, 90% interval −0.0015 to +0.0026.
   - The transfer verdict comes out K-AS-GOOD in 68% of resamples and INCONCLUSIVE in 32%. It is never L-BEATS-K or
     L-TRANSFERS.
   - The which-curve TIE holds in every resample, though L's in-range MAE is below A's in 96% of them.
2. **Cost does not follow k alone across these two models.** At the same k, r5 costs 1.79, 2.09 and 1.52 times what pln5
   costs (k = 1, 2, 4). The seeds don't overlap at k = 1 and 2; at k = 4 they do (pln5 0.012–0.017, r5 0.015–0.028). KP predicts the same cost for both models. It wins on MAE by sitting
   between them, not by matching either. So the prereg's §4 gloss on K-AS-GOOD, "cost follows k across models", does not
   describe this outcome.
3. **Both curves rank the two models correctly, but neither gets the scale.**
   - The r5/pln5 cost ratios were 1.79 / 2.09 / 1.52 observed, against 1.32 / 1.69 / 2.18 for L and 1.60 / 1.93 / 1.97
     for A.
   - L under-predicts every cell: by 15–19% on pln5 at k = 1, 2 and by 35–38% on r5. At k = 4 the relative misses are
     larger, but every cost there is below 0.025.
   - A over-predicts every k = 1, 2 cell, by 32–59%.
   - The measured cost lies between the two curves at k = 1 and 2 on both models.
   - With one pair of models, ranking them correctly is weak evidence: a coin would do it half the time.
4. **No single rescaling of the L curve fits both models.** At k = 1, pln5 needs a factor of 1.18 and r5 needs 1.60.

## What this changes in the paper

- **The prereg's consequence.** For K-AS-GOOD with KP best, the prereg says the paper should state that the lost
  gradient adds nothing for cost, even across models. As a statement about prediction error, that holds on this test.
  The curve fitted on Parcae was no more accurate than Parcae's own cost at the same k.
- **What §4 should not say:** that cost follows k. The two new models differ by up to 2× at the same k (point 2).
- **A possible §4 sentence:** "On two new looped models, the curve fitted on Parcae predicted truncation cost no better
  than Parcae's own cost at the same k (MAE 0.037 vs 0.036, pre-registered). It ranked the two models correctly but
  under-predicted the slower-fading model's cost by about a third."
- **The direction-after-Adam lead is not promoted.** It tied L on the in-range cells and was worse over all cells.
- §4's "What would settle it" names this test. It has now been run, and it did not settle the question in L's favour.
- The paper is unchanged until the user decides.
