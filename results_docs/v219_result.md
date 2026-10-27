# v219 RESULT: Ouro-1.4B truncation cost — ABOVE-BOTH

Prereg: `docs/prereg_v219_ouro_cost.md`, frozen at d765356 and pushed before any scored run.
Driver: `v219_ouro_cost.py`, sha256 (LF) f8543506d909, the same in all five kernels.
Analysis: `python v219_ouro_cost.py --analyze results_v219/s1 ... s5`. The full output is in
`results_v219/v219_analysis_5seeds.txt`.

## Verdict (primary, final-output loss on test)

**ABOVE-BOTH.** At both k, the 90% lower bound of φ_k is above max(L_k, KP_k).

| cell | φ (5 seeds) | 90% CI | L | KP | max of the other trained models |
|---|---|---|---|---|---|
| k = 1 (1 of 4 loops) | **1.133** | 1.012 to 1.255 | 0.258 (extrapolated) | 0.224 | 0.291 (r5) |
| k = 2 (2 of 4 loops) | **0.652** | 0.509 to 0.795 | 0.187 | 0.100 | 0.149 (r5) |

- MAE: L 0.670, KP 0.731.
- Rank among the trained models at each k (descriptive): Ouro > r5 > Parcae > pln5 > SmolLM2.

**Floor gate: PASS.** The gap is 0.0338 nats (s.e. 0.0009; 90% CI 0.0318 to 0.0357).

**Seed rule.** Over the first 3 seeds the s.e. of the k = 1 cost was 0.102, above 0.02, so N = 5 as locked. Seeds
2191004 and 2191005 ran with the code unchanged. All 5 seeds are valid: under full backprop each had at most 3
non-finite steps, and none diverged.

### Per seed: test loss (final output)

| seed | full | k = 2 | k = 1 | floor (k = 0) | φ2 | φ1 |
|---|---|---|---|---|---|---|
| 2191001 | 1.0313 | 1.0478 | 1.0600 | 1.0624 | 0.530 | 0.925 |
| 2191002 | 1.0262 | 1.0505 | 1.0693 | 1.0619 | 0.680 | 1.207 |
| 2191003 | 1.0271 | 1.0588 | 1.0711 | 1.0623 | 0.901 | 1.249 |
| 2191004 | 1.0299 | 1.0487 | 1.0654 | 1.0620 | 0.586 | 1.107 |
| 2191005 | 1.0272 | 1.0468 | 1.0683 | 1.0621 | 0.563 | 1.179 |

### Exit-weighted objective (descriptive, not scored)

- Floor gap 0.0420.
- φ1 1.303 (90% CI 1.204 to 1.402).
- φ2 0.539 (90% CI 0.425 to 0.653).

The same picture.

## What it means

- **k = 1 is no better than not training the loop at all.** Its test loss is above the floor's on 4 of 5 seeds, and
  the 90% CI of φ1 lies entirely above 1. One step of loop credit leaves Ouro worse off than training only the head.
  The excess averages about 0.005 nats.
- **k = 2 recovers only about a third of what full backprop gains over the floor.** Its cost is 0.65; the other
  models' costs at k = 2 range from 0.00 to 0.15.
- **The absolute scale is small.** Full backprop gains only 0.034 nats over the floor on this fine-tune, so these are
  large fractions of a small effect.
- **One model, so weak evidence alone.** At k = 1 the curve's input (err 0.92) is outside its fitted range
  (0.18 to 0.877).
- **A rule based on the fraction of the loop kept does not explain it.** pln5 and r5 keep 1 of 5 loops at k = 1, a
  smaller fraction than Ouro's 1 of 4, yet their costs are 0.16 and 0.29.

### Paper change (frozen mapping, §4 of the prereg)

§5 and the abstract: the model that loses the most gradient paid more than either rule predicts. This is cost
evidence for the lost gradient at the extreme, with k = 1 an extrapolation.

In addition, §1, §5 and §7 drop "we did not train Ouro with truncation" and report φ with the gate. An appendix records:
- the val/test discrepancy;
- the deviation of choosing LR on test (§1 of the prereg). On val, the floor reaches 0.79 while full backprop
  reaches 0.98, which is the opposite of test; this is the same anomaly.

## After the fact: the cost in nats (review note, 2026-10-27)

The cost φ is a share of the loop's benefit, and Ouro's benefit (floor gap 0.034) is small next to the other models'
(0.13–0.18). In absolute terms, truncation added the following over full backprop (Ouro: means of the per-seed
differences; others: φ × gap):

| model | k = 1 | k = 2 |
|---|---|---|
| Ouro | 0.038 | 0.022 |
| Parcae-140m | 0.040 | 0.018 |
| Part II, fixed 5 loops (r5) | 0.041 | 0.021 |
| Part II, sampled loop count (pln5) | 0.021 | 0.009 |

So ABOVE-BOTH comes mainly from the small denominator, not from an unusually expensive truncation. The paper (v17.0)
reports the verdict but does not adopt the frozen mapping's "cost evidence for the lost gradient at the extreme";
App. B records the departure.

## Predictions vs outcome

- I gave ABOVE-BOTH 0.06 and BELOW-BOTH 0.30 at the freeze, so this is a miss in direction.
- I also predicted a k = 1 cost s.e. of about 0.01 over 3 seeds. It was 0.102. The floor gap reproduced as predicted
  (the 0.0338 mean is within 0.002 of the pilot's 0.0325), but the truncated cells vary across seeds far more than
  in v200.

## Execution notes

- **Seed 4's metadata was hand-edited.** `build_kaggle_v219.py study-more` assigns seed 2191004 to aurmerin. Its
  `kernel-metadata.json` id was edited by hand to aurshalevmerin, because that account had the GPU quota. Seed 5 was
  already assigned to aurshalevmerin. The code is the same as frozen; only the account differs.
- **Seeds 1–3 were looked at before seeds 4–5 ran.** That was descriptive only, with no verdict. The seed rule and N
  were locked, so the look could not change what ran.
- **The local copy of the Kaggle console log is empty** for seeds 4 and 5 (a Windows charmap error on download).
  `out/stdout.txt` holds the same stream, so nothing is lost.
- **Selftest PASS on both new kernels:**
  - own forward 1.839788 / 1.839788;
  - checkpointing exact;
  - fp16 vs fp32 within tolerance;
  - synthetic negative control 1.0.
