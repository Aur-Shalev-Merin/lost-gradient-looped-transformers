# v225 RESULT (Part P only): per-k learning rates on the Part II models — SAME-LR

**Provenance.**
- **Prereg:** `docs/prereg_v225_lr_per_arm.md`, frozen and pushed in 6735ddb before any run.
- **Part O (Ouro) was dropped** by the author before any result (`docs/v225_v226_cancelled.md`), so its mapping does
  not apply.
- **Runs:** Part P ran unchanged as frozen, from the Oct 27 builds:
  - `build_kaggle_v200.py` 57dc9f9ccda5 and `v200_ifm_trunc.py` bd9ee68769e0 (hashes re-checked before the push);
  - aurmerin/astrocyte-v200-pln5-v225sel (version 2) and aurshalevmerin/astrocyte-v200-r5-v225sel, run
    2026-11-09 01:09–02:27 UTC.
- **Checks:** the selftest passed on all 4 GPU jobs, and all 18 runs are present.
- **Analysis:** `python v225_lr_per_arm.py --analyze-ifm results_v225`; output in `results_v225/v225_ifm_analysis.txt`.

## Verdict: SAME-LR

Every cell selected full backprop's 1e-4, inside the grid each time. No rescoring runs were needed, and S11's verdict
(K-AS-GOOD) stands as scored.

| model | k | val at 3e-5 | 1e-4 | 3e-4 | selected |
|---|---|---|---|---|---|
| pln5 (sampled loop count) | 1 | 1.4671 | **1.4527** | 1.5543 | 1e-4 |
| pln5 | 2 | 1.4531 | **1.4408** | 1.5461 | 1e-4 |
| pln5 | 4 | 1.4455 | **1.4335** | 1.5334 | 1e-4 |
| r5 (fixed 5 loops) | 1 | 1.4690 | **1.4597** | 1.5816 | 1e-4 |
| r5 | 2 | 1.4465 | **1.4410** | 1.5807 | 1e-4 |
| r5 | 4 | 1.4299 | **1.4242** | 1.5393 | 1e-4 |

The margins over 3e-5 are 0.005–0.015 nats on val; 3e-4 is about 0.1 worse everywhere. One seed (9001) selected the LRs,
as v200 did for its controls.

## Predictions vs outcome

I gave SAME-LR 0.55 and expected any moved cell to move up at k = 1. Nothing moved.

## Paper change (prereg §3)

§5: on Parcae (S2) and both Part II models, tuning the truncated arms' learning rate per k chose full backprop's rate.
The step-size caveat therefore applies only to Ouro (and SmolLM2, whose costs are about 1%).
