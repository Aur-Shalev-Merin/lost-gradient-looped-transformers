# v225 and v226: cancelled before any result (2026-10-27)

**What was frozen.** Both pre-registrations were frozen and pushed in 6735ddb:
- `docs/prereg_v225_lr_per_arm.md`: per-arm learning rates for Ouro, on a split of GSM8K test, plus a per-k LR sweep on
  the Part II models;
- `docs/prereg_v226_ft_seeds.md`: fine-tune seeds 2012 and 2013 for S15–S17 on the Part II models.

**What ran.** Four kernels were pushed at about 21:29–21:30 UTC:
- aurshalevmerin: astrocyte-v225-sel-a and astrocyte-v226-ifm;
- aurmerin: astrocyte-v200-pln5-v225sel and astrocyte-v200-r5-v225sel.

The author cancelled all four the same evening, before any of them finished. Each one's status is
CANCEL_ACKNOWLEDGED. astrocyte-v225-sel-b and -sel-c were never pushed.

**Context.** The author gave no reason. The cancellation came during a discussion of Kaggle quota:
- the Ouro arm alone needed about 57 GPU-hours, roughly a week of both accounts' quota;
- another paper's runs share those accounts and needed a slot that evening;
- the Ouro arm would have firmed up a secondary result that the paper already calls weak evidence.

**Data.** No output was downloaded or read, so no result exists. The paper (v17.2) keeps S18's disclosed caveats:
- learning rates chosen on test;
- truncated arms at full backprop's rate;
- one fine-tune seed per model in S15–S17.

**Rerun of the cheap parts, decided the same evening (about 21:42 UTC).** The author asked to rerun only the cheap
parts, unchanged as frozen, from the same builds, on aurshalevmerin:
- v226 in full (astrocyte-v226-ifm);
- v225 Part P selection (astrocyte-v200-pln5-v225sel, astrocyte-v200-r5-v225sel), plus its rescoring kernels if the
  selection calls for them.

The only change is the Kaggle account; the prereg assigns accounts at push. **v225 Part O (Ouro) is dropped,** and its
outcome mapping does not apply. The cancelled runs above produced no data, so the reruns are the first measurements.
