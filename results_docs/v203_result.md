# v203 result: GENERAL. The fall of the lost gradient during fine-tuning holds on held-out GSM8K problems

**Plan and history.**
- Plan: `docs/prereg_v203_heldout_lost.md`, frozen 97d945f.
  - Amendment 1 (builder fix): the v1 kernels failed at parse, with no data.
  - Amendment 2 (Parcae backup seeds), fc2e374. Both amendments were pushed before the runs they govern.
- Runs, 2026-10-18, T4 × 2:
  - aurmerin: astrocyte-v203-parcae v2, astrocyte-v203-retro v1, astrocyte-v203-parcae-rerun v1;
  - aurshalevmerin: astrocyte-v203-ifm v2.
- Analysis: `python v203_heldout_lost.py --analyze <dirs> --v201 results_v201` (0fb92143fee9). Output in
  `results_v203/v203_analysis.txt`.

**Data note.** These are GSM8K test-split problems: never trained on, but the same kind of math word problem as the
fine-tuning data. This tests memorisation of the trained-on sequences, not other kinds of text such as web pages or
code (review note).

**Validity.**
- All measurements passed the v201 selftests.
- Parcae's seed-2011 fine-tune had 12 non-finite steps, against a guard of 10, so it was not measured. Per amendment 2,
  the backup seeds 2012 (2 non-finite steps) and 2013 (1) both passed, and 2012 is used.
- Test-split batches available: 56 for Parcae and SmolLM2, 48 for Part II (up to 64 allowed).
- A first download of the rerun was silently incomplete, missing the seed-2013 file. It was re-fetched, and the
  verdict is the same on the complete copy.

## Result

Per-sequence lost gradient, k = 1 / 2 / 4, medians:

| model | held-out, stage 0 | held-out, stage 300 | held-out drop | training-text drop (v201) | ratio |
|---|---|---|---|---|---|
| Parcae-140m | 0.80 / 0.64 / 0.41 | 0.60 / 0.42 / 0.20 | 0.20 / 0.22 / 0.21 | 0.20 / 0.22 / 0.18 | 1.02 |
| Part II, sampled loop count | 0.76 / 0.59 / 0.29 | 0.63 / 0.46 / 0.22 | 0.13 / 0.13 / 0.07 | 0.14 / 0.16 / 0.07 | 0.90 |
| Part II, fixed 5 loops | 0.83 / 0.70 / 0.39 | 0.72 / 0.56 / 0.30 | 0.12 / 0.14 / 0.09 | 0.12 / 0.15 / 0.09 | 0.95 |
| SmolLM2 (A), not scored | 0.36 / 0.14 / 0.018 | 0.33 / 0.11 / 0.012 | 0.04 / 0.03 / 0.01 | 0.03 / 0.02 / 0.01 | — |

**Verdict (frozen rule): GENERAL.** The ratio is ≥ 0.5 on all three scored models (1.02, 0.90, 0.95). My prediction
at the freeze was GENERAL 0.45.

**Also reported.**
- The clipped update averaged over 64 held-out sequences falls as well:
  - Parcae 1.05 / 0.82 / 0.56 → 0.55 / 0.37 / 0.15;
  - sampled loop count 0.77 / 0.53 / 0.22 → 0.54 / 0.38 / 0.17;
  - fixed 5 loops 0.88 / 0.67 / 0.30 → 0.62 / 0.47 / 0.21.
- Parcae seed 2013 at stage 300 gives 0.62 / 0.44 / 0.22 per sequence (seed 2012: 0.60 / 0.42 / 0.20), so the backup
  seeds agree.

## What this changes in the paper (frozen mapping: GENERAL)

- **The statement.** The paper can say that fine-tuning lowers the lost gradient. The scope stays explicit: on held-out
  GSM8K problems, the same domain as the fine-tune, after 300 steps of full backprop.
- **The pending caveat comes out.** The "trained-on text only, held-out check pending" wording is replaced with the
  held-out numbers above.
- **Limitations.** Add one item: out-of-domain text was not tested.
