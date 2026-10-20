# PREREG v203: does v201's fall of the lost gradient during fine-tuning hold on held-out text?

**Status: FROZEN** (this commit), before any measurement. Pushed to origin before the first run.

**Code (sha256 prefixes, LF):** `v203_heldout_lost.py` 14b409f03529; `v201_batch_lost.py` 4fe2fff13a39 (unchanged
since v201 amendment 1); `build_kaggle_v203.py` 980c0893e7e7.

**Why.** A review note on v201 (S12). v201 found the per-sequence lost gradient 0.07–0.22 lower after 300
fine-tuning steps on Parcae-140m and both Part II models. But the step-300 measurement used training sequences that
the 300-step fine-tune had just seen about once. Part of the fall may be the model fitting those sequences, not a
change in the loop. S2's earlier note, that ρ fell 0.03–0.05 over fine-tuning, was measured on held-out text, but it
concerns a different quantity.

**Design (LOCKED; `v203_heldout_lost.py`, which imports `v201_batch_lost.py` unchanged for the measurement and the
fine-tune).**
- **Models, settings and fine-tune:** the same as v201 (prereg 38b3203 + amendment 169d8cd). Seed 2011, 300 of 600
  steps, full backprop, the study's recipe on the training split.
- **Measurement:** stage 0 and stage 300 on GSM8K test-split sequences, which no run trains on. These are packed in
  split order, with as many full groups of 8 batches of 8 as exist, up to 64 batches (the count is logged). B = 1 is
  measured on the first 64. The selftests are v201's.

**Decision (LOCKED; `--analyze`).**
- **Models scored:** those whose v201 training-sequence drop (stage 0 minus stage 300 per-sequence median) averages
  more than 0.05 over k = 1, 2, 4: Parcae, pln5 and r5.
- **Ratio** per model = mean held-out drop / mean training-sequence drop.

| verdict | rule | paper wording (v16) |
|---|---|---|
| GENERAL | ratio ≥ 0.5 on every scored model | "fine-tuning lowers the lost gradient", with the held-out numbers |
| MOSTLY-SEEN-SEQUENCES | ratio < 0.5 on every one | the fall is stated as mainly on the sequences the model trained on; the held-out change is reported as the general one |
| MIXED | anything else | the wording is per model |

SmolLM2 is reported but not scored. The held-out clipped B = 64 values are reported. S12's own verdict is not
re-scored.

**Predictions (mine, at freeze).** GENERAL 0.45, MOSTLY-SEEN-SEQUENCES 0.30, MIXED 0.25.

**Cost.** Three kernels on aurmerin, laid out as v201. About 3–4 T4-hours.

**Amendment 1 (2026-10-18, before any measurement).**
- **What happened:** the first launch (v1 kernels) failed at Python's parse step. The builder had written a broken
  string literal into each kernel's launcher (`run.py`), so nothing ran and no data exist.
- **Fix:**
  - the builder's final status line was corrected (`build_kaggle_v203.py` is now f71f7b33174c);
  - every `run.py` is now compile-checked before pushing.
- **Unchanged:** `v203_heldout_lost.py` (14b409f03529), `v201_batch_lost.py` and the rules.
- **Accounts:** Parcae and SmolLM2 run on aurmerin, Part II on aurshalevmerin. Both accounts hold the SmolLM2
  dataset.

**Amendment 2 (2026-10-18; after Part II and SmolLM2 data, before any Parcae stage-300 data).**
- **What happened:** Parcae's stage-300 fine-tune (seed 2011) hit 12 non-finite steps, against the guard of ≤ 10. Per
  the frozen rule, it was not measured, so the verdict is INCOMPLETE. Its loss curve was normal (final 1.41; v201's
  run with the same seed: 1.40). The likely cause is fp16 loss-scale overflows, which GradScaler skips.
- **Fix, following the project's backup-seed practice (v165):**
  - Parcae stage 300 is rerun with pre-listed backup fine-tune seeds 2012 and 2013, one per GPU.
  - The lowest seed that passes the guard is used. If both fail, INCOMPLETE stands.
  - `v203_heldout_lost.py` gains `--ft-seed`, and the analysis keeps, among valid stage-300 runs, the lowest
    fine-tune seed (0fb92143fee9).
  - The builder gains a `parcae-rerun` kernel (7943619a0cd6).
- **Unchanged:** the measurement code, the stage-0 data and the rules.
- **Disclosure:** amended after seeing the other models' results (held-out/training ratios 0.90 and 0.95). Nothing in
  the Parcae measurement can be tuned.
