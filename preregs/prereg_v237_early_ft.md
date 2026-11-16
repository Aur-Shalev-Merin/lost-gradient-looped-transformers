# PREREG v237: when does the fine-tuning fall happen, and does Ouro's signal grow on unseen text?

**Status: FROZEN** (this commit), before any measurement. The freeze commit is pushed to origin before the first
kernel.

**Code:** sha256 prefixes in the "Hashes" section at the end.

## 0. Why

A hostile methods review of the paper (v17.6, 2026-11-09) raised two points that existing data cannot settle.

1. **An alternative explanation for the fall.** The fine-tuning fall of the lost gradient (S12, S13, S15, S17, S20)
   could partly be a quick fix of a mismatch between the trained model and the fine-tuning text, rather than a slower
   change. The evidence: on the sampled-loop-count model, the GSM8K value at step 300 (0.63 / 0.46 / 0.22) lands on
   the trained model's WikiText value (0.64 / 0.47 / 0.22). §4 now names this as not ruled out. A quick fix should
   happen early, while the training loss drops fastest. We measured only steps 0, 300 and 600.
2. **Ouro's growth was measured on text it appears to have seen.** Table 1 and S15's growth checks used the last 473
   GSM8K train problems. Ouro appears to have seen GSM8K's train split (App. K: 0.27 nats easier than test before any
   fine-tuning).

## 1. Part A: the fall over fine-tuning steps (LOCKED)

- **Fine-tune:** v218's full-backprop fine-tune, imported unchanged (`v218_robustness.run`). It uses v201's recipe,
  each model's full-backprop LR, the same data order and seed 2011, the seed v218 used on all three models. The
  checkpoint list is extended from (300, 600) to (10, 25, 50, 100, 200, 300, 600).
- **Measurement:** v218's, unchanged. At step 0 and every checkpoint: the first 64 GSM8K test sequences (state seed
  301000 + s) and the first 64 WikiText-103 test chunks (501000 + s); batch 1, fp32, train mode; arms k = R, 0, 1, 2, 4;
  statistics `all` and `loop`. The global RNG is saved and restored around every measurement, so the training
  trajectory is v218's.
- **Models:** Parcae-140m (R = 32), Part II pln5 and r5 (R = 5).
- **Validity, per model:**
  - the fine-tune passes v218's guard;
  - step 0 matches v218's step 0 within 0.005 (v218's own S0_TOL);
  - steps 300 and 600 match v218 within 0.01 per k, all and loop, GSM8K.

  The 0.01 allows for GPU nondeterminism across kernels; v218 matched v210 within 0.003. A model that fails is
  INVALID and reported. Note that v218's guard reads its non-finite count at the first checkpoint, which is now step 10;
  the step-300 match is the binding check that the run reproduces v218's.
- **Quantity:** F(t) = the mean over k = 1, 2, 4 of the all-parameter per-sequence median on GSM8K at step 0, minus
  the same at step t. The shares are F(50)/F(300) and F(100)/F(300).

**Per-model label:**
- FAST if F(50) ≥ 0.75 · F(300);
- otherwise GRADUAL if F(100) ≤ 0.5 · F(300);
- otherwise INTERMEDIATE.

**Verdict A:** FAST or GRADUAL if all three models get that label; MIXED otherwise; INVALID if any validity check
fails.

**Reported, not scored:**
- F(t) and the WikiText fall at every step;
- the GSM8K value over steps against the step-0 WikiText value;
- the training-loss curve (every 50 steps);
- the loop-only statistic.

**What the speed can and cannot show.** A quick text-mismatch fix predicts a fast fall. A fast fall does not prove
it, since anything that happens early in fine-tuning predicts the same. A gradual fall argues against the quick-fix
reading but not against every form of the mismatch explanation.

## 2. Part B: Ouro on test text (LOCKED)

- **Text:** the first 16 GSM8K TEST problems' chunks: the test split packed in order into 512-token chunks, the same
  packing as v197's validation chunks.
- **Measurement:** v210 Part B (the backward signal and state norms per loop, scale-free versions, exit-objective
  lost gradient) and v197's final-output lost gradient (all parameters, k = 1, 2, 3), both imported unchanged.
  `v197_ouro_allparam.val_chunks` is replaced by the test-chunk loader for both.
- **Model:** Ouro-1.4B at the pinned revision, fp32, one process per T4 (chunks 0–7 and 8–15). Transformers 4.54.1.

**Verdict B** (S15's B1 rule, plus the raw signal):
- GROWS if the median fitted ρ is above 1 for the raw backward signal, the scale-free signal on h and the scale-free
  signal on the pre-norm x;
- NO-GROWTH if all three are at most 1;
- MIXED otherwise.

**Reported:** the per-chunk minima, the exit-objective and final-output lost gradients against Table 1's 0.92 / 0.81,
and the no-growth reference (0.50–0.71 at k = 2 and 0.75–0.87 at k = 1).

## 3. What each outcome changes in the paper

| outcome | paper change |
|---|---|
| A FAST | §4: most of the fall happens within the first 50 steps, while the training loss drops fastest. This fits the text-mismatch explanation but does not prove it, and the paper says so. The abstract's per-sequence sentence stays. |
| A GRADUAL | §4: the fall builds over hundreds of steps, which argues against a quick fix of a text mismatch. The alternative-explanation paragraph is shortened accordingly. |
| A MIXED | §4 states each model's course. |
| A INVALID | Reported. §4 is unchanged. |
| B GROWS | §3 and App. K: Ouro's backward signal grows on test text too, with the numbers. The caveat about seen text is closed. |
| B NO-GROWTH or MIXED | §3, Table 1's footnote, the abstract and §7 qualify Ouro's growth as measured on text it appears to have seen, with the test-text numbers. |

## 4. Checked before the freeze

- **Part A:** a tiny-CPU run of `v237_early_ft.py A --model r5` measured every checkpoint and wrote v218's output
  format.
- **Synthetic worlds** built from the real v218 records, with early checkpoints filled in by a chosen shape. These come
  out as built: FAST (a fall complete by step 40), GRADUAL (linear to 300), MIXED (one model fast), and INVALID (step 300
  shifted by 0.05 from v218).
- **Part B:** a tiny-model run with transformers 4.54.1 measured 2 test chunks through both imported drivers and the
  analysis (a random tiny model, so its NO-GROWTH verdict means nothing).

## 5. Predictions (mine, at the freeze)

- **A:** FAST 0.35, MIXED 0.40, GRADUAL 0.20, INVALID 0.05.
  - On the Part II models I expect most of the fall early, since their WikiText value barely moves and the GSM8K
    value lands on it.
  - On Parcae I expect a slower course, since its WikiText value falls as much as its GSM8K value.
- **B:** GROWS 0.85, MIXED 0.10, NO-GROWTH 0.05. The growth is large on every validation chunk (ρᵢ 1.55–2.27) and
  survives normalisation.

## 6. Cost and runs

| kernel | contents | time |
|---|---|---|
| astrocyte-v237-ifm | gpu0 Part A pln5, gpu1 Part A r5 | about 1 h (8 measurements of about 4.5 min each, plus a 21-min fine-tune) |
| astrocyte-v237-parcae | gpu0 Part A Parcae | about 1.2 h (8 measurements of about 5 min each, plus the fine-tune) |
| astrocyte-v237-ouro | gpu0 Part B chunks 0–7, gpu1 chunks 8–15 | about 0.5 h |

Kaggle accounts are assigned at push, by free slots; the result doc records them.

## Hashes

sha256 prefixes (LF). The first two are new; the others are imported unchanged and match the hashes their own
preregs record.
- `v237_early_ft.py`: 05602fe79dfb
- `build_kaggle_v237.py`: a904d517d9b1
- `v218_robustness.py`: 6aeb0b6c25c5
- `v201_batch_lost.py`: 4fe2fff13a39
- `v200_ifm_trunc.py`: bd9ee68769e0
- `ifm_looped.py`: 502739cdccc8
- `v162_parcae_trunc_train.py`: 0ae4891d7abc
- `v163_parcae_damped_trunc.py`: db639c411892
- `v210_followups.py`: 619235c6b191
- `v197_ouro_allparam.py`: 7f1c32f10bc2
- `loop_grad_probe.py`: 27b80fa0b999
