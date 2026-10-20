# Pre-registration v162 (re-run of v161): does the adjoint decay ρ, measured on the frozen model, predict what truncated backprop through the loops costs in training?

**Author:** Aur Shalev Merin
**Date:** 2026-10-01. **FROZEN at the commit that fills in §9's results. No v162 truncated-credit run had been made at
the freeze.** The v161 truncated runs exist but trained a different (buggy) treatment; see §1.
**Workstream:** misc (looped-transformer credit thread; follows `docs/loop_grad_probe_generality_2026-09-30.md`)
**Trigger:** PI decision 2026-10-01 ("yes set up the rerun"). v161 (frozen e922ed4) was INVALIDATED: under
`torch.autocast` with the default weight-cast cache, Parcae's no-grad loops (which run first) cached the fp16 copies of
the looped-core weights without a grad connection, and the grad loops reused them. In every truncated arm the looped
core therefore got almost no gradient (fp16 relative error of the k = R − 1 gradient 0.87–0.90 vs 0.14 in fp32;
`docs/v161_parcae_truncation_training.md`, `v161_gradcheck.py`). **v162 is the same pre-registered design with the bug
fixed** (`autocast(cache_enabled=False)`) and a selftest that would have caught it. Everything not stated here is
unchanged from v161.

## 0. What this is

**The frozen-model finding.** On two independently built looped LMs trained with truncated backprop (Huginn-3.5B,
Parcae-1.3B), the relative error of the gradient obtained by backpropagating through only the last k of R loops follows
err_k ≈ ρᵏ, where ρ is the per-loop decay of the adjoint ‖dL/dh_t‖ along the actual trajectory (median |err − ρᵏ|
0.03–0.08). The loop Jacobian's spectral norm, the quantity in the published bounds, is ≫ 1 and a weak predictor. No
prior work reports the law or chooses truncation depth from a measured ρ (novelty re-check, 87b6603).

**What is missing.** All of that is gradient error on frozen checkpoints. Nothing says whether that error costs
anything in training. A practitioner choosing k cares about loss, not about gradient cosine.

**Primary question:** when Parcae-140m is trained further with backprop through the last k of R loops, does the
frozen-model law value ρᵏ predict the training cost of truncation, and does it predict it **better than the two simpler
rules a practitioner would otherwise use**: "more backprop loops is better" (k alone) and "backprop the same fraction
of loops" (k/R)?

**Why this design can tell those apart.** The law predicts different costs for cells that the simpler rules tie:
- at the same k, the cell with the larger R has a larger law value if ρ_16 > ρ_8 (on Parcae-1.3B, ρ 0.84 vs 0.78) —
  "k alone" predicts a tie;
- at the same fraction k/R (R 8, k 2 vs R 16, k 4; R 8, k 4 vs R 16, k 8), the law predicts different costs —
  "k/R" predicts a tie.
Each simpler rule fails one family of contrasts; only the law predicts both.

**Scope:**
- One model (Parcae-140m), one fine-tuning task (GSM8K text), one training length. Continued training of a pretrained
  checkpoint, not pretraining from scratch.
- Loss on held-out text, not task accuracy (a 140M model does not solve GSM8K).
- A positive result supports "measure ρ, then choose k" on one model; it does not show it at scale.

## 1. Disclosed prior

- Parcae-1.3B frozen-model numbers are known: ρ 0.78 (R 8), 0.84 (R 16); err at R 8, k 1/2/4: 0.75/0.58/0.34; R 16,
  k 1/2/4/8: 0.80/0.65/0.47/0.23. The Parcae-140m numbers come from the pilot (§8) and fix the contrast directions.
- The pilot's control arms (§8) were inspected before the freeze: full backprop and k = 0 at four LRs, one pilot seed.
- **Seen before this freeze (disclosed):** all v161 results. The v161 control arms (full, floor) are unaffected by the
  bug: full backprop runs no no-grad loop, and in the floor the looped core gets no gradient by design while every
  other module is first used with grad enabled. Their numbers are valid and known (eval seeds: full 1.3747 / 1.3745;
  floor gap 0.197 / 0.171 at R 8 / 16).
- The v161 *truncated* arms trained "looped core frozen, prelude/injection/readout trained": costs 0.062 / 0.049 /
  0.042 (k 1/2/4), flat in R, k = R − 1 also 0.040. These say nothing about real truncated backprop, but they are known
  and could shape expectations. The thresholds and contrasts below were fixed in v161 before any of it was seen, and
  are carried over unchanged.
- The exploratory drift probe (fp32 measurement, valid): training lowers ρ (0.80 → ~0.70 at R 16) and the law keeps
  holding during training.
- Parcae was pretrained with backprop through the last ≤ 4 of ~Poisson(8) loops (`mean_backprop_depth` 4), so k 4 at
  R 8 is the checkpoint's own regime.
- The program's history (Papers 1–2): truncated/immediate credit was often as good as exact credit, and an apparent
  credit effect several times turned out to be a readout effect. The k = 0 floor and gate G2 exist for that reason.
- TRM v4: an under-budgeted training run learned nothing. The pilot's G1 gate checks trainability before anything is
  frozen.

## 2. Apparatus (to be LOCKED at freeze)

### Model
SandyResearch/parcae-140m (prelude 2 + looped core 2 + coda 2 layers, d 768, diagonal injection, tied embeddings),
code from github.com/sandyresearch/parcae at commit 69284c1 (the PyPI wheel is empty). All parameters are trained.

### Truncation
Parcae's own forward with `num_steps_pair = (R − k, k)`: the first R − k loops run under `no_grad`, and the last k
carry gradient. Evaluation always runs all R loops forward. The recurrent state's random initialisation is part of the
model; for evaluation it is fixed by seed, identical across arms.

### Cells
| R | k (backprop loops) |
|---|---|
| 8 | 0 (floor), 1, 2, 4, 8 (full) |
| 16 | 0 (floor), 1, 2, 4, 8, 16 (full) |

The seven **truncated cells** (0 < k < R) are the treatment. k = R (full) and k = 0 (no loop credit; prelude,
injection and core get no gradient, while embeddings, C, coda and head still train) are controls.

### Data
GSM8K (openai/gsm8k, main): question + "\n" + answer, BOS-prefixed, concatenated and packed into 512-token chunks.
Train = first 7,000 train examples (reshuffled per seed and epoch), val = last 473 train examples (LR selection only),
test = all 1,319 test examples (headline).

### Training
AdamW (β 0.9/0.95, ε 1e-8, no weight decay), 20-step linear warm-up then cosine to 0.1 × LR, grad clip 1.0, batch 8 ×
512 tokens, **600 steps** (≈ 2.5M tokens, ≈ 1.5 epochs), fp16 autocast with a grad scaler (checked against fp32 in the
selftest: |Δloss| < 0.01).

### Metric
Per cell and seed: test loss L (mean next-token CE, all R loops forward).
- Cost of truncation: ΔL_{R,k} = L_{R,k} − L_{R,R} (paired: same seed, same data order).
- **Primary: fraction of loop credit lost**, φ_{R,k} = ΔL_{R,k} / (L_{R,0} − L_{R,R}). φ 0 = as good as full
  backprop; φ 1 = as bad as no loop credit. This is scale-free and comparable across R.
- Raw ΔL is reported alongside it.

### Predictor (measured pre-freeze, on the frozen checkpoint)
On 16 GSM8K-val chunks (batch 1, fp32, all parameters), per sequence: the full gradient (k = R), the gradient for each
k < R, the relative error err_k, and the adjoint norms ‖dL/dh_t‖ for t = 0…R−1. ρ_seq = exp(−slope) of a log-linear fit
of the adjoint norms over t. **Law value** law_{R,k} = median over sequences of ρ_seqᵏ. The measured median err_k is a
secondary predictor. The same measurement is repeated post hoc on the end state of a full-backprop run (seed 1,
reported only).

### Seeds and LR selection
- Tuning seeds 1621001, 1621002 (new; v161 used 16110xx): every cell at three LRs {lr*/3, lr*, 3·lr*}, with lr* = the
  full-backprop arm's best val LR in the v161 pilot, per R. **lr* = 3e-4 at both R → grid {1e-4, 3e-4, 9e-4}.**
- Each cell's LR = the grid LR with the lowest mean val loss over the two tuning seeds. **Edge rule:** if that is a
  grid end, the grid is extended one 3× step on that side for that cell (tuning seeds), once.
- Evaluation seeds 1621101–1621105 (5, new): every cell at its selected LR. All headline numbers use evaluation seeds
  only. All 11 cells, including the controls, are re-run on the v162 driver (no v161 run is reused).
- A run whose val or test loss is non-finite counts as diverged; a diverged evaluation run is reported and its seed is
  dropped from that cell's contrasts.

### Budget rule
- If pilot timing × (66 tuning + 55 evaluation runs) exceeds 40 T4-hours, cut evaluation seeds to 4, then the LR grid
  to {lr*, 3·lr*} if still over. **Pilot timing: ≈ 6.7 T4-h tuning + ≈ 5.6 T4-h evaluation (+ extensions) ≈ 13 T4-h:
  no cut.**
- Fixed before the freeze; never revisited after a treatment result.

### Platform
Kaggle T4 ×2 (one run per GPU), both accounts. No laptop runs in the study (the local 3050 ran the TINY pipeline
test only).

### Pre-freeze pilot (v161, controls only; kernel `astrocyte-v161-pilot`; valid for v162, see §1)
1. Selftest: hand-written decomposition = Parcae forward (losses and gradient); fp16 vs fp32 loss.
2. Frozen-model measurement at R 8 and 16 (the predictor above), and step-0 val/test losses.
3. LR sweep {1e-5, 3e-5, 1e-4, 3e-4} of the **two control arms only** (k = R and k = 0) at R 8 and 16, pilot seed 9001
   (not a study seed).

The pilot sets lr* and the run time, and decides gates G1 and G2. No truncated cell is run before the freeze.

## 3. Criteria (to be LOCKED at freeze)

### Gates (decided on the pilot, before the freeze)
- **G1 trainability:** at both R, the full arm's best pilot test loss is ≥ 0.10 nats below the step-0 test loss.
- **G2 loop credit matters:** at both R, L_{R,0} − L_{R,R} ≥ 0.03 nats (best LR of each arm, pilot seed). If the loop's
  credit is worth less than that, no truncation level can cost anything measurable. The study is not run, and the
  outcome is recorded as UNINFORMATIVE (the pilot result is itself reported).
- **Re-checked on evaluation seeds:** L_{R,0} − L_{R,R} > 0, paired 90% CI excluding 0, at both R. If it fails,
  the result is UNINFORMATIVE regardless of the hypotheses.

### Statistics
- Paired over the 5 evaluation seeds.
- A contrast's label is set by its two-sided 90% t-interval (df 4) and its mean:
  - **directional** if the interval excludes 0 and |mean| ≥ 0.002 nats (raw ΔL) or ≥ 0.02 (φ);
  - **TIE** otherwise.

### H0 — truncation costs something
φ_{R,1} > 0 (directional) at both R. If not, last-loop backprop is as good as full backprop for this fine-tune and
there is nothing for the law to predict: UNINFORMATIVE for H1–H3. This is reported as a finding in its own right.

### H1 — consistency
Spearman correlation between law_{R,k} and the seed-mean φ_{R,k} over the seven truncated cells ≥ 0.75.

### H2 — the law against "k alone" (matched-k pairs)
- Pairs: (R 8, k) vs (R 16, k), for k = 1, 2, 4.
- The predicted sign of φ_{16,k} − φ_{8,k} is sign(law_{16,k} − law_{8,k}), fixed from the pilot measurement.
- A pair is **discriminating** only if |law_{16,k} − law_{8,k}| ≥ 0.05. Pairs below that are reported but not scored.

### H3 — the law against "k/R" (matched-fraction pairs)
- Pairs: (R 8, k 2) vs (R 16, k 4), and (R 8, k 4) vs (R 16, k 8).
- Predicted sign and discriminating rule as in H2.

### Outcome labels (first that applies)
1. **UNINFORMATIVE:** G1 or G2 fails (pilot), the floor re-check fails, or H0 fails.
2. **AGAINST:** H1 fails (Spearman < 0.75), **or** any discriminating H2/H3 pair is directional *opposite* to the law.
3. **LAW:** H1 passes, every scored family (H2, H3) with at least one discriminating pair has at least one pair
   directional *in* the law's direction, and none are opposite.
4. **LAW-CONSISTENT:** H1 passes; no discriminating pair is opposite, but none is directional either, or neither
   family had a discriminating pair. The law fits the ordering, but this design cannot separate it from the simpler
   rules.

### Reported, not headline
- Raw ΔL in place of φ, for all of the above.
- Measured err_k in place of law_{R,k} as the predictor.
- A one-parameter fit φ ≈ c·law_{R,k} per R and pooled.
- Training curves; divergence counts; the post-hoc end-state ρ of the full arm; run time per cell (the compute that
  truncation saves).

## 4. Consequences

| outcome | consequence |
|---|---|
| LAW | "Measure ρ, then choose k" has training-level support on one model. The write-up (short note) states the frozen-model law and this test. A larger-scale test (v160-style) becomes worth pricing. |
| LAW-CONSISTENT | The note reports the frozen-model law. The training claim is limited to "the cost ordering is consistent with ρᵏ"; no claim over the simpler rules. |
| AGAINST | The note reports err_k ≈ ρᵏ as a gradient-level fact only and states that it did not predict training cost here. The method claim is dropped. |
| UNINFORMATIVE | As LAW-CONSISTENT for the note, with the reason disclosed. If H0 failed, that is reported: for this fine-tune, last-loop backprop suffices. |

## 5. No-post-hoc clause

No change to cells, steps, data, LR grid, seeds, metrics, gates or labels after the freeze. Additional analyses
are labelled exploratory. A failed gate is not repaired by changing the task, the step count or the floor after
seeing treatment cells. Before the freeze, only the pilot's control arms and the frozen-model measurement may be
inspected.

## 6. Design decisions (disclosed)

- **Parcae-140m:** the smallest open looped LM in a family where the law is already shown at 1.3B. It trains on one
  T4, and its own code accepts (no-grad, with-grad) loop counts.
- **GSM8K text:** a domain shift from FineWeb-Edu, so the loss moves in a few hundred steps, and it is reasoning-style
  text, the setting looped models are built for. The headline is loss, not accuracy.
- **φ primary:** raw ΔL depends on how much the loop's credit is worth at each R. φ normalises by the floor gap, so R 8
  and R 16 cells are comparable, which the H2/H3 contrasts require.
- **All parameters trained:** a practitioner trains the whole model, and truncation also cuts the gradient to the
  prelude and injection through the dropped loops.
- **Per-cell LR:** truncated gradients have different scale and noise. The program's rule is to tune before comparing.

## 7. Predictions (on record at freeze)

Scored pairs (|law difference| ≥ 0.05) and the law's predicted sign, from the locked law values in §8:

| family | pair (a − b) | law_a − law_b | scored | predicted sign of φ_a − φ_b |
|---|---|---|---|---|
| H2 | (16, 1) − (8, 1) | +0.029 | no | (+) |
| H2 | (16, 2) − (8, 2) | +0.046 | no | (+) |
| H2 | (16, 4) − (8, 4) | +0.057 | **yes** | + |
| H3 | (8, 2) − (16, 4) | +0.183 | **yes** | + |
| H3 | (8, 4) − (16, 8) | +0.185 | **yes** | + |

**Weakness, disclosed before any result:** ρ barely differs between R 8 and 16 on the 140m model (0.773 vs 0.802), so
H2 has a single scored pair, and only just (0.057). The design separates the law from "k/R" well and from "k alone"
weakly. In the H2 k = 4 pair, "k/R" makes the same prediction as the law, so only "k alone" is tested there.

**My outcome probabilities for v162** (recorded before any v162 run; v161's buggy truncated results are known):

| outcome | P |
|---|---|
| LAW | 0.30 |
| LAW-CONSISTENT | 0.40 |
| AGAINST | 0.20 |
| UNINFORMATIVE | 0.10 |

**Specific forecasts:**
- With real gradients, truncation costs less than in the buggy v161 arms: φ_{8,4} < 0.10 (P 0.7). v161's buggy value
  was 0.21.
- φ_{8,1} < 0.5 while err 0.81 (P 0.7).
- The H3 pairs show the law's sign directionally: (16, 8) cheaper than (8, 4) (P 0.5); (16, 4) cheaper than (8, 2)
  (P 0.45).
- The H2 pair ties (P 0.6).

## 8. v161 pilot results (controls only; carried over)

Kernel `aurshalevmerin/astrocyte-v161-pilot`, driver sha256 ddb4e99b660b (unchanged for the study), T4 ×2.

**Selftest.** Decomposition = Parcae forward at R 4, k ∈ {4, 2, 0} (2.748588 both ways). fp16-autocast loss 2.68950 vs
fp32 2.68944. Gradient of the hand-written forward vs the model's: relative difference 6.6e-7.

**Frozen-model measurement (the locked predictor; 16 GSM8K-val chunks, 512 tokens, all parameters, fp32):**

| R | ρ (median) | k = 1 err / law | k = 2 | k = 4 | k = 8 |
|---|---|---|---|---|---|
| 8 | 0.773 | 0.805 / 0.773 | 0.648 / 0.597 | 0.386 / 0.356 | — |
| 16 | 0.802 | 0.813 / 0.802 | 0.663 / 0.643 | 0.421 / 0.414 | 0.180 / 0.171 |

The law holds on Parcae-140m too (|err − law| 0.01–0.05): a third checkpoint after Huginn-3.5B and Parcae-1.3B.
These law values are hard-coded in `v161_analyze.py` and, unchanged, in `v162_analyze.py`.

**Control-arm LR sweep (pilot seed 9001, 600 steps; test loss):**

| R | arm | 1e-5 | 3e-5 | 1e-4 | 3e-4 | time/run |
|---|---|---|---|---|---|---|
| 8 | full (k 8) | 1.716 | 1.510 | 1.403 | **1.376** | 416 s |
| 8 | floor (k 0) | 2.126 | 1.905 | 1.728 | **1.623** | 200 s |
| 16 | full (k 16) | 1.623 | 1.493 | 1.420 | **1.376** | 705 s |
| 16 | floor (k 0) | 2.021 | 1.830 | 1.694 | **1.585** | 305 s |

Step-0 test loss: 2.771 (R 8), 2.621 (R 16).
- **G1 PASS:** the full arm drops 1.39 (R 8) and 1.24 (R 16) nats.
- **G2 PASS:** the floor gap is 0.247 (R 8) and 0.209 (R 16) nats, 7–8× the threshold.
- Both arms' best LR is the top of the pilot grid. The study grid is centred there and keeps the edge rule.
- Full-backprop runs had 1–8 non-finite gradient steps out of 600, skipped by the fp16 grad scaler (floor arm: 0). These are
  reported per run.

(v161 freeze contents: superseded by §9.)

## 9. v162 pre-freeze checks and freeze contents

**v162 pilot** (kernel `astrocyte-v162-pilot`, driver `v162_parcae_trunc_train.py`). Two checks, both on controls or
frozen weights only:
1. Selftest, extended. Besides the v161 checks, it computes the **truncated** gradient on the training path (Parcae's
   forward, train mode, fp16 autocast) and in fp32 at (R, k) = (8, 7), (8, 4), (16, 8), (16, 1). The per-group
   relative difference (looped core, injection, prelude, rest) must be < 0.05, and the truncation error must agree to
   0.02. A **negative control** re-runs (8, 4) with the cache on (the v161 setting) and must show the bug (core
   relative difference > 0.5), or the check proves nothing.
2. Control identity. Full backprop at R 8, LR 3e-4, pilot seed 9001, 600 steps, on the v162 driver must reproduce the
   v161 pilot's test loss 1.3759 to within 0.005. That shows the fix does not change the unaffected arms.

If either fails, nothing is frozen.

**Results** (pilot v2, `aurshalevmerin/astrocyte-v162-pilot`, driver sha 0ae4891d7abc LF, both T4s; v1 OOM'd in the
R 16 part of the check, which held four fp32 gradient copies on the GPU; fixed by keeping them on CPU and using batch 4
at R 16):

| (R, k) | per-group rel. diff fp16 vs fp32 (core / injection / prelude / rest) | truncation err fp16 / fp32 |
|---|---|---|
| (8, 7) | 0.0035 / 0.0029 / 0.0030 / 0.0068 | 0.1151 / 0.1151 |
| (8, 4) | 0.0039 / 0.0030 / 0.0032 / 0.0068 | 0.3379 / 0.3380 |
| (16, 8) | 0.0027 / 0.0023 / 0.0026 / 0.0036 | 0.1236 / 0.1234 |
| (16, 1) | 0.0034 / 0.0031 / 0.0032 / 0.0036 | 0.7895 / 0.7895 |

- **Negative control** (cache on, the v161 setting), (8, 4): looped-core relative difference vs fp32 **0.991**. The bug
  reproduces, so the check is sensitive to it.
- **Control identity:** full, R 8, 3e-4, seed 9001 gives test **1.3782** vs v161 pilot 1.3759 (|Δ| 0.0023 < 0.005)
  **PASS**. 4 non-finite steps out of 600, skipped by the grad scaler.
- **Frozen-model measurement** reproduces the locked law values exactly (ρ 0.7726 / 0.8020; every err and law value
  identical to 4 decimals). Step-0 losses identical.
- Selftest decomposition and k = R gradient checks as in v161 (2.748588 both ways; 1.7e-7, 2.1e-7).

All checks pass, so the study is frozen as specified.

**Freeze contents:** this document, `v162_parcae_trunc_train.py`, `v162_analyze.py` (v161's analysis with v162 seeds;
same constants, law values, pairs and labels), and `build_kaggle_v162.py`.
