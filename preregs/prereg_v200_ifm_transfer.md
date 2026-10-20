# PREREG v200: does a cost curve fitted on Parcae predict truncation cost on two new looped models?

**Status: FROZEN** (this commit), before any scored run and before the spread pilot's results were read. The spread
pilot (seeds 9001–9003, still running at the freeze) feeds only the seed rule in §1; the scored study uses other
seeds. The freeze commit is pushed to origin before any scored run.

**Related:**
- v198 (`docs/v198_ifm_fade_result.md`): frozen-model measurements on both models; both TRANSFER-ONLY;
- `docs/v200_plan_notes.md`: apparatus;
- paper v15.1 §4 and Appendix H.

## 0. Why, and what is NOT the test

Within one model, the lost gradient at each k barely depends on the loop count (S10). On its own it is therefore a
relabelling of k, and it can beat "k alone" only through an intervention such as blending. Across models it is
different: the same k costs 22% of the loop's benefit on Parcae-140m and 1% on SmolLM2 (A).
- S5 passed against a weak rival and could not separate the lost gradient from a·k.
- S6 failed under blending, where the gradient mostly shrank.
- The cross-model agreement at S6's unblended cells was "not the test" by S6's own prereg: near-zero cost on a fast
  loop is expected on any account.

v198 found two looped models that fade at rates between the two earlier families. On them, the lost gradient and the
direction after Adam's rescaling give clearly different locked predictions, and those differ again from both
measurement-free rules:

| prediction at k = 1 | pln5 | r5 |
|---|---|---|
| L, lost-gradient curve | 0.138 | 0.182 |
| A, direction-after-Adam curve | 0.258 | 0.413 |
| KP, Parcae's own cost ("k alone") | 0.224 | 0.224 |
| KS, SmolLM2's own cost ("k alone") | 0.009 | 0.009 |

On pln5 alone, A and KP nearly coincide. On r5 alone, L and KP nearly coincide. The two models together separate all
four.

**Not the test:**
- Blending. On these models blending at k = 1 does not act as a pure shrink (v198), so S6's blended contrast cannot be
  repeated.
- Separating the lost gradient from ρᵏ or any other measure of the fade. The ρᵏ-based curve predicts within about 0.02
  of L here, so this test cannot tell them apart.

## 1. Models and apparatus (LOCKED)

**Models:**
- IFM/LoopedLM-P2-huginn-s-fixed-pln5, revision c581f369;
- IFM/LoopedLM-P2-huginn-s-fixed-r5, revision 17cdf78e.

Both use our port `ifm_looped.py` (sha 502739cdccc8), checked against the reference code in v198.

**Driver:** `v200_ifm_trunc.py` (file sha bd9ee68769e0; driver_sha() over driver + port 22a56eba93e9, as logged by the
frozen-embedding pilots). The recipe is v162–v167's:
- GSM8K question + answer (BOS + text + EOS, attention within documents); the last 473 train examples = val; the test
  split = test;
- 600 steps, batch 8 × 512;
- AdamW (0.9, 0.95), weight decay 0; 20-step warm-up, then cosine to 10%; clip 1.0;
- fp16 autocast with the weight cache off, plus GradScaler;
- R = 5, no blending.

**The token embedding is frozen in every arm** (review point, settled before the freeze). IFM's embedding is untied
(98.3M of 303.6M parameters) and reaches the loss only through the loop. So the floor could not train it, while
truncated arms could. That would inflate the floor gap and shrink every cost compared with Parcae, whose tied
embedding trained in every arm. Frozen everywhere, the cost compares loop credit alone. Full backprop then trains
205.3M parameters.

**Cells:**
- k = 0: the floor, no loop gradient. Coda, final norm and head train, 124.7M parameters.
- k = 1, 2, 4: truncated.
- k = 5: full backprop.

**Learning rates** (seed 9001; v167's rule: lowest val loss per arm; truncated cells use the full-backprop LR):
- **Full backprop: 1e-4.** Val loss on pln5 / r5:

  | LR | 1e-5 | 3e-5 | 1e-4 | 3e-4 |
  |---|---|---|---|---|
  | pln5 | 1.5106 | 1.4449 | **1.4320** | 1.5258 |
  | r5 | 1.4875 | 1.4273 | **1.4207** | 1.5256 |

  This was measured with the embedding frozen (pilot-lr3). It is an interior optimum. Pilot 1 (trainable
  embedding) had put 1e-4 at the grid edge; its downward extension, pilot-lr2, agreed on 1e-4.
- **Floor: 3e-4.** Val loss on pln5 / r5:

  | LR | 1e-4 | 3e-4 | 9e-4 |
  |---|---|---|---|
  | pln5 | 1.5759 | **1.5623** | 1.6785 |
  | r5 | 1.5715 | **1.5611** | 1.6834 |

  Pilot 1. The floor never trains the embedding, so the freeze does not change it.
- **Pilot floor gap** (test, seed 9001): 0.128 nats on pln5 and 0.141 on r5. Parcae's was 0.18 and SmolLM2's 0.08.

**Seeds (rule fixed at the freeze; `v200_analyze.seed_rule`).** From the spread pilot, only one number is read: s, the
larger over the two models of the sample s.d. of φ at k = 1 over seeds 9001–9003 (test). Its means are not read
before the scored run.
- N = 5 if s ≤ 0.045 (s.e. ≤ 0.02 at k = 1, half the smallest gap that matters, about 0.04);
- N = 8 if 0.045 < s ≤ 0.06;
- N = 10 otherwise.

The study seeds are pre-listed as 2001101, 2001102, … in order; each model runs the first N + 2 (2 backups). The same
seeds are used in every cell of a model, and the pilot seeds are not reused. N is passed to the analysis as
V200_N_SEEDS.

**Every job runs the selftest before training:**
- the training forward equals the model's own forward;
- checkpointing changes nothing;
- fp16 vs fp32 truncated gradients agree;
- the floor gives the loop zero gradient;
- the embedding is frozen.

The cache-on negative control is informational on Kaggle's PyTorch, where the v161 bug no longer reproduces. The
check's sensitivity is shown on a synthetic version of the same failure, the loop weights getting no gradient, which
must be flagged at ≥ 90%.

**Exclusion (v165 rule).** A seed is valid if its full-backprop control has ≤ 10 non-finite steps, finite val and test
losses, and did not diverge. The first N valid seeds in the pre-listed order are used.

## 2. Predictors (LOCKED in `v200_analyze.py`, sha 8fc6df477ab6)

The curves were fitted on Parcae before v198 existed. The inputs are frozen-model medians at R = 5, a = 1, over the
parameters that train (the embedding frozen). They come from v198's measurement re-run with V198_FREEZE_EMBED=1
(kernels astrocyte-v198-ifm-pln5-pred and -r5-pred; v198_ifm_fade.py sha e7e1e5b75375; selftests PASS; noise floor 0).
They differ from v198's all-parameter values by at most 0.006.

| | pln5 k = 1 / 2 / 4 | r5 k = 1 / 2 / 4 |
|---|---|---|
| lost gradient | 0.7673 / 0.5954 / 0.2923 | 0.8317 / 0.6934 / 0.3668 |
| direction after Adam, Ā (24 clipped batches) | 0.7567 / 0.8646 / 0.9721 | 0.6722 / 0.7946 / 0.9571 |
| **L** = 0.3418 · err^3.4335 (S6 curve, prereg v167) | 0.138 / 0.058 / 0.005 | 0.182 / 0.097 / 0.011 |
| **A** = 2.403 · (1 − Ā)^1.58 (v171b, 29 Parcae cells) | 0.258 / 0.102 / 0.008 | 0.413 / 0.197 / 0.017 |
| **KP**: Parcae-140m's own cost (R = 32) | 0.224 / 0.100 / 0.024 | same |
| **KS**: SmolLM2 (A)'s own cost (R = 8; k = 4 not measured, set to 0) | 0.009 / −0.001 / 0 | same |

**Fitting ranges (review point, settled before the freeze).** The project does not score a curve outside the inputs
it was fitted on; Ouro was not trained for this reason (Appendix E, D3).
- L was fitted on lost gradients 0.180–0.877 (`v167_analyze.FIT`). Every IFM input (0.292–0.832) lies inside, so the
  transfer verdict uses all 6 cells.
- A was fitted on directions after Adam 0.7778–0.9671 (`results_v171b`, 29 Parcae cells). Three IFM inputs lie
  outside:
  - pln5 k = 1: 0.757;
  - pln5 k = 4: 0.972;
  - r5 k = 1: 0.672.
- **The which-curve verdict (L vs A)** is therefore scored on the 3 in-range cells (pln5 k = 2, r5 k = 2, r5 k = 4) as
  the primary result. All 6 cells, with A's 3 extrapolations, are a secondary result reported as such.

## 3. Criteria (LOCKED; `v200_analyze.py --selftest` passes on synthetic worlds)

**Floor gate per model.** Mean(L_floor − L_full) on test ≥ 0.03 nats, with the 90% t-interval excluding 0. A model
that fails is UNINFORMATIVE, and its cells are dropped.

**Per cell.** φ = mean over the valid seeds of (L_k − L_full) / (L_floor − L_full), on test.

**Score.** For each predictor:
- MAE over the scored cells;
- the number of calibrated cells, where calibrated means |φ − prediction| ≤ max(0.02, 0.3·φ).

Fewer than 4 scored cells makes the transfer verdict UNINFORMATIVE.

**Transfer verdict** (all scored cells; applied in order):

| verdict | rule |
|---|---|
| L-TRANSFERS | MAE(L) ≤ 0.5 · min(MAE(KP), MAE(KS)), and L is calibrated in ≥ n − 1 cells |
| L-BEATS-K | MAE(L) ≤ 0.5 · min(MAE(KP), MAE(KS)) |
| K-AS-GOOD | min(MAE(KP), MAE(KS)) ≤ MAE(L) |
| INCONCLUSIVE | otherwise |

**Which-curve verdict.** Primary: MAEs over the cells where both curves' inputs are inside their fitting ranges (at
least 3 cells, else UNINFORMATIVE). Secondary: MAEs over all scored cells, with A extrapolated in 3 of them.

| verdict | rule |
|---|---|
| A-BETTER | MAE(A) ≤ 0.5 · MAE(L) |
| L-BETTER | MAE(L) ≤ 0.5 · MAE(A) |
| TIE | otherwise |

The best predictor (lowest MAE over all scored cells) is reported too.

## 4. What each outcome changes in the paper

- **L-TRANSFERS:** §4 gains a pre-registered cross-model result. The lost-gradient curve fitted on one model
  predicts truncation cost on a new family, against both measurement-free rules. This is the evidence the cost half
  lacks.
- **A-BETTER (in range):** the direction-after-Adam lead becomes a pre-registered finding on the cells where the curve
  applies, and the lost gradient is demoted for cost prediction.
- **K-AS-GOOD, with KP best:** cost follows k across models. The paper states that the lost gradient adds nothing for
  cost even across models.
- **KS best:** these loops barely need gradient in fine-tuning. The paper reports the test as uninformative about the
  curves.

## 5. Predictions (mine, at freeze)

- **Transfer:** L-TRANSFERS 0.30, L-BEATS-K 0.10, K-AS-GOOD 0.40, INCONCLUSIVE 0.15, UNINFORMATIVE 0.05.
- **Which curve (primary, in range):** L-BETTER 0.30, A-BETTER 0.35, TIE 0.35.
- **Main risks:**
  - the curves' scale may not carry across families: different floor gap and recipe defaults, R = 5 instead of 16–32;
  - at R = 5 the loop is short, so k = 4 cells are near zero for every predictor and add little;
  - the A curve's distinctive k = 1 predictions are extrapolations, so the in-range primary rests on 3 cells, mostly
    the two k = 2 cells.

## 6. Cost

2 models × 5 cells × (N + 2) seeds, at the measured 10–12 min per full run and 5–6 min per floor run on a T4. About
14 T4-hours for N = 5, 20 for N = 8 and 24 for N = 10.
