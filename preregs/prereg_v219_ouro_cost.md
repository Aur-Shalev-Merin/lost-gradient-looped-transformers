# PREREG v219: what does truncated backprop cost on Ouro-1.4B, the model that loses the most gradient?

**Status: FROZEN** (this commit), before any scored run. The freeze commit is pushed to origin before the first scored kernel.
Pilots read before the freeze, none on the scored seeds:
- the v219 feasibility pilot: memory, speed, and 40-step held-out losses;
- the LR pilot (seed 9001): validation and test losses per learning rate, and the step-0 lost-gradient measurement.

**Code (sha256 prefixes, LF):**
- `v219_ouro_cost.py` (driver and locked analysis): f8543506d909;
- `build_kaggle_v219.py`: 8fc409c007aa;
- imported unchanged: `loop_grad_probe.py` 27b80fa0b999.

## 0. Why

Section 5 of the paper (arXiv:2610.28871) asks whether the lost gradient says what truncation costs in fine-tuning.
- S11 tested this across models. A curve fitted on Parcae (S6: cost = 0.3418 · err^3.4335) predicted the Part II
  models' costs no better than Parcae's own cost at the same k ("k alone"; K-AS-GOOD, a near-tie).
- Ouro-1.4B is the model whose backward signal grows instead of fading. The paper says "we did not train Ouro with
  truncation", because its lost gradient lies outside the curve's fitted range (Appendix E, D3).
- v219 trains Ouro with truncation and scores the two rules on it.

**Ouro's measured lost gradient** (step 0, embedding frozen as in the study, the parameters that train, final-output
loss, fp32, batch 1, median over the first 32 GSM8K test chunks; `results_v219/lr-a`):

| k | Ouro | r5 | Parcae-140m (R = 32) | pln5 | SmolLM2 (A) |
|---|---|---|---|---|---|
| 1 | **0.921** | 0.832 | 0.81 | 0.767 | 0.36 |
| 2 | **0.839** | 0.693 | 0.65 | 0.595 | 0.13 |

Ouro loses the most gradient at both k. The S6 curve was fitted on lost gradients 0.180–0.877, so Ouro's k = 2 input
is inside the range and its k = 1 input (0.921) is an extrapolation. That is deliberate here: the question is what
happens at the extreme. It is reported as such.

## 1. Models and apparatus (LOCKED)

**Model:** ByteDance/Ouro-1.4B at revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1. That is the revision of every
pilot run, and the selftest fails on any other.
- 24 layers looped 4 times (R = 4); untied embedding and output head; 1.43B parameters.
- Run in fp32 and split across the kernel's two T4s:
  - GPU 0: the embedding and layers 0–11;
  - GPU 1: layers 12–23, the final norm, the exit gate and the output head.
- Transformers 4.54.1.

**Recipe** (Section 5's, as v162–v167 and v200):
- **Data:** GSM8K question + "\n" + answer, packed into 512-token chunks.
  - Training: the train split minus its last 473 problems, 4 shuffled epochs per seed.
  - Val: those last 473 problems.
  - Test: the test split, 1,319 problems.
- **Training:** 600 steps, batch 8 × 512 in micro-batches of 2 with gradient accumulation.
- **Optimiser:** AdamW (0.9, 0.95), weight decay 0; 20-step warm-up, then cosine to 10%; clip 1.0.
- **Precision:** fp16 autocast with the weight-cast cache off, plus GradScaler.
- **Training loss:** cross-entropy of the last loop's output (Section 5's objective).
- **The token embedding is frozen in every arm**, as in v200. It is untied and reaches the loss only through the loop,
  so only full backprop could otherwise train it.

**Cells:**
- k = 4: full backprop.
- k = 2, 1: backprop through the last k of the 4 loops; the earlier loops run without gradient.
- k = 0: the floor, no loop gradient. Only the output head trains (103M parameters). The final norm sits inside every
  loop, so it gets no gradient here either.

Full backprop trains 1.33B parameters.

**Learning rates** (seed 9001; one LR for full and truncated cells, as in Section 5; the floor tuned separately, as in
v200). Losses are final-output cross-entropy:

| full backprop, LR | 3e-6 | 1e-5 | 3e-5 | **1e-4** | 3e-4 |
|---|---|---|---|---|---|
| val | 0.483 | 0.553 | 0.757 | 0.983 | 1.172 |
| test | 1.124 | 1.098 | 1.037 | **1.030** | 1.168 |

| floor, LR | 1e-5 | 3e-5 | **1e-4** | 3e-4 | 1e-3 | 3e-3 |
|---|---|---|---|---|---|---|
| val | 0.838 | 0.810 | 0.790 | 0.798 | 0.840 | 0.958 |
| test | 1.108 | 1.079 | **1.062** | 1.072 | 1.118 | 1.254 |

Base model before fine-tuning: val 0.891, test 1.162.

- **Full backprop LR: 1e-4. Floor LR: 1e-4.** Each is the lowest test loss on the pilot seed, and both are interior
  optima.
- **Deviation from v200 (selection on test, not val).** On Ouro the val split does not behave as held-out text:
  - before any fine-tuning it is 0.27 nats easier than test (0.891 against 1.162);
  - under full backprop it moves opposite to test (val is best at the lowest LR, 3e-6, where test is worst).
  The likely reason is that Ouro saw GSM8K's train split in its own training, and val comes from that split.
  Selecting on val would have chosen 3e-6, where full backprop does worse on test than the floor (1.124 against 1.062),
  and the study could not measure a cost.
- **Why test selection is acceptable here.** The scored seeds are new (§1, Seeds). At 40 steps the seed-to-seed
  spread of full backprop's held-out loss was 0.0003 nats, far below the gaps between the LRs compared. Val losses are
  still reported.

**Pilot floor gap** (test, seed 9001): 1.062 − 1.030 = 0.032 nats. This is small next to the other models' (Parcae
0.18, Part II 0.13–0.14, SmolLM2 0.08). Most of the fine-tuning gain on test (0.10 of 0.13 nats) comes from the output
head alone.

**Seeds (rule fixed at the freeze; `analyze` in the driver).**
- **Pre-listed:** 2191001–2191007, used in order; each seed runs all four cells.
- **N = 3**, raised to **N = 5** if the s.e. of the k = 1 cost over the first 3 valid seeds exceeds 0.02. The rule reads
  only that s.e., and seeds 4 and 5 are then run unchanged.
- **Exclusion (v165 rule):** a seed is valid if its full-backprop run has ≤ 10 non-finite steps and no run of that
  seed has a non-finite val or test loss.

**Every job runs the selftest before training** (`--phase selftest`; all passed on the 4 pilot kernels):
- the model revision is the one above;
- the split training forward equals Ouro's own forward on one device (exit_at_step = 3);
- checkpointing changes no gradient;
- fp16 vs fp32 gradients agree by v200's rule (loop < 0.5 and ≤ 3 × head + 0.02; loss within 0.01; pilot values for
  the loop: 0.007 / 0.009 / 0.028 at k = 1 / 2 / 4);
- a synthetic loss of all loop gradient is flagged (relative difference 1.0 > 0.9);
- the floor gives the loop exactly zero gradient and trains the head;
- k = 1 reaches every looped tensor;
- the embedding never gets a gradient.

## 2. Predictors (LOCKED in the driver)

| | k = 1 | k = 2 |
|---|---|---|
| **L**, the S6 lost-gradient curve at Ouro's measured lost gradient | 0.258 (extrapolated) | 0.187 |
| **KP**, Parcae-140m's own cost at the same k ("k alone"; Table 9) | 0.224 | 0.100 |
| other trained models, for description only: r5 / pln5 / SmolLM2 (A) | 0.291 / 0.162 / 0.009 | 0.149 / 0.071 / −0.001 |

## 3. Criteria (LOCKED; `python v219_ouro_cost.py --analyze <outputs>`)

**Cost per cell:** φ_k = mean over the scored seeds of (L_k − L_full) / (L_floor − L_full), on test, final-output
loss. Each mean comes with a 90% t-interval over seeds.

**Floor gate:** the mean of (L_floor − L_full) on test must be ≥ 0.03 nats, with the 90% lower bound > 0, as in v200.
If it fails, the verdict is UNINFORMATIVE.

**Verdict** (on k = 1 and k = 2; the first rule that applies):

| verdict | rule |
|---|---|
| BELOW-BOTH | at both k, the 90% upper bound of φ_k is below min(L_k, KP_k) |
| ABOVE-BOTH | at both k, the 90% lower bound of φ_k is above max(L_k, KP_k) |
| L-CLOSER | MAE(L) ≤ 0.5 · MAE(KP), over the two cells |
| KP-CLOSER | MAE(KP) ≤ 0.5 · MAE(L) |
| TIE | otherwise |

**Also reported, not scored:**
- φ under Ouro's exit-weighted objective (without its entropy term);
- Ouro's rank among the trained models at each k;
- val losses;
- per-seed values.

**Checked on synthetic worlds before the freeze:** BELOW-BOTH, L-CLOSER, KP-CLOSER, ABOVE-BOTH and TIE each come out as
built, and a noisy world triggers N = 5.

## 4. What each outcome changes in the paper (v2)

In every case §1, §5 and §7 drop "we did not train Ouro with truncation" and report φ with the gate. An appendix
records the val/test discrepancy and the LR-selection deviation.

| verdict | paper change |
|---|---|
| BELOW-BOTH | §5 and the abstract: on the model that loses the most gradient (92% and 84%), truncation cost less than both the lost-gradient curve and "k alone" predict. This is the clearest case so far that the size of the lost gradient and the training cost come apart. |
| ABOVE-BOTH | §5 and the abstract: the model that loses the most gradient paid more than either rule predicts. This is cost evidence for the lost gradient at the extreme, with k = 1 an extrapolation. |
| L-CLOSER | §5: the Parcae-fitted curve predicts Ouro's cost better than "k alone". This is the first cost result where the measurement beats k, on one model, with the k = 1 input outside the fitted range. |
| KP-CLOSER | §5: "k alone" holds on Ouro too, despite its far larger lost gradient. |
| TIE | §5: Ouro's cost lies between the two rules; the test cannot separate them. |
| UNINFORMATIVE | §5 and §7: on Ouro's GSM8K fine-tune, loop credit adds less than 0.03 nats over training the head alone, so no cost can be measured. Reported with the numbers. |

## 5. Predictions (mine, at the freeze; made after the pilots above)

- UNINFORMATIVE 0.10;
- BELOW-BOTH 0.30;
- KP-CLOSER 0.25;
- TIE 0.17;
- L-CLOSER 0.12;
- ABOVE-BOTH 0.06.

**Why UNINFORMATIVE is only 0.10 although the pilot gap (0.0325) is close to the 0.03 gate.**
- The floor gap barely varies across seeds, because a seed changes only the data order (no dropout).
  - In v200 the pilot seed's gaps (0.128 and 0.141) were reproduced by the 5 scored seeds (0.1285 and 0.1407,
    s.e. about 0.0005).
  - Ouro's two 40-step pilot seeds differed by 0.0003 under full backprop.
- So the 3-seed mean should land within about 0.002 of the pilot's.
- The gate stays at v200's 0.03 so that the two studies are comparable. Changing it after seeing the pilot near it
  would look like tuning the rule to the data.
- If the seed noise is like v200's, the s.e. of the k = 1 cost over 3 seeds is about 0.01, below the 0.02 that raises
  N to 5.

At 40 steps with an untuned floor, the feasibility pilot suggested about 0.09 at k = 2. The LR pilot ran only k = 4 and
k = 0, so no truncated cell has been seen at 600 steps.

**Main risks:**
- the floor gap is small, so the cost ratios are noisy, and the gate may fail;
- one model, two cells: a single verdict is weak evidence on its own;
- Ouro's familiarity with GSM8K train may make fine-tuning partly re-learning, unlike the other models.

## 6. Cost and runs

| kernel | account | seed | cells |
|---|---|---|---|
| astrocyte-v219-s1 | aurshalevmerin | 2191001 | 4, 2, 1, 0 |
| astrocyte-v219-s2 | aurmerin | 2191002 | 4, 2, 1, 0 |
| astrocyte-v219-s3 | aurmerin | 2191003 | 4, 2, 1, 0 |

Each kernel takes about 6.3 hours on both T4s: 15.5 / ~9.6 / ~6.5 / 3.6 s per step at k = 4 / 2 / 1 / 0, plus the
selftest and evaluations. About 19 kernel-hours in all, plus about 13 if the seed rule raises N to 5 (seeds 2191004
and 2191005, `build_kaggle_v219.py study-more`).
