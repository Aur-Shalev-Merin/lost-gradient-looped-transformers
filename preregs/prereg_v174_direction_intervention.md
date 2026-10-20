# Pre-registration v174 (DRAFT 3): does Adam-update agreement predict truncation's training cost? A gated dose–collapse test

**Author:** Aur Shalev Merin.
**Date:** 2026-10-05.
**Status:** DRAFT 3.1 (outside review of draft 3 applied). NOT frozen. The pilot (§2) must run first; its B values and the locked pairs are then added
(draft 4), the draft is sent to the outside reviewer, and it is frozen and pushed to origin before any scored run.

**Changes from draft 2** (outside review):
- a gate against collapse-by-construction (B);
- matching tightened to 0.00
- a full-length pilot;
- equivalence via TOST plus high-dose pairs;
- the agreement measure fully specified;
- frozen-model agreement logged;
- both tilt contrasts, with interpretations fixed in advance;
- the backup-seed rule.

## 0. Claim and scope

**Claim tested:** for errors of the truncation type, the training cost of an approximate gradient is a function of
the realised agreement Ā between its Adam update and the full-gradient Adam update.

**Scope:**
- AGREEMENT-PREDICTS is a claim about truncation-type errors (mixes of g_R toward g_k) only, whatever the tilt
  contrasts show.
- It does not extend to arbitrary gradient errors.

## 1. Design

**Model and recipe** (as v162–v166):
- Parcae-140m, R = 32, a = 1;
- GSM8K, 600 steps, batch 8 × 512;
- AdamW (0.9, 0.95), **weight decay 0** (the original v162–v166 recipe, `v163_parcae_damped_trunc.py:102`; not a
  change);
- warm-up + cosine; global-norm clip 1.0;
- fp16 autocast with cache off;
- per-loop checkpointing.
- Learning rates: 3e-4 for every arm except the floor (9e-4).

**Mix arm (k, λ).** At each step, compute g_R and g_k. The update gradient is
  g = g_R + λ(g_k − g_R), rescaled to |g_R|, then clipped, then AdamW.
- λ = 1 is truncation at k.
- λ > 1 is allowed only in the fallback (see the gate).

**Agreement A (the predictor), fully specified:**
- A = cos(u_arm, u_shadow).
  - u_arm is the arm's Adam step direction m̂/(√v̂ + ε) computed from the clipped g, before the learning-rate and
    schedule scaling.
  - u_shadow is the same quantity from a shadow AdamW state that is fed the clipped g_R every step and never updates
    parameters.
- No weight-decay term is included (wd = 0 anyway).
- **Parameter set:** all parameters, as in v171b, so it is the same quantity as the paper's predictor. Coda and head
  parameters receive identical gradients in both and inflate A. A loop-and-prelude-only A is logged and reported as a
  secondary.
- Ā = mean of A over steps 100–600.

**Frozen-model agreement.** Before training, every arm's frozen-model agreement is measured with the v171b procedure,
for the mix of its frozen g_R and g_k. It is reported as a secondary: does the frozen value predict as well as the
in-training Ā?

**Contrasts** (reported, not scored; each matched to one mid-dose mix arm's Ā within 0.005):
- **TILT-FIX:** g_R tilted toward one fixed random direction (a persistent, unstructured bias).
- **TILT-FRESH:** tilted toward a new random direction each step (variance).

**Controls:**
- FULL (λ = 0);
- FLOOR (no loop gradient);
- one-seed pipeline check SIZE-ONLY (|g_k|/|g_R| · g_R must match FULL).

**Seeds** (from the power calculation, §3a):
- pair arms, FULL and FLOOR: 1741101–1741108 (8 evaluation seeds);
- backup 1741109, used only if a seed is excluded by the v165 rule (the FULL control has > 10 non-finite steps), in
  which case it replaces the excluded seed in every arm;
- tilt contrasts (unscored): 3 seeds, 1741101–1741103.

## 2. Pilot (one non-scored seed 1749901, the full 600 steps, Ā logged, final losses NOT recorded)

1. **Selftests:**
   - A(FULL) = 1 exactly;
   - the mix construction is exact;
   - fp16 vs fp32 truncated gradients agree;
   - checkpointed = plain;
   - shadow Adam is exact.
2. **Candidate arms, k = 1 and 2 first:** λ ∈ {0.25, 0.5, 0.75, 1}, with Ā measured for each. The k = 4 candidates
   (λ ∈ {0.5, 0.75, 1, 1.25, 1.5}) run only if the gate fails for k = 1 vs 2.
3. **Matching:** interpolate λ (linear in Ā) to form k = 1 vs k = 2 pairs with |ΔĀ| ≤ 0.005, then re-measure the
   interpolated arms on the pilot seed.
4. **Gate (frozen-model quantities on both sides).** For each candidate pair:
   - B_f = cos between the two arms' Adam-step directions from the same frozen parameters (v171b procedure);
   - A_f = each arm's frozen-model agreement with full-gradient Adam (pair mean).
   - The pair counts as a test only if 1 − B_f ≥ 0.5 · (1 − A_f).
   - Also reported, for interpretability: the cosine between the two arms' error components
     (u_arm − u_full). When the arms are matched on agreement, the gate equals requiring this cosine ≤ 0.75.
5. **Fallback, frozen now:** if fewer than 4 k = 1 vs k = 2 pairs pass the gate, use k = 1 vs k = 4 (λ > 1 allowed for
   k = 4).
6. **Go/no-go:** if neither k = 1 vs 2 nor k = 1 vs 4 yields ≥ 4 gated pairs, STOP. Add one sentence to the paper
   (truncation errors at different k are too collinear to separate by agreement) and submit without v174.
7. **Otherwise lock 4–5 gated pairs.** At least two must have expected φ ≥ 0.05, judged from S2/S5 costs at similar agreement;
   one of them is k = 2 at λ = 1 (or k = 4 at λ = 1 in the fallback).
8. **Lock the TILT-FIX and TILT-FRESH tilts** to match one mid-dose mix arm's Ā within 0.005.

## 3. Outcomes (LOCKED in `v174_analyze.py` at freeze)

Cost φ = (L_arm − L_FULL)/(L_FLOOR − L_FULL), using the 8 evaluation seeds, with the per-seed difference paired by
seed.

**Drift rule.** A locked pair is scored only if its 8-seed mean |ΔĀ| ≤ 0.01 in the scored runs. If fewer than 3 pairs
survive, the outcome is UNINFORMATIVE.

**Primary: COLLAPSE over the locked pairs.**

Tolerance τ(pair) = max(0.015, 0.25 · mean φ of the pair).
- **"same cost"** (TOST): the 90% CI of the seed-paired difference φ₁ − φ₂ lies inside [−τ, +τ].
- **"different cost"**: the 90% CI lies entirely outside [−τ, +τ].

| outcome | rule |
|---|---|
| **AGREEMENT-PREDICTS** | every scored pair is "same cost" (including both high-dose pairs), and within each k no arm inverts the Ā→φ order by more than τ |
| **AGREEMENT-INSUFFICIENT** | at least one locked pair is "different cost" |
| **INCONCLUSIVE** | otherwise |
| **UNINFORMATIVE** | fewer than 3 pairs survive the drift rule, the FLOOR − FULL gap is < 0.03, or the size premise fails (< 95% of mix-arm steps have both norms above the clip) |

### 3a. Power (frozen)

- **Source:** the seed-paired SD of the cost difference between two arms of the same model, from S5 (v166): 0.0033 and
  0.0056, mean 0.0044.
- **Simulation:** TOST, 90% CI inside ±τ, τ = 0.015, true difference 0, 20,000 draws.

| per-seed SD | power per pair at n = 5 / 6 / 8 / 10 | all 5 pairs at n = 8 |
|---|---|---|
| 0.0044 (observed) | 1.00 / 1.00 / 1.00 / 1.00 | 1.00 |
| 0.0088 (2× observed) | 0.86 / 0.94 / 0.99 / 1.00 | 0.96 |
| 0.011 | 0.63 / 0.77 / 0.93 / 0.98 | 0.70 |

- **Choice:** n = 8 gives ≥ 0.99 per pair at twice the observed SD.
- **Reporting rule:** if the scored runs' paired SD exceeds 0.011, power was below target, and an INCONCLUSIVE outcome
  is reported as underpowered.

**Contrast interpretations (fixed now; reported, not scored):**
- **TILT-FIX costs more than its matched mix arm:** a persistent bias in a loss-irrelevant direction is worse than
  truncation's error. Under Adam, low-gradient coordinates drift together, so agreement alone is not enough outside
  truncation-type errors.
- **TILT-FIX costs less:** truncation's structured error is costlier than an unstructured persistent bias at the same
  agreement.
- **TILT-FRESH costs much less than both:** variance is cheap at matched agreement; persistence is what costs.
- None of these change the scope of the primary claim (§0).

**Secondary, reported:**
- a single-curve fit φ = c·(1 − Ā)^p over all mix arms;
- loop-only Ā;
- frozen-model agreement versus in-training Ā as predictors;
- the per-step norm logs.

## 4. Cost

- **Pilot:** 8 candidate arms (k = 1, 2) × 1 seed × ~37 min ≈ 5 T4-h, plus the interpolated re-runs (~2 T4-h); the k = 4
  candidates add ~3 T4-h only if needed.
- **Scored:** (8–10 pair arms + FULL + FLOOR) × 8 seeds + 2 tilts × 3 seeds ≈ 86–102 runs ≈ 50–60 T4-h.
- **Timing:** re-time one run on Alpine before sizing the job array.
- **Platform:** Alpine (one platform for pilot and study).

## 5. Predictions (mine, draft 3)

These are conditional on the gate passing, since gated pairs are by construction not near-identical gradients:
- AGREEMENT-PREDICTS 0.35 / AGREEMENT-INSUFFICIENT 0.40 / INCONCLUSIVE 0.25.
- Unconditionally, the fallback (k = 1 vs 4) is needed: 0.5.
