> **HISTORY ONLY.** Drafts 1–4 and the pilot record of v174, kept verbatim for audit. The frozen contract is
> `docs/prereg_v174_direction_intervention.md`; where this file and that one differ, that one governs.

# Pre-registration v174 (DRAFT 3): does Adam-update agreement predict truncation's training cost? A gated dose–collapse test

**Author:** Aur Shalev Merin.
**Date:** 2026-10-05.
**Status:** DRAFT 3.1 (outside review of draft 3 applied). NOT frozen. The pilot (§2) must run first; its B values and the locked pairs are then added
(draft 4), the draft is sent to the outside reviewer, and it is frozen and pushed to origin before any scored run.

**Changes from draft 2** (outside review):
- a gate against collapse-by-construction (B);
- matching tightened to 0.005;
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

---

## DRAFT 4 ADDENDUM: pilot numbers (2026-10-05). For outside review before the freeze.

Full details: `docs/v174_pilot_result.md`; outputs in `results_v174_pilot/`.

**Pilot selftests.** All OK on the real model; the FULL-arm agreement is 1.000000.

**Realised Ā** (all parameters; steps 100–600; seed 1749901; final losses never computed):

| λ | 0.25 | 0.5 | 0.625 | 0.75 | 0.875 | 1.0 | 1.25 | 1.5 | 2.0 | 2.25 |
|---|---|---|---|---|---|---|---|---|---|---|
| k = 1 | 0.997 | 0.986 | 0.975 | 0.984 (26 skipped steps; unstable) | 0.932 | 0.896 | | | | |
| k = 2 | 0.998 | 0.990 | | 0.974 | | 0.946 | | | | |
| k = 4 | | | | 0.991 | | 0.984 | 0.970 | 0.950 | 0.913 (9 skipped) | 0.753 (13 skipped) |

**Gate (frozen; `v174_gate_a0.json`, `v174_gate_k4.json`).**
- **k = 1 vs k = 2 FAILS.** Agreement-matched pairs have error-component cosines of 0.85–0.87 (limit 0.75).
- **k = 1 vs k = 4 PASSES.** Matched pairs have error cosines of 0.45–0.60. So the pre-registered fallback is used.

**Size premise.** Only 13–31% of steps had both norms above the clip for k = 1 and k = 2, and 62–83% for k = 4.
- The mix arms are rescaled to |g_R|, so the test does not depend on this premise.
- Change: "size premise fails → UNINFORMATIVE" is REMOVED from §3. The clipped fraction is reported as a finding.
  The paper's clipping claim has already been corrected.

**Proposed locked pairs** (k = 1 vs k = 4; λ for k = 4 interpolated to match Ā). Each must be re-measured on the pilot
seed (|ΔĀ| ≤ 0.005) and re-gated (error cosine ≤ 0.75) before the freeze:

| pair | k = 1 λ | k = 4 λ (interpolated) | target Ā | expected cost |
|---|---|---|---|---|
| P1 | 0.5 | ≈ 0.94 | 0.986 | low |
| P2 | 0.625 | ≈ 1.17 | 0.975 | low–mid |
| P3 | 0.875 | ≈ 1.75 | 0.932 | high (φ ≥ 0.05 expected) |
| P4 | 1.0 (plain truncation) | ≈ 2.03 | 0.896 | high (φ ≈ 0.2 at k = 1) |

**Changes requested for the frozen version.**
1. **High-dose anchor.** Draft 3.1 required "k = 4 at λ = 1" in the fallback, but plain k = 4 truncation is low-cost
   (φ ≈ 0.02), so it cannot be the anchor. The anchor becomes P4 (plain k = 1 truncation).
2. **Stability rule for treatment arms.** For each seed, a pair counts only if both its arms have ≤ 10 non-finite
   steps and a finite loss. A pair is scored only if ≥ 6 of 8 seeds remain. P4's k = 4 arm (λ ≈ 2) sits next to the
   unstable zone.
3. **Drift rule kept:** 8-seed mean |ΔĀ| ≤ 0.01. The UNINFORMATIVE threshold stays at fewer than 3 scored pairs.

**Remaining before the freeze:**
- re-measure the 4 interpolated k = 4 arms on the pilot seed (~1.3 quota-h);
- re-gate them;
- outside review of this addendum;
- freeze, push, then the scored study (~50–60 T4-h, on Alpine or over several Kaggle weekly resets).

### Re-measurement 1 (2026-10-06)

Interpolated k = 4 arms on the pilot seed; all stable (1–3 skipped steps).

| pair | k = 1 Ā | k = 4 Ā | \|ΔĀ\| | frozen A_f (k1 / k4) | error cosine (gate ≤ 0.75) |
|---|---|---|---|---|---|
| P1 (k1 0.5 / k4 0.94) | 0.9856 | 0.9851 | 0.0005 ✅ | 0.984 / 0.966 | 0.578 ✅ |
| P2 (k1 0.625 / k4 1.17) | 0.9745 | 0.9750 | 0.0005 ✅ | 0.969 / 0.919 | 0.477 ✅ |
| P3 (k1 0.875 / k4 1.75) | 0.9319 | 0.9407 | 0.0088 ❌ | 0.896 / 0.840 | 0.551 ✅ |
| P4 (k1 1.0 / k4 2.03) | 0.8955 | 0.9177 | 0.0222 ❌ | 0.793 / 0.793 | 0.611 ✅ |

**Finding.** Frozen-model agreement and in-training Ā do not rank the arms the same way:
- P1 is matched in training but not frozen;
- P4 is matched frozen but not in training.

The secondary comparison (frozen vs in-training as predictors) therefore has real stakes.

**Second iteration.**
- P3: k4 λ 1.857.
- P4 is redefined as P4′ = k1 λ 0.924 vs k4 λ 2.03. Matching plain k = 1 truncation (Ā 0.896) would push k = 4 into
  its unstable zone (λ 2.25: 13 skipped steps; Ā falls from 0.92 to 0.75 between λ 2.03 and 2.25).
- The high-dose anchor is therefore P4′ (k1 λ ≈ 0.92, expected φ ≈ 0.15), not plain truncation.

### Re-measurement 2 (2026-10-06)

- **P4′ matched:** k1 λ 0.924 has Ā 0.9188 vs k4 λ 2.03 at 0.9177 (|ΔĀ| 0.0011).
- **P3 not matched:** k4 λ 1.857 gave Ā 0.9438 vs the target 0.9319.
- **Why:** k = 4's Ā is not monotone in this region (λ 1.75: 0.941; 1.857: 0.944; 2.03: 0.918). Single-seed jitter is
  about ±0.005–0.01.
- **Third iteration:** keep k4 λ 1.75 (0.9407), and move the k1 side to λ 0.82 (interpolated). If it is still off
  after this, P3 is locked as measured, and the pre-registered drift rule (8-seed mean |ΔĀ| ≤ 0.01) decides whether it
  is scored.
- **Correction:** the third-iteration kernel was NOT run. aurshalevmerin hit its weekly GPU quota (45 h) at push time,
  and aurmerin was already out.

### Re-measurement 3 (2026-10-10, Kaggle T4, both GPUs; `results_v174_pilot/rm3/`)

| k = 1 λ | Ā | Ā (loop-only) | skipped steps | both clipped |
|---|---|---|---|---|
| 0.78 (bracket) | 0.9517 | 0.900 | 2 | 0.13 |
| 0.82 | 0.9437 | 0.883 | 5 | 0.13 |
| 0.875 (re-measurement 1) | 0.9319 | | | |

- **P3 matched:** k1 λ 0.82 (0.9437) vs k4 λ 1.75 (0.9407), |ΔĀ| 0.0030 ≤ 0.005. k = 1's Ā is monotone in λ here.
- **Still to do before the freeze:** re-gate P3 at k1 λ 0.82 (frozen error cosine ≤ 0.75; it was 0.551 at λ 0.875).
- **Locked pairs (proposed):** P1 k1 0.5 / k4 0.94; P2 k1 0.625 / k4 1.17; P3 k1 0.82 / k4 1.75; P4′ k1 0.924 / k4 2.03.
- First runs of the remeasure3 kernel on one GPU were cancelled after a few minutes and re-pushed with the bracket
  arm on the second GPU; no output was used from the cancelled run.

### Kernel p5 (2026-10-10, Kaggle T4 ×2; `results_v174_pilot/p5/`): P3 re-gate + backup pair P5

**Gate (frozen; error cosine ≤ 0.75):**
- every k1 {0.82, 0.875} × k4 {1.75, 1.93, 1.97} combination passes, with error cosine 0.53–0.55;
- **P3 at k1 0.82 passes: 0.536.**

**k = 4 pilot Ā:**

| k = 4 λ | Ā | skipped steps |
|---|---|---|
| 1.93 | 0.9191 | 3 |
| 1.97 | 0.9223 | 3 |

**P5 is not matched.** Against k1 0.875 (0.9319), the gaps are 0.0128 and 0.0096.

**k = 4 Ā is jumpy across λ 1.75–2.03** (0.941, 0.944, 0.919, 0.922, 0.918):
- It sits on a plateau near 0.94 up to λ ≈ 1.86, then drops to about 0.92.
- The step is about as large as single-seed jitter (±0.01).
- A k = 4 arm at Ā ≈ 0.932 cannot be targeted reliably.

**Proposed P5: k1 λ ≈ 0.91 vs k4 λ 1.97** (target Ā 0.922).
- It is a same-dose backup for P4′ (k1 0.924 / k4 2.03, Ā 0.918), with k4 one step further from the unstable zone.
- It needs one k = 1 re-measurement.

**For the reviewer:** all single-seed matching is within the pilot's seed jitter (±0.005–0.01). The 8-seed drift rule
(|mean ΔĀ| ≤ 0.01) is what actually protects the matching.

### Kernel p5b (2026-10-10; `results_v174_pilot/p5b/`): P5 re-match

| k = 1 λ | Ā | skipped steps |
|---|---|---|
| 0.90 | 0.9257 | 1 |
| 0.91 | 0.9230 | 1 |

- **P5 matched:** k1 λ 0.91 (0.9230) vs k4 λ 1.97 (0.9223), |ΔĀ| 0.0007.
- **Gate:** the neighbouring combinations k1 0.875 / k4 1.97 (0.554) and k1 0.924 / k4 2.03 (0.611) both pass. The
  exact pair is being checked by kernel gate5.
- **gate5 (2026-10-10):** P5 (k1 0.91 / k4 1.97) passes with error cosine 0.567 (B_f 0.866). All five pairs are matched
  and gated.

**Open question for the reviewer: are P4′ and P5 at risk of failing together?** (raised in review)
- P5 sits next to P4′ (k4 λ 1.97 vs 2.03; Ā 0.922 vs 0.918).
- **Skipped steps at k = 4:**

  | k = 4 λ | skipped steps |
  |---|---|
  | 1.93 | 3 |
  | 1.97 | 3 |
  | 2.00 | 9 |
  | 2.03 | 3 |
  | 2.25 | 13 |

  The stability limit is 10 per run.
- **No safer high-dose spot exists for k = 4.** Ā stays on a plateau near 0.94 up to λ ≈ 1.86, and lower agreement
  needs λ ≈ 2. High dose at k = 4 is therefore always near the unstable zone.
- **Options:**
  - (a) keep it as is (≥ 2 of 3 high-dose pairs scored);
  - (b) count P4′ and P5 as one high-dose slot, so AGREEMENT-PREDICTS needs P3 plus either of them;
  - (c) relax to ≥ 1 high-dose pair scored and "same".
- **Author's lean: (b).** It states what P5 was added for (a substitute for P4′, not independent evidence) and does not
  weaken the requirement below draft 3.1's "both high-dose pairs".
