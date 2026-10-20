# Pre-registration v163: does ρᵏ predict the training cost of truncated loop backprop better than "k alone"? ρ manipulated on the same model by loop damping

**Author:** Aur Shalev Merin
**Date:** 2026-10-02. **FROZEN at the commit that fills in §8. No v163 truncated-credit run had been made at the
freeze.** The PI authorised freezing and launching without further review if all pilot gates passed (2026-10-02,
"once it finished and you review it I allow you to send the next thing into kaggle").
**Workstream:** misc (looped-transformer credit thread).
**Builds on:**
- v162 (LAW-CONSISTENT: cost ordering follows ρᵏ and beats k/R, but is not separated from "k alone", because ρ
  barely differed between its cells; `docs/v162_parcae_truncation_training_rerun.md`);
- the damping feasibility probe (`docs/parcae_damping_feasibility_2026-10-02.md`).

**Trigger:** PI decision 2026-10-02 ("yes", to drafting this test).

## 0. What this is

**The open question.** v162 showed that the training cost of backprop through the last k loops is ordered like ρᵏ.
Every v162 contrast was also predicted by the simpler rule "more backprop loops is better" (cost depends on k alone).
Separating the two needs cells with the same k and R but clearly different ρ.

**The manipulation.** Convex damping of Parcae-140m's loop, x ← (1 − a)·x + a·F(x).
- In the feasibility probe on the frozen model, ρ followed 1 − a + a·ρ₀ to ~0.01 (a = 1: 0.80–0.83; a = 0.5:
  0.90).
- The function at convergence is unchanged: held-out loss 2.612 at R = 64 for every a; at R = 32, a = 1: 2.612,
  a = 0.5: 2.616.
- At matched k the gradient error differs strongly (k = 4: 0.42 vs 0.63).

So a = 1 and a = 0.5 at R = 32 are the same model computing the same function, with different ρ.

**Primary question.** At matched k (and R = 32), does the damped variant (higher ρ) lose a larger fraction of the
loop's training credit, as ρᵏ (equivalently, effective depth a·k) predicts? "k alone" predicts equal fractions.

**A third reading this design cannot separate from the law (review review, 2026-10-02, before the pilot).**
- For ρ₀ near 1, damping gives ρ_a = 1 − a + a·ρ₀ ≈ ρ₀ᵃ, so ρ_aᵏ ≈ ρ₀^(a·k).
- "Effective depth" (k damped loops do the work of about a·k undamped loops, so cost depends on a·k) therefore predicts
  the same as the law here: 0.906ᵏ vs ≈ 0.91ᵏ at ρ₀ ≈ 0.82, a = 0.5.
- What this test separates: the law and effective depth together, against **"k alone"** (cost depends only on the raw
  number of backprop loops, the open question from v162). It does not show that ρ itself, rather than effective
  depth, is the operative quantity.
- Separating those needs a knob that changes ρ without shrinking the per-loop step (e.g. spectral regularisation), or
  low ρ₀, where the two diverge (ρ₀ = 0.4, a = 0.5: 0.70ᵏ vs 0.63ᵏ). This is future work. The low-ρ models we have,
  the retrofits, cannot carry a training test: their loops add no capacity.

**Scope.**
- One model (Parcae-140m) and one fine-tune (GSM8K text, 600 steps), as in v162.
- Damping is the manipulation, so a positive result is about ρ as changed by damping. It does not show that ρ
  differences between independently trained models matter.

## 1. Disclosed prior

- **Known before this prereg:**
  - v161 (invalid) and v162 results;
  - the retrofit work (law holds on two retrofits; retrofitted loops add no capacity at 150M tokens);
  - the damping feasibility numbers above.
- v162: cost is steeply convex in the gradient error (φ ≈ 0.18 / 0.08 / 0.01 at k = 1 / 2 / 4 with ρ ≈ 0.77–0.80).
  So at a = 1, R = 32, costs at k ≥ 4 may be near zero. The damped variant's larger errors should then show
  measurable costs at k = 4 and 8, where the law-vs-k-alone contrast is largest.
- Parcae's default R = 8 is not converged on GSM8K (2.805 vs 2.612). R = 32 is used so that both variants run at
  convergence.

## 2. Apparatus (LOCKED at freeze)

**Model and code.**
- SandyResearch/parcae-140m, code at commit 69284c1.
- Driver `v163_parcae_damped_trunc.py`, which reuses `v162_parcae_trunc_train.py` (frozen 89d8758) for loading, data
  and helpers; the driver sha covers both files.

**Loop.** R = 32 loops. Damped update x_{t+1} = (1 − a) x_t + a F(x_t), F = Parcae's core_block_forward.
- Backprop goes through the last k loops only (earlier loops under no_grad).
- Grad loops are checkpointed per loop (identical math; selftest).
- Evaluation runs all 32 loops forward with the same a.

**Cells.** a ∈ {1.0, 0.5} × k ∈ {0 (floor: no loop credit), 1, 2, 4, 8, 32 (full)} = 12 cells. The 8 truncated
cells (0 < k < 32) are the treatment.

**Recipe** = v162 for everything else:
- GSM8K, 512-token packed chunks; train = first 7,000 train examples, val = last 473 (LR selection), test = 1,319
  (headline);
- 600 steps, batch 8;
- AdamW (β 0.9/0.95), 20-step warm-up then cosine to 0.1 × LR, clip 1.0;
- fp16 autocast with cache_enabled=False and a grad scaler.

**Metric.**
- Test loss L.
- Primary: φ_{a,k} = (L_{a,k} − L_{a,32}) / (L_{a,0} − L_{a,32}), i.e. the fraction of the loop's credit lost,
  normalised within each variant.
- Raw ΔL_{a,k} = L_{a,k} − L_{a,32} is reported alongside.

**Predictor** (locked from the pilot's frozen-model measurement):
- ρ_a = median over 16 GSM8K-val sequences of the all-loop log-linear fit of the adjoint norms at R = 32;
- law_{a,k} = median over sequences of ρ_seqᵏ.

**Seeds and LR.**
- Tuning seed 1631001: every cell at LR {1e-4, 3e-4, 9e-4}; per-cell LR = lowest val loss.
- Edge rule: a grid-end winner gets one 3× extension.
- Evaluation seeds 1631101–1631104 (4) at the selected LRs.
- v162's paired seed noise was ~0.0015 in raw ΔL, so 4 evaluation seeds and 1 tuning seed are judged sufficient.
- All 12 cells are run fresh on the v163 driver.

**Platform.** Kaggle T4 ×2, both accounts, one run per GPU. No local runs (TINY pipeline test only).

**Budget.** ~36 tuning + 48 evaluation runs ≈ 20–25 T4-hours (R = 32 is ~4× v162 per run). If pilot timing × 84 runs
exceeds 40 T4-h, drop the tuning grid to {3e-4, 9e-4}.

**Pre-freeze pilot (controls and frozen model only).**
1. Selftest:
   - damped forward at a = 1 equals v162's loss_manual;
   - checkpointed and plain gradients equal;
   - truncated gradient fp16 vs fp32 per parameter group at both a and k ∈ {1, 4}, all < 0.05.
2. Frozen-model measurement at both a (the locked predictor) and step-0 losses.
3. LR sweep {1e-4, 3e-4, 9e-4} of the controls only (k = 0 and k = 32) at both a, pilot seed 9001.

## 3. Criteria (LOCKED at freeze)

**Gates (pilot).**
- **G1 trainability:** full arm drops ≥ 0.10 nats below step 0, at both a.
- **G2 loop credit:** L_{a,0} − L_{a,32} ≥ 0.03 at both a.
- **G3 manipulation:** the measured ρ differs between the variants by ≥ 0.05 (feasibility: 0.08).

If any gate fails, nothing is frozen.

**Floor re-check (evaluation seeds).** L_{a,0} − L_{a,32} > 0, paired 90% CI excluding 0, at both a. Otherwise
UNINFORMATIVE.

**Statistics.** Paired over the 4 evaluation seeds. A contrast is directional if its two-sided 90% t-interval (df 3)
excludes 0 and |mean| ≥ 0.02 (φ) or ≥ 0.002 (raw ΔL); otherwise TIE.

- **H0:** φ_{a,1} > 0 (directional) at both a.
- **H1, primary (law vs k alone):**
  - Matched-k pairs: φ_{0.5,k} − φ_{1.0,k} for k ∈ {1, 2, 4, 8}.
  - The law predicts + (a pair is scored if |law_{0.5,k} − law_{1.0,k}| ≥ 0.05); "k alone" predicts a tie.
- **H2:** Spearman(law, mean φ) over the 8 truncated cells ≥ 0.75.

**Outcome labels (first that applies).**
1. **UNINFORMATIVE:** floor re-check or H0 fails.
2. **AGAINST:** any scored H1 pair directional opposite to the law, or H2 fails.
3. **LAW / EFFECTIVE-DEPTH:** at least min(3, #scored) scored pairs directional in the law's direction, none
   opposite. (The analysis script prints this label as "LAW".)
4. **K-ALONE:** no scored pair directional (all TIE).
5. **MIXED:** otherwise (1–2 pairs in the law's direction, rest TIE).

**Reported only (added 2026-10-02, before freeze): does truncated training reshape the loop?**
- After each run's test evaluation, the driver measures the trained model's adjoint decay ρ_end (fp32, 8 val sequences, all 32 loops with grad, all-loop fit).
- Prediction on record: within each a, smaller k ends with smaller ρ_end (Spearman(k, ρ_end) over k > 0 positive).
- Motivation: every truncation-trained checkpoint has a decaying adjoint while full-backprop-trained Ouro's grows, and ρ fell during v161's drift probe (0.80 → ~0.70).
- This touches no hypothesis or label. The measurement runs after val/test are recorded, so the pilot's control runs (driver without it) are unaffected.

**Reported, not headline:** H1 on raw ΔL; the measured err_k as predictor; training curves; non-finite steps; run
time per cell.

## 4. Consequences

| outcome | consequence |
|---|---|
| LAW / EFFECTIVE-DEPTH | "k alone" is refuted: truncation cost tracks the gradient error (effective backprop depth), not the raw loop count. The write-up claims ρᵏ as a predictor of truncation cost on Parcae under damping, and states explicitly that this design cannot distinguish ρ from effective depth a·k. The separating test (a non-damping ρ knob) is the next step. The method question ("measure ρ, choose k") is reopened, scoped to that. |
| K-ALONE | ρᵏ is a gradient-level fact only. Training cost is set by k, not by how fast the backward signal decays. The method claim is dropped for good. |
| AGAINST | As K-ALONE, and the convex-cost reading of v162 is reconsidered. |
| MIXED | The law's direction is supported only at some k. Reported as such; no method claim. |
| UNINFORMATIVE | Reported with the reason. |

## 5. No-post-hoc clause

No change to cells, R, a values, steps, data, LR grid, seeds, metric, gates or labels after the freeze. Before the
freeze, only the pilot's control arms and the frozen-model measurement may be inspected.

## 6. Design decisions (disclosed)

- **a = 0.5 rather than 0.3:** a = 0.3 needs R = 64 to converge (R = 32: 2.888), doubling the cost. a = 0.5 gives a ρ
  gap of 0.08 and a k = 4 error gap of 0.21 at R = 32.
- **φ primary.** It normalises out a possible difference in how much loop credit the two variants have. Raw ΔL is
  reported, after v161's lesson that normalisation can create spurious effects.
- **One tuning seed:** v162's selection was unanimous (every truncated cell 3e-4; floor 9e-4) and its seed noise
  tiny.

## 7. Predictions (on record at freeze, before any truncated run)

**Scored pairs** (locked law values, §8): law_{0.5,k} − law_{1.0,k}

| k | difference | scored | predicted sign of φ_{0.5,k} − φ_{1.0,k} |
|---|---|---|---|
| 1 | +0.078 | yes | + |
| 2 | +0.134 | yes | + |
| 4 | +0.200 | yes | + |
| 8 | +0.225 | yes | + |

**Final outcome probabilities:**

| outcome | P |
|---|---|
| LAW / EFFECTIVE-DEPTH | 0.45 |
| K-ALONE | 0.20 |
| MIXED | 0.20 |
| AGAINST | 0.05 |
| UNINFORMATIVE | 0.10 |

Reasoning:
- v162's cost was steeply convex in the gradient error.
- Here the a = 0.5 error at k = 4 (0.645) ≈ the a = 1.0 error at k = 2 (0.675), so I expect φ_{0.5,4} ≈ φ_{1.0,2}.
  That gives clear separation at k = 2–8.
- k = 1 may tie: both errors are ≥ 0.82, the top of the convex curve.

**Specific forecasts:**
- φ_{1.0,4} < 0.05 and φ_{0.5,4} > 0.05 (P 0.55).
- k = 1 pair TIE (P 0.5).
- Reported ρ_end: Spearman(k, ρ_end) > 0 within each a (P 0.6).

Pre-pilot forecast (kept for the record):

| outcome | P |
|---|---|
| LAW / EFFECTIVE-DEPTH | 0.35 |
| K-ALONE | 0.30 |
| MIXED | 0.20 |
| AGAINST | 0.05 |
| UNINFORMATIVE | 0.10 |

## 8. Pilot results and freeze contents

**Kernels.**
- `aurmerin/astrocyte-v163-pilot-b`: selftest, measurement, k = 0 sweeps.
- `aurshalevmerin/astrocyte-v163-pilot-a`: k = 32 sweeps.
- Pilot seed 9001, driver sha 888f185a0c54.

**Selftest (real model, R = 32).**
- a = 1 damped forward = v162 loss_manual: 2.675809 at k ∈ {32, 16, 0}.
- Checkpointed vs plain gradient: 8.9e-7.
- fp16 vs fp32 truncated gradient, all groups, a ∈ {1.0, 0.5}, k ∈ {1, 4}: 0.0022–0.0032.

**Frozen-model measurement (locked predictor):**

| a | ρ | k = 1 err / law | k = 2 | k = 4 | k = 8 | k = 16 |
|---|---|---|---|---|---|---|
| 1.0 | 0.825 | 0.820 / 0.825 | 0.675 / 0.681 | 0.443 / 0.463 | 0.205 / 0.215 | 0.043 / 0.046 |
| 0.5 | 0.903 | 0.877 / 0.903 | 0.791 / 0.815 | 0.645 / 0.663 | 0.418 / 0.440 | 0.187 / 0.194 |

Step-0 test loss: 2.6123 (a = 1.0), 2.6353 (a = 0.5).

**Control-arm LR sweep (test loss, pilot seed):**

| a | arm | 1e-4 | 3e-4 | 9e-4 | time/run |
|---|---|---|---|---|---|
| 1.0 | full (k = 32) | 1.3987 | **1.3753** | 1.4318 | ~1,615 s |
| 1.0 | floor (k = 0) | 1.6742 | 1.5811 | **1.5382** | ~505 s |
| 0.5 | full (k = 32) | 1.3977 | **1.3769** | 1.4311 | ~1,550 s |
| 0.5 | floor (k = 0) | 1.6852 | 1.5888 | **1.5451** | ~475 s |

**Gates.**
- **G1 PASS:** full arm −1.24 / −1.26 nats from step 0.
- **G2 PASS:** floor gap 0.163 (a = 1.0) and 0.168 (a = 0.5).
- **G3 PASS:** ρ difference 0.078.
- Non-finite gradient steps: 0–6 of 600, skipped by the scaler.

**Budget.** Estimated per run ≈ 530 + 34·k s (k = 0: ~500 s, k = 32: ~1,600 s). About 7.7 T4-h tuning + 10.2 T4-h
evaluation + extensions ≈ 18–19 T4-h, under the 40 T4-h cap, so the grid is kept.

**Change after the pilot (before freeze; disclosed).** The reported-only end-of-run ρ measurement (§3) was added to the
driver after the pilot's control runs. It runs after val/test are recorded, so the control-arm numbers above are
unaffected. The frozen driver includes it.

**Freeze contents.** This document; `v163_parcae_damped_trunc.py` + `v162_parcae_trunc_train.py` (frozen 89d8758) — **study driver sha d5c2647126de** (the combined sha the driver logs and stamps on every run);
`v163_analyze.py` (law values locked; selftest LAW / K-ALONE / AGAINST / UNINFORMATIVE all as expected);
`build_kaggle_v163.py`.
