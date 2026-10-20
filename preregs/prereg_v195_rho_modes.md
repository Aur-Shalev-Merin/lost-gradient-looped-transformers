# PREREG v195: why do the two offsetting effects behind ρᵏ match? (frozen before any v195 data)

**Status.** Pre-registration, requested by the user ("yes") after `docs/rho_match_exploration_2026-10-13.md`. Frozen by
the commit that adds this file, together with `v195_rho_modes.py` (measurement), `v195_analyze.py` (analysis, with a
synthetic check) and `build_kaggle_v195.py` (kernel). Pushed to origin before the kernel is launched.

## Background and what had been seen at freeze time

**The finding being explained.** v191 / v192 showed that ρᵏ lands close to the lost gradient because two effects of
similar size offset each other:
- **Directions raise the lost gradient** above the signal-size form, ×1.05–1.47. The "angle factor".
- **The single fitted ρ raises ρᵏ** above the same form, ×1.05–1.77. The "fit factor".

They match within 13% on 4 of 5 model–loop-count pairs. Parcae-370m at 32 loops is the exception.

**Seen before freeze:**
- all v191 / v192 outputs, including the per-loop contribution Grams;
- the exploration doc:
  - toys reproduce both known misses, as a slower far fade inflating the fit;
  - real contributions turn 10–400 times more than a mix of real fade rates allows;
  - snapshot DMD on the stored Grams is unreliable;
  - an after-the-fact link check came out UNRESOLVED (Huginn shows no across-sequence link);
- a literature check ("partly known": the ingredients exist, the two-sided offset was not found);
- the blended Parcae-140m lost gradients used in decision D, already in the paper's Figure 1 and §3. D's model is built
  from unblended data only.

**Not seen:** any v195 output. TINY random-model outputs were seen; they test code paths only.

## Question and candidate answers

Is the match **structural**, with both effects coming from the same parts of the backward signal, or a coincidence of
separate effects? Candidate sources for the turning, which drives the angle factor:
- parts of the backward signal that rotate or flip from loop to loop (complex or negative modes of the loop's backward
  map);
- non-normal transients;
- the map from the signal to the weights changing from loop to loop, plausible on Huginn, whose state still moves far
  from the output.

## Measurements (`v195_rho_modes.py`)

**Models.** Frozen; R = 32; the v191 / v192 sequences and setups:
- Parcae-140m and Parcae-370m: 16 GSM8K validation chunks, fp32, all parameters, a = 1, seed 777 + i;
- Huginn-3.5B: probe sequences, WikiText 0–7 and GSM8K 16–23, bf16, adapter + core_block[0].

**Per sequence:**
- **Trajectory.** The forward states, and the backward signal λ_t at every loop from one full backward pass. The vectors
  are stored and their exact float64 Gram computed.
- **c_t = P_tᵀ λ_t:** each loop's contribution, as a parameter VJP through that loop alone. For Parcae, P_t reaches the
  prelude through the injected e, as in v191.
- **p_t = P_tᵀ u:** the same per-loop maps applied to one fixed vector, u = λ_out / |λ_out|.
- **Constant-map model.** κ_i = (Jᵀ)ⁱ λ_out for i ≤ K, with Jᵀ the last loop's backward map, and ĉ_i = Pᵀ κ_i. K = 64 for
  Parcae (for decision D), 32 for Huginn.
- **Arnoldi.** m = 32 steps of Jᵀ from λ_out (two Gram–Schmidt passes, float64): H, and Pᵀ q_l.
- **g_0** (Parcae), the gradient with no loop gradient.

**Storage.** Parameter vectors go into one CountSketch Gram (2^20 buckets, hashes fixed per model), with exact norms.

**TINY tests (random tiny Parcae, CPU, one process), all passed:**
- the per-loop contributions reproduce the lost gradient of truncated backward passes within 1.7e-5;
- they sum to g_R − g_0 within 1e-5;
- g_0 within 4e-6;
- sketch cosines within 2.5e-3;
- the Arnoldi reconstruction within 2e-7;
- the blending formula matches direct blended iteration within 8e-6;
- the read-out reproduces the model's loss to CPU noise (≤ 4e-6 relative; the same function differs by up to 7e-6
  between calls).

## Decisions (`v195_analyze.py`, per model)

**Definitions.** Index j counts loops before the output. err_k is the lost share over the loop part. ρ is a straight-line
fit of log signal size over j = 0 … 31. A0 and A1 are the aligned forms on signal sizes and on contribution sizes. The
angle factor is err / A1 and the fit factor ρᵏ / A0. "Actual" uses λ and c; "constant-map" uses κ and ĉ.

**A. Where the turning comes from.** T_c = mean of 1 − cos(c_j, c_{j+1}) over j = 0, 1, 2 (near band); T_p is the same
for p_j. With r = median T_p / median T_c:
- SIGNAL if r < 0.2;
- MAP if r ≥ 0.5;
- MIXED otherwise.

The same rule is applied to the far band, j = 8–14.

**B. Do one loop's maps capture the dynamics?** PASS if |median err_C − median err_A| ≤ 0.02 at k = 1, 2, 4, 8 and
|median ρ_C − median ρ_A| ≤ 0.02. FAIL otherwise.

**C. Do both effects come from the turning parts?** Only for models where B passes.

*Model.* The modal model from H: Ritz values T, amplitudes b. Guards (else UNINFORMATIVE):
- it reproduces |κ_i| within 2% and err_C within 0.01 (medians);
- the median non-normality ratio Σ|b_m| / |λ_out| is ≤ 10;
- the turning modes (|arg T| > π/4) carry ≥ 5% of Σ|b_m|. If not, the verdict is NO-TURNING-MODES.

*Test.* Knock out the turning modes. In each sequence where both full excesses (factor − 1) are ≥ 0.05 at k = 4 (≥ 8
sequences needed), d = reduced excess / full excess. Using the medians of d:

| verdict | rule | meaning |
|---|---|---|
| SHARED | both d ≤ 1/3 | the turning parts carry both effects, so the match is structural for that model |
| NOT-SHARED | otherwise; labelled SEPARATE when one d ≤ 1/3 and the other ≥ 2/3 | no structural claim |

Descriptive: ρ⁴ / err_4 with the turning modes' amplitudes scaled by s = 0, 0.5, 1, 1.5, 2.

**D. Out-of-sample check.** Only for Parcae-140m, and only if B passes. Blending maps Jᵀ to (1 − a)I + aJᵀ and Pᵀ to aPᵀ,
so ĉᵃ_i = a Σ_l C(i, l)(1 − a)^(i−l) a^l ĉ_l. The all-parameter lost gradient, |Σ_{i≥k} ĉᵃ_i| / |g_0 + Σ_{i<R} ĉᵃ_i|, is
predicted for the four measured settings (same 16 sequences and protocol):
- R 32, a 0.5 (v163);
- R 32, a 0.7 (v164);
- R 64, a 0.3 (v164, v166 pre);
- R 48, a 0.4 (v166 pre).

That is 20 points. PASS if every predicted median is within 0.03 of the measured one.

**Validity** (a model's results are not used if it fails):
- read-out loss = model loss within 1e-4 relative (Parcae);
- signal sizes reproduce v191 / v192: median over loops of the relative difference, max over sequences, 1e-3 for
  Parcae and 2e-2 for Huginn;
- err_A reproduces v191 err_loop / v192 err at k = 1, 2, 4, 8 within 0.005 (Parcae) and 0.01 (Huginn);
- sketch norms within 1%;
- Krylov step 1 equals the backward pass: cos ≥ 0.9999 (Parcae), 0.995 (Huginn);
- Arnoldi orthogonality ≤ 1e-8.

**Synthetic check of the decision code** (`python v195_analyze.py --synthetic`; linear toys with exact Grams):

| toy | result |
|---|---|
| fast turning parts | A SIGNAL, B PASS, C SHARED (kept 0.03 / 0.12); the match holds as the turning parts are dialled (ρ⁴/err 1.02 → 1.00) |
| turning parts plus an ultra-slow tail | C NOT-SHARED (kept 0.26 / 0.62) |
| per-loop weight map that changes | A MAP, B FAIL |
| fade rate that changes far back | B FAIL |

D's algebra matches direct blended iteration to 8e-17.

**Disclosure (changed after the synthetic check, before any data).** C first had three outcomes: SHARED, SEPARATE and
MIXED. Toys showed that real modes with different fade rates always produce some of both effects, so SEPARATE is nearly
unreachable. C was therefore restated as SHARED vs NOT-SHARED, which is the distinction that changes the paper. The
"separate" toy was retuned to make its ground truth clean. The read-out bar became relative (1e-4) after the TINY test
showed that CPU noise alone moves a random tiny model's loss by up to 7e-6 relative.

## What each outcome changes in the paper

- **C = SHARED on Parcae (B passes):** §3 says that on Parcae the two effects come from the same parts of the backward
  signal, those that turn near the output, so the agreement of ρᵏ is structural there. Removing those parts in a model
  of the measured backward map removes both effects. If D passes, §3 adds that the same model predicts the blended
  settings (out of sample).
- **C = NOT-SHARED:** §3 says the two effects come partly from different parts of the backward signal, so the agreement
  is an empirical regularity of these models rather than a consequence of their dynamics. The ~20% claim stands as
  measured.
- **B fails on Parcae:** even settled Parcae is not captured by one loop's maps, and the mechanism stays open. One
  sentence in §3.
- **Huginn:**
  - A tells whether its turning comes from the signal or from the map to the weights;
  - if B fails, §3 says Huginn's backward map changes along the loops, so a one-map account does not apply, and its
    agreement stays unexplained (as now).
- In every case these are diagnostics on frozen models, with no new prediction rule.

## Compute

One Kaggle kernel (aurmerin):
- **GPU 0:** Parcae-140m and -370m (about 1 h), then Huginn sequences 20–23;
- **GPU 1:** Huginn sequences 0–7 and 16–19.

About 1.5 h of wall time, 2–3 T4-hours.
