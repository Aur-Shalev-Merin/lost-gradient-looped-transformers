# v170 result: outside the pre-registered outcomes. SGD makes truncation catastrophic on the retrofit (2026-10-05)

- **Prereg:** `docs/prereg_v170_sgd_vs_adam.md` (frozen 29ecbb2; amendment A1 1c67bdf; LR record 07feb48).
- **Runs:** `results_v170/{a,b}` (Parcae) and `{c,d}` (retrofit).
- **Completeness:** 35 + 56 runs; 8/8 selftests OK; all 91 runs logged `optimizer = SGD`.

## What the locked analysis printed, and why it is not the verdict

- **Floor gates:** both retrofit variants failed (a 1: 0.0266; a 0.3: 0.0256), so amendment A1's raw-loss fallback
  applied.
- **The printed label is an artifact.** The fallback returned NaN, and the code labelled it NOT-OPTIMISER-DIRECTIONAL
  only because `lo > 0` is False for NaN.
- **Cause:** every undamped k = 1 run diverged. The pre-registered exclusion rule screens only the full-backprop
  controls. Nothing pre-registered covers divergence of a *treatment* run.
- **Status:** the result falls OUTSIDE the pre-registered outcome set. It is reported descriptively below.

## Retrofit under SGD (test loss; floor LR 0.3, full/truncated 0.03)

| variant | seeds | k 0 (floor) | k 8 (full) | k 1 | k 2 |
|---|---|---|---|---|---|
| a 1.0 | 7 | 1.2681–1.2693 | 1.2353–1.2590 | **NaN in 7/7** (454–524 non-finite steps) | 1.2672–1.2875 |
| a 0.3 | 7 | 1.2674–1.2689 | 1.2413–1.2430 | 1.2599–1.2626 (stable) | 1.2567–1.2580; 1 of 7 NaN |

- **Undamped, k 1:** truncated SGD training diverges every time.
- **Undamped, k 2:** extra loss over full ≈ 0.04. That is **larger than the floor gap**, so φ ≈ 1.3: keeping 2 loops
  is worse than giving the loop no gradient at all.
- **Damped (a 0.3):** stable. φ ≈ 0.69 at k 1 and ≈ 0.6 at k 2. The extra loss is 0.018 / 0.015 nats.
- **Adam on the same cells (v167):** φ ≈ 0.009 (k 1) and ≈ 0 (k 2), stable everywhere.

## Reading

1. **The optimiser matters enormously (H2-type comparison).** On the retrofit, truncation under Adam costs ~1% of the
   loop's credit. Under SGD it costs ~60–70% damped, and is catastrophic undamped. H2 as written
   (φ_SGD(a 0.3, k 1) − φ_Adam ≥ 0.02) holds by a wide margin: 0.69 vs 0.009.
2. **The predicted mechanism (H1) is reversed in sign.** H1 predicted that damping, which raises the frozen gradient
   error, would raise SGD's cost. Instead the undamped cells are far worse (divergence, φ > 1), and damping
   *stabilises* truncated SGD.
3. **Confound:** the undamped retrofit is near SGD's stability edge. Full backprop already diverged at LR 0.1, and LR
   0.03 was tuned on full backprop. The truncated gradient may cross that edge where the full gradient does not. So
   v170 cannot separate "truncation error hurts SGD" from "truncated SGD is less stable at a fixed LR".
4. **Parcae (context):** SGD and Adam costs are similar (k 1 / 2 / 4: 0.263 / 0.104 / 0.013 vs 0.224 / 0.100 /
   0.024). This is consistent with v168: Adam's step deficit on Parcae is itself large (~0.32 at k 1), so Adam gets no
   free pass there.

## What it means

- **Supported:** "Adam is why the retrofit tolerated truncation". On the same model and cells, truncation that was
  nearly free under Adam is severe or catastrophic under SGD.
- **Not supported:** that the SGD cost follows the first-order SGD deficit (v168). The ordering across damping is
  opposite, and stability dominates.
- **Needed before any claim:** the "sweep before concluding" rule applies. The next step is an SGD learning-rate sweep
  of the truncated cells themselves (a 1 and a 0.3, k 1 and k 2, at LRs ≤ 0.03), so that cost is separated from
  stability.

## Follow-up: SGD LR sweep of the truncated retrofit cells (interpretation fixed before launch, 2026-10-05)

- **Cells:** a ∈ {1, 0.3} × k ∈ {1, 2} × LR ∈ {0.003, 0.01, 0.02} × seeds {1701101, 1701102}. The LR 0.03 runs come
  from the main v170 run.
- **Controls:** floor and full stay at their best LRs (0.3 / 0.03, main-run seeds). φ is computed per seed against
  them.
- **Best-LR cost:** for each cell, the minimum over LR of the 2-seed mean φ, counting only stable runs (finite, ≤ 10
  non-finite steps).

**Rule** (undamped, k 1, best-LR φ):
- ≤ 0.05 → mostly a STABILITY ARTIFACT;
- ≥ 0.20 → a GENUINE optimiser effect (truncation costly under SGD, ~free under Adam);
- in between → PARTIAL.

**Damping at best LR** (reported): φ(a 0.3) vs φ(a 1) at each k.

## Follow-up result (2026-10-05): GENUINE optimiser effect by the pre-stated rule

24/24 sweep runs, all logging SGD. φ is the 2-seed mean, against the controls at their best LRs (floor 0.3, full
0.03).

| cell | LR 0.003 | 0.01 | 0.02 | 0.03 | best (stable) |
|---|---|---|---|---|---|
| a 1, k 1 | 3.66 | **2.18** | diverged | diverged | **2.18** (LR 0.01) |
| a 1, k 2 | 3.85 | 3.31 | 1.56 | 1.27 | 1.27 (0.03, grid edge) |
| a 0.3, k 1 | 4.28 | 2.14 | 1.23 | 0.70 | 0.70 (0.03, grid edge) |
| a 0.3, k 2 | 4.14 | 1.97 | 1.07 | 0.58 | 0.58 (0.03, grid edge) |

**Rule:** undamped k 1 best-LR φ = 2.18 ≥ 0.20, so **GENUINE**.
- Truncated SGD on the retrofit has no learning rate that is both stable and cheap. Above 0.01 it diverges. At 0.01
  and below it trains so slowly that the result is worse than the no-loop-gradient floor (φ > 1). The floor itself
  trains stably at LR 0.3.
- Under Adam the same cell costs φ ≈ 0.009.
- Adding the k 1 loop gradient is what destabilises SGD: the floor (k 0) is stable at LR 0.3, while k 1 diverges at
  0.02.

**Caveats.**
- The best LR for the damped cells and for a 1, k 2 sits at the top of the swept grid (0.03). Their best costs may be
  somewhat lower at 0.04–0.05; the pilot showed the damped full run stable at 0.05. The decisive undamped k 1 cell is
  bracketed by divergence, so its result is not edge-limited.
- 2 seeds per sweep cell.

**Damping at best LR:** damping still *helps* SGD (a 0.3, k 1: 0.70 vs a 1, k 1: 2.18). It stabilises the truncated
updates, which reverses H1's predicted sign.

## Overall v170 conclusion

**Supported:** on the retrofit, whether truncating the loop is cheap depends on the optimiser.
- **Under Adam:** about 1% of the loop's credit.
- **Under SGD with momentum:** no stable, cheap learning rate exists. The undamped k 1 cell is unstable or worse than
  not training the loop. Damping helps by stabilising the truncated updates.
- **On Parcae:** SGD and Adam cost about the same.

**Not supported:** the first-order mechanism (SGD cost tracking the SGD progress deficit). Under SGD, stability, not
first-order progress, dominates.

## Scope limit (review check, 2026-10-05)

- **Adam's "free pass" was shown on the retrofit, a model whose loops add little.** The never-loop control matched the
  looped model for the drop-6 retrofit (2.907 vs 2.905). The no-drop retrofit used in v167/v170 has no never-loop
  control, but it also converges within ~4 loops.
- **On Parcae, a natively looped model, Adam and SGD paid about the same.** So "with Adam you can safely cut loops"
  is NOT supported for models whose loops matter.
- **What the retrofit result does show:** the optimiser contrast is real on that model. If "the loops do little"
  were the whole story, truncation would be cheap under SGD too, and instead it diverged or did worse than no loop
  gradient.
- **Claim scope for any write-up:** optimiser-dependence of truncation cost, demonstrated on one weak-loop model.
  Generality to load-bearing loops is untested.
