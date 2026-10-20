# v211 result: the k = 0 fine-tune control (S16; outside review 7, M1)

**Prereg:** `docs/prereg_v211_k0_control.md`, frozen 07446dd and pushed to origin (12:27, 2026-10-19) before any run.
No amendments.
**Code:** `v211_k0_control.py` c516ee2fd492, `build_kaggle_v211.py` af7b6fd7bbf9, run unchanged.
**Runs:** Kaggle T4×2, astrocyte-v211-parcae and astrocyte-v211-ifm, 2026-10-19. Two operational deviations from the
prereg's cost table, neither touching code, inputs or rules:
- The ifm kernel ran on aurmerin, not aurshalevmerin, because the latter's two GPU sessions were taken by another
  thread's v209 runs.
- The Parcae kernel sat in Kaggle's queue for about 1 h 40 min while the later ifm kernel ran. It was re-pushed
  unchanged as version 2, and the downloaded output is the completed run.

**Outputs:** `results_v211/` (JSONs, stdout, logs). `python v211_k0_control.py --analyze results_v210 results_v211`
reproduces `results_v211/v211_analysis.txt`.

## Verdict (frozen rule): **PARTIAL**

r = loop-only fall after the k = 0 fine-tune / loop-only fall after full backprop (v210), means over k = 1, 2, 4:

| model | full-backprop fall (loop) | k = 0 fall (loop) | r | all-parameter analogue |
|---|---|---|---|---|
| Parcae-140m | 0.157 | 0.076 | **0.48** | 0.43 |
| pln5 (sampled loop count) | 0.081 | 0.041 | **0.50** | 0.62 |
| r5 (fixed 5 loops) | 0.086 | 0.012 | **0.14** | 0.27 |

The rule needed r < 0.25 on all three (LOOP-NEEDED) or r ≥ 0.75 on all three (OUTSIDE-SUFFICES); neither holds.
Prediction at freeze: PARTIAL 0.45.

## Medians at k = 1 / 2 / 4 (64 held-out GSM8K test sequences)

| model | state | all | loop |
|---|---|---|---|
| Parcae-140m | s0 (v211 = v210) | 0.798 / 0.636 / 0.408 | 0.818 / 0.657 / 0.417 |
| | zero300 (seed 2011) | 0.707 / 0.543 / 0.338 | 0.740 / 0.571 / 0.354 |
| | zero300b (seed 2012, seed check) | 0.711 / 0.555 / 0.346 | 0.744 / 0.579 / 0.363 |
| pln5 | s0 | 0.764 / 0.591 / 0.291 | 0.805 / 0.626 / 0.308 |
| | zero300 | 0.671 / 0.510 / 0.256 | 0.756 / 0.574 / 0.286 |
| r5 | s0 | 0.833 / 0.702 / 0.387 | 0.867 / 0.730 / 0.400 |
| | zero300 | 0.790 / 0.662 / 0.378 | 0.848 / 0.709 / 0.403 |

## Reading

- The loop-only statistic shows where the fall appears, not what causes it. Training only the weights outside the loop
  reproduces about half of the loop-only fall on Parcae and on pln5, and little of it on r5 (0.14; at k = 4 the
  loop-only value does not move at all, 0.400 → 0.403).
- So on two of three models about half of the fall needs no change to the loop's weights. On the fixed-5-loop model it
  is mostly a change in the loop.
- r5 is also the model whose truncated (k = 2) fine-tune moved the loop-only statistic least (0.51 of the full fall, flat
  at k = 4; v210). On r5 the fall comes with training the loop through all its loops.
- **What trained at k = 0.**
  - Part II: the post-loop layer and the read-out (11 tensors, 124.65M parameters, dominated by the output matrix; the
    embedding is frozen in this recipe).
  - Parcae: the coda (2 layers), the projection C, ln_f, a coda value embedding, and the token embedding wte. The embedding
    is tied to the read-out, so it changed (|ΔW| 98.9) and with it the loop's input, though not the loop's weights. So on
    Parcae "loop fixed" means the loop's weights, not its inputs.
- The seed check (Parcae, seed 2012) gives r = 0.44 against 0.48 for seed 2011.
- Clipping in the k = 0 runs was rare (3–5% of steps), against 97–100% for full backprop.

## Validity (LOCKED checks)

- **Loop fixed:** PASS on all three models (and on zero300b). Every changed parameter had a nonzero k = 0 gradient.
- **s0 agreement with v210:** max |diff| 0.0000 on every model, k and statistic (bar 0.005).
- **Guard:** every run used seed 2011 with 0 non-finite steps (zero300b: seed 2012, 0).

## Paper change (frozen mapping, PARTIAL) → v16.8

- "A change in the loop itself" is dropped from the abstract, §1, §3 and §6.
- The paper says the fall is not just the read-out's share of the gradient, and that training only outside the loop
  gives 0.48, 0.50 and 0.14 of it, stated per model.
- S16 row in Table 3; Table 10 gains the k = 0 rows; the Parcae tied-embedding caveat is stated.
