# v217 result: CPU reanalyses of saved data (exploratory)

**Script:** `v217_cpu_reanalysis.py` (CPU only, no new measurement). **Output:** `results_v217/v217_output.txt`.
**Status:** EXPLORATORY. Part 1 was suggested by outside reviews 7 and 8 after the long-loop data (S10, S14) had been
seen, so it tests nothing. Part 2 adds uncertainty to numbers the paper already reports.

## Part 1: ρ fitted on a fixed window next to the output

Question (reviews 7 Q5, 8 minor 4): the all-loop fit of ρ drifts up with the loop count R, so ρᵏ overstates the lost
gradient at long R. Does a ρ fitted only on the W loops next to the output avoid that, and is its ρᵏ a better summary?

Per sequence, ρ_W = exp(slope of log|λ_j| over j < W). Rule = median over sequences of ρ_W,iᵏ, as in Table 1. The data
are Parcae-140m and -370m at R = 32, 48, 64, 96, 128 (the same 16 sequences; v196, v204) and Huginn at R = 48.

| model | fit | median ρ at R = 32 / 48 / 64 / 96 / 128 | worst relative error of the rule (k with lost > 0.05), same R |
|---|---|---|---|
| Parcae-140m | all R loops | 0.825 / 0.852 / 0.871 / 0.898 / 0.908 | 5% / 35% / 61% / 105% / 125% (overstates) |
| | W = 16 | 0.790 / 0.786 / 0.786 / 0.785 / 0.785 | 26% / 29% / 29% / 30% / 30% (understates) |
| | W = 8 | 0.737 at every R | 57–58% (understates) |
| Parcae-370m | all R loops | 0.868 / 0.890 / 0.907 / 0.931 / 0.932 | 35% / 66% / 93% / 136% / 178% |
| | W = 16 | 0.782 / 0.783 / 0.786 / 0.784 / 0.784 | 41% / 40% / 39% / 40% / 30% |
| Huginn (R = 48) | all / W = 16 / W = 8 | 0.902 / 0.858 / 0.832 | 5% / 29% / 45% |

**Reading.**
- The window removes the drift entirely: ρ_16 is constant to ±0.005 from 32 to 128 loops.
- But the window ρᵏ understates the lost gradient at every R, by 30–40% at W = 16 and more for smaller windows. For
  example, on Parcae-140m at k = 4 it gives 0.38 against a measured 0.43, and at k = 8 it gives 0.15 against 0.21.
- The reason is that near the output the backward signal shrinks faster than the lost gradient does. The loops' gradient
  contributions turn as well as shrink (Appendix J), so the dropped sum keeps more weight than the per-loop sizes imply.
- So neither choice of fit window gives a ρᵏ that is accurate across loop counts. The all-loop fit is close at moderate R
  and drifts; the near-output fit is stable but biased low. This supports the paper's recommendation to measure the
  lost gradient itself rather than summarise it with ρᵏ.
- Paper (v2): one sentence in §3, replacing "we did not test whether its ρᵏ is a closer summary".

## Part 2: paired intervals for the fine-tuning falls

Falls are means over k = 1, 2, 4 of (median at stage 0 − median at step 300), with a paired bootstrap over sequences
(4,000 resamples, 90% intervals). Sequences only, not seeds.

| study | Parcae-140m | Part II, sampled loop count | Part II, fixed 5 loops | SmolLM2 (A) |
|---|---|---|---|---|
| S12, training sequences | 0.202 [0.187, 0.222] | 0.123 [0.113, 0.131] | 0.121 [0.112, 0.132] | 0.021 [0.017, 0.026] |
| S13, held-out (Parcae: seed 2012) | 0.207 [0.182, 0.221] | 0.111 [0.102, 0.123] | 0.114 [0.104, 0.121] | 0.023 [0.018, 0.026] |
| S15, full backprop, all | 0.196 [0.172, 0.211] | 0.112 [0.102, 0.123] | 0.114 [0.105, 0.121] | |
| S15, full backprop, loop only | 0.157 [0.135, 0.171] | 0.081 [0.072, 0.091] | 0.086 [0.077, 0.092] | |
| S15, last 2 loops, all | 0.193 [0.168, 0.208] | 0.095 [0.082, 0.107] | 0.070 [0.060, 0.075] | |
| S15, last 2 loops, loop only | 0.158 [0.133, 0.172] | 0.065 [0.052, 0.076] | 0.044 [0.036, 0.049] | |
| S16, no loop credit, loop only | 0.076 [0.056, 0.086] | 0.041 [0.030, 0.049] | 0.012 [0.003, 0.021] | |

**Reading.**
- Every fall excludes zero, including SmolLM2's small one.
- The smallest is the fixed-5-loop model's no-loop-credit fall (0.003–0.021), which is the paper's "little on the third
  model".
- Paper (v2): intervals for the S12/S13 falls in §4 or Appendix I.
