# v192: per-loop contributions on Huginn-3.5B (plan, written before data)

**Type.** Exploratory measurement on a frozen model, no training. It is the second-family counterpart of v191 (Parcae,
`docs/v191_parcae_contrib_plan.md`), launched while v191 was still running, before any v191 data was seen.

**Why.** On Huginn, as on every model where we stored the per-loop backward signal, the lost gradient is above both
reference forms of Appendix A computed from that signal. At R = 32 the gap at k = 1 / 2 / 4 is +0.066 / +0.083 / +0.095.
Huginn's gradient is measured over loop parameters only (input-injection adapter + first core layer), so the gradient of
the layers after the loop plays no part. v191 decides between sizes, angles and cancellation on Parcae. v192 asks the
same question on a model from another lab, trained differently (Huginn samples its loop count; Parcae uses a fixed mean
of 8).

**This run.** `v192_huginn_contrib.py` reuses `loop_grad_probe.py` unchanged:
- the Sep-30 probe setup: 128-token sequences, bf16, the same parameter subset;
- R = 32, Huginn's mean training loop count and Table 1's setting;
- 16 sequences: probe WikiText indices 0–7 on GPU 0 and GSM8K indices 16–23 on GPU 1, from one Kaggle kernel.

Because the subset holds loop parameters only, g_0 = 0. For every k = 1 … 32 we compute g_k. From those:
- c_j = g_{j+1} − g_j, exactly;
- |c_j|, the cosine of neighbouring contributions and |g_R − g_k|, in float64;
- a CountSketch Gram matrix (2^20 buckets) of all c_j;
- the probe's per-loop backward signal and forward residual.

**Checks.**
- Per sequence, the lost gradient at k = 1, 2, 4, 8 and 16 must reproduce the Sep-30 probe (same sequences, same code)
  within 0.01. bf16 kernels are not bit-deterministic.
- The sketch must match the exact quantities: norms within 1%, neighbouring cosines within 0.01, the lost gradient
  rebuilt from the Gram within 0.01.
- The local TINY test (Parcae adapter, same code path) passed with errors below 1e-3.

**Analysis and decisions.** These are v191's, unchanged (`v191_analyze.py`, extended to read v192's files). Q1: SIZES,
ANGLES, SIZES-OVERSHOOT or MIXED, by the same ±0.02 rules at k = 1, 2, 4, 8. Q3: does gain_j track the forward residual
of loop j? Spearman correlation, median over sequences.

**What each outcome changes in the paper.**
- **Same Q1 verdict as v191 on both Parcae models at R = 32** → §3 states the cause as measured on two model families.
- **Different verdict** → §3 reports the cause per family and says it is not shared.

Either way it stays a diagnostic on frozen models; no new prediction rule (as in the v190 amendment).

**Compute.** About 34 passes per sequence, roughly 5–8 minutes per sequence at R = 32. That is about 1 hour per GPU,
run on two GPUs in parallel.

**Amendment (2026-10-13, before launch and before any v192 data).** A review pointed out that in bf16 a contribution
formed as g_{k+1} − g_k carries rounding noise of order 0.4% of the whole gradient. That is as large as the far loops'
contributions, and it would inflate them and make them look less aligned, biasing toward the "far loops push harder"
answer. The first version (commit c24a147) was never launched. The method is now direct:
- one backward pass with frozen parameters records the backward signal at every loop;
- each loop's contribution is one vector-Jacobian product through that loop alone, so its rounding is relative to its
  own size;
- tail sums from the farthest loop give the exact lost gradient at every k.

The cost drops from about 34 full passes per sequence to about 6. In the TINY fp32 test the direct contributions match
the subtraction ones within 1.3e-5 (relative, every loop), the lost gradients agree to 4 decimals, and their sum matches
the full gradient within 1e-7. Also recorded:
- the probe's own truncated passes at k = 1, 2, 4, 8, 16, for the reproduction check (within 0.01 per sequence);
- |Σ c_t − g_R| / |g_R| with g_R from one bf16 backward, i.e. the accumulation noise that subtraction would have
  carried.

Decisions and outcome mapping are unchanged.
