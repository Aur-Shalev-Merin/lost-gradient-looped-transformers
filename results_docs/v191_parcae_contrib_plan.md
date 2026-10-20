# v191: what each loop actually contributes to the weight gradient (plan, written before data)

**Type.** Exploratory measurement on frozen models, no training.

**Why.** On every model and loop count where we stored the per-loop backward signal, the lost gradient is above both
reference forms of Appendix A computed from that signal (aligned Σ_{j≥k}sⱼ/Σsⱼ and orthogonal), by 0.05–0.15 at
k = 1–2. The models checked are Huginn at 8 / 16 / 32 / 64 loops, Parcae-1.3B at 16, Parcae-140m at 8 / 16 / 32,
Parcae-370m at 32 and SmolLM2 (A) at 8. If every loop's contribution were proportional to its backward signal, with no
cancellation and even angles, the lost gradient would lie between those forms. So at least one of three things holds:
- **(a) sizes:** loops far from the output contribute more per unit of signal than near ones;
- **(b) angles:** the kept loops' contributions are less aligned with each other than the dropped loops' are;
- **(c) cancellation:** the kept loops' contributions partly cancel.

ρᵏ with a single fitted ρ still lands within about 20% because that ρ decays more slowly than the signal does near the
output. Separately, v190 left open why ρᵏ overshoots on Parcae-370m at 32 loops but not on Parcae-140m.

**This run.** `v191_parcae_contrib.py`, on Parcae-370m (GPU 0) and Parcae-140m (GPU 1), each at R = 32 and then
R = 16. Same 16 GSM8K validation sequences, seeds, fp32, all parameters and a = 1 as v164 / v189 / v190.

For every sequence we compute the gradient g_k for every k = 0 … R. The per-loop contributions are
c_j = g_{j+1} − g_j. This is exact: truncation does not change the backward signal inside the kept loops.

Saved per sequence:
- the backward signal sⱼ;
- the forward step of each loop;
- |cⱼ| and the cosine between neighbouring contributions (exact);
- the lost gradient at every k (exact);
- a CountSketch (2^20 buckets) Gram matrix of all contributions, for the full set of angles.

**Checks.**
- The lost gradient must reproduce v190 (R = 32) and v189 (R = 16) within 0.001 at k = 1 / 2 / 4 / 8.
- The sketch must match the exact quantities per sequence: contribution norms within 1%, neighbouring cosines within
  0.01, and the lost gradient rebuilt from the sketched Gram within 0.01.
- The TINY local test compares the sketched Gram with the exact one.

If a check fails, the affected quantities are not used.

**Analysis** (`v191_analyze.py`, committed with this plan). Index j counts loops before the output. Definitions:
- gain_j = (|cⱼ| / sⱼ) relative to the last loop;
- err_loop_k = |g_R − g_k| / |g_R − g_0|, i.e. without the coda/head gradient g_0 in the denominator;
- P0 = the two forms on sⱼ (sizes proportional to the signal);
- P1 = the two forms on the measured |cⱼ|.

**Q1 decision, per model and R** (k = 1, 2, 4, 8):

| outcome | rule |
|---|---|
| SIZES | err_loop within P1's band ± 0.02 at every k: measured sizes explain the shortfall, so (a) |
| ANGLES | err_loop more than 0.02 above P1's aligned form at two or more k: sizes do not explain it, so (b) or (c). The Gram matrix shows which: alignment among kept vs dropped loops, and the share of negative cosines. |
| SIZES-OVERSHOOT | err_loop more than 0.02 below P1's orthogonal form at two or more k |
| MIXED | otherwise; report how much of the P0 gap the measured sizes close |

**Q2 (descriptive, no threshold).** What differs between Parcae-370m and Parcae-140m at R = 32, given that ρᵏ misses
only on 370m: gain profile, size profile, angle structure? The R = 16 runs give the same comparison where 370m is on
the line.

**Q3 (descriptive).** Does gain_j track the forward step of loop j? Spearman correlation across loops, median over
sequences.

**What each outcome changes in the paper.** §3 already says (as of this commit) that the agreement of ρᵏ is partly an
offset, and that at least one of (a)–(c) holds.
- **SIZES on both models** → §3 adds that, measured directly on Parcae, loops further from the output contribute more
  per unit of backward signal (gain from 1 to X by 8 loops back), and that the measured sizes account for the lost
  gradient.
- **ANGLES** → §3 says the contributions follow the signal in size but the kept loops are less aligned (or partly
  cancel), with the numbers.
- **MIXED / SIZES-OVERSHOOT** → §3 reports the split as measured.
- **Q2:** a clear difference goes into §2 in one sentence as a description, not a rule. Otherwise the 370m miss stays
  unexplained.

In every case these are diagnostics on two frozen Parcae models. Huginn and SmolLM2 are not measured this way, and no
new prediction rule is proposed (as in the v190 amendment).

**Compute.** About 35 passes per sequence. Roughly 15–25 minutes per model on a T4, under 1 T4-hour in total.
