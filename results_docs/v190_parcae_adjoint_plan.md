# v190: why ρᵏ overshoots on Parcae-370m at 32 loops (plan, written before data)

**Type.** Exploratory measurement on frozen models, no training. Follow-up to v189: at R = 32, Parcae-370m's lost
gradient (0.67 / 0.45 / 0.24 at k = 2 / 4 / 8) is the same as at R = 16, but its fitted ρ rose from 0.81 to 0.87, so
ρᵏ overstates the loss by 0.08–0.12. Parcae-140m at R = 32 sits on the line (0.44 against 0.46 at k = 4). v164's
measure() does not save the per-loop backward signal, so the shape of the fade was not visible.

**This run.** `v190_parcae_adjoint.py` repeats v164's measure() exactly (same 16 GSM8K validation sequences, seeds,
fp32, all parameters, a = 1) and saves adj[t] for every loop. One kernel, two T4s: Parcae-370m at R = 32 on GPU 0,
Parcae-140m at R = 32 on GPU 1. Check: medians must reproduce v189 (370m: 0.674 / 0.451 / 0.238) and v163's R 32
measurement (140m: 0.675 / 0.443 / 0.205 at k = 2 / 4 / 8) within 0.005.

**Hypothesis.** On 370m the backward signal fades faster near the output than far from it. A single fit over 32 loops
is pulled up by the slow far loops, which carry little of the gradient.

**Measures** (median over sequences, computed per sequence): ρ_near from a log-linear fit over the 16 loops nearest
the output (t = 16..31), ρ_far over t = 0..15, and the lost gradient against median(ρ_near,ᵢᵏ).

**Expectation if the hypothesis holds.**
- On 370m, ρ_near < ρ_far by at least 0.03, and median(ρ_near,ᵢᵏ) is within the paper's ±20% of the measured loss at
  k = 2 / 4 / 8.
- On 140m, ρ_near and ρ_far differ by less than on 370m, consistent with its single fit sitting on the line.

**What each outcome changes in the paper.**
- Holds → §2 says the 370m miss comes from the slower fade far from the output, and that a ρ fitted on the loops
  nearest the output fixes it. The advice becomes "fit ρ on the loops nearest the output". If Huginn at 64 behaves the
  same way (fit over the last 32 of 64 loops, from v187), say so too.
- ρ_near ≈ ρ_far on 370m, or ρ_near still overshoots → the 370m miss is not explained by the fade's shape. §2 says
  the cause is unknown and reports these numbers. The paper's current advice ("fit near the loop count in use") stays.
- Either way the figures go in `docs/v190_parcae_adjoint_result.md`, and §2 changes by at most two sentences.

**Compute.** Under 30 T4-minutes (v189: 133 s of measurement at R = 32, plus model and data loading).

**Amendment (2026-10-13, before any v190 data was seen).** Two reasons to downgrade the "Holds" branch above.
- A review pointed out that the near-loops fit would be checked on the same two misses that suggested it, with a
  split we chose (last half), so a pass is not independent evidence.
- Applying the same split to Huginn data we already had shows it is not a general fix. Over the last 32 of 64 loops it
  brings Huginn within 4% at k ≤ 8 (from −16%). Over the last 16 of 32 loops it makes Huginn worse (+16% / +25% at
  k = 4 / 8, from +6% / +4%).

So whatever v190 shows, the paper presents the shape of the fade as a likely explanation for the two misses, not as a
validated rule, and does not advise "fit ρ on the loops nearest the output". A rule would need a fitting window named
in advance and tested on a model or loop count it was not derived from.
