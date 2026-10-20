# v189 result: Parcae-370m sits on the ρᵏ line at 16 loops; at 32 loops ρᵏ overstates the loss

**Plan:** `docs/v189_parcae370_rho_plan.md` (committed and pushed 3e08af3, before data). **Run:** Kaggle
aurshalevmerin/astrocyte-v189-parcae370-rho v1, 2026-10-13, one kernel, two T4s. Outputs in `kaggle-output-v189/`
(gitignored). Analysis: `make_paper_figs.py` (Table 1 row, Figure 1, all-points summary).

**Checks (both pass).**
- Parcae-140m at R = 16 reproduces Table 1 exactly: 0.8132 / 0.6632 / 0.4210 at k = 1 / 2 / 4 (ρ 0.802).
- Parcae-370m at R = 32 reproduces the v166 pilot exactly: 0.674 / 0.451 / 0.238 at k = 2 / 4 / 8 (ρ 0.868).

**Result** (16 GSM8K validation sequences, all parameters, fp32; "rule" = median of per-sequence ρᵢᵏ):

| R | ρ | lost gradient k = 1 / 2 / 4 / 8 (/ 16) | rule | relative |
|---|---|---|---|---|
| 16 | 0.81 | 0.82 / 0.67 / 0.45 / 0.21 | 0.81 / 0.66 / 0.43 / 0.19 | +1 / +1 / +3 / +13% |
| 32 | 0.87 | 0.83 / 0.67 / 0.45 / 0.24 / 0.074 | 0.87 / 0.75 / 0.57 / 0.32 / 0.10 | −5 / −11 / −21 / −26 / −29% |

Measurement time: 61 s (R = 16) and 133 s (R = 32) for 16 sequences on a T4.

**Against the plan.** R = 16 is within the spread of the other models at k = 1 / 2 / 4. At k = 8 it is 0.025 above in
absolute terms and +13% relative, just past the +12% edge named in the plan. Read as the plan's first outcome, with that
caveat. As in Huginn (v187), the lost gradient barely moves between 16 and 32 loops while the fitted ρ rises
(0.81 → 0.87). The v164 measure() does not save the per-loop backward signal, so we cannot check here whether the
signal fades more slowly far from the output, as it does in Huginn.

**Over every measured point** (38 unblended points with per-sequence data, shrinking signal): measured vs ρᵏ from −29%
to +13%, max |diff| 0.117; ρᵏ outside the bootstrap 95% interval of the median for 14 of 38. At the loop counts of
Table 1 (43 points incl. SmolLM2 and blended Parcae): −18% to +13%, within 0.068.

**Paper change (draft v12).**
- Table 1 and Figure 1 get a Parcae-370m row at R = 16.
- §2 reports R = 32: the lost gradient carries over and ρ drifts.
- §3's "how close is close" covers every measured point, with bootstrap intervals.
- The abstract says "to within about 20% on six looped models … when ρ is measured at a loop count near the one in use".
  [Superseded 2026-10-13: that condition does not hold (the misses used ρ measured at the loop count in use); the
  abstract now reports the misses at the largest loop counts instead.]
