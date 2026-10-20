# v190 result: the fade's shape does not fully explain the Parcae-370m miss

**Plan:** `docs/v190_parcae_adjoint_plan.md` (d95ebbb), with an amendment committed before data (ea67e37) that rules out
presenting a near-loops fit as a validated rule. **Run:** Kaggle aurshalevmerin/astrocyte-v190-parcae-adjoint v1,
2026-10-13. Outputs in `kaggle-output-v190/` (gitignored). **Analysis:** `python v190_analyze.py`.

**Checks (pass).** Both models reproduce earlier measurements exactly: 370m 0.674 / 0.451 / 0.238 (v189), 140m
0.675 / 0.443 / 0.205 at k = 2 / 4 / 8 (v163).

**Result** (R = 32, 16 GSM8K validation sequences, all parameters; per-sequence fits, medians):

| model | ρ all 32 | ρ near (last 16) | ρ far (first 16) | measured k = 1 / 2 / 4 / 8 | vs ρ_allᵏ | vs ρ_nearᵏ |
|---|---|---|---|---|---|---|
| Parcae-370m | 0.868 | 0.782 | 0.892 | 0.83 / 0.67 / 0.45 / 0.24 | −5 / −11 / −21 / −26% | +6 / +10 / +21 / +70% |
| Parcae-140m | 0.825 | 0.790 | 0.876 | 0.82 / 0.68 / 0.44 / 0.20 | −1 / −1 / −4 / −5% | +4 / +8 / +14 / +35% |
| Huginn-3.5B, R 32 | 0.875 | 0.855 | 0.884 | 0.90 / 0.80 / 0.62 / 0.36 | +3 / +4 / +6 / +4% | +6 / +9 / +16 / +25% |
| Huginn-3.5B, R 64 (near = last 32) | 0.907 | 0.884 | 0.918 | 0.91 / 0.81 / 0.63 / 0.39 | 0 / −1 / −6 / −16% | +3 / +4 / +4 / +4% |

**Against the plan.**
- ρ_near < ρ_far on 370m by 0.11, passing the ≥ 0.03 bar.
- ρ_nearᵏ is within ±20% at k = 2 (+10%) but not at k = 4 (+21%) or k = 8 (+70%). It overcorrects.
- 140m curves almost as much (0.086 against 0.110), yet its all-loop fit is on the line.

So the curved fade is real on both Parcae models, and on 370m the measured loss lies between the all-loop and near-loop
predictions. But the shape alone does not explain why 370m misses and 140m does not. This is the plan's second branch:
the 370m miss is not explained by the fade's shape. Its cause is unknown. One possibility we did not test is that the
far loops' contributions to the weight gradient are smaller on 370m than its backward signal implies.

**Paper change (draft §2, three sentences).** It says the slower far fade is a likely cause for Huginn, but does not
fully explain 370m, and gives these numbers. Per the amendment, there is no "fit on the nearest loops" advice. The advice
stays "fit ρ near the loop count in use". [Superseded the same day after a review note: both misses (Parcae-370m
at 32, Huginn at 64) used ρ measured at exactly the loop count in use, so that advice is not supported and was removed
from the paper. The abstract now says the lost gradient fell up to about 30% below ρᵏ at the largest loop counts.]
