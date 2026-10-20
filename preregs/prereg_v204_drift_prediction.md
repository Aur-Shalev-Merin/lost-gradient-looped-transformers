# PREREG v204: does the two-mode window account predict ρ at 96 and 128 loops?

**Status: FROZEN** (this commit), before any measurement at 96 or 128 loops. Pushed to origin before the run.

**Derivation and predictions:** `docs/drift_derivation_2026-10-15.md`. The predictions below were computed by
`paper_drift_derivation.py` (sha 1cf92b90f83b) from the stored 64-loop S10 profiles only.

**Run (LOCKED).** The unchanged S10 driver `v196_band.py` (6b6495ce3d1e; v195_rho_modes 69efe65fd6fa), built by
`build_kaggle_v204.py` (be88bba260e1). One Kaggle T4 × 2 kernel:
- GPU 0: Parcae-140m at R = 96 and 128;
- GPU 1: Parcae-370m at R = 96 and 128;
- the same 16 GSM8K validation sequences and protocol as S10.

**Predictions (LOCKED).** Median over the 16 sequences of ρ_fit (the paper's fit over all R loops), with bootstrap
95% intervals:

| model | R = 96 | R = 128 |
|---|---|---|
| Parcae-140m (primary) | 0.891 [0.875, 0.904] | 0.900 [0.875, 0.910] |
| Parcae-370m (secondary; wide interval) | 0.922 [0.858, 0.951] | 0.924 [0.866, 0.954] |

Lost gradient: at k = 1, 2, 4, 8 the median stays within 0.03 of its 64-loop value (S10's bar). The 64-loop values
are 0.818 / 0.677 / 0.429 / 0.206 on 140m and 0.834 / 0.679 / 0.462 / 0.237 on 370m.

**Verdict (per model; Parcae-140m decides).**

| verdict | rule | paper change |
|---|---|---|
| PREDICTED | both measured ρ_fit inside their intervals, and the lost gradient within 0.03 at every k ≤ 8 | §3 gains one sentence; Appendix J gains the closed form, presented as predicting 96 and 128 loops before they were measured |
| RHO-ABOVE | a measured ρ_fit above its interval | the account is incomplete: a slower far mode keeps pulling the fit up. The closed form is reported as describing 32–64 loops only. |
| RHO-BELOW | a measured ρ_fit below its interval | the far mode fades beyond 64 loops. Same reporting as RHO-ABOVE. |
| LOST-MOVES | the lost gradient moves by more than 0.03 at some k ≤ 8 | reported. The paper's "carries over to longer loop counts" is limited to 64 loops. |

Also reported, no verdict: the window check, i.e. the 128-loop profile's slope over its first 96 loops against ρ
fitted in the 96-loop run.

**Predictions (mine).** Parcae-140m PREDICTED 0.65, RHO-ABOVE 0.25, other 0.10.

**Amendment 1 (2026-10-15, before any 96- or 128-loop data exist; prompted by a review note).**

**The problem.**
- Parcae-370m's intervals contain its own 64-loop value (0.907), so a pass there cannot tell "ρ keeps rising" from
  "ρ stops rising".
- Parcae-140m's 128-loop interval only just excludes its 64-loop value (0.875 vs 0.871).

**The changes.**
- **Parcae-370m is descriptive only.** It is reported, with no verdict.
- **PREDICTED on Parcae-140m needs a third condition.** Besides its two interval conditions, the paired rise must
  hold: per sequence, ρ_fit at 128 loops minus ρ_fit at 64 loops (the stored S10 run, same sequences), with the
  bootstrap 95% interval of the median above 0. The account predicts about +0.03.
- **A rise that fails the intervals.** If ρ_fit lies inside both intervals but the paired rise's interval includes 0,
  the verdict is RHO-FLAT. This is reported like RHO-BELOW: the closed form describes 32–64 loops only.

Nothing else changes.

**Cost.** About 20 GPU-minutes; it runs when the Kaggle quota resets, together with v203.
