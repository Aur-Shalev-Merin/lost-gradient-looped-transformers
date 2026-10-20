# v174 interim look (unplanned, disclosed)

2026-10-11 ~12:50, after scored-1/2/3 finished (81 of 109 runs) and before scored-4 (28 runs) finished,
the user asked to see results early. The frozen analysis (`v174_analyze.py --verdict`) was run on the 81
runs into a scratch directory (no `v174_verdict.json` written in the repo), plus an informal per-seed table.

Nothing was changed as a result: the prereg, code and run list were already frozen (095bd38) and all runs
launched. The prereg has no interim-analysis or stopping rule; this look has no decision attached to it.
The scored verdict is the one computed on all 109 runs. The paper discloses this look.

What was seen (3-5 seeds per pair; FULL/FLOOR missing for seeds that run in scored-4):
- frozen script: every unit below 6 seeds -> UNINFORMATIVE (expected with partial data).
- informal: on non-collapsed seeds k1 cost slightly more than k4 at matched agreement in every pair
  (dphi about +0.002 to +0.022); a few k4 high-dose seeds had agreement collapse (Abar 0.26-0.78) with
  doubled cost; P3/P4'/P5 mean dAbar above the 0.01 drift guard so far.
