# v240 RESULT: TURN at k = 1 (S23, clean-up of S22)

**Provenance.**
- **Prereg:** `docs/prereg_v240_k1_cleanup.md`. It was frozen in d92473a and pushed to origin as a freeze-only commit
  (0a25691, branch `v240-freeze`) at 2026-11-11 17:31:39 UTC, before any run. Amendment 1 (report-only) was pushed as
  e58679d at 17:40:04 UTC, before any v240 output was read.
- **Kernels:** pushed about 17:32 UTC, all four complete by 20:37 UTC:
  - aurshalevmerin/astrocyte-v240-1 and -2;
  - aurmerin/astrocyte-v240-3 and -4.
- **Code:** every lane passed v239's selftest on the real model, with combined driver sha ac4e7f8570b8. The kernels'
  copies of all five files match the frozen hashes (v239 7f5262e51b75, v240 37ad1840140b, v174 f3cdba92e749,
  v163 db639c411892, v162 0ae4891d7abc).
- **Analysis:** `python v240_k1_cleanup.py --analyze results_v240`; output in `results_v240/v240_analysis.txt`.
  Amendment 1 counts come from a short script over `results_v239` and `results_v240` (the same rows, no new rule).

## Scored result

All 35 runs completed. All seven seeds were valid under the new rule (every run at most 10 non-finite steps and a
finite test loss), so seeds 2401101–2401105 were scored and the backups 2401108–09 were not run.

The floor gap was 0.165 nats (90% lower bound 0.163), so the gate passed.

| arm | cost φ (mean ± s.e., 5 seeds) | per seed | non-finite steps | Abar |
|---|---|---|---|---|
| trunc1 | 0.224 ± 0.006 | 0.244 0.219 0.229 0.216 0.213 | 0 0 1 5 1 | 0.893 |
| turn1 | 0.224 ± 0.005 | 0.242 0.217 0.229 0.218 0.211 | 0 0 1 4 1 | 0.894 |
| shrink1 | 0.001 ± 0.001 | 0.003 0.003 −0.000 −0.003 0.002 | 1 2 0 4 0 | 0.998 |

Margin m = 0.045. Turn share 0.997, shrink share 0.004.

**Verdict (frozen rule): TURN.**

## Unscored seeds

| seed | trunc1 | turn1 | shrink1 |
|---|---|---|---|
| 2401106 | 0.207 (0 non-finite) | 0.206 (0) | −0.001 (0) |
| 2401107 | 0.222 (5) | 0.221 (5) | 0.012 (10) |

`shrink1` on seed 2401107 had exactly 10 non-finite steps, at the limit but not over it. Its test loss was normal and
its cost 0.012.

## Amendment 1: unstable runs per arm (reported, not scored)

A run counts as unstable if it has more than 10 non-finite steps or a non-finite test loss. All runs made are counted,
backups included, over v239 (k = 1, 2, 4; 66 runs) and v240 (k = 1; 35 runs).

| arm | all k: unstable / runs | mean non-finite steps | k = 1: unstable / runs | k = 1 mean non-finite |
|---|---|---|---|---|
| trunc | 0 / 25 | 1.40 | 0 / 13 | 1.69 |
| turn | 0 / 25 | 1.36 | 0 / 13 | 1.77 |
| shrink | 2 / 25 | 3.08 | 1 / 13 | 3.77 |
| full | 0 / 13 | 0.85 | (k-independent) | |
| floor | 0 / 13 | 0.00 | | |

- **The two unstable size-only runs** are both from v239: `shrink1` on seed 2391105 (24 non-finite steps, test loss 20.4,
  diverged) and `shrink4` on seed 2391106 (11 steps, test loss 1.372, normal). v240 added no unstable run in any arm.
- **The comparison:** size-only 2 of 25 against 0 of 50 for truncated and direction-only (Fisher exact, two-sided,
  p ≈ 0.11). The size-only arm's mean non-finite count is higher (3.08 against about 1.4), partly from the diverged
  run's 24 steps. Without it, the mean is 2.21. The highest non-diverged size-only count, 11, is close to the next
  highest (10, also size-only).
- **Reading:** a weak hint that the size-only arm hits non-finite steps somewhat more often, not a shown difference.
  All three two-gradient arms compute g_R and g_k on every step and differ only in which update they apply, so this
  is a difference in how the update reacts to a spike, if it is anything.

## Descriptive, pooled (after the fact, not a verdict)

Pooling every valid k = 1 seed of both studies under v240's validity rule (12 seeds: v239's 2391101–04 and 2391106,
and all seven of v240's):
- trunc1 0.223 ± 0.003;
- turn1 0.222 ± 0.003;
- shrink1 0.002 ± 0.001.

Turn share 0.995, shrink share 0.007.

## Predictions vs outcome

| | predicted (prereg §4) | outcome |
|---|---|---|
| verdict | TURN 0.85, PARTIAL 0.08, SHRINK 0.02, INVALID or INCOMPLETE 0.05 | TURN |
| a diverged run | possible; would be replaced by a backup | none occurred; no backups were needed |

## Paper change (prereg §2, TURN)

§5: on Parcae-140m under the paper's recipe, the turn carries truncation's cost at k = 1 (S23, a clean-up of S22) and
at k = 2 (S22). S22's MIXED verdict and its diverged run are reported, and k = 4 is reported descriptively. The
abstract gains one sentence: what costs is the change of direction, not the lost size, which is why the lost
gradient's size is a poor guide to cost. The limits stay explicit:
- one model;
- AdamW with clipping;
- the reason the shrink is cheap (Adam or clipping) is untested.

Amendment 1's counts are reported alongside. App. G gains S22 and S23, and Table 4 gains their rows.
