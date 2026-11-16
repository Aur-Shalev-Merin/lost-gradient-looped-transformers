# v241 RESULT: MIXED (k = 1 PARTIAL, k = 2 TURN), S24: S22's split under SGD

**Provenance.**
- **Prereg:** `docs/prereg_v241_sgd_split.md`.
  - The pilot rule was pushed first, before the controls-only pilot ran (freeze-only commit 2e01de8, branch
    `v241-freeze`, 2026-11-11 21:15:39 UTC).
  - The design was frozen in 39ca63e and pushed as freeze-only commit c9d8c82 to the same branch at 2026-11-12
    17:39:27 UTC, before any treatment run.
- **Kernels:** pushed 17:40:05 UTC, all four complete by 22:05 UTC:
  - aurshalevmerin/astrocyte-v241-1 and -2;
  - aurmerin/astrocyte-v241-3 and -4.
- **Code:** every lane passed the v241 selftest on the real model, with combined driver sha 3b446a33effc (the same as
  the local frozen set). Each kernel's copy of `v241_sgd_split.py` hashes to the frozen e505057ac886 (LF).
- **Analysis:** `OUT_DIR=results_v241 python v241_sgd_split.py --analyze results_v241`; the output is reproduced below
  and is also in `results_v241/v241_analysis.txt`.
  - Run it with the astrocyte env's own python: a repo `.venv` now shadows `conda run`.
  - The unscored diagnostics come from a scratch script over the same rows, with no new rule.

## Scored result

All 60 runs completed. Seed 2411106 is invalid: its full-backprop run diverged (582 non-finite steps, NaN test loss).
Seeds 2411101–2411105 are valid and scored, so the backups 2411107/08 were not needed.

The floor gap was 0.227 nats (90% lower bound 0.224), so the gate passed.

| arm | cost φ (mean ± s.e., 5 seeds) | per seed | non-finite steps | Mbar | step vs full | clipped |
|---|---|---|---|---|---|---|
| trunc1 | 0.257 ± 0.012 | 0.258 0.302 0.248 0.226 0.252 | 1 2 1 0 0 | 0.799 | 0.692 | 0.02 |
| turn1 | 0.201 ± 0.007 | 0.199 0.186 0.188 0.219 0.212 | 0 3 0 0 0 | 0.794 | 1.000 | 0.05 |
| shrink1 | 0.039 ± 0.038 | 0.047 0.014 −0.070 0.167 0.034 | 1 2 0 2 0 | 0.997 | 0.691 | 0.01 |
| trunc2 | 0.095 ± 0.014 | 0.100 0.068 0.067 0.145 0.095 | 0 3 1 1 0 | 0.906 | 0.833 | 0.03 |
| turn2 | 0.085 ± 0.012 | 0.087 0.097 0.055 0.063 0.122 | 1 2 1 0 1 | 0.915 | 1.000 | 0.04 |
| shrink2 | 0.012 ± 0.020 | 0.031 −0.057 −0.008 0.056 0.040 | 0 1 1 1 0 | 0.999 | 0.835 | 0.02 |

Labels:
- **k = 1:** margin 0.0515, turn share 0.78, shrink share 0.15.
  - PARTIAL. The shrink-only arm met its condition (0.039 ≤ 0.0515).
  - The turn-only arm missed its condition by 0.005: 0.201 against a bar of 0.257 − 0.0515 = 0.206.
- **k = 2:** margin 0.019, turn share 0.89, shrink share 0.13. TURN.

**Verdict: MIXED {1: PARTIAL, 2: TURN}.**

## Reported, not scored

**Matched-step controls** (full backprop at learning rate 0.1 × R_MATCH):

| arm | cost φ | n | per seed | non-finite steps |
|---|---|---|---|---|
| fullm1 (0.0702) | 0.037 ± 0.053 | 4 | 0.192 −0.044 −0.006 0.007 | 11 1 0 0 |
| fullm2 (0.0836) | −0.033 ± 0.013 | 5 | −0.033 −0.008 −0.080 −0.024 −0.022 | 0 1 1 1 0 |

**Step readings, as frozen:**
- **k = 1, AS-STEP:** shrink 0.039 against 0.037 (difference 0.002, margin 0.0515).
- **k = 2, BEYOND-STEP:** shrink 0.012 against −0.033 (difference 0.046, margin 0.019).

**Both step readings are weak. Read them with these caveats.**
- **k = 1.** The frozen analysis counts a reported run whenever its test loss is finite. It does not apply S23's ≤10
  non-finite rule to reported cells.
  - fullm1 on seed 2411101 had 11 non-finite steps and cost 0.192, and that one run makes up the whole frozen mean.
  - fullm1 on seed 2411103 diverged (528 non-finite steps) and is out of both readings.
  - With S23's rule applied (unscored), fullm1 is −0.015 ± 0.015 on 3 seeds. The shrink-minus-control difference is
    then 0.053, just above the margin of 0.0515, which would read BEYOND-STEP.
  - So at k = 1 the shrink cost 0.04 ± 0.04, and the matched-step control fell somewhere between −0.015 and 0.037,
    depending on one run. The step reading is not determined.
- **k = 2.** BEYOND-STEP here does not mean the shrink costs more than a smaller step. It means the matched-step
  control came out *below* full backprop (−0.033 ± 0.013): a constant learning rate 16% lower trained slightly better
  than 0.1 on these seeds.
  - The pilot's grid (0.03, 0.1, 0.3) was too coarse to place 0.1 above the optimum.
  - Full backprop at 0.1 also diverged on seed 2411106.
  - The shrink-only arm (0.012 ± 0.020) is within its margin of zero.

**Unscored diagnostics** (same rows, paired by seed):
- **k = 1:** truncation minus turn-only is 0.057 ± 0.018. Under SGD, giving the truncated direction back its full
  length recovered about a fifth of truncation's cost. Under AdamW (S23) that difference was 0.000.
- **Additivity:** truncation minus turn-only minus shrink-only is 0.018 ± 0.051 at k = 1 and −0.002 ± 0.018 at
  k = 2, so the two parts add up to truncation's cost within the noise.
- **k = 2:** truncation minus turn-only is 0.010 ± 0.020.
- **Steps actually taken:** median step length relative to full backprop was 0.69 (k = 1) and 0.83 (k = 2) for
  truncation and the shrink-only arm, matching the pilot's 0.70 / 0.84.
- **Against S7** (clip 1, indicative only): truncation cost 0.257 / 0.095 at k = 1 / 2, against S7's 0.263 / 0.104.

**Stability** (all runs; unstable means more than 10 non-finite steps or a non-finite test loss):
- full backprop: 1 of 6 (seed 2411106 diverged);
- fullm1: 2 of 6 (one diverged, one with 11 non-finite steps);
- fullm2: 0 of 6;
- floor: 0 of 6;
- all six two-gradient arms: 0 of 36.

This reverses the AdamW picture, where the shrink-only arm was the least stable. Under SGD with only a spike guard at
norm 5, the single-gradient runs at 0.07–0.1 were the fragile ones (3 of 18 against 0 of 36). The pilot's 0.3
diverged for both controls, so 0.1 is near the edge.

## Predictions (from the prereg, §4)

- **Overall:** TURN 0.65, **MIXED 0.12**, PARTIAL 0.10, SHRINK 0.03, INVALID/INCOMPLETE 0.10. MIXED is what came out.
- **Per k:** **k = 1 PARTIAL** was 0.20; **k = 2 TURN** was 0.80.
- **φ_fullm,1 ≤ 0.02** (0.8): **missed** as frozen (0.037). It would hold under S23's rule (−0.015), but the frozen
  number is the one that counts.

## What it means

Under SGD with only a spike guard, the change of direction still carries most of truncation's cost: 78% at k = 1 and
89% at k = 2. At k = 1 the turn alone fell short of truncation's cost by more than the margin (the PARTIAL label). Whether the
lost size carries the rest is unclear: the scored test of the size, the shrink alone, cost 0.04 ± 0.04, inside its
margin. The only sign that size costs is the paired truncation-minus-turn difference (0.057 ± 0.018), which is
unscored and was chosen after seeing the data; it is a lead, not a finding.

How that size cost compares with an equally smaller learning rate is undetermined. The control is too noisy at k = 1
and came out on the wrong side of zero at k = 2. The S22/S23 statement "the lost size costs nothing" holds under AdamW;
under SGD the pre-set test found no clear cost from size, but cannot rule out a small one.

## Paper changes (prereg §2, MIXED row)

- **§5:** report the per-k labels (k = 1 partial, k = 2 turn), the turn and shrink shares, and the step readings with
  their caveats.
- **Abstract:** keep the AdamW qualifier and say the SGD split is mixed.
- **In every case:**
  - App. G gains S24 with the pilot's shrink sizes;
  - Table 4 gains an S24 row;
  - §1 counts twenty-four studies.
- **Consistency edits, not in the mapping and recorded in App. B:**
  - §7 and §8 said the split was "untested without Adam". That is no longer true, so both now cite S24 briefly.
  - §5's sentence calling the SGD case untested is replaced.
