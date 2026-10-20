# v174 result: UNINFORMATIVE (2026-10-11)

- **Prereg:** `docs/prereg_v174_direction_intervention.md`, frozen 095bd38 and pushed to origin before any scored run.
- **Analysis:** `v174_analyze.py` (sha a01b030d06ef, unchanged since the freeze), `--verdict` on all 8 jsonl files;
  output `v174_verdict.json`.
- **Runs:** `results_v174_scored/scored-{1..4}/gpu*/v174_train_s*.jsonl`, 109/109 complete; every run carries driver
  sha 17b54eecf438 (= the frozen driver + v163); no duplicates; every test loss finite.
- **Interim look:** an unplanned look at 81/109 runs happened before scored-4 finished (`docs/v174_interim_look.md`).
  Nothing was changed after it; the paper discloses it.

## Validity
- FULL controls: 0–3 non-finite steps (limit 10); no seed excluded; evaluation seeds 1741101–1741108; backup unused.
- Floor gap 0.1647 over 8 seeds (≥ 0.03: pass). FULL pipeline check: v174 1.3734 vs v163 `train_cell` 1.3738.
- Underpowered flag: max paired SD 0.0054 (< 0.011): not flagged.

## Frozen verdict

| pair | seeds | mean ΔĀ | status | adjusted Δφ (90% CI) | τ | unit verdict |
|---|---|---|---|---|---|---|
| P1 (k1 0.5 vs k4 0.94) | 8 | +0.0007 | eligible | +0.0046 [+0.0028, +0.0065] | 0.015 | **same** |
| P2 (k1 0.625 vs k4 1.17) | 6 | −0.0001 | eligible | +0.0116 [+0.0076, +0.0155] | 0.015 | **undecided** (upper end 0.0005 over τ) |
| P3 (k1 0.82 vs k4 1.75) | 8 | +0.0108 | **DRIFTED** | — | — | not scored |
| P4′ (k1 0.924 vs k4 2.03) | 6 | +0.0378 | **DRIFTED** | — | — | H not scored |
| P5 (k1 0.91 vs k4 1.97) | 8 | +0.0219 | **DRIFTED** | — | — | H not scored |

Scored units 2/4 (P1, P2) < 3 → **OUTCOME: UNINFORMATIVE.** (Predicted, conditional on the gate: PREDICTS 0.35 /
INSUFFICIENT 0.40 / INCONCLUSIVE 0.25; the null-world simulation gave UNINFORMATIVE 0.015.)

P2 has 6 seeds: two seeds dropped by the stability rule (k1 0.625 seed 1741107: 16 non-finite steps; k4 1.17 seed
1741102: 25).

## Why three pairs drifted
The k = 4 arms' agreement on the scored seeds fell below what the single pilot seed showed. Per-seed ΔĀ (k1 − k4):
- P3: +0.003 to +0.022, no outlier; mean +0.0108, just over the 0.01 guard. k4 λ 1.75 averaged Ā 0.9335 against the
  pilot's ~0.941.
- P4′ / P5: mostly +0.004 to +0.012, plus one seed each where the k = 4 arm's agreement collapsed (ΔĀ +0.185 and
  +0.142; that arm's cost roughly doubled). These are the near-unstable high-dose k = 4 arms the prereg warned about
  (§2).
The pilot matched each pair on one seed. The simulation assumed drift 0.005 and Ā noise 0.007; the k = 4 arms' real
seed-to-seed spread at λ ≥ 1.75 was larger and one-sided.

## Reported, not scored (descriptive only)
Ā-adjusted Δφ = φ_k1 − φ_k4 at equal agreement, every pair (drifted pairs computed the same way, not scored):

| pair | φ k1 | φ k4 | adjusted Δφ (90% CI) | as share of mean φ |
|---|---|---|---|---|
| P1 | 0.027 | 0.022 | +0.005 [+0.003, +0.007] | ~19% |
| P2 | 0.043 | 0.032 | +0.012 [+0.008, +0.016] | ~31% |
| P3 | 0.097 | 0.083 | +0.012 [+0.005, +0.019] | ~14% |
| P4′ | 0.150 | 0.169 | +0.022 [+0.010, +0.034] | ~14% |
| P5 | 0.143 | 0.140 | +0.006 [−0.001, +0.014] | ~4% |

- In every pair, k = 1 costs a little more than k = 4 at matched agreement; the CI excludes zero in four of five. None
  reaches "different" (every CI's lower end is inside τ).
- One curve over all 10 mix arms: φ = 1.334 (1 − Ā)^0.94, MAE 0.011. Agreement explains most of the cost across both
  truncation types.
- Clipped fractions (both gradients above the clip): k1 0.12–0.13, k4 0.64–0.76.

## What it means
- The frozen test cannot say whether matched agreement means matched cost: matching did not hold on the scored seeds
  for the three higher-dose pairs.
- Descriptively: agreement accounts for most of the cost of both truncation types (one curve, MAE 0.011), with a small,
  consistent extra cost for k = 1 (about 0.005–0.02 φ, 4–31% of the cost) that no scored unit could call "different".
- For the paper: the reviewer advised keeping the direction measure for a follow-up with a third model. This result
  supports that: report v174 in an appendix as UNINFORMATIVE, with the descriptive table, or hold it for the follow-up.
  Either way the after-Adam direction measure stays a labelled lead.
- A follow-up that wants this answer needs multi-seed matching in the pilot, or k = 4 doses below the unstable zone.
