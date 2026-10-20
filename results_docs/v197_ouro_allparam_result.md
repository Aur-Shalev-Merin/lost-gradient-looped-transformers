# v197 result: Ouro-1.4B on the Table 1 protocol (all parameters)

**Run:** Kaggle aurmerin/astrocyte-v197-ouro v1, 2026-10-14, T4 x2 (sequences 0–7 and 8–15), driver 7f1c32f10bc2.
**Analysis:** `python v197_ouro_allparam.py --analyze kaggle-output-v197`.
**Why:** Table 1's Ouro row was measured on one middle layer with the probe's 128-token texts, unlike the other unstarred
rows; Appendix E's all-parameter value (0.996 / 0.893) used batch-averaged, clipped gradients. This puts Ouro on Table 1's
protocol: 16 GSM8K validation chunks of 512 tokens, batch 1, fp32, full vs truncated gradients at R = 4. Not
pre-registered: a descriptive re-measurement of one table cell, with no decision rule.

**Checks.** The repeated full pass reproduced the first exactly (err 0.0) on both GPUs; on a tiny random model the
checkpointed and plain loops agree exactly.

| | k = 1 | k = 2 | k = 3 |
|---|---|---|---|
| lost gradient, all parameters (median [95% CI]) | 0.921 [0.916, 0.949] | 0.812 [0.795, 0.882] | 0.715 [0.655, 0.799] |
| loop parameters only | 0.921 | 0.813 | 0.714 |
| one middle layer (old Table 1 subset), same sequences | 0.870 | 0.681 | 0.438 |
| size ratio / cosine, all parameters | 0.28 / 0.40 | 0.49 / 0.59 | 0.65 / 0.70 |
| geometric forms from ρᵢ (aligned / orthogonal) | 0.923 / 0.992 | 0.779 / 0.962 | 0.508 / 0.849 |
| measured-profile forms (aligned / orthogonal) | 0.948 / 0.996 | 0.710 / 0.901 | 0.388 / 0.701 |

- **ρ:** median 1.88 [1.74, 2.06]; every sequence has ρᵢ > 1 (1.55–2.27). Most of the growth is over the last loop
  (backward-signal size about 4 times larger before the last loop than after it), then 1.2–1.4 times per loop.
  The probe's 128-token mixed texts gave 1.52 on the one-layer subset.
- **Reading.** Keeping one or two of Ouro's four loops loses 92% or 81% of the gradient. The all-parameter value lies
  between the two geometric finite-loop forms at k = 2 and 3 and at the aligned one at k = 1. The single middle layer
  reads 0.05–0.28 lower than all parameters on the same sequences, as single layers did on SmolLM2. Batch-averaged
  gradients (Appendix E, D3) read higher again (0.996 / 0.893 / 0.750).
- **Paper (v15):** Table 1's Ouro row becomes all parameters on this protocol (ρ 1.88; 0.92 / 0.81); the one-layer and
  batch-averaged values are named where they appear.
