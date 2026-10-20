# v182 result: review checks for the looped-transformer paper (exploratory, 2026-10-11)

Measurement only, frozen models, no training. Script `v182_review_checks.py` (sha 1c6d1edbf41e). Kernels
astrocyte-v182-parcae (aurmerin) and astrocyte-v182-retro (aurshalevmerin). Outputs in `results_v182/` (gitignored).

## 1. Part of the loop vs all parameters (Table 1 protocol: 16 GSM8K-val seqs, batch 1, fp32, median)

| model | k | all | adapter+core0 | whole loop | single loop layers |
|---|---|---|---|---|---|
| Parcae-140m R16 | 1 / 2 / 4 / 8 | 0.813 / 0.663 / 0.421 / 0.180 | 0.822 / 0.650 / 0.398 / 0.182 | 0.838 / 0.681 / 0.430 / 0.182 | core1 0.842 / 0.692 / 0.442 / 0.184 |
| SmolLM2 (A) R8 | 1 / 2 / 4 | 0.358 / 0.137 / 0.016 | 0.383 / 0.147 / 0.018 | 0.378 / 0.144 / 0.017 | 0.22–0.41 / 0.06–0.15 / 0.006–0.020 |

- adapter+core0 (the Huginn / Parcae-1.3B choice) is within 0.025 of all parameters on both models.
- One loop layer alone (the Ouro choice) varies widely across layers on SmolLM2.

## 2. The starred rows' probe (loop_grad_probe.py) on Parcae-140m

- WikiText failed to load (new `datasets` rejects 'wikitext'), so 16 GSM8K-test seqs per R, not 32.

| variant | R16 rho | k 1 / 2 / 4 / 8 |
|---|---|---|
| subset fp32 | 0.818 | 0.857 / 0.740 / 0.532 / 0.203 |
| whole loop fp32 | 0.818 | 0.868 / 0.735 / 0.535 / 0.204 |
| subset bf16 | 0.820 | 0.852 / 0.737 / 0.519 / 0.200 |

- Subset vs whole loop: within 0.011. bf16 vs fp32: within 0.013.
- Probe setup vs Table 1 setup: +0.04 / +0.08 / +0.11 at k 1 / 2 / 4. Rule from the probe's own ρ: 0.82 / 0.67 / 0.45.
- In the original Sep-30 probe runs, WikiText vs GSM8K texts differ by up to ~0.1 (Parcae-1.3B R16 k4 0.49 vs 0.38).

## 3. Table 2 in float64

- v168's float32 CPU cosines ran 0.004–0.007 high; after-Adam cos up to 1.017 on SmolLM2.
- float64 values now in `paper_looped/table2.md`. SmolLM2 k=2 row reconciles: r 0.9505, cos 0.9921 → lost 0.132.
- After-Adam agreement: SmolLM2 0.967–0.997, Parcae 0.772–0.918.

## Paper changes (draft v6)

§3 "Checking the starred rows", Limitations bullet, Table 2 rebuilt with an after-Adam column, App. F note on float32 sums.
Fig 1 blended points (from existing measurements, no GPU): Parcae blended on the line; SmolLM2 blended (R 8) below by
up to 0.22.

## Correction (2026-10-11, v183 item 2)
Section 2's "+0.04 / +0.08 / +0.11" is withdrawn as a protocol effect. These runs loaded only the GSM8K half of the
probe's sequences (WikiText failed). In the real 32-sequence condition (WikiText fixed) the probe reads 0.83 / 0.67 /
0.45 / 0.19, within 0.03 of Table 1. Both scripts give identical numbers on identical sequences. See
`docs/v183_item2_probe_gap.md`.
