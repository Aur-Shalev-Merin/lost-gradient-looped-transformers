# v187: does Huginn's ρ carry over across loop counts? (plan, written before data)

**Type.** Exploratory measurement on a frozen model, no training. Asked for by an outside review of the looped-truncation
paper ("ρ carries over to other loop counts rests on one comparison; Huginn samples R in training, show it there").

**What exists.** The 2026-09-30 probe (`docs/loop_grad_probe_huginn_2026-09-30.md`) measured Huginn-3.5B at R = 16 and
32. Recomputed with the paper's script (`make_paper_figs.py`, `probe()`):

| R | ρ | lost gradient at k = 1 / 2 / 4 / 8 |
|---|---|---|
| 16 | 0.86 | 0.90 / 0.79 / 0.61 / 0.32 |
| 32 | 0.88 | 0.90 / 0.80 / 0.62 / 0.36 |

**This run.** Same probe (`loop_grad_probe.py`), same 32 sequences (16 WikiText-103 test chunks, 16 GSM8K test, 128
tokens), same parameter subset (adapter + core_block[0]), bf16, at R = 8 and R = 64 (R = 48 if 64 does not fit). One
Kaggle kernel, one loop count per T4. Probe changes since Sep 30, none of which change the measured quantities: float64
sums done in chunks (checked equal to the direct float64 computation to 1e-12), an `LOOP_R` override, a WikiText
dataset-id fallback, and k = 32 added where k < R.

**Expectation.** If ρ is a property of the loop rather than of R, then at R = 64 the median lost gradient at
k = 1 / 2 / 4 / 8 should sit within 0.03 of the R = 32 values, with ρ in 0.85–0.89. At R = 8 the lost gradient at
k = 1 / 2 / 4 should also match, except that finite-R terms pull k = 4 down: the aligned form (ρᵏ − ρᴿ)/(1 − ρᴿ) with
ρ = 0.87 gives about 0.37 at k = 4, R = 8, against 0.57 for ρ⁴ alone. So at R = 8, compare k = 4 against the
finite-R form, not against R = 32. Huginn was trained with R sampled around 32, so R = 8 is far from its training
regime and may move ρ itself.

**What each outcome changes in the paper.**
- R = 64 (or 48) within 0.03 at every k, ρ within 0.85–0.89 → §2 says the lost gradient and ρ hold across 16–64 loops on
  Huginn (and 8–16 on Parcae).
- R = 64 off by more than 0.03, or ρ outside that range → §2's "ρ carries over to other loop counts" is narrowed to the
  range where it holds, and the shift is reported.
- R = 8 is reported either way, against the finite-R form.

**Compute.** About 3 T4-hours in one kernel (R = 16 + 32 took 1.8 h on one T4 on Sep 30).
