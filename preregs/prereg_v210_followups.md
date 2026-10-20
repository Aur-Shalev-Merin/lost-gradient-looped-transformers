# PREREG v210: two follow-ups from outside review 6

**Status: FROZEN** (this commit), before any measurement. Pushed to origin before the first run.

**Code (sha256 prefixes, LF):**
- `v210_followups.py` 619235c6b191 (measurement and analysis);
- `build_kaggle_v210.py` 35cd86b26463;
- imported unchanged: `v201_batch_lost.py` 4fe2fff13a39 (v201/v203 setup and fine-tune), `v197_ouro_allparam.py`
  7f1c32f10bc2 and `loop_grad_probe.py` 27b80fa0b999 (Ouro adapter).

## Part A: is the fall during fine-tuning a change in the loop? (review 6, M1)

**Why.** The paper's lost gradient, |g − g_k| / |g|, is over all parameters. Its denominator includes gradient that
reaches the weights with no loop credit at all: the layers after the loop, the read-out. If fine-tuning shifts gradient
toward those, the statistic falls with no change in the loop. Only full-backprop fine-tuning was measured (S12, S13).

**Models and setting.** Parcae-140m (R = 32), Part II pln5 and r5 (R = 5). The setups, LRs and fine-tune recipe are
those of v201/v203, imported unchanged.

**Measurement.**
- Data: the first 64 GSM8K test-split sequences (held out, as S13), batch 1, fp32, train mode, state draw seed
  301000 + s.
- Arms per sequence: k = R (full), 0 (no loop credit), 1, 2, 4.
- Two statistics:
  - all = |G − T_k| / |G| (the paper's);
  - loop = |G − T_k| / |G − T_0| (denominator without the no-loop-credit gradient).
- Three states:
  - s0: the trained model;
  - full300: after 300 of 600 steps of full-backprop fine-tuning;
  - trunc300: after 300 of 600 steps of fine-tuning with backprop through the last 2 loops. Same recipe, data order
    and LR as full300.
- Fine-tune seeds are pre-listed as 2011, 2012, 2013. The first that passes the guard (≤ 10 non-finite steps, finite
  last loss) is used, separately for full300 and trunc300.

**Decisions (LOCKED; `--analyze`).** Per model, using medians over sequences averaged over k = 1, 2, 4:
- fall_all = all(s0) − all(full300);
- fall_loop = loop(s0) − loop(full300);
- fall_trunc = all(s0) − all(trunc300).

| verdict | rule | paper change |
|---|---|---|
| A1: IN-LOOP | fall_loop / fall_all ≥ 0.5 on all three models | §3 and §1 drop the confound caveat and report the loop-only fall |
| A1: READOUT-SHARE | ratio < 0.5 on all three | the abstract, §1 and §3 say the fall is mostly the read-out's growing share, not a change in the loop |
| A1: MIXED | otherwise | stated per model |
| A2: TRUNC-FALLS | fall_trunc ≥ 0.5·fall_all on all three | §3: it also falls along a truncated fine-tune |
| A2: TRUNC-FLAT | \|fall_trunc\| < 0.03 on all three | §3: it does not fall along a truncated fine-tune |
| A2: TRUNC-RISES | fall_trunc ≤ −0.03 on all three | §3: it rises along a truncated fine-tune, which reverses the practical reading of S12 |
| A2: PARTIAL / MIXED | otherwise | stated per model |

## Part B: Ouro (review 6, M5)

**Why.**
- Ouro's backward signal grows (ρ = 1.88, about fourfold across the last loop). Ouro re-normalises its state between
  loops, so the growth might reflect state scale rather than sensitivity.
- Ouro's 92% / 81% are for the final-output loss, but Ouro trains on a weighted sum of the losses at every loop's exit.

**Measurement.** Ouro-1.4B (R = 4), the 16 GSM8K-validation chunks of v197 / Table 1, batch 1, fp32.
- **B1.** Per loop t, record:
  - the pre-norm residual x_t (output of the 24 layers) and the loop output h_t = norm(x_t), with their norms;
  - the backward signal on each, under the final-output loss with k = R.
  - Scale-free sizes are |dL/dx_t|·|x_t| and |dL/dh_t|·|h_t|. ρ̃ is fitted to each as for ρ.
- **B2.** The lost gradient of Ouro's training objective without its entropy term, over all parameters:
  - the objective is the per-token Σ_t p_t L_t;
  - p_t comes from the model's exit gate as in its code;
  - L_t is the next-token CE of lm_head(h_t);
  - k = 3, 2, 1, with loops before R − k run without gradient, while every loop's exit read-out (gate, lm_head) stays
    differentiable.
- Noise check: a repeated full pass of the exit objective.

**Decisions (LOCKED).**

| verdict | rule | paper change |
|---|---|---|
| B1: GROWS-SCALE-FREE | median ρ̃ > 1 on both h and pre-norm x | the Ouro paragraph says the growth survives normalising for state scale |
| B1: NORMALISATION | both ≤ 1 | the paper says Ouro's growth reflects state scale (normalisation between loops), and rewrites the ρ > 1 warning accordingly |
| B1: MIXED | otherwise | reported |
| B2: MOST-LOST | median lost gradient of the exit objective at k = 2 ≥ 0.5 | "under its training objective too, keeping two of four loops loses most of the gradient" |
| B2: LESS-LOST | < 0.5 | Ouro's 92% / 81% are explicitly limited to the final-output loss, with the exit-objective numbers |

## Predictions (mine, at freeze)

- A1: IN-LOOP 0.5, READOUT-SHARE 0.2, MIXED 0.3.
- A2: TRUNC-FALLS 0.4, PARTIAL/MIXED 0.35, TRUNC-FLAT 0.2, TRUNC-RISES 0.05.
- B1: GROWS-SCALE-FREE 0.55, NORMALISATION 0.25, MIXED 0.2.
- B2: MOST-LOST 0.6.

## Cost and runs

| kernel | account | contents |
|---|---|---|
| parcae | aurmerin | GPU0 s0 + full300, GPU1 trunc300 |
| ifm | aurshalevmerin | pln5 and r5 |
| ouro | aurmerin | 16 sequences over two GPUs |

About 4 T4-hours. TINY pipeline tests pass on CPU for Part A (tiny Part II model) and Part B (tiny random Ouro).
