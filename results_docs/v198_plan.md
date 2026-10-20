# v198 plan: does a Part II looped LM give a usable third model for the cost test? (frozen before any v198 output)

**What.** A frozen-model fade measurement on two of Huang et al.'s Part II looped LMs (arXiv 2610.06833; Apache-2.0),
ported from their xLLM code (`ifm_looped.py`, validated: WikiText PPL 21.464 against their 21.47; notes in
`docs/v198_ifm_port_notes.md`). The model first named for this, `IFM/LoopedLM-P2-d4-s`, is their non-looped "Untied 4"
baseline, so it has no loop to measure.
- **huginn-s-fixed-pln5** (revision c581f369): Huginn's depth prior (capped Poisson-lognormal, mean 5); backprop
  through the last min(R, 10) loops in training. Kernel aurmerin/astrocyte-v198-ifm-pln5.
- **huginn-s-fixed-r5** (revision 17cdf78e): always R = 5, full backprop through all 5 loops. Kernel
  aurshalevmerin/astrocyte-v198-ifm-r5.

Both are about 304M parameters, small enough to fine-tune on a T4 with our recipe.

**Protocol.** `v198_ifm_fade.py`, Table 1's protocol, with these details:
- **Text:** 16 GSM8K-val chunks of 512 tokens, each example encoded in the model's training format (BOS + text + EOS),
  with attention kept within documents.
- **Gradients:** batch 1, fp32, all parameters. Also reported over the loop's parameters and over Table 1's starred
  part.
- **Settings:**
  - R = 5 (their training mean and evaluation depth) and R = 10;
  - blending a = 1, 0.5, 0.3;
  - every k at a = 1; k = 1, 2, 4 when blended.
- **Direction after Adam:** as in Appendix E (D2/v172), B = 24 clipped batches per arm, at k = 1, 2, 4 for each a, at
  R = 5.

**Validity.** A variant's numbers are used only if three checks pass:
- its selftest passes (k = R reproduces the full gradient; checkpointing changes nothing; k = 0 gives the loop no
  gradient);
- the in-kernel check of our port against their reference code does not fail;
- the noise floor (a repeated full pass) is below 1e-5 in all-parameter err.

**Statistics.** All are medians over the 16 sequences, all parameters, R = 5:
- err₁ and err₂, the lost gradient at k = 1 and 2 (a = 1);
- ρ;
- c(k) = 0.3418 · err_k^3.4335, the locked S6 cost curve (prereg v167, refit on 29 Parcae cells);
- the measurement-free references:
  - Parcae-140m's own unblended costs, 0.224 at k = 1 and 0.100 at k = 2 (R = 32, Table 5);
  - SmolLM2 (A)'s own unblended costs, 0.009 at k = 1 and −0.001 at k = 2;
- at k = 1: the change in err₁ and in the direction after Adam (Ā₁) from a = 1 to a = 0.3.

**Decision rule, per variant (applied in order):**

| label | rule |
|---|---|
| GROWS | median ρ > 1 at R = 5. Reported like Ouro; not a candidate for the curve-transfer test. |
| CANDIDATE | TRANSFER and CONTRAST both hold. |
| TRANSFER-ONLY | only TRANSFER holds. |
| CONTRAST-ONLY | only CONTRAST holds. |
| NOT-SUITABLE | neither holds. |

- **TRANSFER:** at k = 1 or at k = 2, c(k) differs from both references at that k by at least 0.03. The unblended
  settings of a fine-tuning test on this model could then tell the lost-gradient curve from rules that need no
  measurement.
- **CONTRAST:** blending at k = 1 is mostly a shrink: from a = 1 to a = 0.3, err₁ rises by at least 0.10 while Ā₁
  changes by at most 0.02. Then the lost-gradient and direction-after-Adam curves disagree there.

**Descriptive (no decision weight).**
- ρ and the lost gradient at R = 5 and R = 10, and whether the curve carries over (k ≤ 2).
- Whether the backward signal shrinks in a model trained with (nearly) full backprop. This bears on the paper's
  remark that every ρ < 1 model so far was trained with truncation.
- The summed-signal estimate of the loop's lost share.
- The blended forms.

**What follows.** A CANDIDATE or TRANSFER-ONLY variant is the model for the follow-up cost test, which gets its own
pre-registration with both curves locked. That test is the one the paper's Section 4 describes and the user parked as a
follow-up, not part of paper v15. NOT-SUITABLE on both variants ends the search on this release.

**Cost.** About 20–30 min per variant on T4 x2; two kernels in parallel, one per account.
