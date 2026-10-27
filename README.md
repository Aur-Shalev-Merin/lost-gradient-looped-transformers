# How Much Gradient Does Truncated Backprop Lose in Looped Transformers?

Code, pre-registrations, results and paper source for the paper of that title (Aur Shalev Merin, 2026; [arXiv:2610.28871](https://arxiv.org/abs/2610.28871)). The paper measures the *lost gradient*, |g − g_k| / |g|: the share of the weight gradient that truncated
backprop through the last k loops of a looped transformer throws away. It does this on nine looped language models, follows the
lost gradient through fine-tuning, and tests whether it predicts the training cost of truncation.

## Layout

| folder | contents |
|---|---|
| `paper/` | `draft.md` (the source; `main.tex` is generated from it; the development changelog is not included), the figures, `arxiv/` (preprint TeX, bib, bbl, PDF, arXiv source archive), `tmlr/` (style files and bib), and the build scripts `make_paper_tex.py`, `make_paper_pdf.py`, `make_paper_figs.py` |
| `preregs/` | the pre-registrations (`prereg_vNNN_*.md`), as in the development repository, except that notes saying where a review point came from were reworded and an implementation credit line was removed |
| `results_docs/` | result write-ups (`vNNN_*result*.md`), plans and derivation notes, with the same rewording |
| `code/` | experiment and analysis scripts and the Kaggle kernel builders (`build_kaggle_vNNN.py`), byte-identical to the versions in the development repository (except one reworded comment in `v167_analyze.py`, a file no pre-registration hashes), so the code hashes recorded in the pre-registrations verify; `code/frozen_versions/` holds versions that were later changed by an amendment |
| `results/` | the saved outputs (per-sequence JSON, logs), under their original folder names |
| `tools/verify_hashes.py` | checks every code hash recorded in `preregs/` against `code/` |
| `FREEZES.md` | each freeze and amendment commit, when it was committed, when GitHub received it, and when its first scored run was submitted |

## Studies

Internal IDs (vNNN) are experiment numbers in the development repository. S-numbers are the paper's (Table 4).

| study | question | prereg | result doc | main scripts | results | paper |
|---|---|---|---|---|---|---|
| S1 (v162) | does cost follow ρᵏ ordering? | `prereg_v162_*` | `v162_*` | `v162_parcae_trunc_train.py`, `v162_analyze.py` | `kaggle-output-v162`, `v162_*.json` | §5 |
| S2 (v163) | does raising ρ by blending raise cost at matched k? | `prereg_v163_*` | `v163_*` | `v163_parcae_damped_trunc.py`, `v163_analyze.py` | `kaggle-output-v163`, `v163_*.json` | §5 |
| S3 (v164) | out-of-sample cost prediction, 10 settings | `prereg_v164_*` | `v164_*` | `v164_parcae_predict.py`, `v164_analyze.py` | `kaggle-output-v164`, `v164_verdict.json` | §5, App. B |
| S4 (v165) | rerun of S3's unblended settings with an exclusion rule | `prereg_v165_*` | `v165_*` | `v165_analyze.py` | `kaggle-output-v165`, `v165_verdict.json` | App. B |
| S5 (v166) | cost prediction on Parcae-140m | `prereg_v166_*` | `v166_*` | `v166_analyze.py` | `results_v166*`, `v166_verdict.json` | §5 |
| S6 (v167) | cost prediction on SmolLM2 | `prereg_v167_*` | `v167_*` | `v167_retrofit_trunc.py`, `v167_analyze.py` | `results_v167*`, `v167_verdict.json` | §5 |
| S7 (v170) | SGD vs Adam | `prereg_v170_*` | `v170_*` | `v170_sgd.py`, `v170_analyze.py` | `results_v170*`, `v170_verdict.json` | App. D |
| S8 (v174) | matched direction after Adam, k = 1 vs 4 | `prereg_v174_*` | `v174_*` | `v174_mix_driver.py`, `v174_analyze.py` | `results_v174_*`, `v174_verdict.json` | App. G |
| S9 (v195) | where the turning comes from | `prereg_v195_*` | `v195_*` | `v195_rho_modes.py`, `v195_analyze.py` | `kaggle-output-v195` | §3, App. J |
| S10 (v196) | does ρᵏ overshoot more with the loop count? | `prereg_v196_*` | `v196_*` | `v196_band.py`, `v196_analyze.py` | `kaggle-output-v196` | §3, App. J |
| S11 (v200) | cost transfer to two new models | `prereg_v200_*` | `v200_*` | `v200_ifm_trunc.py`, `v200_analyze.py` | `results_v200`, `kaggle-output-v200-*` | §5, App. H |
| S12 (v201) | lost gradient of the averaged, clipped update, before and during fine-tuning | `prereg_v201_*` | `v201_result.md` | `v201_batch_lost.py` | `results_v201` | §4, App. I |
| S13 (v203) | does S12's fall hold on held-out problems? | `prereg_v203_*` | `v203_result.md` | `v203_heldout_lost.py` | `results_v203` | §4, App. I |
| S14 (v204) | two-mode prediction of ρ at 96 and 128 loops | `prereg_v204_*` | `v204_result.md` | `v204_score.py`, `v196_band.py` | `results_v204`, `kaggle-output-v204` | §3, App. J |
| S15 (v210) | is the fall in the loop's part of the gradient; truncated fine-tune; Ouro checks | `prereg_v210_*` | `v210_result.md` | `v210_followups.py` | `results_v210` | §3, §4, App. I |
| S16 (v211) | the same fine-tune with the loop's weights fixed | `prereg_v211_*` | `v211_result.md` | `v211_k0_control.py` | `results_v211` | §4, App. I |
| S17 (v218) | does the fall persist to step 600, hold on WikiText, and vary with the random initial state? | `prereg_v218_*` | `v218_result.md` | `v218_robustness.py` | `results_v218`, `kaggle-output-v218-*` | §4, App. I |
| S18 (v219) | truncation cost on Ouro-1.4B | `prereg_v219_*` | `v219_result.md` | `v219_ouro_cost.py` | `results_v219`, `kaggle-output-v219-pilot` | §5, App. K |

Unregistered measurements behind Tables 1–2 and the appendices come from `loop_grad_probe.py`
(`kaggle-output-loopgrad`), `v187`–`v192`, `v197`, `v198`, `v168`, `v171`, `v182`, `v217` and `retrofit_rho_probe.py`.
v161 is the study the paper reports as invalidated by a precision bug (Appendix F).

## Reproducing

- **Measurements and fine-tunes** ran on Kaggle (2 × T4 per kernel). Each `code/build_kaggle_vNNN.py` writes a
  self-contained kernel (`kaggle-kernel-vNNN-*/run.py`, the scripts embedded) to push with
  `kaggle kernels push -p <dir> --accelerator NvidiaTeslaT4`; set `KAGGLE_API_TOKEN` and the kernel ids in the
  builder to your own account. Models are downloaded from Hugging Face at the revisions pinned in the builders.
- **Analyses** run on a CPU from the saved outputs. Run them from `results/` with `code/` on the path, e.g.
  `cd results && PYTHONPATH=../code python ../code/v210_followups.py --analyze results_v210`.
- **Paper:** from `results/`, `python ../paper/make_paper_figs.py` redraws the figures; from `paper/`,
  `python make_paper_tex.py --preprint` rebuilds the TeX and PDF (needs `tectonic`).
- **Hashes:** `python tools/verify_hashes.py` from the repository root.

Environment: `environment.yml` (Python 3.12, PyTorch 2.5, NumPy, Matplotlib; the Ouro runs used
`transformers==4.54.1`, installed by their kernels).

## Not included

- The per-sequence output of `retrofit_rho_probe.py` behind Table 1's two SmolLM2 rows, which was saved outside the
  repository. Its logged medians are in `results_docs/retrofit_rho_probe_2026-10-02.md`.
- Model weights; the builders download them.
- The development repository's full history, which holds the freeze commits. It is private and available to reviewers
  on request; `FREEZES.md` gives GitHub's record of each push.

## Licence

Code: MIT (`LICENSE`). Text, documents and figures: CC BY 4.0.

## Citation

See `CITATION.cff`.
