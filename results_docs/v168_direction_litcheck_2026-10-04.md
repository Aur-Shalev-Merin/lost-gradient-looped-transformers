# Literature check: direction metrics vs truncation cost (2026-10-04)

Search, about 9 searches and fetches. Most papers were checked at abstract level only.

## Already known

- **Norm error decays geometrically.** Truncation error falls off geometrically, and the bias level controls the SGD
  rate: Aicher, Foti & Fox 2019, arXiv 1905.07473. This is a norm-bias framing.
- **A descent direction is enough.** A descent or cos > 0 condition on the approximate gradient suffices to train
  implicit (fixed-point) models:
  - JFB, Fung et al. 2022, arXiv 2103.12803;
  - phantom gradients, Geng et al. 2021, arXiv 2111.05177. This is the closest work: it argues direction over exact
    gradient, within one model family.
- **Biased-SGD theory** uses relative norm-bias conditions and covers SGD only: Ajalloeian & Stich 2020, arXiv
  2008.00051.
- **Truncation is costly in looped LMs.** Truncation under-trains the loop in looped LMs: Schwethelm et al. 2026, arXiv
  2604.21106. They report no gradient-cosine measurement.
- **Adam's update has a sign part:** "Dissecting Adam", Balles & Hennig 2018, arXiv 1705.07774. No paper links this to
  tolerance of truncated gradients.

## Not found

- A direction metric (cosine, sign agreement, Adam-preconditioned cosine, or first-order loss-decrease ratio)
  validated as a predictor of truncation cost **across models**.
- The observation that norm error predicts cost on one model and fails on another.

## Verify before claiming novelty

- Geng 2021 full text (the exact descent condition; any cosine-vs-accuracy measurement).
- The Huginn (2502.05171) and McLeish (2511.07384) appendices.
- Tallec & Ollivier 2017 (1705.08209) and the Ouro, HRM and TRM papers were cited from memory and are unverified.
