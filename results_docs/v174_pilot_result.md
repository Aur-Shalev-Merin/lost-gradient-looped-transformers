# v174 pilot result (2026-10-05; non-scored seed 1749901; final losses never computed)

**Run details.**
- Kernels astrocyte-v174-pilot-a/b (aurshalevmerin); outputs in `results_v174_pilot/`.
- Selftests all OK on the real model, including the FULL-arm agreement of exactly 1.000000.
- Loop-affected parameters: 29.3M of 144.3M.

**Realised agreement Ā (all parameters / loop-affected), steps 100–600:**

| k | λ 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|
| 1 | 0.997 / 0.995 | 0.986 / 0.974 | 0.984 / 0.965 (unstable: 26 skipped steps) | 0.896 / 0.769 |
| 2 | 0.998 / 0.996 | 0.990 / 0.980 | 0.974 / 0.945 | 0.946 / 0.884 |

**Size premise FAILS during training.**
- Both |g_R| and |g_k| exceeded the clip on only 14% of steps for k = 1 and about 30% for k = 2. The k = 1, λ = 0.75
  run reads 67%, but that run was unstable.
- Frozen-model norms were 2–9× the clip, but gradient norms fall below it once fine-tuning starts.
- So the paper's sentence "every update is clipped, which erases the size error" is wrong for training; it is
  corrected in the draft.
- The mix arms are unaffected, because they are rescaled to |g_R| by construction. The "size premise →
  UNINFORMATIVE" rule in draft 3.1 is therefore reclassified as a reported finding (draft 4).

**Gate (frozen-model A_f, B_f, error cosine; `results_v174_pilot/a/gpu0/v174_gate_a0.json`).**
- Agreement-matched k = 1 vs k = 2 pairs (same λ, similar A_f) all FAIL. The error-component cosines are 0.85–0.87,
  above the 0.75 limit.
- Only pairs whose A_f differs pass, and those are useless for the collapse test.
- So k = 1 and k = 2 truncation errors are too collinear, and the pre-registered fallback is k = 1 vs k = 4.
- Next: the k = 4 frozen gate first (cheap, no training), before any k = 4 pilot arms.

## Fallback gate k = 1 vs k = 4 (frozen; kernel astrocyte-v174-gate4): PASSES

Matched pairs (|ΔA_f| ≤ 0.01) have error-component cosines of 0.45–0.60, well under the 0.75 limit:
- k1 λ 0.5 ↔ k4 λ 0.75: A_f 0.984 / 0.981; error cosine 0.59;
- k1 λ 0.75 ↔ k4 λ 1.0: A_f 0.942 / 0.940; error cosine 0.45.

The high-dose pairs need k = 4 beyond λ = 1.5; for example, k1 λ 1.0 (A_f 0.793) needs roughly k4 λ ≈ 2.2. Next is
the k = 4 pilot (agreement only), with λ ∈ {0.75, 1.0, 1.25, 1.5, 2.0, 2.25}, plus k = 1 at λ ∈ {0.625, 0.875} as
interpolation points.
