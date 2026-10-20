# v198: porting a Part II looped LM (Huang et al. 2026) for the fade measurement

*2026-10-14. Status: port written and checked on CPU (plus a seconds-long selftest on the laptop GPU); measurement
script and Kaggle kernel built (`kaggle-kernel-v198-ifm/`), **not pushed**. Nothing committed.*

Files: `ifm_looped.py` (our port), `v198_ifm_fade.py` (measurement, selftest, reference check, analysis),
`build_kaggle_v198.py` (kernel builder). Downloaded files live only in the session scratchpad.

## 1. The requested checkpoint has no loop (decision needed)

`IFM/LoopedLM-P2-d4-s` (and `-d4-m`) is not a looped model. Its `config.json` says `arch: "transformer"`,
`loop_times: 1`, `num_layers: 4`, and it has no injection weights. 301,217,280 parameters is exactly
4 Llama blocks plus untied embedding and head. Their `xllm/paper_part2/recipes.py` calls it "The dense baseline D4:
the four Huginn blocks' width and numerics, run once in sequence, untied head". In the paper it is "Untied 4", which
"stacks four distinct blocks, matching the physical depth and parameter count" (Sec. 5.1, Tables 2 and 5; the model
card and their code call it "Table 4", which is Table 5 in the arXiv PDF). With no loop there is no fade to measure.

The looped models of the same release are the `IFM/LoopedLM-P2-huginn-*` checkpoints. They have the same width,
heads, FFN, tokenizer and format, at S (303.6M), M (819M) and L scale. I ported and validated the S model trained
with Huginn's depth prior, **`IFM/LoopedLM-P2-huginn-s-fixed-pln5`** (Table 2, S, "Fixed PLN-5"; revision `c581f369`).
The same code loads every Part II checkpoint with diagonal injection. You need to pick which one v198 should
measure (`build_kaggle_v198.py --variant`):

| variant (S) | training depth | backprop window in training | native Wiki PPL (Table 8) |
|---|---|---|---|
| `huginn-s-fixed-pln5` (built) | capped Poisson-lognormal, mean 5 | last min(R, 10) loops | 21.47 |
| `huginn-s-learned-entropy0p01` | learned prior, mean 5 (paper's flagship; Table 5 teacher) | last min(R, 10) | 21.12 |
| `huginn-s-learned-entropy0` | learned prior, no entropy term | last min(R, 10) | 21.14 |
| `huginn-s-fixed-r5` | always R = 5 | all 5 loops (full BPTT) | 20.62 |

`versions.json` still describes v198 as a measurement on `d4-s`. I did not edit it.

## 2. The model

**Layout.** Huginn-style, written 1 + 2xR + 1 in the paper: one prelude block, a core of two blocks run R times, one
coda block. Every block is a pre-norm Llama block: RMSNorm, then grouped-query attention (24 query and 6 key-value
heads of size 64) with RoPE (base 1e6, the full 64-dim head, adjacent pairs rotated as complex numbers), residual add;
RMSNorm, then a SwiGLU FFN (1,536 -> 4,352 -> 1,536), residual add. No biases and no dropout. RMSNorm eps is 1e-6, and
the gain is stored as an offset (gain = weight + 1; the prelude's norms sit at about 0.5). Width 1,536, vocabulary
64,256, untied embedding and head.

**The loop.** e = prelude(embed(x)), the injected input, computed once (no normalisation for this variant). The
state starts at h_0 ~ N(0, 1/d). Training draws it at random; evaluation uses a deterministic hash of (channel,
position, row) with the same distribution. Each recurrence is

  h <- core( exp(-dt*a) * h + dt * (W e) ),  dt = softplus(dt_bias), a = exp(A_log) per channel, W full d x d.

The injection sits inside the loop (it is part of F). In the trained model exp(-dt*a) has median 0.35 (range
0.31-0.53), dt has median 0.72, and W is about 0.9 I plus off-diagonal terms (spectral norm 7.1). The final state
goes straight to the coda (no readout matrix), then the final RMSNorm and the head. Attention stays within a document
(a BOS token starts one; their training sets `multi_segments=True`). Positions run on across documents.

**Parameters.** 303,579,648 in total: embedding 98.7M, head 98.7M, prelude 26.0M, coda 26.0M, loop 54.3M (injection
2.36M + two blocks of 25.96M). Table 1's starred-row choice (injection + first loop block) is 28.3M.

**Loop counts.** Training draws R per micro-batch from a capped Poisson-lognormal with mean 5 (sigma 0.5, cap 64; the
exact PMF is in their `looped_depth.py`; mode 3, P(R > 5) = 35%, P(R > 10) = 5.0%). The paper's tables evaluate
at R = 5 (`EVAL_DEPTH = 5`); its Figure 4 sweeps 1 to 64.

**Training backprop window.** Backprop goes through the last min(R, 10) recurrences
(`huginn_backprop_depth = 10`; paper App. B.3, Algorithm 2 line 4). Earlier states are detached, and the prelude
still gets gradient through the injection at each kept recurrence. With this depth distribution, 95.0% of training
forwards backpropagated through *all* their loops, and 97% of all recurrence steps received gradient
(E[min(R, 10)] = 4.85 of E[R] = 5). This model was trained with essentially full backprop through a loop of
sampled depth. That is unlike Huginn-3.5B (last 8 of about 32) and Parcae (last half). It bears on the paper's
remark that every model with rho < 1 so far was trained with truncation.

**Training.** 5,120 updates of 512 x 8,192 tokens (21.5B tokens). AdamW (0.9, 0.98), eps 1e-8, weight decay 0.1,
clip 1.0, warmup-stable-decay. BF16 compute under FSDP mixed precision; the released weights are the FP32 checkpoint
rounded to BF16. Documents are encoded as BOS + text + EOS and packed.

**Tokenizer.** "Jais64k": a byte-level BPE with a 64,000-token vocabulary plus 256 added special tokens (64,256 in
all), read with `tokenizers` from `tokenizer/tokenizer.json`. BOS `<|begin_of_text|>` = 0 and EOS
`<|endoftext|>` = 1 are added by hand (their wrapper encodes with `add_special_tokens=False`, then inserts them).

## 3. License

Weights and tokenizer: Apache-2.0 (model card, `LICENSE`, `NOTICE`). xLLM code (github.com/ifm-ai/xllm-loop): Apache-2.0
(repo `LICENSE`; third-party pieces are MIT/Apache, listed in `THIRD_PARTY_NOTICES.md`). Research use, modification
and redistribution are allowed with attribution. Our port is a reimplementation from reading their code. Two
algorithms are transcribed op for op: their RoPE table and their hashed initial state. The file header says where
they come from.

## 4. The port (`ifm_looped.py`)

`IFMLooped(cfg)` builds from the artifact's `config.json` (model section) and refuses settings it does not implement.
`load_model(dir)` checks every file against `artifact_manifest.json` (size and SHA-256), then loads the Safetensors
shards strictly. Parameter names are theirs, and BF16 weights are cast to FP32 exactly. The interface matches the
adapters in `loop_grad_probe.py`:

    h, ctx = m.prepare(ids)            # embedding, prelude, e, h_0 (with gradients, unless wrapped in no_grad)
    h = m.step(h, t, ctx, a, ckpt)     # one recurrence, optionally blended (1 - a) h + a F(h); ckpt = per-block checkpoints
    logits = m.readout(h, ctx)         # coda, final norm, head (float32)

Implemented: diagonal injection (validated), and Parcae-Decay prelude norm, OrthoInj and Huginn-Linear with its
exit norm (written to their code, not yet validated; the Kaggle reference check would cover them). Not implemented:
the hierarchical H/L state of Part I, MoE, partial RoPE, QK-norm.

## 5. Assumptions (where their code left a choice)

1. **Initial state.** Their eval-mode state (`Huginn.initialize_recurrent_state`): a deterministic hash of channel,
   position and batch row, with the N(0, 1/d) distribution of the training draw. It is computed on the input's
   device, so CPU and GPU give slightly different (equally valid) states. The paper's own evaluation seeds the state
   by example identity instead; `prepare(state='randn', seed=...)` gives a seeded training-style draw.
2. **Precision.** FP32, so the gain is weight + 1 exactly. Under their BF16 compute the gain was rounded to BF16
   (steps of 2^-8 to 2^-7). This matches their code run in FP32. BF16 vs FP32 on 512 WikiText tokens: 3.2545 vs 3.2529 nats.
3. **Document masking** is on by default, as in their training: a token attends only within its own GSM8K example.
   Table 1's other rows ran plain causal attention over the packed chunk. `V198_DOCMASK=0` reproduces that. On GSM8K
   chunk 0 the loss is 2.684 masked and 2.557 unmasked (the unmasked model reads earlier examples).
4. **Text encoding.** Each GSM8K example is BOS + question + "\n" + answer + EOS, the format the model was trained on.
   Packing is in order, as in v162. Table 1's other models used their own tokenizer conventions.
5. **Attention kernel.** PyTorch SDPA with a boolean mask instead of their FlashAttention 3 or their plain-matmul
   path. It is the same math (scale 1/8, GQA head j reads key-value head j // 4).
6. **Gradient path under truncation** is copied from their training: the first R - k recurrences run under no_grad,
   the state is detached, and e keeps its graph. The kept recurrences therefore send gradient into the prelude and
   the embedding through the injection.

## 6. Validation

**CPU, FP32, the real model** (`python -I ifm_looped.py --weights DIR --wikitext ... --gsm8k ...`):

- All 9 files match the manifest. 303,579,648 parameters, as the card says. Strict state-dict load.
- WikiText-103 test, first 512 tokens: 3.253 nats at R = 5 (PPL 25.9). First 2,048 tokens: 2.773 (PPL 16.0).
- **The paper's own protocol** (all 38 windows of 8,192 tokens, BOS + text + EOS, R = 5; our FP32 CPU forward):
  PPL **21.464** over 310,315 targets, against **21.47** in the paper (Table 8, native; BF16, FlashAttention 3,
  identity-seeded states). The target count is the paper's own (310,315; App. D.2). This is the strongest check
  available without running their code.
- Loss against loop count (first 512 WikiText tokens): R = 1 3.593, 2 3.363, 3 3.289, 4 3.271, **5 3.253**, 6 3.243,
  8 3.235, 10 3.232, 16 3.229, 32 3.228, 64 3.228. The loss falls monotonically and settles by about 16 loops (a
  fixed point in the forward pass). The default R = 5 is within 0.025 nats of the best. This matches the paper's
  Figure 4, where PLN-5 stays good at every depth.
- GSM8K validation chunk 0 (512 tokens, 4 examples), document-masked: R = 1 3.036, R = 5 2.684, R = 10 2.673.
- Greedy continuations at R = 5, coherent with small-model repetition:
  "The capital of France is located in the heart of the country, with a population of approximately 1.5 million
  people..."; "Water boils at a temperature of 100°C, which is 1.5 times the temperature of the Earth's
  atmosphere..."; "Once upon a time, there was a little girl named Lizzy, who was a little girl. She was a little
  girl, and she was a little girl...".

**Selftest on the real model, laptop GPU (seconds), FP32** (`v198_ifm_fade.py --selftest`): PASS. k = R through the
measurement path (checkpointed, hooks) matches plain backprop of the whole forward to 3.6e-7 (a = 1) and 3.0e-7
(a = 0.5). Checkpointed matches plain at k = 2, 1, 0 to at most 2.1e-7. Truncation leaves the loss unchanged
(difference 0) and the readout gradient unchanged (5e-9). At k = 0 the loop gets exactly zero gradient. The document
mask is active.

**TINY=1 end to end on CPU** (random tiny model of the same architecture): selftest PASS. Two measurement processes
ran (sequences 0-1 with Adam a = 1, 0.5; sequences 2-3 with a = 0.3) at R = 5 and 10 with a = 1, 0.5 and 0.3.
The reference-check plumbing (`V198_REFCHECK_MOCK=1`) PASSes, and `--analyze` prints every table. The RMSNorm
shim used by the reference check matches autograd (forward exact, backward 2e-8 in FP32).

**Not verified yet: equality with their code.** I did not execute their code locally. The reference check runs on
Kaggle only (section 8).

## 7. The measurement (`v198_ifm_fade.py`)

The Table 1 protocol (v164 `measure()`, v197). Frozen model, FP32 with TF32 off, batch 1, all parameters. Text is
the last 473 GSM8K train examples, packed into 512-token chunks; the first 16 are used. g_R is the full gradient,
and g_k backpropagates through the last k loops only. A full-gradient copy is kept on the CPU (pinned when
possible), and all reductions are in float64. The first sequence of each process repeats every full pass as a noise
floor. Per sequence and per cell:

- loop counts R = 5 (trained mean and eval depth) and 2R = 10. Blending a = 1, 0.5 and 0.3;
- k: every k = 0 .. R-1 at a = 1 (this includes 1, 2, 4 and 8, and k = 0 gives |g_R - g_0|); k = 0, 1, 2, 4 for
  blended cells;
- err_k, |g_k| / |g_R| and cos(g_k, g_R) over all parameters, the loop's parameters (injection + both core blocks),
  and the starred part (injection + first core block). Also |g_k - g_k'| from the previous k, which at a = 1 is the
  exact per-loop contribution |c_j|;
- the backward signal at each loop output: its norms (adj), rho_i from a line fit to log adj (as v164), and the
  float64 Gram matrix of the signals. The Gram matrix gives the summed-signal estimate of the lost share (v196);
- direction after Adam, as in v172. R = 5, arms k in {5, 1, 2, 4} at each a, B = 24 batches (`V198_B`) of 8 x 512
  GSM8K training tokens (fixed shuffle, rng 172). Each batch is clipped to norm 1. u = mean / (rms + 1e-8), and the
  output is cos(u_k, u_R), plus the raw lost gradient, size ratio and cosine of the batch-mean gradients.

`--analyze DIR` prints medians with bootstrap 95% CIs of rho and err_k per group, err over the loop-dependent part,
size ratios and cosines, the rho_i^k rule, the finite-R geometric forms, the measured-profile forms (as v197), the
summed-signal share, the per-loop contribution profile, the aligned form on measured contribution sizes, and the
Adam table.

Beyond the spec, all cheap: every k at a = 1, the k = 0 pass, the starred-part group, the Gram matrix and the
contribution steps.

## 8. Kaggle kernel (`build_kaggle_v198.py`, built, not pushed)

Kernel `aurmerin/astrocyte-v198-ifm-fade`, T4 x2, internet on. Push with
`kaggle kernels push -p kaggle-kernel-v198-ifm --accelerator NvidiaTeslaT4` (not done).

1. Setup. pip installs datasets, huggingface_hub, safetensors, tokenizers, einops and typeguard>=4.5. It downloads
   the checkpoint at its pinned revision into `/tmp/ifm_weights` (symlinks replaced by files, since their loader
   rejects symlinks), clones xllm-loop at `df0ed1ea` into `/tmp/xllm_ref`, and caches GSM8K.
2. Stage 1, in parallel. GPU 0 runs the selftest on the real model. GPU 1 runs the **reference check**: their own
   `DepthControlledHuginn`, built by their code from the same artifact through their `load_artifact`, against our
   port on the same inputs and device. It compares FP32 logits at R = 1, 5 and 10, with and without document masking,
   on a packed chunk and a single document; BF16 logits at R = 5; and FP32 gradients of the next-token loss. The
   gradients are the full one at R = 5 and the truncated one at R = 10, k = 2, the latter through their own
   `backprop_depth`, i.e. their training's truncation code. PASS needs logits within 2e-3 (relative to scale), CE
   within 1e-4 and gradients within 1e-4.
   Their code is built around a CUDA extension compiled for Hopper GPUs (sm90) and FlashAttention 3. FlashAttention 3
   does not run on a T4, and I did not try to compile the extension for it. So their attention runs on its own plain
   PyTorch path (`causal_attn_backend=None`), and their compiled group RMSNorm is replaced by a PyTorch
   transcription of their kernel (forward and analytic backward, checked against autograd). Every other missing
   kernel module is stubbed to raise if called. The check is therefore their Python model
   against our port, except for those few lines of RMSNorm.
3. Stage 2 runs only if the selftest PASSes and the reference check does not FAIL. A reference ERROR (their code
   would not import or run on Kaggle's torch) is reported and does not block. GPU 0 takes sequences 0-11 plus Adam
   at a = 1. GPU 1 takes sequences 12-15 plus Adam at a = 0.5 and 0.3.
4. `--analyze` output goes to `v198_analysis.txt`, and per-GPU stdout to `gpuN/stdout_stage*.txt`.

**Expected T4 runtime.** Forward cost per token is 150.6M + 54.3M x R multiply-adds: 0.42G at R = 5 and 0.69G at
R = 10. That is 0.43 / 0.71 TFLOP per 512-token sequence, and a full gradient pass with per-block recompute is
about 1.6 / 2.6 TFLOP. Per sequence: 6 full and 31 truncated passes, about 33 full-pass (R = 5) equivalents, roughly
50 TFLOP. Adam: 3 a-values x 4 arms x 24 x 8 = 2,304 micro-passes, about 1,600 full-pass equivalents, roughly 2.5 PFLOP.
Calibration: on the laptop's RTX 3050 Ti a full R = 5 pass took 0.78 s (2 TFLOPS effective), R = 10 1.15 s, and a
k = 2 pass 0.42 s. Peak memory was 2.7 GB. I assume a T4 is about as fast (8.1 TFLOPS FP32 peak, against about 7.6 for
the laptop GPU). Moving the 1.2 GB gradient copies cost 1.8 s per truncated pass with pageable host memory
(measured on the laptop, where pinning 1.2 GB failed). Pinned memory on Kaggle should cut that to roughly 0.5 s (an
estimate).

| part | GPU 0 | GPU 1 |
|---|---|---|
| setup + stage 1 | about 5 min | about 5 min |
| per-sequence cells (about 40 s each with pinned memory, about 85 s pageable) | 12 seq: 8-17 min | 4 seq: 3-6 min |
| direction after Adam (about 7 min per a) | a = 1: 7 min | a = 0.5, 0.3: 14 min |
| **total** | **about 20-30 min** | **about 22-25 min** |

One session of about half an hour.

## 9. Blockers, risks, decisions

- **Decision: which checkpoint** (section 1). Fixed PLN-5 is built by default. The fixed-R5 model is the closest
  analogue of Ouro's regime (full backprop at a fixed depth), and the learned-prior one is the paper's main model.
  All four diagonal-injection S variants are one flag away.
- **Decision: document masking** on (training semantics, default) or off (how Table 1's other rows were measured).
- **Risk: the reference check may ERROR** if their package does not import on Kaggle's torch or Python. It uses
  private FSDP imports, torch >= 2.6 APIs, typeguard and einops. The measurement then still runs. The port would
  then rest on the CPU checks and the paper's WikiText number (section 6). A FAIL blocks the measurement.
- **Risk: R = 5 is short.** A five-point fit makes rho_i noisy, which is why 2R = 10 and the every-k curve are
  included.
- The M scale (`huginn-m-*`, 819M) is not prepared. Two processes would need about 13 GB of host RAM each for the
  CPU gradient copies, which is tight on Kaggle; one GPU per run would do.
- Pre-registration: v198 is a frozen feasibility measurement, but if its numbers will gate the choice of the third
  cost-test model, freeze that rule before the first run (project rule).
