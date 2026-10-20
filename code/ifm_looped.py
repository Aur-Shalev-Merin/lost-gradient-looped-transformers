"""Our own minimal PyTorch port of the looped models of Huang et al. 2026, "Towards Looped Models Done Right, Part II:
Rethinking at Fixed Points" (arXiv 2610.06833), released as IFM/LoopedLM-P2-huginn-* on Hugging Face in xLLM's native
BF16 Safetensors format. Written from a reading of their code (github.com/ifm-ai/xllm-loop @ df0ed1ea,
xllm/models/looped.py `DepthControlledHuginn`, xllm/models/transformer.py, xllm/modules/*); none of their code is
imported. Their code and weights are Apache-2.0; the RoPE table and the hashed initial state below transcribe their
algorithms op for op. Notes, assumptions and validation: docs/v198_ifm_port_notes.md.

NOT a port of IFM/LoopedLM-P2-d4-s / -d4-m: those are the paper's non-looped "Untied 4" baseline
(config arch='transformer', loop_times=1; recipes.py: "The dense baseline D4 ... run once in sequence").

Architecture (Huginn layout 1 + 2 x R + 1, every block a pre-norm Llama block):
  e   = prelude(embed(ids))                        layers[0]; Parcae-Decay variant: e = RMSNorm(e) (learned gain)
  h_0 = N(0, 1/d)-like state                       eval: their deterministic hash of (dim, position, batch row)
  h_{t+1} = F(h_t) = core(inj(h_t, e))             core = layers[1], layers[2]; R recurrences
     diagonal inj:  exp(-dt*a) * h + dt * (W e),   dt = softplus(dt_bias), a = exp(A_log)  (per channel)
     OrthoInj:      decayed state minus its component along dt*(W e), then + dt*(W e)
     Huginn-Linear: W [h; e] + b, then a parameter-free RMSNorm after the core
  logits = head(RMSNorm(coda(h_R)))                coda = layers[3]; untied head; logits in float32
Block: x + Wo attn(RoPE(Wq n(x)), RoPE(Wk n(x)), Wv n(x)), then x + W2 (silu(W1 n(x)) * W3 n(x)); n = RMSNorm whose
gain is stored as an offset (gain = weight + 1); GQA (24 query / 6 key-value heads at S); RoPE base 1e6 on the full
64-dim head, interleaved (complex) pairs; no biases; no dropout. Attention stays within documents (a BOS token starts
a new document), as in their training (`multi_segments=True`); positions are not reset at document starts.

Interface for truncated backprop through the last k loops (same shape as the Ouro / Huginn / Parcae adapters in
loop_grad_probe.py):  h, ctx = m.prepare(ids);  h = m.step(h, t, ctx, a, ckpt) for t < R;  logits = m.readout(h, ctx).

  python -I ifm_looped.py --weights DIR [--wikitext PARQUET] [--gsm8k PARQUET] [--out JSON]   (CPU validation)
"""
import argparse, hashlib, json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.checkpoint import checkpoint

SUPPORTED = dict(arch=('huginn',), huginn_depth_control=(True,), apply_rmsnorm=(True,), swiglu=(True,),
                 attn_act_func=('softmax',), qknorm=(False,), apply_attn_gate=(False,), residual_func=('base', 'add'),
                 scale_emb=(False,), num_experts=(0,), num_values=(0,), layernorm_num_groups=(1,), norm_affine=(True,),
                 huginn_hierarchical_state=('none',), huginn_split_module_repeats=(1,), output_size=(-1,),
                 huginn_input_injection=('diagonal', 'linear'), huginn_prelude_norm=('none', 'rms'),
                 huginn_recurrent_exit_norm=('none', 'parameterless_rms'), huginn_state_init=('normal', 'zero', 'input'),
                 v_head_dim=(None,), dropout=(0.0,), hidden_dropout=(0.0,), attention_dropout=(0.0,))


# ------------------------------------------------------------------------------------------------------- modules
class RMSNorm(nn.Module):
    """xLLM GroupRMSNorm with one group: y = x * rsqrt(mean(x^2) + eps) * (weight + 1), computed in float32 and cast
    back (their CUDA kernel accumulates in float32; the gain `weight + 1.0` is formed in the parameter dtype)."""

    def __init__(self, dim, eps, affine=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim)) if affine else None

    def forward(self, x):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            y = y * (self.weight + 1.0).float()
        return y.to(x.dtype)


def rope_table(head_dim, base, length, device):
    """Their RotaryEmbedding table, op for op (python-float powers, float32 reciprocal, outer product, polar)."""
    freqs = torch.tensor([base ** (j / head_dim) for j in range(0, head_dim, 2)], dtype=torch.float32, device=device)
    freqs = 1.0 / freqs
    ang = torch.outer(torch.arange(length, dtype=torch.float32, device=device), freqs)
    return torch.polar(torch.ones_like(ang), ang)                       # complex64 [L, head_dim / 2]


def apply_rope(x, freqs_cis):
    """x [B, L, H, D]; consecutive pairs (x[2i], x[2i+1]) rotate as one complex number (their apply_rotary_embeddings)."""
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * freqs_cis.unsqueeze(1)).flatten(3).type_as(x)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        d, self.H, self.KV, self.hd = c['model_dim'], c['num_heads'], c['num_kv_heads'], c['head_dim']
        self.norm = RMSNorm(d, c['rmsnorm_eps'])
        self.wq = nn.Linear(d, self.H * self.hd, bias=False)
        self.wk = nn.Linear(d, self.KV * self.hd, bias=False)
        self.wv = nn.Linear(d, self.KV * self.hd, bias=False)
        self.wo = nn.Linear(self.H * self.hd, d, bias=False)

    def forward(self, x, freqs, mask):
        B, L, _ = x.shape
        n = self.norm(x)
        q = self.wq(n).view(B, L, self.H, self.hd)
        k = self.wk(n).view(B, L, self.KV, self.hd)
        v = self.wv(n).view(B, L, self.KV, self.hd)
        q, k = apply_rope(q, freqs), apply_rope(k, freqs)
        rep = self.H // self.KV                                         # query head j reads key-value head j // rep
        k, v = k.repeat_interleave(rep, dim=2), v.repeat_interleave(rep, dim=2)
        o = Fn.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask,
                                            is_causal=mask is None, scale=1.0 / math.sqrt(self.hd))
        return x + self.wo(o.transpose(1, 2).reshape(B, L, self.H * self.hd))


class FFN(nn.Module):
    def __init__(self, c):
        super().__init__()
        d, f = c['model_dim'], c['ffn_hidden_dim']
        self.norm = RMSNorm(d, c['rmsnorm_eps'])
        self.fc1 = nn.Linear(d, f, bias=False)
        self.fc2 = nn.Linear(f, d, bias=False)
        self.fc3 = nn.Linear(d, f, bias=False)

    def forward(self, x):
        n = self.norm(x)
        return x + self.fc2(Fn.silu(self.fc1(n)) * self.fc3(n))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attention = Attention(c)
        self.nffn = FFN(c)

    def forward(self, x, freqs, mask):
        return self.nffn(self.attention(x, freqs, mask))


class DiagonalInjection(nn.Module):
    """exp(-dt * a) * h + dt * (W e), dt = softplus(dt_bias), a = exp(A_log); with `orthogonal`, OrthoInj."""

    def __init__(self, d, orthogonal=False, eps=1e-6):
        super().__init__()
        self.input_adapter = nn.Linear(d, d, bias=False)
        self.A_log = nn.Parameter(torch.zeros(d))
        self.dt_bias = nn.Parameter(torch.zeros(d))
        self.orthogonal, self.eps = orthogonal, eps

    def forward(self, h, e):
        upd = self.input_adapter(e)
        dt = Fn.softplus(self.dt_bias).to(h.dtype)
        retained, injected = h * torch.exp(-dt * torch.exp(self.A_log).to(h.dtype)), dt * upd
        if self.orthogonal:                                             # per token, in float32 (float64 if fp64)
            cd = torch.float64 if retained.dtype == torch.float64 else torch.float32
            v, q = retained.to(cd), injected.to(cd)
            coef = (v * q).sum(-1, keepdim=True) / (q.square().sum(-1, keepdim=True) + self.eps)
            retained = (v - coef * q).to(retained.dtype)
        return retained + injected


class LinearInjection(nn.Module):
    """Huginn-Linear: W [h; e] + b."""

    def __init__(self, d):
        super().__init__()
        self.adapter = nn.Linear(2 * d, d, bias=True)

    def forward(self, h, e):
        return self.adapter(torch.cat([h, e], -1))


class Output(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.final_norm = RMSNorm(c['model_dim'], c['rmsnorm_eps'])
        self.output = nn.Linear(c['model_dim'], c['vocab_size'], bias=False)

    def forward(self, x):
        return self.output(self.final_norm(x)).float()


# --------------------------------------------------------------------------------------------------------- model
def hashed_normal_state(like, cache_len, batch_term, scale):
    """Their `Huginn._hashed_normal_state`, op for op: fract(43758.5453123 * sin(x)) with x = 0.013579 * dim
    + 0.021713 * pos + batch_term + 0.123457, mapped to a normal by erfinv, times `scale`; float32 on like's device."""
    B, L, d = like.shape
    dim = torch.arange(0, d, device=like.device, dtype=torch.float32).view(1, 1, -1)
    pos = torch.arange(cache_len, cache_len + L, device=like.device, dtype=torch.float32).view(1, -1, 1)
    s = dim.expand(B, L, d).clone()
    s.mul_(0.013579); s.add_(pos * 0.021713); s.add_(batch_term); s.add_(0.123457)
    s.sin_(); s.mul_(43758.5453123); s.sub_(s.floor()); s.clamp_(1e-6, 1.0 - 1e-6)
    s.mul_(2.0).sub_(1.0); s.erfinv_(); s.mul_(scale)
    return s.to(dtype=like.dtype)


def doc_mask(ids, bos_id):
    """Boolean [B, 1, L, L] attention mask (True = attend): causal and within a document, a document starting at
    every BOS (their `multi_segments`: segment = cumsum(ids == BOS)). None when no row has a BOS after position 0."""
    seg = torch.cumsum((ids == bos_id).long(), dim=1)
    if bool((seg == seg[:, :1]).all()):
        return None
    L = ids.shape[1]
    causal = torch.ones(L, L, dtype=torch.bool, device=ids.device).tril()
    return (causal & (seg[:, :, None] == seg[:, None, :])).unsqueeze(1)


class IFMLooped(nn.Module):
    """Parameter names match the released state dict (strict load)."""

    def __init__(self, cfg):
        super().__init__()
        c = self.cfg = dict(cfg)
        for key, allowed in SUPPORTED.items():
            if key in c and c[key] not in allowed:
                raise NotImplementedError('ifm_looped does not implement %s=%r' % (key, c[key]))
        c.setdefault('head_dim', c['model_dim'] // c['num_heads'])
        if c.get('rope_head_dim') not in (None, c['head_dim']):
            raise NotImplementedError('partial RoPE (rope_head_dim < head_dim) is not implemented')
        self.n_pre = c['huginn_prelude_layers']
        self.n_core = c['huginn_recurrent_layers'] if c.get('huginn_recurrent_layers') is not None else \
            c['num_layers'] - c['huginn_prelude_layers'] - c['huginn_coda_layers']
        assert self.n_pre + self.n_core + c['huginn_coda_layers'] == c['num_layers']
        self.R = c['loop_times']
        self.bos_id = c.get('bos_id', 0)
        self.embed = nn.Embedding(c['vocab_size'], c['model_dim'])
        self.layers = nn.ModuleList([Block(c) for _ in range(c['num_layers'])])
        d = c['model_dim']
        if c['huginn_input_injection'] == 'linear':
            self.input_injection = LinearInjection(d)
        else:
            self.input_injection = DiagonalInjection(d, bool(c.get('huginn_prelude_orthogonal')),
                                                     c.get('huginn_ortho_projection_eps', 1e-6))
        if c.get('huginn_prelude_norm', 'none') == 'rms':
            self.huginn_prelude_norm_module = RMSNorm(d, c['rmsnorm_eps'])
        self.exit_norm = RMSNorm(d, c['rmsnorm_eps'], affine=False) \
            if c.get('huginn_recurrent_exit_norm', 'none') == 'parameterless_rms' else None
        self.output = Output(c)

    # parameter groups -----------------------------------------------------------------------------------------
    def prelude_layers(self):
        return list(self.layers[:self.n_pre])

    def core_layers(self):
        return list(self.layers[self.n_pre:self.n_pre + self.n_core])

    def coda_layers(self):
        return list(self.layers[self.n_pre + self.n_core:])

    def loop_parameters(self):
        """Parameters applied inside every recurrence: the input injection and the recurrent (core) blocks."""
        return list(self.input_injection.parameters()) + [p for b in self.core_layers() for p in b.parameters()]

    # forward pieces --------------------------------------------------------------------------------------------
    def prepare(self, ids, docmask=True, state='hashed', seed=0):
        """Embedding, prelude, the injected input e and the initial state h_0. Runs with gradients (wrap in no_grad
        if none are wanted): the truncated loops' injection carries gradient back into prelude and embedding, as in
        their training. state='hashed': their eval-mode state (`Huginn.initialize_recurrent_state`, batch row b);
        'randn': N(0, 1/d) from a CPU generator seeded by `seed` (their training draw, made reproducible)."""
        c = self.cfg
        L = ids.shape[1]
        freqs = rope_table(c['head_dim'], c['rope_base'], L, ids.device)
        mask = doc_mask(ids, self.bos_id) if docmask else None
        x = self.embed(ids)
        for b in self.prelude_layers():
            x = b(x, freqs, mask)
        e = self.huginn_prelude_norm_module(x) if hasattr(self, 'huginn_prelude_norm_module') else x
        init = c.get('huginn_state_init', 'normal')
        if init == 'zero':
            h0 = torch.zeros_like(e)
        elif init == 'input':
            h0 = e
        elif state == 'randn':
            g = torch.Generator().manual_seed(seed)
            h0 = (torch.randn(e.shape, generator=g) * c['model_dim'] ** -0.5).to(e.device, e.dtype)
        else:
            bt = torch.arange(e.shape[0], device=e.device, dtype=torch.float32).view(-1, 1, 1) * 0.037719
            h0 = hashed_normal_state(e.detach(), 0, bt, math.sqrt(2.0 / c['model_dim']))
        return h0.detach(), dict(e=e, freqs=freqs, mask=mask)

    def F(self, h, t, ctx):
        """One recurrence: input injection, then the core blocks (then the exit norm for Huginn-Linear). t unused."""
        h = self.input_injection(h, ctx['e'])
        for b in self.core_layers():
            h = b(h, ctx['freqs'], ctx['mask'])
        return self.exit_norm(h) if self.exit_norm is not None else h

    def F_ckpt(self, h, t, ctx):
        """= F, with each core block checkpointed (same math, less activation memory)."""
        h = self.input_injection(h, ctx['e'])
        for b in self.core_layers():
            h = checkpoint(b, h, ctx['freqs'], ctx['mask'], use_reentrant=False)
        return self.exit_norm(h) if self.exit_norm is not None else h

    def step(self, h, t, ctx, a=1.0, ckpt=False):
        """Blended recurrence h <- (1 - a) h + a F(h); a = 1 is the model's own loop (returned exactly)."""
        y = self.F_ckpt(h, t, ctx) if ckpt else self.F(h, t, ctx)
        return y if a == 1.0 else (1.0 - a) * h + a * y

    def readout(self, h, ctx):
        for b in self.coda_layers():
            h = b(h, ctx['freqs'], ctx['mask'])
        return self.output(h)

    def forward(self, ids, R=None, docmask=True, a=1.0):
        """Full forward, all R recurrences (default: the model's loop_times), eval semantics; float32 logits."""
        h, ctx = self.prepare(ids, docmask)
        for t in range(self.R if R is None else R):
            h = self.step(h, t, ctx, a)
        return self.readout(h, ctx)


def ce(logits, ids):
    """Mean next-token cross-entropy over positions 0..L-2 (as loop_grad_probe.ce)."""
    return Fn.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))


# --------------------------------------------------------------------------------------------------- loading
def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_artifact(wdir):
    """Every file listed in artifact_manifest.json must match its size and SHA-256 (their loader's check, minus
    the closed-file-set rule). Returns the manifest."""
    man = json.load(open(os.path.join(wdir, 'artifact_manifest.json')))
    for f in man['files']:
        p = os.path.join(wdir, f['path'])
        if os.path.getsize(p) != f['size_bytes'] or _sha256(p) != f['sha256']:
            raise ValueError('artifact file does not match its manifest: %s' % f['path'])
    return man


def load_model(wdir, dtype=torch.float32, device='cpu', verify=True):
    """Build from <wdir>/config.json and load the BF16 Safetensors shards (strict); parameters cast to `dtype`.
    Returns (model, info)."""
    from safetensors.torch import load_file
    man = verify_artifact(wdir) if verify else json.load(open(os.path.join(wdir, 'artifact_manifest.json')))
    cfg = json.load(open(os.path.join(wdir, 'config.json')))['model']
    model = IFMLooped(cfg)
    index = json.load(open(os.path.join(wdir, 'model.safetensors.index.json')))['weight_map']
    sd = {}
    for shard in sorted(set(index.values())):
        sd.update(load_file(os.path.join(wdir, shard)))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise ValueError('state dict mismatch: missing %s unexpected %s' % (missing[:8], unexpected[:8]))
    model = model.to(device=device, dtype=dtype).eval()
    info = dict(checkpoint_id=man['source'].get('checkpoint_id'), table_row=man['source'].get('table_row'),
                manifest_sha256=man.get('manifest_sha256'), n_params=sum(p.numel() for p in model.parameters()),
                weight_dtype=str(next(iter(sd.values())).dtype), cfg=cfg)
    return model, info


class Tokenizer:
    """Their HuggingFace tokenizer wrapper: encode = tokenizer(text, add_special_tokens=False) with an optional BOS
    in front and EOS at the end (training documents use both)."""

    def __init__(self, wdir):
        from tokenizers import Tokenizer as T
        tdir = os.path.join(wdir, 'tokenizer')
        self.tok = T.from_file(os.path.join(tdir, 'tokenizer.json'))
        sp = json.load(open(os.path.join(tdir, 'special_tokens_map.json')))
        self.bos_id, self.eos_id = self.tok.token_to_id(sp['bos_token']), self.tok.token_to_id(sp['eos_token'])
        self.vocab_size = self.tok.get_vocab_size()

    def encode(self, text, bos=True, eos=False):
        ids = self.tok.encode(text, add_special_tokens=False).ids
        return ([self.bos_id] if bos else []) + ids + ([self.eos_id] if eos else [])

    def decode(self, ids):
        return self.tok.decode(list(ids), skip_special_tokens=False)


def tiny_config(**over):
    c = dict(arch='huginn', huginn_depth_control=True, num_layers=4, model_dim=64, num_heads=4, num_kv_heads=2,
             head_dim=16, ffn_hidden_dim=128, vocab_size=1000, swiglu=True, apply_rmsnorm=True, rmsnorm_eps=1e-6,
             rope_base=1_000_000.0, rope_head_dim=16, loop_times=5, huginn_prelude_layers=1, huginn_recurrent_layers=2,
             huginn_coda_layers=1, huginn_state_init='normal', huginn_input_injection='diagonal',
             huginn_prelude_norm='none', huginn_recurrent_exit_norm='none', huginn_prelude_orthogonal=False)
    c.update(over)
    return c


def tiny_model(seed=0, dtype=torch.float32, device='cpu', **over):
    """Random-weight model of the same architecture (pipeline tests only); weights large enough that the loop's
    Jacobian is far from the identity."""
    torch.manual_seed(seed)
    m = IFMLooped(tiny_config(**over))
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith('norm.weight') or n.endswith('norm_module.weight'):
                p.normal_(0.0, 0.1)
            elif n.endswith('A_log'):
                p.normal_(0.0, 0.3)
            elif n.endswith('dt_bias'):
                p.normal_(0.5, 0.3)
            elif n.endswith('input_adapter.weight'):
                p.copy_(torch.eye(p.shape[0]) + 0.1 * torch.randn_like(p))
            elif n == 'embed.weight':
                p.normal_(0.0, 1.0)
            elif p.dim() == 2:
                p.normal_(0.0, p.shape[1] ** -0.5)
            else:
                p.normal_(0.0, 0.1)
    return m.to(device=device, dtype=dtype).eval()


# ------------------------------------------------------------------------------------------- CPU validation
@torch.no_grad()
def _loss_by_depth(m, ids, depths, docmask=True):
    """Losses after R recurrences for every R in depths, from one trajectory."""
    h, ctx = m.prepare(ids, docmask)
    out, t = {}, 0
    for R in sorted(depths):
        while t < R:
            h = m.F(h, t, ctx); t += 1
        out[R] = float(ce(m.readout(h, ctx), ids))
    return out


@torch.no_grad()
def _greedy(m, tok, prompt, n=40, R=5):
    ids = tok.encode(prompt, bos=True)
    for _ in range(n):
        logits = m.forward(torch.tensor([ids]), R=R)
        nxt = int(logits[0, -1].argmax())
        ids.append(nxt)
        if nxt == tok.eos_id:
            break
    return tok.decode(ids[1:])


def validate(a):
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    rep, t0 = {}, time.time()
    m, info = load_model(a.weights, verify=True)
    tok = Tokenizer(a.weights)
    rep['info'] = {k: v for k, v in info.items() if k != 'cfg'}
    rep['groups'] = dict(loop=sum(p.numel() for p in m.loop_parameters()),
                         prelude=sum(p.numel() for b in m.prelude_layers() for p in b.parameters()),
                         coda=sum(p.numel() for b in m.coda_layers() for p in b.parameters()),
                         embed=m.embed.weight.numel(), head=m.output.output.weight.numel())
    rep['tokenizer'] = dict(bos=tok.bos_id, eos=tok.eos_id, vocab=tok.vocab_size)
    print('loaded %s (%s): %d params %s | tokenizer %s (%.0fs)' % (info['checkpoint_id'], info['table_row'],
          info['n_params'], rep['groups'], rep['tokenizer'], time.time() - t0), flush=True)
    depths = [int(x) for x in a.depths.split(',')]
    if a.wikitext:
        import pandas as pd
        lines = list(pd.read_parquet(a.wikitext)['text'])
        wt = tok.encode('\n'.join(lines), bos=True, eos=True)      # their eval_data: test lines joined by newlines
        rep['wikitext'] = {}
        for L in [int(x) for x in a.wiki_len.split(',')]:
            ids = torch.tensor([wt[:L]])
            ts = time.time()
            res = _loss_by_depth(m, ids, depths if L == min(int(x) for x in a.wiki_len.split(',')) else [5])
            rep['wikitext'][L] = res
            print('wikitext first %d tokens: CE by depth %s (PPL at R=5: %.2f) (%.0fs)' % (
                L, {k: round(v, 4) for k, v in res.items()}, math.exp(res[5]), time.time() - ts), flush=True)
        ids = torch.tensor([wt[:512]])
        mb = m.to(torch.bfloat16)
        lb = float(ce(mb(ids, R=5), ids))
        m = mb.float()
        rep['wikitext_bf16_R5_512'] = lb
        print('bf16 forward, same 512 tokens, R=5: CE %.4f (fp32 %.4f)' % (lb, rep['wikitext'][512][5]), flush=True)
    if a.gsm8k:
        import pandas as pd
        df = pd.read_parquet(a.gsm8k)
        ex = [tok.encode(q + '\n' + s, bos=True, eos=True) for q, s in zip(df['question'][-473:], df['answer'][-473:])]
        flat = [t for e in ex for t in e]
        ids = torch.tensor([flat[:512]])
        r1 = _loss_by_depth(m, ids, [1, 5, 10], docmask=True)
        r0 = _loss_by_depth(m, ids, [5], docmask=False)
        rep['gsm8k_val_chunk0'] = dict(docmask=r1, no_docmask_R5=r0[5], n_bos=int((ids == tok.bos_id).sum()))
        print('GSM8K val chunk 0 (512 tokens, %d documents): CE doc-masked %s; plain causal R=5 %.4f' % (
            rep['gsm8k_val_chunk0']['n_bos'], {k: round(v, 4) for k, v in r1.items()}, r0[5]), flush=True)
    rep['greedy'] = {}
    for p in a.prompts.split('|'):
        ts = time.time()
        rep['greedy'][p] = _greedy(m, tok, p, n=a.new_tokens)
        print('greedy R=5 (%.0fs): %r' % (time.time() - ts, rep['greedy'][p]), flush=True)
    rep['seconds'] = round(time.time() - t0)
    if a.out:
        json.dump(rep, open(a.out, 'w'), indent=1)
    return rep


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True); ap.add_argument('--wikitext'); ap.add_argument('--gsm8k')
    ap.add_argument('--out'); ap.add_argument('--depths', default='1,2,3,4,5,6,7,8,10,12,16,24,32,64')
    ap.add_argument('--wiki_len', default='512,2048'); ap.add_argument('--new_tokens', type=int, default=40)
    ap.add_argument('--prompts', default='The capital of France is|Once upon a time, there was a little girl named|'
                    'Water boils at a temperature of')
    validate(ap.parse_args())
