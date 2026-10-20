"""Step 1 of the contractivity-vs-truncation idea (docs/gate0_contractivity_truncation_result_2026-09-30.md, NARROW):
on OPEN, TRAINED looped LMs, measure (a) how contractive the loop map is and (b) how far truncated gradients through
the loop are from the full gradient -- no training.

Models: Ouro-1.4B (ByteDance/Ouro-1.4B; 24-layer stack looped 4x, no input injection) and Huginn-3.5B
(tomg-group-umd/huginn-0125; 4-layer core looped R times with linear input injection; trained with backprop through
the last ~8 of ~32 loops). The loop is driven directly: the pre-loop part (embeddings / prelude) is computed once
without grad; state h_0 -> F(h, t) repeated R times -> readout (norm / coda / lm_head) -> next-token CE loss.

Per sequence:
  gradient of the loss w.r.t. a fixed subset of the SHARED loop parameters (disclosed below) with backprop through
  the last k loops (earlier loops under no_grad), k in {1,2,4,..,R}; k = R is the full gradient. Report relative error
  ||g_k - g_R|| / ||g_R|| and cosine.
  contractivity: (i) spectral norm sigma of dF/dh at a few loop steps (power iteration on J^T J, Jv by double-backward),
  (ii) trajectory stretch ||h_{t+1}-h_t|| / ||h_t-h_{t-1}||, (iii) fixed-point residual ||h_{t+1}-h_t|| / ||h_t||,
  (iv) adjoint norms ||dL/dh_t|| along the full backward pass (their per-loop decay predicts truncation bias).
Generality run (2026-09-30, docs/loop_grad_probe_generality_2026-09-30.md) adds: Ouro-2.6B and Ouro-1.4B-Thinking (same
Ouro adapter), and Parcae-1.3B (SandyResearch/parcae-1.3b; 8-layer core looped with a diagonal, spectrally constrained
injection x <- exp(-dt*A) x + dt*B e, trained with backprop through the last <=4 of ~Poisson(8) loops; needs the
parcae_lm source on PARCAE_SRC, the PyPI wheel is empty). Cosine is computed in float64 (the float32 CPU dot over
~10^8 entries inflated cosines by 0.05-0.09 in the Huginn v2 run).
Data: 16 WikiText-103 test chunks + 16 GSM8K test (question + answer), first S tokens.
Env: TINY=1 builds a tiny random-weight model of the same architecture (local pipeline test only).
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as Fn
from torch.utils.checkpoint import checkpoint

MODEL = os.environ.get('LOOP_MODEL', 'ouro')
TINY = os.environ.get('TINY') == '1'
REPO = {'ouro': 'ByteDance/Ouro-1.4B', 'huginn': 'tomg-group-umd/huginn-0125', 'ouro26': 'ByteDance/Ouro-2.6B',
        'ouro14t': 'ByteDance/Ouro-1.4B-Thinking', 'parcae': 'SandyResearch/parcae-1.3b'}[MODEL]
FAMILY = 'ouro' if MODEL.startswith('ouro') else MODEL
S = 32 if TINY else 128
N_PER_SET = 2 if TINY else 16
R_LIST = {'ouro': [4], 'huginn': [16, 32], 'parcae': [8, 16]}[FAMILY] if not TINY else ([4] if FAMILY == 'ouro' else [4, 8])
if os.environ.get('LOOP_R'):                          # v187: override, e.g. LOOP_R=64
    R_LIST = [int(r) for r in os.environ['LOOP_R'].split(',')]
POWER_ITERS = 3 if TINY else 12
BUDGET_S = float(os.environ.get('BUDGET_S', 60 if TINY else 9.5 * 3600))
OUT = os.environ.get('OUT_DIR', '/kaggle/working' if os.path.isdir('/kaggle/working') else '.')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32 if (MODEL in ('ouro', 'ouro14t', 'parcae') or TINY or DEV == 'cpu') else torch.bfloat16


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'loop_grad_probe_%s_log.txt' % MODEL), 'a').write(s + '\n')


def load():
    if FAMILY == 'parcae':
        return load_parcae()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
    if TINY:
        if FAMILY == 'ouro':
            cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers = 64, 128, 2
            cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 2, 2, 32
            cfg.layer_types = cfg.layer_types[:2]
        else:
            cfg.n_embd, cfg.n_heads, cfg.num_key_value_heads, cfg.intermediate_size = 64, 2, 2, 128
            cfg.head_dim, cfg.n_layers_in_prelude, cfg.n_layers_in_recurrent_block, cfg.n_layers_in_coda = 32, 1, 1, 1
            cfg.n_layers = 3
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(REPO, trust_remote_code=True, torch_dtype=DTYPE)
    model = model.to(DEV).to(DTYPE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


class Ouro:
    def __init__(self, m):
        self.m, self.b = m, m.model
        self.mod = sys.modules[type(self.b).__module__]
        L = self.b.config.num_hidden_layers
        self.subset = list(self.b.layers[L // 2].parameters())           # one shared middle layer (all its weights)
        self.subset_name = 'layers[%d] (all params)' % (L // 2)

    @torch.no_grad()
    def prepare(self, ids):
        b = self.b; h = b.embed_tokens(ids)
        cp = torch.arange(ids.shape[1], device=ids.device); pos = cp.unsqueeze(0)
        mask = self.mod.create_causal_mask(config=b.config, input_embeds=h, attention_mask=None, cache_position=cp,
                                           past_key_values=None, position_ids=pos)
        return h, dict(mask=mask, pos=pos, cp=cp, pe=b.rotary_emb(h, pos))

    def F(self, h, t, ctx):
        b = self.b
        for layer in b.layers[: b.config.num_hidden_layers]:
            h = layer(h, attention_mask=ctx['mask'], position_ids=ctx['pos'], past_key_value=None, use_cache=False,
                      cache_position=ctx['cp'], position_embeddings=ctx['pe'], current_ut=t)
            h = h[0] if isinstance(h, tuple) else h
        return b.norm(h)

    def readout(self, h, ctx):
        return self.m.lm_head(h)


def load_parcae():
    """parcae_lm is not on PyPI in usable form (the 0.1.0 wheel is empty); import it from a source checkout
    (github.com/sandyresearch/parcae, pinned commit in the Kaggle runner) given by PARCAE_SRC."""
    sys.path.insert(0, os.environ.get('PARCAE_SRC', 'parcae_src'))
    from huggingface_hub import hf_hub_download
    from transformers import PreTrainedTokenizerFast
    from parcae_lm.models.config import RoPESettings
    from parcae_lm.models.parcae import Parcae as ParcaeModel, ParcaeConfig
    tok = PreTrainedTokenizerFast(tokenizer_file=hf_hub_download('SandyResearch/parcae-tokenizer', 'tokenizer.json'))
    from tokenizers.processors import TemplateProcessing   # prepend '<|bos|>' (id 0), as their eval code does
    tok._tokenizer.post_processor = TemplateProcessing(single='<|bos|> $A', special_tokens=[('<|bos|>', 0)])
    cfg = json.load(open(hf_hub_download(REPO, 'config.json')))
    for key in ('_class_name', 'init'):
        cfg.pop(key, None)
    cfg['rope_settings'] = RoPESettings(**cfg['rope_settings'])
    if TINY:
        cfg.update(n_embd=64, intermediate_size=128, num_attention_heads=2, num_key_value_heads=2, n_layers_in_prelude=1,
                   n_layers_in_recurrent_block=1, n_layers_in_coda=1, recurrent_embedding_dimension=64,
                   recurrent_intermediation_embedding_dimension=128, block_size=256)
    model = ParcaeModel(ParcaeConfig(**cfg))
    if not TINY:
        sd = torch.load(hf_hub_download(REPO, 'pytorch_model.bin'), map_location='cpu', weights_only=True)
        sd = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        log('parcae load: missing %s unexpected %s' % (list(missing)[:10], list(unexpected)[:10]))
    freqs = model.freqs_cis
    model = model.to(DEV).to(DTYPE).eval()
    model.freqs_cis = freqs.to(DEV)                         # keep the rope cache in its original precision
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


class Parcae:
    """Parcae: prelude (+ln_prelude) -> e; x_0 ~ like-init (seeded); x <- core_block(adapter(x, e)) R times;
    readout = C -> coda -> ln_f -> lm_head * logit_scale. Value embeddings are indexed by the input ids."""
    def __init__(self, m):
        self.m = m
        uses_ve = str(m.config.n_layers_in_prelude) in m.value_embeds    # ve_gate exists but is unused without a ve
        core = [p for n, p in m.transformer.core_block[0].named_parameters() if uses_ve or 've_gate' not in n]
        self.subset = list(m.transformer.adapter.parameters()) + core
        self.subset_name = 'adapter (diagonal injection) + core_block[0] (all params that receive gradient)'

    def ve(self, i, ids):
        return self.m.value_embeds[str(i)](ids) if str(i) in self.m.value_embeds else None

    @torch.no_grad()
    def prepare(self, ids):
        m = self.m; freqs = m.freqs_cis[:, : ids.shape[1]]
        m._current_input_ids = ids
        e = m.transformer.wte(ids)
        if m.emb_scale != 1:
            e = e * m.emb_scale
        for i, block in enumerate(m.transformer.prelude):
            e = block(e, freqs, None, ve=self.ve(i, ids))
        if m.config.prelude_norm:
            e = m.transformer.ln_prelude(e)
        torch.manual_seed(0)
        return m.initialize_state(e), dict(e=e, freqs=freqs, ids=ids)

    def F(self, x, t, ctx):
        self.m._current_input_ids = ctx['ids']
        return self.m.core_block_forward(x, ctx['e'], ctx['freqs'], None, torch.tensor(t), torch.tensor(0))

    def readout(self, x, ctx):
        m = self.m; x = m.transformer.C(x); off = m.config.n_layers_in_prelude + m.config.n_layers_in_recurrent_block
        for i, block in enumerate(m.transformer.coda):
            x = block(x, ctx['freqs'], None, ve=self.ve(off + i, ctx['ids']))
        logits = m.lm_head(m.transformer.ln_f(x)).float() * m.config.init.logit_scale
        if m.config.logit_softcap is not None:
            logits = m.config.logit_softcap * torch.tanh(logits / m.config.logit_softcap)
        return logits

    @torch.no_grad()
    def check(self, ids):
        """Decomposition vs the model's own eval forward (eval mode: mean_recurrence loops, all without grad)."""
        h, ctx = self.prepare(ids)
        for t in range(self.m.config.mean_recurrence):
            h = self.F(h, t, ctx)
        mine = float(ce(self.readout(h, ctx), ids))
        torch.manual_seed(0)
        lab = torch.cat([ids[:, 1:], torch.full_like(ids[:, :1], -100)], 1)
        steps = torch.tensor([self.m.config.mean_recurrence, 0])
        ref = float(self.m(ids, labels=lab, num_steps_pair=steps)['loss'])
        return mine, ref


class Huginn:
    def __init__(self, m):
        self.m = m
        self.subset = list(m.transformer.adapter.parameters()) + list(m.transformer.core_block[0].parameters())
        self.subset_name = 'adapter + core_block[0] (all params)'

    @torch.no_grad()
    def prepare(self, ids):
        m = self.m; freqs = m.freqs_cis[:, : ids.shape[1]]
        e = m.transformer.wte(ids)
        if m.emb_scale != 1:
            e = e * m.emb_scale
        bi = torch.tensor(-1, dtype=torch.long)
        for block in m.transformer.prelude:
            bi += 1; e = block(e, freqs, bi, None, None)
        torch.manual_seed(0)
        return m.initialize_state(e, scale=1.0), dict(e=e, freqs=freqs, bi=bi)

    def F(self, x, t, ctx):
        return self.m.core_block_forward(x, ctx['e'], ctx['freqs'], None, None, ctx['bi'].clone(), t)[0]

    def readout(self, x, ctx):
        m = self.m; x = m.transformer.ln_f(x); bi = torch.tensor(0, dtype=torch.long)
        for block in m.transformer.coda:
            bi -= 1; x = block(x, ctx['freqs'], bi, None, None)
        return m.lm_head(m.transformer.ln_f(x))


def ce(logits, ids):
    return Fn.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))


def variant(ad, h0, ctx, ids, R, k, adj=None):
    """Loss with backprop through the last k of R loops; returns (loss, flat grad of the subset)."""
    for p in ad.subset:
        p.requires_grad_(True); p.grad = None
    h = h0
    with torch.no_grad():
        for t in range(R - k):
            h = ad.F(h, t, ctx)
    h = h.detach()
    for t in range(R - k, R):
        h = checkpoint(ad.F, h, t, ctx, use_reentrant=False)
        if adj is not None:
            h.register_hook(lambda g, t=t: adj.__setitem__(t, float(g.float().norm())))
    loss = ce(ad.readout(h, ctx), ids)
    loss.backward()
    g = torch.cat([p.grad.detach().float().flatten().cpu() for p in ad.subset])   # gathered on CPU (451M params for Huginn)
    for p in ad.subset:
        p.grad = None; p.requires_grad_(False)
    return float(loss), g


def spectral(ad, h, t, ctx):
    """sigma_max of dF/dh at state h (power iteration on J^T J; Jv via double backward). The math SDPA backend is
    forced here because the fused attention kernels have no double-backward."""
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with sdpa_kernel(SDPBackend.MATH):
        return _spectral(ad, h, t, ctx)


def _spectral(ad, h, t, ctx):
    h = h.detach().float().to(DTYPE).requires_grad_(True)
    out = ad.F(h, t, ctx)
    v = torch.randn_like(h); v = v / v.norm(); s = torch.tensor(0.)
    for _ in range(POWER_ITERS):
        u = torch.zeros_like(out, requires_grad=True)
        g = torch.autograd.grad(out, h, u, create_graph=True)[0]
        Jv = torch.autograd.grad(g, u, v, retain_graph=True)[0]
        JtJv = torch.autograd.grad(out, h, Jv, retain_graph=True)[0]
        s = JtJv.float().norm(); v = (JtJv / (s + 1e-30)).to(h.dtype)
    return float(s.sqrt())


def data(tok):
    seqs = []
    if TINY:                                          # local pipeline test: random tokens, no downloads
        g = torch.Generator().manual_seed(0)
        return [('random', torch.randint(0, 1000, (S,), generator=g)) for _ in range(2 * N_PER_SET)]
    from datasets import load_dataset
    try:
        try:
            wt = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')
        except Exception:                             # the bare id stopped resolving on the Hub (seen in v182)
            wt = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='test')
        ids = tok(' '.join(t for t in wt['text'][:4000] if t.strip()), return_tensors='pt').input_ids[0]
        seqs += [('wikitext', ids[i * S:(i + 1) * S]) for i in range(N_PER_SET)]
    except Exception as e:
        log('wikitext load failed', repr(e)[:200])
    try:
        gs = load_dataset('openai/gsm8k', 'main', split='test')
        for ex in gs:
            ids = tok(ex['question'] + '\n' + ex['answer'], return_tensors='pt').input_ids[0]
            if len(ids) >= S:
                seqs.append(('gsm8k', ids[:S]))
            if sum(1 for s, _ in seqs if s == 'gsm8k') >= N_PER_SET:
                break
    except Exception as e:
        log('gsm8k load failed', repr(e)[:200])
    if not seqs:                                      # offline fallback (local test only)
        g = torch.Generator().manual_seed(0)
        seqs = [('random', torch.randint(0, tok.vocab_size, (S,), generator=g)) for _ in range(2 * N_PER_SET)]
    return seqs


def main():
    t0 = time.time()
    tok, model = load()
    ad = {'ouro': Ouro, 'huginn': Huginn, 'parcae': Parcae}[FAMILY](model)
    n_sub = sum(p.numel() for p in ad.subset)
    rep = dict(model=REPO, tiny=TINY, dtype=str(DTYPE), device=DEV, S=S, R_list=R_LIST, subset=ad.subset_name,
               subset_params=n_sub, gpu=(torch.cuda.get_device_name(0) if DEV == 'cuda' else 'cpu'), seqs=[])
    log('model %s subset %s (%d params) dtype %s' % (REPO, ad.subset_name, n_sub, DTYPE))
    seqs = data(tok)
    rep['data'] = {k: sum(1 for s, _ in seqs if s == k) for k in set(s for s, _ in seqs)}
    if FAMILY == 'parcae':
        mine, ref = ad.check(seqs[0][1].unsqueeze(0).to(DEV))
        rep['decomposition_check'] = dict(mine=mine, model_forward=ref)
        log('parcae decomposition check: loss mine %.5f vs model forward %.5f' % (mine, ref))
        if not abs(mine - ref) <= 1e-3 * abs(ref):
            json.dump(rep, open(os.path.join(OUT, 'loop_grad_probe_%s.json' % MODEL), 'w'), indent=1)
            raise SystemExit('decomposition check FAILED: the adapter does not reproduce the model forward')
    for si, (src, ids) in enumerate(seqs):
        if time.time() - t0 > BUDGET_S:
            log('budget reached'); break
        ids = ids.unsqueeze(0).to(DEV)
        h0, ctx = ad.prepare(ids)
        for R in R_LIST:
            ts = time.time()
            # trajectory (no grad)
            traj = [h0]
            with torch.no_grad():
                for t in range(R):
                    traj.append(ad.F(traj[-1], t, ctx))
            d = [float((traj[t + 1] - traj[t]).float().norm()) for t in range(R)]
            resid = [d[t] / max(float(traj[t].float().norm()), 1e-30) for t in range(R)]
            stretch = [d[t] / max(d[t - 1], 1e-30) for t in range(1, R)]
            sig = {}
            for t in sorted({R // 2, R - 1}):
                sig[t] = spectral(ad, traj[t], t, ctx)
            adj = {}
            lossR, gR = variant(ad, h0, ctx, ids, R, R, adj=adj)
            ks = sorted({k for k in (1, 2, 4, 8, 16, 32) if k < R})
            trunc = {}
            for k in ks:
                _, gk = variant(ad, h0, ctx, ids, R, k)
                # float64 sums (the float32 CPU dot inflated cosines), in chunks so two probes fit in one kernel's RAM
                dd = kk = rr = kr = 0.0
                for a in range(0, gk.numel(), 1 << 26):
                    x, y = gk[a:a + (1 << 26)].double(), gR[a:a + (1 << 26)].double()
                    dd += float(((x - y) ** 2).sum()); kk += float((x * x).sum())
                    rr += float((y * y).sum()); kr += float((x * y).sum())
                trunc[k] = dict(rel_err=math.sqrt(dd) / (math.sqrt(rr) + 1e-300),
                                cos=kr / (math.sqrt(kk) * math.sqrt(rr) + 1e-300))
                del gk
            adj_norms = [adj.get(t) for t in range(R)]
            rec = dict(i=si, src=src, R=R, loss=lossR, resid=resid, stretch=stretch, sigma={str(k): v for k, v in sig.items()},
                       adjoint=adj_norms, trunc={str(k): v for k, v in trunc.items()}, s=round(time.time() - ts, 1))
            rep['seqs'].append(rec)
            log('seq %2d %-8s R=%-2d loss %.3f sigma %s resid_end %.2e stretch_med %.3f | %s  (%.0fs)' % (
                si, src, R, lossR, {k: round(v, 3) for k, v in sig.items()}, resid[-1], float(np.median(stretch)) if stretch else float('nan'),
                ' '.join('k%d:err %.3f cos %.3f' % (k, v['rel_err'], v['cos']) for k, v in trunc.items()), time.time() - ts))
            json.dump(rep, open(os.path.join(OUT, 'loop_grad_probe_%s.json' % MODEL), 'w'), indent=1)
        del h0, ctx
        if DEV == 'cuda':
            torch.cuda.empty_cache()
    rep['wall_s'] = round(time.time() - t0)
    json.dump(rep, open(os.path.join(OUT, 'loop_grad_probe_%s.json' % MODEL), 'w'), indent=1)
    log('done', rep['wall_s'], 's')


if __name__ == '__main__':
    main()
