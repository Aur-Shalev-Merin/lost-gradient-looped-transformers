"""v198: how fast the loop fades in a Part II looped LM (Huang et al. 2026, arXiv 2610.06833), frozen, no training.
Feasibility check for a third-model cost test. Port and assumptions: ifm_looped.py, docs/v198_ifm_port_notes.md.

Model: IFM/LoopedLM-P2-huginn-s-fixed-pln5 (default; any diagonal-injection Part II checkpoint loads). NOT the
requested IFM/LoopedLM-P2-d4-s, which is the paper's non-looped "Untied 4" baseline (no loop to measure).

Protocol (Table 1 / v164 measure() / v197):
  - text: GSM8K train, last 473 examples (= v162's val split), question + '\\n' + answer, each encoded as the model was
    trained (BOS + text + EOS), packed in order into 512-token chunks; the first 16 chunks. Attention stays within
    documents (V198_DOCMASK=1, their training's multi_segments; 0 = plain causal over the chunk).
  - per sequence, batch 1, fp32 (TF32 off), all parameters: full gradient g_R and truncated gradients g_k (backprop
    through the last k loops; earlier loops under no_grad; the prelude and embedding get gradient only through the
    kept loops' injection, as in their training). Initial state: their eval-mode deterministic N(0, 1/d) state.
  - loop counts R = 5 (the trained mean depth and their eval depth) and 2R = 10; blending h <- (1 - a) h + a F(h),
    a = 1, 0.5, 0.3. a = 1: every k = 0 .. R-1 (k = 0: no loop credit, for the loop part |g_R - g_0|); blended:
    k = 0, 1, 2, 4 (k < R).
  - per cell: err_k = |g_k - g_R| / |g_R|, |g_k| / |g_R|, cos(g_k, g_R), and the step |g_k - g_k'| from the previous
    k' (= the exact per-loop contribution c_k' when k = k' + 1), over all parameters, the loop's parameters (input
    injection + both recurrent blocks) and Table 1's starred part (injection + first recurrent block); backward signal
    lam[t] = dL/d(output of loop t): norms adj[t], rho_i = exp(-slope) of a line fit to log adj[t] (as v164), and
    the float64 Gram of the lam[t] (v196: the summed-signal estimate of the lost share).
  - noise floor: the first sequence of each process repeats every full pass.
  - direction after Adam (v172): per arm (a, k), k in {R, 1, 2, 4} at R = 5, B batches of 8 x 512-token GSM8K
    training chunks (fixed shuffle, rng 172); each batch gradient clipped to global norm <= 1; gbar = mean clipped
    gradient, s = sqrt(mean g^2) + 1e-8, u = gbar / s; cos(u_k, u_R), and the raw lost gradient, size ratio and cosine
    of gbar_k vs gbar_R (float64 dots).

  python v198_ifm_fade.py --weights DIR --selftest --tag g0                  checks (GPU on Kaggle, CPU with TINY=1)
  python v198_ifm_fade.py --weights DIR --refcheck XLLM_DIR                  port vs their xLLM code (Kaggle only)
  python v198_ifm_fade.py --weights DIR --seqs 0:8 --adam 1.0,0.5 --tag g0   measurement (one process per T4)
  python v198_ifm_fade.py --analyze DIR
Env: TINY=1 (tiny random model, CPU pipeline test), OUT_DIR, V198_B (Adam batches, 24), V198_R (5), V198_RLIST,
V198_DOCMASK (1), V198_ACC (Adam accumulators: cuda|cpu), GSM8K_PARQUET (local GSM8K train parquet instead of
datasets), V198_REFCHECK_MOCK=1 (refcheck against our own model through a fake xLLM interface: plumbing test).
"""
import argparse, glob, hashlib, json, math, os, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # our modules, also under python -I
import numpy as np
import torch

import ifm_looped as M

TINY = os.environ.get('TINY') == '1'
OUT = os.environ.get('OUT_DIR', '/kaggle/working' if os.path.isdir('/kaggle/working') else '.')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
R0 = int(os.environ.get('V198_R', 5))
R_LIST = tuple(int(x) for x in os.environ.get('V198_RLIST', '%d,%d' % (R0, 2 * R0)).split(','))
ALPHAS = tuple(float(x) for x in os.environ.get('V198_ALPHAS', '1.0,0.5,0.3').split(','))
FREEZE_EMBED = os.environ.get('V198_FREEZE_EMBED') == '1'    # v200 predictor inputs: the embedding does not train
KS_BLEND = (1, 2, 4)
KS_ADAM = (1, 2, 4)
B = int(os.environ.get('V198_B', 2 if TINY else 24))
MICRO = 2 if TINY else 8
SEQ = 32 if TINY else 512
N_VAL = 473
N_SEQ = 4 if TINY else 16
DOCMASK = os.environ.get('V198_DOCMASK', '1') == '1'
CHUNK = 8_000_000                                   # elements per float64 reduction chunk


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v198_log.txt'), 'a').write(s + '\n')


def driver_sha():
    h = hashlib.sha256()
    for f in (__file__, M.__file__):
        h.update(open(os.path.abspath(f), 'rb').read().replace(b'\r\n', b'\n'))
    return h.hexdigest()[:12]


# ------------------------------------------------------------------------------------------------------- loading
def load(weights):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    if TINY:
        m = M.tiny_model(seed=0, device=DEV)
        info = dict(checkpoint_id='tiny-random', table_row=None, manifest_sha256=None,
                    n_params=sum(p.numel() for p in m.parameters()), cfg=m.cfg)
        tok = None
    else:
        m, info = M.load_model(weights, dtype=torch.float32, device=DEV)
        tok = M.Tokenizer(weights)
        assert tok.bos_id == m.bos_id, (tok.bos_id, m.bos_id)
    for p in m.parameters():
        p.requires_grad_(False)
    return m, info, tok


def gsm8k_train():
    path = os.environ.get('GSM8K_PARQUET')
    if path:
        import pandas as pd
        df = pd.read_parquet(path)
        return list(zip(df['question'], df['answer']))
    from datasets import load_dataset
    return [(ex['question'], ex['answer']) for ex in load_dataset('openai/gsm8k', 'main', split='train')]


def tiny_docs(n, seed):
    rng = np.random.default_rng(seed)
    return [[0] + [int(x) for x in rng.integers(2, 1000, size=int(rng.integers(4, 20)))] + [1] for _ in range(n)]


def val_chunks(tok):
    """The last 473 GSM8K train examples (v162's val split), BOS + q + '\\n' + a + EOS, packed in order."""
    docs = tiny_docs(40, 0) if TINY else [tok.encode(q + '\n' + a, bos=True, eos=True) for q, a in gsm8k_train()[-N_VAL:]]
    flat = [t for d in docs for t in d]
    n = len(flat) // SEQ
    return torch.tensor(flat[:n * SEQ], dtype=torch.long).view(n, SEQ)


def train_chunks(tok):
    """v172 chunks(): GSM8K train in a fixed shuffled order (rng 172), packed, B * MICRO chunks of SEQ tokens."""
    ex = tiny_docs(400, 1) if TINY else gsm8k_train()
    flat = []
    for i in np.random.default_rng(172).permutation(len(ex)):
        flat += ex[i] if TINY else tok.encode(ex[i][0] + '\n' + ex[i][1], bos=True, eos=True)
        if len(flat) >= B * MICRO * SEQ:
            break
    return torch.tensor(flat[:B * MICRO * SEQ], dtype=torch.long).view(B * MICRO, SEQ)


# ------------------------------------------------------------------------------------------------- measurement
class Meas:
    """Gradients of all parameters with backprop through the last k of R (blended) loops, and float64 comparisons."""

    def __init__(self, m):
        self.m = m
        self.params = list(m.parameters())
        for p in self.params:
            p.requires_grad_(True)
        if FREEZE_EMBED:                                # its gradient is then None: zero in every comparison and in
            for p in m.embed.parameters():              # the Adam measurement's clipping, as in v200's training
                p.requires_grad_(False)
        self.sizes = [p.numel() for p in self.params]
        self.offs = np.cumsum([0] + self.sizes)
        self.ntot = int(self.offs[-1])
        loop = {id(p) for p in m.loop_parameters()}
        part = {id(p) for p in m.input_injection.parameters()} | {id(p) for p in m.core_layers()[0].parameters()}
        readout = {id(p) for b in m.coda_layers() for p in b.parameters()} | {id(p) for p in m.output.parameters()}
        self.member = {'all': lambda p: True, 'loop': lambda p: id(p) in loop, 'part': lambda p: id(p) in part,
                       'readout': lambda p: id(p) in readout}
        self.groups = ('all', 'loop', 'part')

    def zero(self):
        for p in self.params:
            p.grad = None

    def loss(self, ids, R, k, a, lam=None, ckpt=True):
        h, ctx = self.m.prepare(ids, DOCMASK)
        with torch.no_grad():
            for t in range(R - k):
                h = self.m.step(h, t, ctx, a)
        h = h.detach()
        for t in range(R - k, R):
            h = self.m.step(h, t, ctx, a, ckpt=ckpt)
            if lam is not None:
                h.register_hook(lambda g, t=t: lam.__setitem__(t, g.detach().double()))
        return M.ce(self.m.readout(h, ctx), ids)

    def backward(self, ids, R, k, a, lam=None, ckpt=True):
        self.zero()
        loss = self.loss(ids, R, k, a, lam, ckpt)
        loss.backward()
        return float(loss)

    def buffer(self):
        """Full-gradient copy on the CPU (pinned when the host allows it, for faster transfers)."""
        if DEV == 'cuda':
            try:
                return torch.empty(self.ntot, dtype=torch.float32, pin_memory=True)
            except RuntimeError:
                pass
        return torch.empty(self.ntot, dtype=torch.float32)

    def to_buf(self, buf):
        for p, o, n in zip(self.params, self.offs[:-1], self.sizes):
            if p.grad is None:
                buf[o:o + n].zero_()
            else:
                buf[o:o + n].copy_(p.grad.detach().float().flatten())

    def compare(self, gR, gP=None, groups=None):
        """Per group: err = |g - gR| / |gR|, ratio = |g| / |gR|, cos, nR = |gR|, step = |g - gP| (float64)."""
        groups = groups or self.groups
        acc = {g: np.zeros(5) for g in groups}
        for p, o, n in zip(self.params, self.offs[:-1], self.sizes):
            mine = [g for g in groups if self.member[g](p)]
            if not mine:
                continue
            x_all = p.grad.detach().float().flatten() if p.grad is not None else None
            part = torch.zeros(5, dtype=torch.float64, device=DEV)
            for s in range(0, n, CHUNK):
                e = min(n, s + CHUNK)
                r = gR[o + s:o + e].to(DEV, non_blocking=True).double()
                x = x_all[s:e].double() if x_all is not None else torch.zeros_like(r)
                part[0] += ((x - r) ** 2).sum(); part[1] += (x * x).sum()
                part[2] += (x * r).sum(); part[3] += (r * r).sum()
                if gP is not None:
                    part[4] += ((x - gP[o + s:o + e].to(DEV, non_blocking=True).double()) ** 2).sum()
            part = part.cpu().numpy()
            for g in mine:
                acc[g] += part
        out = {}
        for g, a in acc.items():
            out[g] = dict(err=float(np.sqrt(a[0] / (a[3] + 1e-300))), ratio=float(np.sqrt(a[1] / (a[3] + 1e-300))),
                          cos=float(a[2] / np.sqrt(a[1] * a[3] + 1e-300)), nR=float(np.sqrt(a[3])))
            if gP is not None:
                out[g]['step'] = float(np.sqrt(a[4]))
        return out

    def gdot(self, x, y=None, diff=False):
        """Per-group float64 sum of x*y (y on CPU or DEV; y=None: x*x); diff=True: sum (x - y)^2."""
        acc = {g: 0.0 for g in self.groups}
        for p, o, n in zip(self.params, self.offs[:-1], self.sizes):
            v = torch.zeros((), dtype=torch.float64, device=DEV)
            for s in range(o, o + n, CHUNK):
                e = min(o + n, s + CHUNK)
                a = x[s:e].to(DEV, non_blocking=True).double()
                b = a if y is None else y[s:e].to(DEV, non_blocking=True).double()
                v += ((a - b) ** 2).sum() if diff else (a * b).sum()
            v = float(v)
            for g in self.groups:
                if self.member[g](p):
                    acc[g] += v
        return acc


def measure(ms, data, seqs, out, path):
    gR, gP = ms.buffer(), ms.buffer()
    lo, hi = seqs
    for i in range(lo, min(hi, data.shape[0])):
        ids = data[i:i + 1].to(DEV)
        for R in R_LIST:
            for a in ALPHAS:
                t0, lam = time.time(), {}
                loss = ms.backward(ids, R, R, a, lam=lam)
                ms.to_buf(gR)
                Lm = torch.stack([lam[t].flatten() for t in range(R)])
                G = (Lm @ Lm.T).cpu().numpy()
                del Lm, lam
                adj = np.sqrt(np.diag(G))
                rho = float(np.exp(-np.polyfit(np.arange(R), np.log(adj + 1e-30), 1)[0]))
                row = dict(i=i, R=R, a=a, loss=loss, adj=adj.tolist(), rho=rho, gram=G.tolist(), k={})
                if i == lo:                              # noise floor: the identical full pass, run again
                    ms.backward(ids, R, R, a)
                    out['checks']['repeat_full_R%d_a%g' % (R, a)] = ms.compare(gR)['all']['err']
                ks = list(range(R)) if a == 1.0 else [0] + [k for k in KS_BLEND if k < R]
                prev = None
                for k in ks:
                    ms.backward(ids, R, k, a)
                    res = ms.compare(gR, gP if prev is not None else None)
                    row['nR'] = {g: v.pop('nR') for g, v in res.items()}
                    if prev is not None:
                        res['prev_k'] = prev
                    row['k'][str(k)] = res
                    ms.to_buf(gP)
                    prev = k
                row['seconds'] = round(time.time() - t0, 1)
                out['rows'].append(row)
                show = [k for k in (1, 2, 4, 8) if str(k) in row['k']]
                log('seq %2d R=%-2d a=%.1f: loss %.3f rho %.3f adj %s | err all %s | loop %s (%.0fs)' % (
                    i, R, a, loss, rho, ' '.join('%.3g' % x for x in adj),
                    ' / '.join('k%d %.3f' % (k, row['k'][str(k)]['all']['err']) for k in show),
                    ' / '.join('%.3f' % row['k'][str(k)]['loop']['err'] for k in show), row['seconds']))
                json.dump(out, open(path, 'w'))
    ms.zero()


def adam_direction(ms, data, alphas, out, path, R=None):
    """v172: per arm, B clipped batch gradients (8 x SEQ tokens each); u = mean / (rms + 1e-8); dots vs the full arm."""
    R = R or R0
    arms = [R] + [k for k in KS_ADAM if k < R]
    acc_dev = os.environ.get('V198_ACC') or (DEV if ms.ntot < 5e8 else 'cpu')
    for a in alphas:
        ref = None
        for k in arms:
            t0 = time.time()
            s1 = torch.zeros(ms.ntot, device=acc_dev)
            s2 = torch.zeros(ms.ntot, device=acc_dev)
            norms = []
            for b in range(B):
                ms.zero()
                for mi in range(MICRO):
                    ids = data[b * MICRO + mi:b * MICRO + mi + 1].to(DEV)
                    (ms.loss(ids, R, k, a) / MICRO).backward()
                nrm = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in ms.params if p.grad is not None)))
                scale = min(1.0, 1.0 / (nrm + 1e-6))
                norms.append(nrm)
                for p, o, n in zip(ms.params, ms.offs[:-1], ms.sizes):
                    if p.grad is not None:
                        g = p.grad.detach().float().flatten().to(acc_dev) * scale
                        s1[o:o + n].add_(g)
                        s2[o:o + n].addcmul_(g, g)
            s1.div_(B)
            s2.div_(B).sqrt_().add_(1e-8)
            u = s1 / s2
            del s2
            res = dict(a=a, k=k, R=R, B=B, micro=MICRO, norm_median=float(np.median(norms)), norm_max=max(norms),
                       clipped=int(sum(x > 1.0 for x in norms)))
            if k == R:
                ref = (s1.cpu(), u.cpu())
                res.update(gg=ms.gdot(s1), uu=ms.gdot(u))
            else:
                gR_, uR_ = ref
                res.update(gg=ms.gdot(s1), uu=ms.gdot(u), gRg=ms.gdot(s1, gR_), uRu=ms.gdot(u, uR_),
                           d2=ms.gdot(s1, gR_, diff=True), gRu=ms.gdot(u, gR_))
                base = out['adam']['a%g_R' % a]
                res['cos_u'] = {g: res['uRu'][g] / math.sqrt(res['uu'][g] * base['uu'][g] + 1e-300) for g in ms.groups}
                res['cos_g'] = {g: res['gRg'][g] / math.sqrt(res['gg'][g] * base['gg'][g] + 1e-300) for g in ms.groups}
                res['err_g'] = {g: math.sqrt(res['d2'][g] / (base['gg'][g] + 1e-300)) for g in ms.groups}
                res['ratio_g'] = {g: math.sqrt(res['gg'][g] / (base['gg'][g] + 1e-300)) for g in ms.groups}
            out['adam']['a%g_%s' % (a, 'R' if k == R else k)] = res
            del s1, u
            log('adam a=%.1f k=%s: batch norms median %.3f max %.3f (%d of %d clipped) %s (%.0fs)' % (
                a, 'R' if k == R else k, res['norm_median'], res['norm_max'], res['clipped'], B,
                '' if k == R else '| cos_u %.4f cos_g %.4f err_g %.4f ratio %.4f' % (
                    res['cos_u']['all'], res['cos_g']['all'], res['err_g']['all'], res['ratio_g']['all']),
                time.time() - t0))
            json.dump(out, open(path, 'w'))
    ms.zero()


# ---------------------------------------------------------------------------------------------------- selftest
def selftest(ms, data, tag):
    """k = R through the measurement path equals plain backprop of the whole forward; checkpointed = plain; the
    truncation leaves the loss and the readout gradient unchanged; k = 0 gives the loop no gradient; the doc mask
    is active. Tolerances: 1e-5 (CPU) / 1e-4 (GPU: attention backward kernels are not bitwise deterministic)."""
    rep, tol = dict(driver_sha=driver_sha(), device=DEV, docmask=DOCMASK), (1e-5 if DEV == 'cpu' else 1e-4)
    ids = data[0:1].to(DEV)
    R, buf = R0, ms.buffer()
    for a in (1.0, 0.5):
        ms.backward(ids, R, R, a, ckpt=True); ms.to_buf(buf)
        ms.zero(); M.ce(ms.m(ids, R=R, docmask=DOCMASK, a=a), ids).backward()
        rep['kR_vs_plain_a%g' % a] = ms.compare(buf)['all']['err']
    for a, k in ((1.0, 2), (0.5, 1), (1.0, 0)):
        ms.backward(ids, R, k, a, ckpt=True); ms.to_buf(buf); ms.backward(ids, R, k, a, ckpt=False)
        rep['ckpt_vs_plain_a%g_k%d' % (a, k)] = ms.compare(buf)['all']['err']
    lR = ms.backward(ids, R, R, 1.0); ms.to_buf(buf)
    for k in (2, 0):
        lk = ms.backward(ids, R, k, 1.0)
        rep['loss_diff_k%d' % k] = abs(lk - lR)
        rep['readout_grad_k%d_vs_kR' % k] = ms.compare(buf, groups=('readout',))['readout']['err']
    rep['loop_grad_norm_k0'] = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in ms.m.loop_parameters()
                                                    if p.grad is not None)) if any(p.grad is not None for p in
                                                    ms.m.loop_parameters()) else 0.0)
    with torch.no_grad():
        n_doc = int((ids[0, 1:] == ms.m.bos_id).sum())
        l_doc = float(M.ce(ms.m(ids, R=R, docmask=True), ids)); l_cau = float(M.ce(ms.m(ids, R=R, docmask=False), ids))
        one = ids[:, :max(2, int(torch.nonzero(ids[0, 1:] == ms.m.bos_id)[0]) + 1)] if n_doc else ids
        same = float((ms.m(one, R=R, docmask=True) - ms.m(one, R=R, docmask=False)).abs().max())
    rep.update(doc_boundaries=n_doc, loss_docmask=l_doc, loss_causal=l_cau, single_doc_mask_vs_causal=same)
    ms.zero()
    fails = [key for key in rep if (key.startswith(('kR_vs', 'ckpt_vs', 'readout_grad')) and not rep[key] < tol)
             or (key.startswith('loss_diff') and not rep[key] < 1e-5 * max(1.0, lR))]
    if rep['loop_grad_norm_k0'] != 0.0:
        fails.append('loop_grad_norm_k0')
    if n_doc and not abs(l_doc - l_cau) > 1e-6:
        fails.append('docmask_inactive')
    if same != 0.0:
        fails.append('single_doc_mask_vs_causal')
    rep['fails'] = fails
    rep['status'] = 'PASS' if not fails else 'FAIL'
    json.dump(rep, open(os.path.join(OUT, 'v198_selftest_%s.json' % tag), 'w'), indent=1)
    log('selftest %s: %s' % (rep['status'], {k: (round(v, 9) if isinstance(v, float) else v) for k, v in rep.items()
                                             if k not in ('driver_sha',)}))
    return rep['status'] == 'PASS'


# ------------------------------------------------------------------------------------------- reference check
class _Stub:
    """Placeholder for something the reference shim does not provide; fails loudly if used."""

    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        if attr.startswith('__'):
            raise AttributeError(attr)
        return _Stub(self._name + '.' + attr)

    def __call__(self, *a, **k):
        raise RuntimeError('%s is not available in the v198 reference shim' % self._name)


def _install_ref_shims():
    """Their model code imports a compiled CUDA extension (xllm_extension, built for sm90 with FlashAttention 3) and
    optional kernels at import time. For the reference forward on a T4: xllm_extension.ops provides their group
    RMSNorm (forward and backward) in PyTorch, transcribed from xllm/csrc/ops/group_rms_norm_kernel.cu; every other
    missing kernel module becomes a stub that raises if called. Their attention runs on its plain PyTorch path
    (causal_attn_backend=None). Returns the names stubbed."""
    import importlib.abc, importlib.machinery, types
    stubbed = []

    def fwd(x, num_features, num_groups, weight, eps):
        """y = x * rsqrt(mean_group(x^2) + eps) * gamma, float32 accumulation, output in x's dtype; rstd [rows, groups]."""
        xg = x.float().reshape(-1, num_groups, num_features // num_groups)
        rstd = torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + eps)
        y = (xg * rstd).reshape(x.shape)
        if weight is not None:
            y = y * weight.float()
        return y.to(x.dtype), rstd.reshape(-1, num_groups)

    def bwd(y_grad, x, num_features, num_groups, rstd, weight, memory_efficient):
        """Analytic RMSNorm gradient from the saved rstd, as their kernel: dx = r*(gy*w) - x*r^3*mean(gy*w*x),
        dw = sum over rows of gy*x*r."""
        if memory_efficient:
            raise RuntimeError('memory-efficient RMSNorm backward is not in the shim')
        n = num_features // num_groups
        gy = y_grad.float().reshape(-1, num_groups, n)
        xg = x.float().reshape(-1, num_groups, n)
        r = rstd.float().reshape(-1, num_groups, 1)
        w = weight.float().reshape(num_groups, n) if weight is not None else None
        gyw = gy * w if w is not None else gy
        dx = r * gyw - xg * r ** 3 * (gyw * xg).mean(-1, keepdim=True)
        dw = (gy * xg * r).sum(0).reshape(-1).to(weight.dtype) if w is not None else None
        return dx.reshape(x.shape).to(x.dtype), dw

    def fwd_affine(x, nf, ng, weight, eps):
        return fwd(x, nf, ng, weight, eps)

    def fwd_plain(x, nf, ng, eps):
        return fwd(x, nf, ng, None, eps)

    def bwd_affine(y_grad, x, nf, ng, rstd, weight, me):
        return bwd(y_grad, x, nf, ng, rstd, weight, me)

    def bwd_plain(y_grad, x, nf, ng, rstd, me):
        return bwd(y_grad, x, nf, ng, rstd, None, me)[0]

    class StubModule(types.ModuleType):
        def __getattr__(self, attr):
            if attr.startswith('__'):
                raise AttributeError(attr)
            return _Stub(self.__name__ + '.' + attr)

    ext, ops = StubModule('xllm_extension'), StubModule('xllm_extension.ops')
    ext.__path__ = []
    ext.ops = ops
    ops.group_rms_norm_fwd_affine, ops.group_rms_norm_fwd = fwd_affine, fwd_plain
    ops.group_rms_norm_bwd_affine, ops.group_rms_norm_bwd = bwd_affine, bwd_plain
    sys.modules['xllm_extension'], sys.modules['xllm_extension.ops'] = ext, ops
    stubbed += ['xllm_extension (group RMSNorm in PyTorch; everything else raises)']
    prefixes = ('flash_attn_2_cuda', 'flash_attn_3', 'flash_attn', 'fla', 'xattn', 'sentencepiece', 'tiktoken',
                'nltk', 'rouge_score', 'pynvml', 'wandb', 'submitit', 'fire', 'blobfile')

    class Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, name, path, target=None):
            if name.split('.')[0] in prefixes:
                return importlib.machinery.ModuleSpec(name, self, is_package=True)
            return None

        def create_module(self, spec):
            mod = StubModule(spec.name)
            mod.__path__ = []
            return mod

        def exec_module(self, module):
            stubbed.append(module.__name__)
    sys.meta_path.append(Finder())                   # last: installed packages win
    return stubbed


class _MockRef(torch.nn.Module):
    """V198_REFCHECK_MOCK=1: our own model behind their interface (forward(tokens, multi_segments) -> (logits, aux,
    cache); loop_times; backprop_depth). Tests the reference-check plumbing without their code."""

    def __init__(self, m):
        super().__init__()
        self.inner = m
        self.loop_times, self.backprop_depth = m.R, None

    def forward(self, tokens, multi_segments, **kw):
        m, R = self.inner, self.loop_times
        k = R if self.backprop_depth is None else min(R, self.backprop_depth)
        h, ctx = m.prepare(tokens, multi_segments)
        with torch.no_grad():
            for t in range(R - k):
                h = m.F(h, t, ctx)
        for t in range(R - k, R):
            h = m.F(h, t, ctx)
        return m.readout(h, ctx), None, None


def refcheck(xllm_dir, weights):
    """Our port vs their xLLM model on the same inputs and device: logits (fp32 at R = 1, 5, 10, with and without
    document masking; bf16 at R = 5), and the full (R = 5) and truncated (R = 10, last 2 loops, via their own
    backprop_depth) fp32 gradients of the next-token loss. Kaggle only: it executes their code."""
    mock = os.environ.get('V198_REFCHECK_MOCK') == '1'
    if not (mock or os.path.isdir('/kaggle/working') or os.environ.get('V198_ALLOW_REFCHECK') == '1'):
        raise SystemExit('refcheck executes their code: Kaggle only (or V198_ALLOW_REFCHECK=1 inside a sandbox)')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rep = dict(driver_sha=driver_sha(), device=DEV, mock=mock, xllm_dir=xllm_dir)
    path = os.path.join(OUT, 'v198_refcheck.json')
    ours, info, tok = load(weights)
    data = val_chunks(tok)
    ids_doc = data[0:1].to(DEV)
    try:
        if mock:
            ref = _MockRef(M.tiny_model(seed=0, device=DEV) if TINY else M.load_model(weights, device=DEV)[0])
        else:
            rep['stubbed'] = _install_ref_shims()
            sys.path.insert(0, os.path.abspath(xllm_dir))
            from xllm.config import ModelConf
            from xllm.models.build import get_model_cls
            from xllm.paper_part2.artifacts import load_artifact
            fields, sd = load_artifact(weights)
            fields = dict(fields, causal_attn_backend=None)      # their plain PyTorch attention (no FA3 on a T4)
            cfg = ModelConf.from_dict(fields)

            class _Tok:
                bos_id, eos_id, pad_id = ours.bos_id, 1, -1
            with torch.device(DEV):
                ref = get_model_cls(cfg.arch, cfg)(cfg, _Tok())
            ref.load_state_dict(sd, strict=True)
            rep['ref_class'] = type(ref).__name__
        ref = ref.float().eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    except Exception:
        rep['status'], rep['error'] = 'ERROR', traceback.format_exc()[-4000:]
        json.dump(rep, open(path, 'w'), indent=1)
        log('refcheck ERROR (their code did not build):', rep['error'][-600:])
        return 'ERROR'
    try:
        _ref_compare(rep, ref, ours, ids_doc, mock)
    except Exception:
        rep['status'], rep['error'] = 'ERROR', traceback.format_exc()[-4000:]
        json.dump(rep, open(path, 'w'), indent=1)
        log('refcheck ERROR during the comparison:', rep['error'][-600:])
        return 'ERROR'
    json.dump(rep, open(path, 'w'), indent=1)
    fp = {k: v for k, v in rep['logits'].items() if k.startswith('fp32')}
    log('refcheck %s: fp32 logits max |diff| %.2e, CE diff max %.2e (R = 1, 5, 10; doc mask on/off); bf16 %s; '
        'gradients rel err %s' % (rep['status'], max(v['max_abs'] for v in fp.values()),
                                  max(abs(v['ce_ref'] - v['ce_ours']) for v in fp.values()),
                                  rep['logits'].get('bf16_chunk_R5_doc1', rep.get('bf16_error', '')[-200:]),
                                  {k: '%.2e' % v['rel_err_all'] for k, v in rep['grads'].items()}))
    return rep['status']


def _ref_compare(rep, ref, ours, ids_doc, mock):
    pos = torch.nonzero(ids_doc[0, 1:] == ours.bos_id)
    first_doc = int(pos[0]) + 1 if len(pos) else ids_doc.shape[1]
    inputs = dict(chunk=ids_doc, one_doc=ids_doc[:, :max(8, first_doc)])
    logits = rep['logits'] = {}

    def cmp(lr, lo, ids):
        d = (lr - lo).abs()
        return dict(max_abs=float(d.max()), mean_abs=float(d.mean()), logit_absmax=float(lr.abs().max()),
                    ce_ref=float(M.ce(lr, ids)), ce_ours=float(M.ce(lo, ids)),
                    argmax_agree=float((lr.argmax(-1) == lo.argmax(-1)).float().mean()))
    with torch.no_grad():
        for name, ids in inputs.items():
            for R in (1, 5, 10):
                for dm in (True, False):
                    ref.loop_times = R
                    logits['fp32_%s_R%d_doc%d' % (name, R, dm)] = cmp(ref(ids, dm)[0].float(),
                                                                      ours(ids, R=R, docmask=dm), ids)
        ref.loop_times = 5
        try:                                             # their numerics: BF16 weights and activations
            rb, ob = ref.to(torch.bfloat16), ours.to(torch.bfloat16)
            logits['bf16_chunk_R5_doc1'] = cmp(rb(ids_doc, True)[0].float(), ob(ids_doc, R=5, docmask=True), ids_doc)
        except Exception:
            rep['bf16_error'] = traceback.format_exc()[-1500:]
        ref.float(); ours.float()
    # gradients: full at R = 5 (their eval forward backpropagates through all loops when R <= backprop_depth), and
    # truncated at R = 10 through their own backprop_depth = 2 (their training's truncation) vs our k = 2
    ms = Meas(ours)
    for p in ref.parameters():
        p.requires_grad_(True)
    names = [n for n, _ in ours.named_parameters()]
    rparams = dict(ref.named_parameters())
    if mock:
        rparams = {n[len('inner.'):]: p for n, p in rparams.items()}
    grads = rep['grads'] = {}
    old_depth = getattr(ref, 'backprop_depth', None)
    for R, k in ((5, 5), (10, 2)):
        ref.loop_times, ref.backprop_depth = R, (None if k == R else k)
        for p in ref.parameters():
            p.grad = None
        M.ce(ref(ids_doc, DOCMASK)[0].float(), ids_doc).backward()
        ms.backward(ids_doc, R, k, 1.0, ckpt=False)
        num = den = 0.0
        worst = (0.0, None)
        for n, p in zip(names, ms.params):
            gr = rparams[n].grad
            gr = torch.zeros_like(p, dtype=torch.float64) if gr is None else gr.double()
            go = torch.zeros_like(p, dtype=torch.float64) if p.grad is None else p.grad.double()
            dn, rn = float(((go - gr) ** 2).sum()), float((gr ** 2).sum())
            num += dn; den += rn
            if rn > 0 and math.sqrt(dn / rn) > worst[0]:
                worst = (math.sqrt(dn / rn), n)
        grads['R%d_k%d' % (R, k)] = dict(rel_err_all=math.sqrt(num / (den + 1e-300)), worst_param=worst[1],
                                         worst_rel_err=worst[0])
    ref.loop_times, ref.backprop_depth = 5, old_depth
    ok = all(v['max_abs'] <= 2e-3 * max(1.0, v['logit_absmax'] / 10) and abs(v['ce_ref'] - v['ce_ours']) <= 1e-4
             for key, v in logits.items() if key.startswith('fp32')) and \
        all(v['rel_err_all'] <= 1e-4 for v in grads.values())
    rep['status'] = 'PASS' if ok else 'FAIL'


# ----------------------------------------------------------------------------------------------------- analysis
def analyze(d):
    rng = np.random.default_rng(198)
    files = sorted(glob.glob(os.path.join(d, '**', 'v198_ifm_*.json'), recursive=True))
    rows, adam, checks = [], {}, {}
    for f in files:
        h = json.load(open(f))
        rows += h['rows']; adam.update(h.get('adam', {})); checks[os.path.relpath(f, d)] = h['checks']
        print('%s: %d rows, driver %s, model %s, docmask %s' % (os.path.relpath(f, d), len(h['rows']), h['driver_sha'],
              h['model'].get('checkpoint_id'), h.get('docmask')))
    for f in sorted(glob.glob(os.path.join(d, '**', 'v198_selftest_*.json'), recursive=True)) + \
            sorted(glob.glob(os.path.join(d, '**', 'v198_refcheck.json'), recursive=True)):
        h = json.load(open(f))
        print('%s: %s %s' % (os.path.relpath(f, d), h.get('status'), {k: v for k, v in h.items() if k in (
            'fails', 'grads', 'stubbed', 'error')}))
    print('noise floors (repeat of the full pass, err over all params):',
          {k: '%.1e' % v for c in checks.values() for k, v in c.items()})
    cells = sorted({(r['R'], r['a']) for r in rows}, key=lambda c: (c[0], -c[1]))

    def med_ci(x):
        x = np.asarray(x, float)
        bs = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(4000)]
        return float(np.median(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    for R, a in cells:
        rs = sorted({r['i']: r for r in rows if r['R'] == R and r['a'] == a}.values(), key=lambda r: r['i'])
        n = len(rs)
        ks = sorted({int(k) for r in rs for k in r['k']} - {0})
        rho = np.array([r['rho'] for r in rs])
        print('\n=== R = %d, a = %.1f: n = %d sequences, loss median %.3f' % (R, a, n, np.median([r['loss'] for r in rs])))
        print('rho: median %.3f [%.3f, %.3f]; range %.3f - %.3f; rho_i > 1 in %d of %d' % (
            *med_ci(rho), rho.min(), rho.max(), int((rho > 1).sum()), n))
        for g in ('all', 'loop', 'part'):
            print('%-5s err  ' % g + '  '.join('k=%d %.3f [%.3f, %.3f]' % (k, *med_ci([r['k'][str(k)][g]['err'] for r in rs]))
                                              for k in ks))
        print('all   err_loop = |gR - gk| / |gR - g0| (readout gradient, equal at every k, left out): ' + '  '.join(
            'k=%d %.3f' % (k, np.median([r['k'][str(k)]['all']['err'] / max(r['k']['0']['all']['err'], 1e-30)
                                        for r in rs])) for k in ks))
        for q in ('ratio', 'cos'):
            print('all   %-5s ' % q + '  '.join('k=%d %.3f' % (k, np.median([r['k'][str(k)]['all'][q] for r in rs]))
                                              for k in ks))
        print('rule rho_i^k (median):        ' + '  '.join('k=%d %.3f' % (k, np.median(rho ** k)) for k in ks))
        al = lambda p, k: (p ** k - p ** R) / (1 - p ** R)
        ort = lambda p, k: np.sqrt(np.abs((p ** (2 * k) - p ** (2 * R)) / (1 - p ** (2 * R))))
        print('geometric aligned / orthogonal: ' + '  '.join('k=%d %.3f / %.3f' % (
            k, np.median([al(p, k) for p in rho]), np.median([ort(p, k) for p in rho])) for k in ks))
        pa, po, ss = {k: [] for k in ks}, {k: [] for k in ks}, {k: [] for k in ks}
        for r in rs:
            s = np.array(r['adj'][::-1])               # s_j: signal j loops back from the output
            G = np.array(r['gram'])[::-1, ::-1]       # same order
            for k in ks:
                pa[k].append(s[k:].sum() / s.sum()); po[k].append(np.sqrt((s[k:] ** 2).sum() / (s ** 2).sum()))
                ss[k].append(np.sqrt(max(G[k:, k:].sum(), 0.0) / G.sum()))   # summed signal |sum_{j>=k} lam_j| / |sum lam_j|
        print('measured-profile aligned / orthogonal: ' + '  '.join('k=%d %.3f / %.3f' % (
            k, np.median(pa[k]), np.median(po[k])) for k in ks))
        print('summed backward signal (hidden-state share): ' + '  '.join('k=%d %.3f' % (k, np.median(ss[k])) for k in ks))
        if a == 1.0 and all(str(k) in rs[0]['k'] for k in range(R)):
            c = np.array([[r['k'][str(j + 1)]['loop']['step'] if j + 1 < R else r['k'][str(R - 1)]['loop']['err'] *
                           r['nR']['loop'] for j in range(R)] for r in rs])      # |c_j|, j = 0 (last loop) .. R-1
            prof = np.median(c / c[:, :1], 0)
            print('per-loop contribution |c_j| / |c_0| (loop params, median): ' + ' '.join('%.3g' % x for x in prof))
            print('aligned form on measured contribution sizes: ' + '  '.join('k=%d %.3f' % (k, np.median(
                c[:, k:].sum(1) / c.sum(1))) for k in ks))
    if adam:
        print('\n=== direction after Adam (v172 procedure, R = %s): cos(u_k, u_R) | raw: lost gradient of gbar, size ratio, '
              'cosine | loop params cos_u' % sorted({v['R'] for v in adam.values()}))
        for key in sorted(adam, key=lambda s: (-adam[s]['a'], -1 if adam[s]['k'] == adam[s]['R'] else adam[s]['k'])):
            v = adam[key]
            if v['k'] == v['R']:
                print('a=%.1f full: batch norm median %.2f, %d of %d batches clipped' % (v['a'], v['norm_median'],
                                                                                       v['clipped'], v['B']))
                continue
            print('a=%.1f k=%d: cos_u %.4f | err %.3f ratio %.3f cos %.4f | loop cos_u %.4f' % (
                v['a'], v['k'], v['cos_u']['all'], v['err_g']['all'], v['ratio_g']['all'], v['cos_g']['all'],
                v['cos_u']['loop']))


# --------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=None); ap.add_argument('--seqs', default='0:16')
    ap.add_argument('--adam', default='1.0,0.5,0.3'); ap.add_argument('--tag', default='x')
    ap.add_argument('--selftest', action='store_true'); ap.add_argument('--refcheck', default=None)
    ap.add_argument('--analyze', default=None)
    A = ap.parse_args()
    if A.analyze:
        return analyze(A.analyze)
    if not TINY and not A.weights:
        raise SystemExit('--weights DIR is required (or TINY=1)')
    if A.refcheck:
        st = refcheck(A.refcheck, A.weights)
        raise SystemExit(0 if st in ('PASS', 'ERROR') else 3)
    m, info, tok = load(A.weights)
    ms = Meas(m)
    data = val_chunks(tok)
    log('v198 driver %s | %s (%s) | dev %s fp32 | params %.1fM (loop %.1fM, part %.1fM) | docmask %s | %d val chunks of %d '
        'tokens, %d boundaries in chunk 0' % (driver_sha(), info['checkpoint_id'], info['table_row'], DEV, ms.ntot / 1e6,
                                              sum(p.numel() for p in m.loop_parameters()) / 1e6,
                                              sum(p.numel() for p in ms.params if ms.member['part'](p)) / 1e6, DOCMASK,
                                              data.shape[0], data.shape[1], int((data[0, 1:] == m.bos_id).sum())))
    if A.selftest:
        ok = selftest(ms, data, A.tag)
        raise SystemExit(0 if ok else 3)
    out = dict(driver_sha=driver_sha(), model={k: v for k, v in info.items() if k != 'cfg'}, cfg=info['cfg'], R_list=R_LIST,
               alphas=ALPHAS, docmask=DOCMASK, seq=SEQ, rows=[], checks={}, adam={}, freeze_embed=FREEZE_EMBED,
               gpu=torch.cuda.get_device_name(0) if DEV == 'cuda' else 'cpu')
    path = os.path.join(OUT, 'v198_ifm_%s.json' % A.tag)
    lo, hi = (int(x) for x in A.seqs.split(':'))
    if hi > lo:
        measure(ms, data[:N_SEQ], (lo, hi), out, path)
    if A.adam:
        adam_direction(ms, train_chunks(tok), [float(x) for x in A.adam.split(',')], out, path)
    json.dump(out, open(path, 'w'))
    log('done', A.tag)


if __name__ == '__main__':
    main()
