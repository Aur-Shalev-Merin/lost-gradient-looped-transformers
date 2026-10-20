"""v192: per-loop contributions to the weight gradient on frozen Huginn-3.5B (plan: docs/v192_huginn_contrib_plan.md).
The v191 measurement on a second model family, through loop_grad_probe.py's adapter (the Sep-30 probe setup:
128-token sequences, gradient over the input-injection adapter + first core layer, bf16 on a T4).

bf16 rounding is ~0.4% of each gradient entry, so a contribution obtained as g_{k+1} - g_k would carry noise of order
0.4% of the whole gradient, as large as the far loops' contributions. Instead each loop's contribution is computed
DIRECTLY, so its rounding is relative to its own size:
  1. one backward pass through all R loops records the backward signal lam_t = dL/d(output of loop t) (fp32 copies);
  2. for each loop t, c_t = (dF_t/dtheta)^T lam_t, one vector-Jacobian product through that loop alone, gathered in
     fp32. Exact arithmetic gives the same c as g_{j+1} - g_j (truncation leaves the kept loops' signal unchanged);
  3. going from the farthest loop to the last, the tail sums T_j = sum_{j' >= j} c_{j'} = g_R - g_j are accumulated in
     fp32, so the lost gradient |g_R - g_k| / |g_R| comes out for every k in the same pass.
The subset holds loop parameters only, so g_0 = 0. Also recorded: the probe's truncated passes at k = 1, 2, 4, 8, 16
(reproduction check against the Sep-30 probe), |sum_t c_t - g_R| / |g_R| with g_R from one bf16 backward (the
accumulation noise that subtraction would have suffered), the forward residual per loop, and a CountSketch Gram
(2^20 buckets, GPU) of all contributions. Rows use v191's format (j = loops before the output), so v191_analyze.py
reads them. CHECK_SUBTRACT=1 (TINY test) also forms every c_j by subtraction and compares.
  LOOP_MODEL=huginn LOOP_R=32 SEQS=0-7 TAG=huginn_part0 python v192_huginn_contrib.py
"""
import json, math, os, time
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

import loop_grad_probe as L

SKETCH_D, CHUNK, F64 = 2 ** 20, 2 ** 24, 1 << 26
CHECK_SUBTRACT = os.environ.get('CHECK_SUBTRACT') == '1'


def f64(fn, *vecs):
    """Chunked float64 reduction over flat CPU vectors."""
    return sum(float(fn(*[v[a:a + F64].double() for v in vecs])) for a in range(0, vecs[0].numel(), F64))


def sketch(vec, dev):
    out = torch.zeros(SKETCH_D, dtype=torch.float64, device=dev)
    gen = torch.Generator(device=dev); gen.manual_seed(10007)
    for a in range(0, vec.numel(), CHUNK):
        x = vec[a:a + CHUNK].to(dev, torch.float64)
        h = torch.randint(0, SKETCH_D, (x.numel(),), generator=gen, device=dev)
        sg = torch.randint(0, 2, (x.numel(),), generator=gen, device=dev).to(torch.float64) * 2 - 1
        out.index_add_(0, h, x * sg)
    return out


def signals(ad, h0, ctx, ids, R):
    """Full backward with the parameters frozen: lam[t] = dL/d(output of loop t), fp32 copies."""
    lam = {}
    h = h0.detach().requires_grad_(True)
    for t in range(R):
        h = checkpoint(ad.F, h, t, ctx, use_reentrant=False)
        h.register_hook(lambda g, t=t: lam.__setitem__(t, g.detach().float().clone()))
    L.ce(ad.readout(h, ctx), ids).backward()
    return lam


def contribution(ad, x, t, ctx, lam_t):
    for p in ad.subset:
        p.requires_grad_(True)
    out = ad.F(x.detach(), t, ctx)
    gs = torch.autograd.grad(out, ad.subset, grad_outputs=lam_t.to(out.dtype), allow_unused=True)
    for p in ad.subset:
        p.requires_grad_(False)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).detach().float().flatten().cpu() for g, p in zip(gs, ad.subset)])


def one_sequence(ad, ids, R, dev):
    h0, ctx = ad.prepare(ids)
    traj = [h0]
    with torch.no_grad():
        for t in range(R):
            traj.append(ad.F(traj[-1], t, ctx))
    step = [float((traj[t + 1] - traj[t]).float().norm()) / max(float(traj[t].float().norm()), 1e-30) for t in range(R)]
    lam = signals(ad, h0, ctx, ids, R)
    cn, cadj, tail2 = [0.0] * R, [0.0] * (R - 1), [0.0] * R
    sk = [None] * (R + 1); sk[0] = torch.zeros(SKETCH_D, dtype=torch.float64, device=dev)      # g_0 = 0
    T, c_prev, direct = None, None, {}
    for t in range(R):                              # farthest loop first: j = R - 1 - t
        j = R - 1 - t
        c = contribution(ad, traj[t], t, ctx, lam[t])
        T = c.clone() if T is None else T.add_(c)
        tail2[j] = f64(lambda x: (x * x).sum(), T)
        cn[j] = math.sqrt(f64(lambda x: (x * x).sum(), c))
        if c_prev is not None:                       # c_prev = c_{j+1}
            cadj[j] = f64(lambda x, y: (x * y).sum(), c, c_prev) / (cn[j] * cn[j + 1] + 1e-300)
        sk[j + 1] = sketch(c, dev)
        if CHECK_SUBTRACT:
            direct[j] = c.clone()
        c_prev = c
    del traj, c, c_prev
    err = {k: math.sqrt(tail2[k] / tail2[0]) for k in range(1, R)}
    # the probe's own method: truncated bf16 backward passes, differences of full gradients
    _, gR = L.variant(ad, h0, ctx, ids, R, R)
    nR2 = f64(lambda x: (x * x).sum(), gR)
    err_sub = {}
    for k in sorted({k for k in (1, 2, 4, 8, 16) if k < R} | (set(range(1, R)) if CHECK_SUBTRACT else set())):
        _, gk = L.variant(ad, h0, ctx, ids, R, k)
        err_sub[k] = math.sqrt(f64(lambda x, y: ((x - y) ** 2).sum(), gR, gk) / nR2)
        if CHECK_SUBTRACT:
            direct.setdefault('g', {})[k] = gk
    acc = math.sqrt(f64(lambda x, y: ((x - y) ** 2).sum(), T, gR) / nR2)          # |sum_t c_t - g_R| / |g_R|
    S = torch.stack(sk); G = (S @ S.T).cpu().numpy(); del S, sk
    row = dict(adj=[float(lam[t].norm()) for t in range(R)], step=step, cnorm=cn, cos_adjacent=cadj, err=err,
               err_subtraction=err_sub, sum_vs_full_backward=acc, g0_norm=0.0, gR_norm=math.sqrt(tail2[0]),
               gR_minus_g0_norm=math.sqrt(tail2[0]), gram_sketch=G.tolist())
    dg = np.sqrt(np.clip(np.diag(G), 0, None))
    row['val_norm_maxrel'] = float(np.max(np.abs(dg[1:] / np.array(cn) - 1)))
    sk_adj = [G[j + 1, j + 2] / (dg[j + 1] * dg[j + 2]) for j in range(R - 1)]
    row['val_cos_adjacent_maxabs'] = float(np.max(np.abs(np.array(sk_adj) - np.array(cadj))))
    tot = G.sum()
    row['val_err_maxabs'] = float(max(abs(math.sqrt(max(G[k + 1:, k + 1:].sum(), 0) / tot) - err[k]) for k in err))
    if CHECK_SUBTRACT:                               # c_j by subtraction vs direct, every j (TINY, fp32)
        g = direct['g']; g[R] = gR; g[0] = torch.zeros_like(gR)
        row['check_subtract_maxrel'] = max(
            math.sqrt(f64(lambda x, y, z: ((x - y - z) ** 2).sum(), g[j + 1], g[j], direct[j])) / (cn[j] + 1e-300)
            for j in range(R))
    return row


def main():
    t0 = time.time(); tok, model = L.load(); dev = L.DEV
    ad = {'huginn': L.Huginn, 'parcae': L.Parcae}[L.FAMILY](model)
    seqs = L.data(tok)
    lo, hi = [int(x) for x in os.environ.get('SEQS', '0-%d' % (len(seqs) - 1)).split('-')]
    tag = os.environ.get('TAG', L.MODEL)
    L.log('v192: %s subset %s (%d params) dtype %s, seqs %d-%d of %d' % (
        L.REPO, ad.subset_name, sum(p.numel() for p in ad.subset), L.DTYPE, lo, hi, len(seqs)))
    for R in L.R_LIST:
        out = dict(model=L.REPO, R=R, subset=ad.subset_name, sketch_d=SKETCH_D, method='direct per-loop VJP', rows=[])
        for si in range(lo, min(hi, len(seqs) - 1) + 1):
            src, ids = seqs[si]; ts = time.time()
            r = one_sequence(ad, ids.unsqueeze(0).to(dev), R, dev); r.update(i=si, src=src, s=round(time.time() - ts, 1))
            out['rows'].append(r)
            L.log('R=%d seq %2d %-8s err direct k1/2/4 %.4f %.4f %.4f | subtraction %.4f %.4f %.4f | sum-vs-full %.1e | '
                  'sketch: norm %.1e cos %.1e err %.1e%s (%.0fs, total %.0fs)' % (
                      R, si, src, *[r['err'].get(k, float('nan')) for k in (1, 2, 4)],
                      *[r['err_subtraction'].get(k, float('nan')) for k in (1, 2, 4)], r['sum_vs_full_backward'],
                      r['val_norm_maxrel'], r['val_cos_adjacent_maxabs'], r['val_err_maxabs'],
                      ' | subtract-vs-direct %.1e' % r['check_subtract_maxrel'] if CHECK_SUBTRACT else '',
                      time.time() - ts, time.time() - t0))
            json.dump(out, open(os.path.join(L.OUT, 'v192_%s_R%d.json' % (tag, R)), 'w'))
            if dev == 'cuda':
                torch.cuda.empty_cache()
    L.log('done %.0f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
