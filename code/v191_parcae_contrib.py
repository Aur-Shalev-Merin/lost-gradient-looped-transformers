"""v191: per-loop contributions to the weight gradient on a frozen Parcae model (plan: docs/v191_parcae_contrib_plan.md).
Same setup as v164 measure() / v190: 16 GSM8K validation sequences, batch 1, fp32, all parameters, a = 1, seed 777 + i
before every pass. For each sequence:
  g_k     gradient with backprop through the last k loops, k = 0 .. R (g_0: no loop gradient, only coda / head; g_R: full);
  c_j     = g_{j+1} - g_j, the contribution of the loop j steps before the output. Exact: truncation leaves the backward
          signal in the kept loops unchanged, so g_k = g_0 + c_0 + ... + c_{k-1};
  adj[t]  backward-signal size at the output of loop t (t = R-1 is the last loop), as v190;
  step[t] ||x_{t+1} - x_t|| / ||x_t|| in the forward pass (a = 1, so x_{t+1} is the core block's output).
Exact (float64 sums): |c_j|, cos(c_j, c_{j-1}), |g_R - g_k| for every k, |g_R|, |g_0|, |g_R - g_0|.
Sketched (CountSketch, 2^20 buckets, the same hashes for every vector): the Gram matrix of (g_0, c_0, ..., c_{R-1}),
validated per sequence against the exact quantities. --exact_gram also stores every c_j and computes the exact Gram
(TINY test only; too large on the real models).
  python v191_parcae_contrib.py --repo SandyResearch/parcae-370m --R 32,16 --tag 370m
"""
import argparse, json, math, os, time
import numpy as np
import torch

import v162_parcae_trunc_train as v
import v163_parcae_damped_trunc as d
import v164_parcae_predict as p

SKETCH_D = 2 ** 20
CHUNK = 2 ** 24
STEP, REC = {}, [False]


def instrument(m):
    """Record the forward step of every loop while REC[0] is set (damped_loss calls core_block_forward(x, e, freqs,
    None, tensor(t), tot); with a = 1 the new state is its output)."""
    orig = m.core_block_forward

    def wrapped(x, e, *args, **kw):
        out = orig(x, e, *args, **kw)
        if REC[0]:
            with torch.no_grad():
                STEP[int(args[2])] = float((out - x).float().norm() / x.float().norm().clamp_min(1e-30))
        return out
    m.core_block_forward = wrapped


def grad_pass(m, ids, i, R, k, params, adj=None):
    d.R = R; torch.manual_seed(777 + i); m.zero_grad(set_to_none=True)
    d.damped_loss(m, ids, k, 1.0, adj=adj, ckpt=False).backward()
    g = [q.grad if q.grad is not None else torch.zeros_like(q) for q in params]
    for q in params:
        q.grad = None
    return g


def dot(a, b):
    return sum(float(torch.sum(x * y, dtype=torch.float64)) for x, y in zip(a, b))


def sqdiff(a, b):
    return sum(float(torch.sum((x - y) ** 2, dtype=torch.float64)) for x, y in zip(a, b))


def sketch(vecs, dev):
    out = torch.zeros(SKETCH_D, dtype=torch.float64, device=dev)
    for pi, t in enumerate(vecs):
        gen = torch.Generator(device=dev); gen.manual_seed(10007 * (pi + 1)); flat = t.flatten()
        for s0 in range(0, flat.numel(), CHUNK):
            x = flat[s0:s0 + CHUNK]
            h = torch.randint(0, SKETCH_D, (x.numel(),), generator=gen, device=dev)
            sg = torch.randint(0, 2, (x.numel(),), generator=gen, device=dev).to(torch.float64) * 2 - 1
            out.index_add_(0, h, x.double() * sg)
    return out


def one_sequence(m, ids, i, R, params, dev, exact_gram=False):
    STEP.clear(); adj = {}; REC[0] = True
    gR = grad_pass(m, ids, i, R, R, params, adj=adj)
    REC[0] = False
    nR2 = dot(gR, gR)
    g_prev = grad_pass(m, ids, i, R, 0, params)
    n02, nRm02 = dot(g_prev, g_prev), sqdiff(gR, g_prev)
    sk = [sketch(g_prev, dev)]; keep = [[x.detach().cpu().clone() for x in g_prev]] if exact_gram else None
    err, cn, cadj, c_prev = {}, [], [], None
    for k in range(1, R + 1):
        g = gR if k == R else grad_pass(m, ids, i, R, k, params)
        if k < R:
            err[k] = math.sqrt(sqdiff(gR, g) / nR2)
        c = [y.neg_().add_(x) for x, y in zip(g, g_prev)]           # c_{k-1} = g_k - g_{k-1}, in g_prev's storage
        cn.append(math.sqrt(dot(c, c)))
        if c_prev is not None:
            cadj.append(dot(c, c_prev) / (cn[-1] * cn[-2] + 1e-300))
        sk.append(sketch(c, dev))
        if exact_gram:
            keep.append([x.detach().cpu().clone() for x in c])
        c_prev, g_prev = c, g
    S = torch.stack(sk); G = (S @ S.T).cpu().numpy(); del S, sk
    row = dict(i=i, adj=[adj[t] for t in range(R)], step=[STEP.get(t) for t in range(R)], cnorm=cn, cos_adjacent=cadj,
               err=err, g0_norm=math.sqrt(n02), gR_norm=math.sqrt(nR2), gR_minus_g0_norm=math.sqrt(nRm02),
               gram_sketch=G.tolist())
    # validation of the sketch against exact quantities
    dg = np.sqrt(np.clip(np.diag(G), 0, None))
    row['val_norm_maxrel'] = float(np.max(np.abs(dg[1:] / np.array(cn) - 1)))
    sk_adj = [G[j + 1, j + 2] / (dg[j + 1] * dg[j + 2]) for j in range(R - 1)]
    row['val_cos_adjacent_maxabs'] = float(np.max(np.abs(np.array(sk_adj) - np.array(cadj)))) if R > 1 else 0.0
    tot = G.sum()
    rec = {k: math.sqrt(max(G[k + 1:, k + 1:].sum(), 0) / tot) for k in err}
    row['val_err_maxabs'] = float(max(abs(rec[k] - err[k]) for k in err))
    if exact_gram:
        E = np.array([[sum(float(torch.sum(x.double() * y.double())) for x, y in zip(a, b)) for b in keep] for a in keep])
        nrm = np.sqrt(np.diag(E)); C_ex = E / np.outer(nrm, nrm); C_sk = G / np.outer(dg, dg)
        row['val_exact_gram_cos_maxabs'] = float(np.max(np.abs(C_ex - C_sk)))
    return row


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--repo', required=True); ap.add_argument('--R', default='32')
    ap.add_argument('--tag', default='x'); ap.add_argument('--exact_gram', action='store_true'); A = ap.parse_args()
    v.REPO = A.repo
    tok, m = v.load_model(); data = v.gsm8k_chunks(tok); instrument(m); m.eval()
    params = [q for q in m.parameters() if q.requires_grad]
    print('v191: repo %s | params %.1fM | v164 driver %s' % (v.REPO, sum(q.numel() for q in params) / 1e6, p.driver_sha()),
          flush=True)
    for R in [int(r) for r in A.R.split(',')]:
        out = dict(repo=v.REPO, R=R, a=1.0, sketch_d=SKETCH_D, rows=[]); t0 = time.time()
        for i in range(min(v.N_MEASURE, len(data['val']))):
            ids = data['val'][i:i + 1].to(v.DEV)
            r = one_sequence(m, ids, i, R, params, v.DEV, A.exact_gram); out['rows'].append(r)
            print('R=%d seq %2d: err k1/2/4 %.4f %.4f %.4f | |c_0..2| %s | sketch check: norm %.1e cos %.1e err %.1e%s (%ds)' % (
                R, i, r['err'].get(1, float('nan')), r['err'].get(2, float('nan')), r['err'].get(4, float('nan')),
                [round(x, 4) for x in r['cnorm'][:3]], r['val_norm_maxrel'], r['val_cos_adjacent_maxabs'], r['val_err_maxabs'],
                ' exact-gram cos %.1e' % r['val_exact_gram_cos_maxabs'] if A.exact_gram else '', time.time() - t0), flush=True)
            out['seconds'] = round(time.time() - t0)
            json.dump(out, open(os.path.join(v.OUT, 'v191_%s_R%d.json' % (A.tag, R)), 'w'))


if __name__ == '__main__':
    main()
