"""v196: band-of-R test of the rho^k account, and a head-to-head of lost-gradient estimators (prereg
docs/prereg_v196_band.md). Frozen models, no training. Per sequence, at each loop count R:
  lam[t]     backward signal dL/d(output of loop t), t = 0 .. R-1, one checkpointed backward with parameters frozen
             (v195's signals()); exact float64 state Gram of all lam[t]; their norms (adj)
  per-token  for each loop: per-position norms of lam[t] (a T x d array) -> overlap of neighbouring loops' position
             profiles BC_t = sum_i |lam_t(i)| |lam_t+1(i)| / (|lam_t| |lam_t+1|) and within-position cosine
             W_t = sum_i <lam_t(i), lam_t+1(i)> / sum_i |lam_t(i)| |lam_t+1(i)|, so cos(lam_t, lam_t+1) = BC_t x W_t;
             position centre of mass, participation ratio, effective rank (exp entropy of normalised singular values^2)
  err        lost gradient |g_R - g_k| / |g_R| at k = 1, 2, 4, 8, 16, the Table 1 protocol (v164 measure(): seed
             777 + i, fp32, all parameters, a = 1; per-loop checkpointing, as in training), plus g_0 for the loop part
             err_loop = |g_R - g_k| / |g_R - g_0|. Huginn: the probe's own truncated passes (loop-only subset, bf16).
  python v196_band.py parcae --repo SandyResearch/parcae-140m --R 32,48,64 --tag 140m
  LOOP_MODEL=huginn python v196_band.py probe --R 48 --seqs 0-7,16-23 --tag huginn
"""
import argparse, json, math, os, time
import numpy as np
import torch

import v195_rho_modes as V


def per_token(lam, check=False):
    X = torch.stack([l.reshape(l.shape[-2], -1).double() for l in lam])          # R x T x d
    n2 = (X ** 2).sum(-1); tot = n2.sum(-1); nt = n2.sqrt()
    prod = (nt[:-1] * nt[1:]).sum(-1)
    BC = prod / (tot[:-1] * tot[1:]).sqrt()
    W = (X[:-1] * X[1:]).sum((-1, -2)) / prod
    pos = torch.arange(X.shape[1], dtype=torch.float64, device=X.device)
    com = (n2 * pos).sum(-1) / tot; pr = tot ** 2 / (n2 ** 2).sum(-1)
    sv2 = torch.linalg.svdvals(X.float()).double() ** 2; p = sv2 / sv2.sum(-1, keepdim=True)
    erank = torch.exp(-(p * torch.log(p.clamp_min(1e-300))).sum(-1))
    out = dict(BC=BC.tolist(), W=W.tolist(), com=com.tolist(), pr=pr.tolist(), erank=erank.tolist(), T=int(X.shape[1]))
    if check:                                                                    # BC x W must equal the plain cosine
        cos = (X[:-1] * X[1:]).sum((-1, -2)) / (tot[:-1] * tot[1:]).sqrt()
        out['check_bcw'] = float((BC * W - cos).abs().max())
    return out


def state_gram(lam):
    Vm = torch.stack([l.flatten().double() for l in lam]); return (Vm @ Vm.T).cpu().numpy()


def parcae_seq(fam, ids, i, R):
    x0, e = fam.start(ids, i, R)
    lam = V.signals(fam, x0, e, R)
    row = dict(i=i, R=R, adj=[float(l.norm()) for l in lam], state_gram=state_gram(lam).tolist(), tok=per_token(lam, True))
    del lam
    d = fam.d; m = fam.m; d.R = R
    params = [q for q in m.parameters()]
    for q in params:
        q.requires_grad_(True)

    def grad(k):
        torch.manual_seed(777 + i); m.zero_grad(set_to_none=True)
        d.damped_loss(m, ids, k, 1.0, ckpt=True).backward()
        g = torch.cat([(q.grad if q.grad is not None else torch.zeros_like(q)).flatten() for q in params])
        m.zero_grad(set_to_none=True); return g
    gR = grad(R); g0 = grad(0)
    nR, nR0 = float(gR.norm()), float((gR - g0).norm()); del g0
    err, errl = {}, {}
    for k in (1, 2, 4, 8, 16):
        if k < R:
            dk = float((grad(k) - gR).norm()); err[k] = dk / nR; errl[k] = dk / nR0
    for q in params:
        q.requires_grad_(False)
    row.update(err=err, err_loop=errl, gR_norm=nR, gR_minus_g0_norm=nR0)
    return row


def probe_seq(fam, ids, si, R):
    L = fam.L
    h0, ctx = fam.start(ids, si, R)
    lam = V.signals(fam, h0, ctx, R)
    row = dict(i=si, R=R, adj=[float(l.norm()) for l in lam], state_gram=state_gram(lam).tolist(), tok=per_token(lam, True))
    del lam
    _, gR = L.variant(fam.ad, h0, ctx, ids, R, R)
    nR = math.sqrt(sum(float((gR[a:a + (1 << 26)].double() ** 2).sum()) for a in range(0, gR.numel(), 1 << 26)))
    err = {}
    for k in (1, 2, 4, 8, 16):
        if k < R:
            _, gk = L.variant(fam.ad, h0, ctx, ids, R, k)
            dk = math.sqrt(sum(float(((gk[a:a + (1 << 26)] - gR[a:a + (1 << 26)]).double() ** 2).sum()) for a in range(0, gR.numel(), 1 << 26)))
            err[k] = dk / nR; del gk
    row.update(err=err, err_loop=dict(err), gR_norm=nR, gR_minus_g0_norm=nR)
    return row


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('family', choices=['parcae', 'probe'])
    ap.add_argument('--repo', default=None); ap.add_argument('--R', default='48'); ap.add_argument('--seqs', default='0-15')
    ap.add_argument('--tag', default='x'); ap.add_argument('--limit', type=int, default=None)
    A = ap.parse_args(); t0 = time.time()
    fam = V.ParcaeFam(A.repo) if A.family == 'parcae' else V.ProbeFam(A.seqs)
    print('v196: %s | dtype %s' % (fam.name, fam.dtype), flush=True)
    for R in [int(r) for r in A.R.split(',')]:
        out = dict(model=fam.name, R=R, rows=[])
        for n, (i, src, ids) in enumerate(fam.seqs()):
            if A.limit is not None and n >= A.limit:
                break
            ts = time.time()
            r = (parcae_seq if A.family == 'parcae' else probe_seq)(fam, ids.to(fam.dev), i, R)
            r['src'] = src; r['s'] = round(time.time() - ts, 1); out['rows'].append(r)
            print('R=%d seq %2d %-8s err k1/2/4/8 %s | rho %.4f | BCxW check %.1e (%.0fs, total %.0fs)' % (
                R, i, src, ' '.join('%.4f' % r['err'][k] for k in (1, 2, 4, 8) if k in r['err']),
                math.exp(np.polyfit(np.arange(R), np.log(np.array(r['adj'][::-1])), 1)[0]), r['tok']['check_bcw'],
                r['s'], time.time() - t0), flush=True)
            json.dump(out, open(os.path.join(fam.out_dir, 'v196_%s_R%d.json' % (A.tag, R)), 'w'))
            if fam.dev == 'cuda':
                torch.cuda.empty_cache()
    print('done %.0f s' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
