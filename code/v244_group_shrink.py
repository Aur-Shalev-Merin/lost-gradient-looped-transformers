"""v244 (unregistered, descriptive; a check behind a caveat in the paper's Section 5): where does truncation's shrink sit
under SGD? S24/S25 (v241, v243) split the truncated gradient into a turn and a shrink using norms over ALL parameters.
Truncation changes the gradient of only the "loop-affected" parameters (v174's mask: those whose g_1 differs from g_R);
the rest get the same gradient. So under SGD the turn-only arm (g_k * |g_R| / |g_k|) also scales the unaffected
parameters' step by 1 / r_all, and the shrink-only arm by r_all, where r_all = |g_k| / |g_R|.

This script reruns v241's setting G for full backprop unchanged (SGD momentum 0.9, lr 0.1 with warm-up + cosine, global
clip 5, S8's recipe on Parcae-140m at R = 32, 600 steps) on new unscored seeds. Every PROBE_EVERY steps it also computes
g_1 and g_2 on the same batch (not used for training) and records, per k:
  r_all, r_loop, r_rest   |g_k| / |g_R| over all / loop-affected / other parameters
  cos_all, cos_loop       cosine of g_k with g_R over all / loop-affected parameters
  rest_diff               max |g_k - g_R| over the other parameters, relative to max |g_R| (should be ~0)
  share_loop              |g_R[loop]|^2 / |g_R|^2
Summaries are medians over probes at steps >= 100 (as the v241 pilot's R_MATCH). No threshold, no verdict.
  python v244_group_shrink.py --seed S          # one run (GPU), writes OUT_DIR/v244_seed<S>.json
  python v244_group_shrink.py --summary DIR...  # medians over the saved runs
"""
import argparse, glob, json, math, os, time

SEEDS = (2441101, 2441102)
PROBE_EVERY = 10
MASK_BATCHES = 3


def run(seed):
    import numpy as np
    import torch
    import v241_sgd_split as G
    import v174_mix_driver as M
    v = M.v
    tok, m0 = v.load_model(); data = v.gsm8k_chunks(tok); del m0
    _, m = v.load_model(); m.train()
    named = list(m.named_parameters()); params = [p for _, p in named]
    v174_mask = M.affected_mask(m, params, data)            # v174's element mask (one validation sequence), for reference
    steps, lr, clip = v.STEPS, G.LR_G, G.CLIP_G
    g = np.random.default_rng(seed)
    chunks = torch.cat([v.pack(data['train'], g.permutation(len(data['train']))) for _ in range(4)])
    # per-tensor mask: a tensor is loop-affected if any element of g_1 differs from g_R on the first MASK_BATCHES
    # training batches (v174's element mask misses rows of the tied embedding for tokens absent from its one sequence)
    aff = torch.zeros(len(params), dtype=torch.bool)
    for b in range(MASK_BATCHES):
        ids = chunks[b * v.BATCH:(b + 1) * v.BATCH].to(v.DEV)
        gR, g1 = M.grad_pair(m, params, ids, 1, 1.0, False, seed=seed * 1000 + 999000 + b)
        tol = 1e-7 * float(gR.abs().max()); i = 0
        for j, p in enumerate(params):
            n = p.numel(); aff[j] |= bool((gR[i:i + n] - g1[i:i + n]).abs().max() > tol); i += n
    mask = torch.cat([torch.full((p.numel(),), bool(a), dtype=torch.bool) for p, a in zip(params, aff)]).to(v.DEV)
    rest = ~mask
    M.log('v244 seed %d: v241 driver %s; lr %g clip %g momentum %g; loop-affected parameters %d of %d (%d of %d tensors;'
          ' v174 element mask %d); other tensors: %s' % (
              seed, G.driver_sha(), G.LR_G, G.CLIP_G, G.MOM, int(mask.sum()), mask.numel(), int(aff.sum()), len(params),
              int(v174_mask.sum()), ', '.join(nm for (nm, _), a in zip(named, aff) if not a)[:1500]))
    del gR, g1
    gname = lambda nm: '.'.join(nm.split('.')[:2]) if nm.startswith('transformer.') else nm.split('.')[0]
    groups = {}
    i = 0
    for nm, p in named:
        groups.setdefault(gname(nm), []).append((i, i + p.numel())); i += p.numel()
    gidx = {k: torch.cat([torch.arange(a, b) for a, b in sp]).to(v.DEV) for k, sp in groups.items()}
    M.log('groups: %s' % ', '.join('%s %d' % (k, ix.numel()) for k, ix in gidx.items()))
    torch.manual_seed(seed)
    opt = torch.optim.SGD(params, lr=lr, momentum=G.MOM, weight_decay=0.0)
    sched = lambda s: min(1.0, (s + 1) / v.WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / v.STEPS)))
    S = 65536.0 if v.AMP else 1.0
    nonfinite, probes, t0 = 0, [], time.time()
    nrm = lambda x: float(x.double().norm())
    for s in range(steps):
        for gp in opt.param_groups:
            gp['lr'] = lr * sched(s)
        ids = chunks[s * v.BATCH:(s + 1) * v.BATCH].to(v.DEV)
        gR = M.grad_one(m, params, ids, M.R, S, v.AMP, seed=seed * 1000 + s)
        if not torch.isfinite(gR).all():
            nonfinite += 1; S /= 2; continue
        if s % PROBE_EVERY == 0:
            p = dict(step=s, nR=nrm(gR), share_loop=nrm(gR[mask]) ** 2 / max(nrm(gR) ** 2, 1e-30),
                     share_group={k: nrm(gR[ix]) ** 2 / max(nrm(gR) ** 2, 1e-30) for k, ix in gidx.items()})
            for k in G.KS:
                gk = M.grad_one(m, params, ids, k, S, v.AMP, seed=seed * 1000 + s)
                if not torch.isfinite(gk).all():
                    continue
                p['k%d' % k] = dict(
                    r_all=nrm(gk) / nrm(gR), r_loop=nrm(gk[mask]) / nrm(gR[mask]),
                    r_rest=nrm(gk[rest]) / max(nrm(gR[rest]), 1e-30),
                    cos_all=M.cos(gk, gR), cos_loop=M.cos(gk, gR, mask),
                    rest_diff=float((gk[rest] - gR[rest]).abs().max() / (gR.abs().max() + 1e-30)) if rest.any() else 0.0,
                    r_group={k: nrm(gk[ix]) / max(nrm(gR[ix]), 1e-30) for k, ix in gidx.items()})
            probes.append(p)
        gu, _ = G.clipc(gR, clip)
        M.assign(params, gu); opt.step()
        if s % 100 == 0:
            q = probes[-1] if probes and probes[-1]['step'] == s else {}
            M.log('  seed %d step %d: |gR| %.2f %s S %g (%ds)' % (seed, s, nrm(gR), ' '.join(
                'k%d r_all %.3f r_loop %.3f cos_loop %.3f' % (k, q['k%d' % k]['r_all'], q['k%d' % k]['r_loop'],
                                                               q['k%d' % k]['cos_loop']) for k in G.KS if 'k%d' % k in q),
                S, time.time() - t0))
    out = dict(seed=seed, steps=steps, lr=lr, clip=clip, nonfinite=nonfinite, n_loop=int(mask.sum()),
               n_all=mask.numel(), n_v174=int(v174_mask.sum()), probes=probes, v241_sha=G.driver_sha(),
               other_tensors=[nm for (nm, _), a in zip(named, aff) if not a],
               test=M.d.eval_loss(m, data['test'], 1.0), s=round(time.time() - t0))
    fn = os.path.join(v.OUT, 'v244_seed%d.json' % seed)
    json.dump(out, open(fn, 'w'))
    M.log('v244 seed %d done: test %.4f nonfinite %d (%ds) -> %s' % (seed, out['test'], nonfinite, out['s'], fn))
    print(summary_of([out]))


def summary_of(runs):
    import numpy as np
    L = []
    for r in runs:
        pr = [p for p in r['probes'] if p['step'] >= 100] or r['probes']
        L.append('seed %d: %d probes (steps >= 100 where available), test %.4f, nonfinite %d, loop-affected %d of %d' % (
            r['seed'], len(pr), r['test'], r['nonfinite'], r['n_loop'], r['n_all']))
        L.append('  share of |g_R|^2 on loop-affected parameters: median %.3f' % np.median([p['share_loop'] for p in pr]))
        for k in (1, 2):
            q = [p['k%d' % k] for p in pr if 'k%d' % k in p]
            med = lambda key: float(np.median([x[key] for x in q]))
            ra, rl = med('r_all'), med('r_loop')
            L.append('  k=%d: r_all %.3f  r_loop %.3f  r_rest %.4f  cos_all %.3f  cos_loop %.3f  max rest_diff %.2e' % (
                k, ra, rl, med('r_rest'), med('cos_all'), med('cos_loop'), max(x['rest_diff'] for x in q)))
            L.append('        step vs full backprop -- truncation: loop %.3f, other 1.000 | turn-only: loop %.3f, other %.3f'
                     ' | shrink-only: loop %.3f, other %.3f' % (rl, rl / ra, 1 / ra, ra, ra))
            if q and 'r_group' in q[0]:
                L.append('        by group (|g_k|/|g_R|; share of |g_R|^2): ' + '; '.join(
                    '%s %.3f (%.3f)' % (gk_, float(np.median([x['r_group'][gk_] for x in q])),
                                        float(np.median([p['share_group'][gk_] for p in pr])))
                    for gk_ in q[0]['r_group']))
    return '\n'.join(L)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int); ap.add_argument('--summary', nargs='+')
    A = ap.parse_args()
    if A.summary:
        rs = [json.load(open(f)) for d in A.summary for f in sorted(glob.glob(os.path.join(d, '**', 'v244_seed*.json'),
                                                                               recursive=True))]
        txt = summary_of(rs); print(txt)
        open(os.path.join(os.environ.get('OUT_DIR', '.'), 'v244_summary.txt'), 'w').write(txt + '\n')
    else:
        run(A.seed)
