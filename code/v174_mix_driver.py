"""v174 driver (prereg docs/prereg_v174_direction_intervention.md, draft 3.1): mix-gradient training on Parcae-140m with
realised Adam-update agreement, plus the frozen-model gate.

Recipe = v162-v166 (the v163 module's damped_loss, a = 1, R = 32; GSM8K 600 steps, batch 8 x 512, AdamW (0.9, 0.95),
eps 1e-8, weight decay 0, warm-up + cosine, global-norm clip 1.0, fp16 autocast with cache off, per-loop checkpoints).

Mix arm (k, lam): every step compute g_R (full backprop) and g_k (last k loops); update gradient
    g = g_R + lam * (g_k - g_R), rescaled to |g_R|, clipped to global norm 1, then AdamW.
lam = 0 is FULL. Loss scaling is manual (fp16): both passes share scale S; a step with a non-finite g_R or g_k is skipped
and S halves (as torch's GradScaler does).
Agreement: a shadow Adam (same betas / eps) is fed the clipped g_R every step and never updates parameters.
    A_t = cos(u_arm, u_shadow), u = m_hat / (sqrt(v_hat) + eps), the Adam step direction before lr scaling,
over ALL parameters (as v171b) and over the "loop-affected" parameters only (those whose frozen g_R != g_1; parameters
downstream of the loop get identical gradients and inflate the all-parameter value). Abar = mean A_t over steps >= 100.

  --selftest                    v163 selftest at R 32 + mix construction + shadow-Adam exactness (A(FULL) == 1)
  --gate                        frozen-model per-arm Adam directions (v171b procedure, 24 clipped batches) for the pilot
                                grid; writes A_f per arm, B_f and error-component cosines for every k1-vs-k2 pair
  --pilot CELLS --tag T         cells 'k:lam' (seed 1749901): logs Abar only; final losses are NOT evaluated (prereg §2)
"""
import argparse, hashlib, json, math, os, time
import numpy as np
import torch

import v162_parcae_trunc_train as v
import v163_parcae_damped_trunc as d

R = d.R
BETAS, EPS = (0.9, 0.95), 1e-8
PILOT_SEED = 1749901
GRID = [(1, 0.25), (1, 0.5), (1, 0.75), (1, 1.0), (2, 0.25), (2, 0.5), (2, 0.75), (2, 1.0)]
A_FROM = 2 if v.TINY else 100


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(v.OUT, 'v174_log.txt'), 'a').write(s + '\n')


def flat_grad(params):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().float().flatten() for p in params])


def grad_pair(m, params, ids, k, scale, amp, seed=1234):
    """(g_R, g_k), unscaled fp32 flat vectors; both passes use the same state-init noise (seed varies per step)."""
    out = []
    for kk in (R, k):
        m.zero_grad(set_to_none=True); torch.manual_seed(seed)
        (d.damped_loss(m, ids, kk, 1.0, amp=amp) * scale).backward()
        out.append(flat_grad(params) / scale)
    m.zero_grad(set_to_none=True)
    return out


def mix(gR, gk, lam):
    g = gR + lam * (gk - gR)
    return g * (gR.norm() / (g.norm() + 1e-30))


def clip1(g):
    n = float(g.norm()); return g * min(1.0, 1.0 / (n + 1e-6)), n


def assign(params, g):
    i = 0
    for p in params:
        n = p.numel(); p.grad = g[i:i + n].view_as(p).to(p.dtype); i += n


def adam_dir(m1, m2, t):
    return (m1 / (1 - BETAS[0] ** t)) / ((m2 / (1 - BETAS[1] ** t)).sqrt() + EPS)


def cos(a, b, mask=None):
    if mask is not None:
        a, b = a[mask], b[mask]
    return float(torch.dot(a.double(), b.double()) / (a.double().norm() * b.double().norm() + 1e-30))


def affected_mask(m, params, data):
    ids = data['val'][:1].to(v.DEV)
    gR, g1 = grad_pair(m, params, ids, 1, 1.0, False)
    return (gR - g1).abs() > 1e-9 * gR.abs().max()


def grad_one(m, params, ids, k, scale, amp, seed):
    m.zero_grad(set_to_none=True); torch.manual_seed(seed)
    (d.damped_loss(m, ids, k, 1.0, amp=amp) * scale).backward()
    g = flat_grad(params) / scale; m.zero_grad(set_to_none=True)
    return g


def unit_noise(n, seed):
    """Random unit direction over all parameters (Gaussian, normalised), reproducible from seed."""
    gen = torch.Generator(device=v.DEV); gen.manual_seed(seed)
    r = torch.randn(n, generator=gen, device=v.DEV)
    return r / r.norm()


def train_mix(data, k, lam, seed, steps=None, mode='mix', evaluate=False):
    """mode 'mix': g = mix(g_R, g_k, lam). mode 'tiltfix' / 'tiltfresh': g = mix(g_R, |g_R| r, lam) with r a random unit
    direction fixed for the run (tiltfix) or drawn fresh each step (tiltfresh); g_k is not computed. lam 0 = FULL (g_k
    not computed). evaluate: val/test loss at the end (scored runs)."""
    steps = steps or v.STEPS
    _, m = v.load_model(); m.train(); params = [p for p in m.parameters()]
    mask = affected_mask(m, params, data)
    n_par = sum(p.numel() for p in params)
    r_fix = unit_noise(n_par, seed * 7 + 1) if mode == 'tiltfix' else None
    g = np.random.default_rng(seed)
    chunks = torch.cat([v.pack(data['train'], g.permutation(len(data['train']))) for _ in range(4)])
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(params, lr=3e-4, betas=BETAS, eps=EPS, weight_decay=0.0)
    sched = lambda s: min(1.0, (s + 1) / v.WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / v.STEPS)))
    sm1 = sm2 = None; ts = 0; S = 65536.0 if v.AMP else 1.0
    A, A_loop, both_clipped, nonfinite = [], [], 0, 0; t0 = time.time()
    for s in range(steps):
        for gp in opt.param_groups:
            gp['lr'] = 3e-4 * sched(s)
        ids = chunks[s * v.BATCH:(s + 1) * v.BATCH].to(v.DEV)
        if mode == 'mix' and lam != 0:
            gR, gk = grad_pair(m, params, ids, k, S, v.AMP, seed=seed * 1000 + s)
        else:
            gR = grad_one(m, params, ids, R, S, v.AMP, seed=seed * 1000 + s)
            if mode == 'mix':
                gk = gR
            else:
                gk = (r_fix if mode == 'tiltfix' else unit_noise(n_par, seed * 100003 + s)) * gR.norm()
        if not (torch.isfinite(gR).all() and torch.isfinite(gk).all()):
            nonfinite += 1; S /= 2; continue
        both_clipped += int(gR.norm() > 1 and gk.norm() > 1)
        gm, _ = clip1(mix(gR, gk, lam)); gRc, _ = clip1(gR)
        assign(params, gm); opt.step()
        ts += 1
        sm1 = (1 - BETAS[0]) * gRc if sm1 is None else sm1.mul_(BETAS[0]).add_(gRc, alpha=1 - BETAS[0])
        sm2 = (1 - BETAS[1]) * gRc * gRc if sm2 is None else sm2.mul_(BETAS[1]).addcmul_(gRc, gRc, value=1 - BETAS[1])
        st = [opt.state[p] for p in params]
        u_arm = adam_dir(torch.cat([x['exp_avg'].float().flatten() for x in st]),
                         torch.cat([x['exp_avg_sq'].float().flatten() for x in st]), float(st[0]['step']))
        u_sh = adam_dir(sm1, sm2, ts)
        A.append(cos(u_arm, u_sh)); A_loop.append(cos(u_arm, u_sh, mask))
        if s % 100 == 0:
            log('  k=%d lam=%.3f step %d: A %.4f A_loop %.4f |gR| %.2f |gk| %.2f S %g (%ds)' % (
                k, lam, s, A[-1], A_loop[-1], float(gR.norm()), float(gk.norm()), S, time.time() - t0))
    w = [x for i, x in enumerate(A) if i >= A_FROM] or A; wl = [x for i, x in enumerate(A_loop) if i >= A_FROM] or A_loop
    out = dict(mode=mode, k=k, lam=lam, seed=seed, steps=steps, Abar=float(np.mean(w)), Abar_loop=float(np.mean(wl)),
               frac_both_clipped=both_clipped / max(1, steps - nonfinite), nonfinite=nonfinite, driver_sha=driver_sha())
    if evaluate:
        out.update(val=d.eval_loss(m, data['val'], 1.0), test=d.eval_loss(m, data['test'], 1.0))
    out['s'] = round(time.time() - t0)
    return out


def driver_sha():
    """v174 sha: this file + the v163 driver sha (which covers v162)."""
    h = hashlib.sha256(open(os.path.abspath(__file__), 'rb').read().replace(b'\r\n', b'\n'))
    h.update(d.driver_sha().encode())
    return h.hexdigest()[:12]


def parse_cell(c):
    """'full' | 'floor' | 'k:lam' | 'tiltfix:lam' | 'tiltfresh:lam' -> (arm name, mode, k, lam)."""
    if c in ('full', 'floor', 'full163'):
        return c, c, 0, 0.0
    a, lam = c.split(':'); lam = float(lam)
    if a in ('tiltfix', 'tiltfresh'):
        return '%s_l%.3f' % (a, lam), a, 0, lam
    return 'k%d_l%.3f' % (int(a), lam), 'mix', int(a), lam


def run_scored(data, cell, seed):
    """One scored run. FULL = mix arm at lam 0 (same manual-scaling pipeline as the pair arms; A must be 1).
    FLOOR = the S2/S5 floor exactly (v163 train_cell, k 0, lr 9e-4)."""
    name, mode, k, lam = parse_cell(cell)
    if mode == 'full163':                     # reported pipeline check: FULL through the original v163 train_cell
        r = d.train_cell(data, 1.0, R, 3e-4, seed)
        return dict(arm='full163', mode='full163', k=R, lam=0.0, seed=seed, steps=r['steps'], val=r['val'], test=r['test'],
                    nonfinite=r['nonfinite_steps'], s=r['s'], driver_sha=driver_sha())
    if mode == 'floor':
        r = d.train_cell(data, 1.0, 0, 9e-4, seed)
        r = dict(mode='floor', k=0, lam=0.0, seed=seed, steps=r['steps'], val=r['val'], test=r['test'],
                 nonfinite=r['nonfinite_steps'], s=r['s'], driver_sha=driver_sha())
    else:
        r = train_mix(data, 1 if mode == 'full' else k, lam, seed, mode='mix' if mode == 'full' else mode, evaluate=True)
        r['mode'] = 'full' if mode == 'full' else r['mode']
    r['arm'] = name
    return r


def gate(m, data, B=None, grid=None):
    """Frozen-model Adam directions per arm (v171b procedure) and pairwise k1-vs-k2 gate quantities."""
    B = B or (2 if v.TINY else 24); params = [p for p in m.parameters()]; m.eval()
    mask = affected_mask(m, params, data)
    train = v.pack(data['train'], np.random.default_rng(171).permutation(len(data['train'])))
    grid = grid or GRID; ks = sorted({k for k, _ in grid})
    arms = [('full', 0, 0.0)] + [('k%d_l%.2f' % (k, lam), k, lam) for k, lam in grid]
    s1 = {a[0]: None for a in arms}; s2 = dict(s1)
    for i in range(B):
        ids = train[i * v.BATCH:(i + 1) * v.BATCH].to(v.DEV)
        gk = {}
        for kk in ks:
            gR, gk[kk] = grad_pair(m, params, ids, kk, 1.0, False, seed=4242 + i)
        for name, k, lam in arms:
            gg, _ = clip1(gR if k == 0 else mix(gR, gk[k], lam)); gg = gg.cpu()
            s1[name] = gg.clone() if s1[name] is None else s1[name].add_(gg)
            s2[name] = gg * gg if s2[name] is None else s2[name].addcmul_(gg, gg)
    maskc = mask.cpu()
    u = {n: (s1[n] / B) / ((s2[n] / B).sqrt() + EPS) for n in s1}
    out = dict(A_f={n: cos(u[n], u['full']) for n in u}, A_f_loop={n: cos(u[n], u['full'], maskc) for n in u}, pairs={})
    lo, hi = ks[0], ks[-1]
    for k1, l1 in [g for g in grid if g[0] == lo]:
        for k2, l2 in [g for g in grid if g[0] == hi]:
            a, b = 'k%d_l%.2f' % (k1, l1), 'k%d_l%.2f' % (k2, l2)
            out['pairs']['%s|%s' % (a, b)] = dict(B_f=cos(u[a], u[b]), err_cos=cos(u[a] - u['full'], u[b] - u['full']),
                                                 B_f_loop=cos(u[a], u[b], maskc))
    m.train()
    return out


def selftest(m, data):
    d.selftest(m, data)                                         # forward = v162, checkpoint = plain, fp16 vs fp32
    params = [p for p in m.parameters()]; ids = data['val'][:1].to(v.DEV)
    gR, g1 = grad_pair(m, params, ids, 1, 1.0, False)
    assert torch.allclose(mix(gR, g1, 0.0), gR)
    e = mix(gR, g1, 1.0); assert abs(cos(e, g1) - 1) < 1e-6 and abs(float(e.norm() / gR.norm()) - 1) < 1e-5
    log('selftest mix construction OK (lam 0 = g_R; lam 1 = g_1 direction at |g_R|)')
    mask = affected_mask(m, params, data)
    log('selftest loop-affected parameters: %d of %d' % (int(mask.sum()), mask.numel()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true'); ap.add_argument('--gate', action='store_true')
    ap.add_argument('--gate-grid', default=None); ap.add_argument('--pilot', default=None); ap.add_argument('--tag', default='x'); ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--train', default=None, help="scored cells, e.g. 'full,floor,1:0.5,4:0.94'"); ap.add_argument('--seeds', default='')
    ap.add_argument('--runs', default=None, help="explicit scored runs 'cell@seed,...', e.g. 'full@1741101,4:0.94@1741102'")
    A = ap.parse_args()
    tok, m = v.load_model(); data = v.gsm8k_chunks(tok)
    log('v174 driver %s (v163 %s) dev %s amp %s R %d' % (driver_sha(), d.driver_sha(), v.DEV, v.AMP, R))
    if A.selftest:
        selftest(m, data)
        r = train_mix(data, 1, 0.0, PILOT_SEED, steps=4 if not v.TINY else 3)    # FULL arm: A must be 1
        log('selftest FULL arm agreement %.6f (must be 1)' % r['Abar']); assert abs(r['Abar'] - 1) < 1e-4
        log('selftest v174 OK')
    if A.gate:
        grid = [(int(c.split(':')[0]), float(c.split(':')[1])) for c in A.gate_grid.split(',')] if A.gate_grid else None
        out = gate(m, data, grid=grid); out['driver_sha'] = d.driver_sha()
        json.dump(out, open(os.path.join(v.OUT, 'v174_gate_%s.json' % A.tag), 'w'), indent=1)
        log('gate A_f %s' % {n: round(x, 4) for n, x in out['A_f'].items()})
    if A.pilot:
        del m; torch.cuda.empty_cache() if v.DEV == 'cuda' else None
        for cell in A.pilot.split(','):
            name, mode, k, lam = parse_cell(cell); r = train_mix(data, k, lam, PILOT_SEED, A.steps, mode=mode)
            log('pilot %s: Abar %.4f Abar_loop %.4f both-clipped %.3f nonfinite %d (%ds)' % (
                name, r['Abar'], r['Abar_loop'], r['frac_both_clipped'], r['nonfinite'], r['s']))
            open(os.path.join(v.OUT, 'v174_pilot_%s.jsonl' % A.tag), 'a').write(json.dumps(r) + '\n')
    if A.train or A.runs:
        # scored runs: --runs list, or every (cell, seed) in --train x --seeds; resumable (runs already in the output
        # file are skipped)
        if not A.pilot:
            del m; torch.cuda.empty_cache() if v.DEV == 'cuda' else None
        fn = os.path.join(v.OUT, 'v174_train_%s.jsonl' % A.tag)
        done = set()
        if os.path.exists(fn):
            done = {(json.loads(x)['arm'], json.loads(x)['seed']) for x in open(fn) if x.strip()}
        todo = ([(c.split('@')[0], int(c.split('@')[1])) for c in A.runs.split(',')] if A.runs else
                [(c, int(sd)) for sd in A.seeds.split(',') for c in A.train.split(',')])
        for cell, seed in todo:
            if (parse_cell(cell)[0], seed) in done:
                log('skip %s seed %d (already done)' % (parse_cell(cell)[0], seed)); continue
            r = run_scored(data, cell, seed)
            log('scored %s seed %d: test %.4f val %.4f Abar %s nonfinite %d (%ds)' % (
                r['arm'], seed, r['test'], r['val'], '%.4f' % r['Abar'] if 'Abar' in r else '-', r['nonfinite'], r['s']))
            open(fn, 'a').write(json.dumps(r) + '\n'); done.add((r['arm'], seed))


if __name__ == '__main__':
    main()
