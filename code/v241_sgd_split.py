"""v241 (plan in docs/prereg_v241_sgd_split.md): S22's shrink-versus-turn split under SGD instead of AdamW.

Under AdamW the split came out almost by construction (Adam ignores a constant rescale of the gradient), and clipping
at norm 1 hides most of a shrink too (the full gradient is cut to 1 while the truncated one mostly is not). Under SGD
with clipping only as a spike guard, a shorter gradient is a shorter step, so a shrink can cost. Parcae-140m, R = 32.

Pipeline: S8's (v174, imported unchanged): GSM8K, 600 steps, batch 8 x 512, warm-up + cosine, fp16 autocast with manual
loss scaling (a step with a non-finite g_R or g_k is skipped and the scale halved). Every step of a two-gradient arm
computes g_R and g_k on the same batch and state draw (v174.grad_pair), builds the update gradient with v239's
constructions (imported unchanged), clips it to the setting's norm and steps SGD (momentum 0.9, no weight decay):
  trunc  g = g_k
  turn   g = g_k * |g_R| / |g_k|
  shrink g = g_R * |g_k| / |g_R|
FULL is g = g_R through the same loop (only g_R computed). FLOOR is v170's floor: v163 train_cell, k 0, with
torch.optim.AdamW replaced by v170's SGD(momentum 0.9) for that call only, and v162's CLIP set to the setting's norm.

Settings:
  G  (scored)   clip at CLIP_G = 5 (a spike guard), SGD rates LR_G / LR_G_FLOOR chosen by the pilot below
  C1 (dropped at the freeze; code kept) clip at 1, S7's (v170) Parcae SGD rates 0.1 / 0.3, k = 1 only
Cells: G: full, floor, trunc1, turn1, shrink1, trunc2, turn2, shrink2 (scored); fullm1, fullm2 (reported): FULL at
       LR_G * R_MATCH[k], a constant step shrink matched to the pilot's median |g_k| / |g_R|, so that a shrink cost
       can be told apart from a smaller step.
       C1: full_c1, floor_c1, trunc1_c1, turn1_c1, shrink1_c1 (reported).

Logged per run: Mbar = mean over steps >= 100 of cos(arm's momentum buffer, a shadow SGD momentum buffer fed the clipped
g_R) (FULL must give 1); step = mean |clipped update| / |clipped g_R|; frac_clip = share of steps whose update was
clipped; spikes = steps with |g_R| > CLIP_G.

  python v241_sgd_split.py --selftest
  python v241_sgd_split.py --pilot 'full@0.03,full@0.1,full@0.3,floor@0.1,floor@0.3,floor@1.0' --tag P
  python v241_sgd_split.py --pick DIR [DIR ...]        (applies the pilot rule; prints LR_G, LR_G_FLOOR)
  python v241_sgd_split.py --runs 'trunc1@2411101,full@2411101,...' --tag T
  python v241_sgd_split.py --analyze DIR [DIR ...]
Env: TINY=1, OUT_DIR, PARCAE_SRC (as v162).
"""
import argparse, glob, hashlib, json, math, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))

MOM = 0.9
CLIP_G, CLIP_C1 = 5.0, 1.0
LR_G, LR_G_FLOOR = 0.1, 0.1                        # from the pilot (--pick, two rounds), before the freeze
LR_C1, LR_C1_FLOOR = 0.1, 0.3                      # S7 (v170) Parcae SGD rates at clip 1
R_MATCH = {1: 0.702, 2: 0.836}                     # from the pilot (--pick), before the freeze
PILOT_SEED = 2419901
PILOT_GRID_FULL, PILOT_GRID_FLOOR = (0.03, 0.1, 0.3), (0.1, 0.3, 1.0)
PROBE_EVERY = 20                                   # pilot full runs: |g_1| / |g_R| and |g_2| / |g_R| every 20 steps
KS = (1, 2)
ARMS2 = ('trunc', 'turn', 'shrink')
SCORED = ('full', 'floor') + tuple('%s%d' % (a, k) for k in KS for a in ARMS2)
REPORTED_G = ('fullm1', 'fullm2')
C1 = ('full_c1', 'floor_c1', 'trunc1_c1', 'turn1_c1', 'shrink1_c1')
CELLS = SCORED + REPORTED_G                        # C1 dropped at the freeze (prereg §1); its code is kept
SEEDS = (2411101, 2411102, 2411103, 2411104, 2411105, 2411106)   # all run; first 5 valid in this order are scored
EXTRA_BACKUPS = (2411107, 2411108)                               # run only if fewer than 5 of SEEDS are valid
N_USE, N_MIN = 5, 4
MAX_NONFINITE = 10
GATE_NATS = 0.03


def m_k(phi_trunc):
    return max(0.015, 0.2 * phi_trunc)                   # v239's margin, unchanged


def label(t, u, s):
    """v239's per-k rule, unchanged."""
    mk = m_k(t)
    return 'TURN' if u >= t - mk and s <= mk else 'SHRINK' if s >= 0.5 * t else 'PARTIAL'


def sha(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(open(p, 'rb').read().replace(b'\r\n', b'\n'))
    return h.hexdigest()[:12]


FILES = ('v241_sgd_split.py', 'v239_shrink_turn.py', 'v170_sgd.py', 'v174_mix_driver.py', 'v163_parcae_damped_trunc.py',
         'v162_parcae_trunc_train.py')


def driver_sha():
    return sha([os.path.join(HERE, f) for f in FILES])


def parse(cell):
    """-> (arm, k, lr, clip)."""
    c1 = cell.endswith('_c1'); c = cell[:-3] if c1 else cell
    clip = CLIP_C1 if c1 else CLIP_G
    lr, lr_floor = (LR_C1, LR_C1_FLOOR) if c1 else (LR_G, LR_G_FLOOR)
    if c == 'full':
        return 'full', 0, lr, clip
    if c in ('fullm1', 'fullm2') and not c1:
        r = R_MATCH[int(c[-1])]
        return 'full', 0, (lr * r if lr and r else None), clip
    if c == 'floor':
        return 'floor', 0, lr_floor, clip
    for a in ARMS2:
        if c.startswith(a) and c[len(a):].isdigit():
            return a, int(c[len(a):]), lr, clip
    raise ValueError(cell)


def clipc(g, c):
    n = float(g.norm()); return g * min(1.0, c / (n + 1e-6)), n


# ------------------------------------------------------------------------------------------------ training
def train_sgd(M, data, arm, k, lr, clip, seed, steps=None, evaluate=False, trace=False, probe=False):
    """v174.train_mix's loop with SGD(momentum 0.9) in place of AdamW, v239's constructions in place of the mix, and the
    setting's clip norm. probe (FULL only, pilot): every PROBE_EVERY steps also compute g_1 and g_2 for their norms (not
    used for training)."""
    import numpy as np
    import torch
    from v239_shrink_turn import build
    v = M.v
    assert lr is not None, 'learning rate not set (run the pilot and --pick first)'
    steps = steps or v.STEPS
    _, m = v.load_model(); m.train(); params = [p for p in m.parameters()]
    g = np.random.default_rng(seed)
    chunks = torch.cat([v.pack(data['train'], g.permutation(len(data['train']))) for _ in range(4)])
    torch.manual_seed(seed)
    opt = torch.optim.SGD(params, lr=lr, momentum=MOM, weight_decay=0.0)
    sched = lambda s: min(1.0, (s + 1) / v.WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / v.STEPS)))
    S = 65536.0 if v.AMP else 1.0
    sbuf = None; Mc, ratio, nR, clipped, spikes, nonfinite, tr, pr = [], [], [], 0, 0, 0, [], []
    t0 = time.time()
    for s in range(steps):
        for gp in opt.param_groups:
            gp['lr'] = lr * sched(s)
        ids = chunks[s * v.BATCH:(s + 1) * v.BATCH].to(v.DEV)
        if arm == 'full':
            gR = M.grad_one(m, params, ids, M.R, S, v.AMP, seed=seed * 1000 + s); gk = gR
        else:
            gR, gk = M.grad_pair(m, params, ids, k, S, v.AMP, seed=seed * 1000 + s)
        if not (torch.isfinite(gR).all() and torch.isfinite(gk).all()):
            nonfinite += 1; S /= 2; continue
        if probe and s % PROBE_EVERY == 0:
            q = {kk: float(M.grad_one(m, params, ids, kk, S, v.AMP, seed=seed * 1000 + s).norm()) for kk in KS}
            pr.append(dict(step=s, nR=float(gR.norm()), **{'n%d' % kk: q[kk] for kk in KS}))
        gu, n_pre = clipc(gR if arm == 'full' else build(gR, gk, arm), clip)
        gRc, _ = clipc(gR, clip)
        clipped += int(n_pre > clip); spikes += int(float(gR.norm()) > CLIP_G); nR.append(float(gR.norm()))
        ratio.append(float(gu.norm() / (gRc.norm() + 1e-30)))
        if trace:
            tr.append(dict(nR=float(gR.norm()), nk=float(gk.norm()), nu=float(gu.norm()), cos_uk=M.cos(gu, gk),
                           cos_uR=M.cos(gu, gR)))
        M.assign(params, gu); opt.step()
        sbuf = gRc.clone() if sbuf is None else sbuf.mul_(MOM).add_(gRc)       # torch SGD momentum, dampening 0
        buf = torch.cat([opt.state[p]['momentum_buffer'].float().flatten() for p in params])
        Mc.append(M.cos(buf, sbuf))
        if s % 100 == 0:
            M.log('  %s k=%d lr %g clip %g step %d: M %.4f step %.3f |gR| %.2f |gk| %.2f S %g (%ds)' % (
                arm, k, lr, clip, s, Mc[-1], ratio[-1], float(gR.norm()), float(gk.norm()), S, time.time() - t0))
    w = [x for i, x in enumerate(Mc) if i >= M.A_FROM] or Mc
    out = dict(k=k, lr=lr, clip=clip, seed=seed, steps=steps, optimizer=type(opt).__name__,
               Mbar=float(np.mean(w)) if w else float('nan'), step=float(np.mean(ratio)) if ratio else float('nan'),
               frac_clip=clipped / max(1, steps - nonfinite), spikes=spikes,
               nR_median=float(np.median(nR)) if nR else float('nan'), nonfinite=nonfinite)
    if trace:
        out['trace'] = tr
    if probe:
        out['probe'] = pr
    if evaluate:
        out.update(val=M.d.eval_loss(m, data['val'], 1.0), test=M.d.eval_loss(m, data['test'], 1.0))
    out['s'] = round(time.time() - t0)
    return out


ORIG_ADAMW = None


def train_floor(M, data, lr, clip, seed, steps=None):
    """v170's floor: v163 train_cell (k 0) with AdamW -> v170's SGD(momentum 0.9) and v162's CLIP -> clip, for this
    call only."""
    global ORIG_ADAMW
    import torch
    assert lr is not None, 'floor learning rate not set (run the pilot and --pick first)'
    if ORIG_ADAMW is None:
        ORIG_ADAMW = torch.optim.AdamW
    import v170_sgd                                          # its import patches torch.optim.AdamW; restored below
    torch.optim.AdamW = v170_sgd.sgd_momentum
    old_steps, old_clip = M.v.STEPS, M.v.CLIP
    try:
        if steps:
            M.v.STEPS = steps
        M.v.CLIP = clip
        r = M.d.train_cell(data, 1.0, 0, lr, seed)
    finally:
        torch.optim.AdamW = ORIG_ADAMW
        M.v.STEPS, M.v.CLIP = old_steps, old_clip
    return dict(k=0, lr=lr, clip=clip, seed=seed, steps=r['steps'], optimizer='SGD (v170 patch)', val=r['val'],
                test=r['test'], nonfinite=r['nonfinite_steps'], s=r['s'])


def run_one(M, data, cell, seed):
    arm, k, lr, clip = parse(cell)
    r = (train_floor(M, data, lr, clip, seed) if arm == 'floor' else
         train_sgd(M, data, arm, k, lr, clip, seed, evaluate=True))
    r.update(arm=cell, k=k, seed=seed, v241_sha=driver_sha())
    return r


# ------------------------------------------------------------------------------------------------ selftest
def selftest(M, data):
    import torch
    from v239_shrink_turn import build
    v = M.v
    _, m = v.load_model(); m.train()
    M.selftest(m, data)                                      # v163 selftest + v174 mix construction
    params = [p for p in m.parameters()]; ids = data['val'][:1].to(v.DEV)
    for k in KS:
        gR, gk = M.grad_pair(m, params, ids, k, 1.0, False)
        t, u, s = build(gR, gk, 'trunc'), build(gR, gk, 'turn'), build(gR, gk, 'shrink')
        assert torch.equal(t, gk)
        assert abs(M.cos(u, gk) - 1) < 1e-5 and abs(float(u.norm() / gR.norm()) - 1) < 1e-5
        assert abs(M.cos(s, gR) - 1) < 1e-5 and abs(float(s.norm() / gk.norm()) - 1) < 1e-5
        M.log('v241 selftest k=%d: |g_k|/|g_R| %.3f cos %.3f; constructions OK' % (k, float(gk.norm() / gR.norm()), M.cos(gk, gR)))
        for c in (CLIP_G, CLIP_C1):                              # below the clip each arm keeps its own length
            a, b = gR * (0.5 * c / float(gR.norm())), gk * (0.5 * c / float(gR.norm()))
            for arm, want in (('trunc', float(b.norm())), ('turn', 0.5 * c), ('shrink', float(b.norm()))):
                got = float(clipc(build(a, b, arm), c)[0].norm())
                assert abs(got - want) < 1e-5 * c, (arm, c, got, want)
    M.log('v241 selftest: below the clip, trunc and shrink keep |g_k| and turn keeps |g_R| (clip %g and %g)' % (CLIP_G, CLIP_C1))
    del m
    n = 3
    lr_t = LR_G or 0.1
    for clip in (CLIP_G, CLIP_C1):
        r = train_sgd(M, data, 'full', 0, lr_t, clip, PILOT_SEED, steps=n)
        assert r['optimizer'] == 'SGD' and abs(r['Mbar'] - 1) < 1e-4 and abs(r['step'] - 1) < 1e-5, r
        M.log('v241 selftest full clip %g: SGD, Mbar %.6f (must be 1), step %.4f (must be 1)' % (clip, r['Mbar'], r['step']))
        for arm in ARMS2:
            r = train_sgd(M, data, arm, 1, lr_t, clip, PILOT_SEED, steps=n, trace=True)
            assert r['optimizer'] == 'SGD'
            for t in r['trace']:                                 # the update applied is the arm's construction, clipped
                want_n = min({'trunc': t['nk'], 'turn': t['nR'], 'shrink': t['nk']}[arm], clip)
                assert abs(t['nu'] - want_n) < 1e-3 * max(1.0, want_n), (arm, clip, t)
                assert abs((t['cos_uR'] if arm == 'shrink' else t['cos_uk']) - 1) < 1e-4, (arm, clip, t)
            M.log('v241 selftest %s1 clip %g: SGD, %d steps, Mbar %.4f, step %.3f, nonfinite %d; norms and directions OK' % (
                arm, clip, n, r['Mbar'], r['step'], r['nonfinite']))
    r = train_floor(M, data, LR_G_FLOOR or 0.3, CLIP_G, PILOT_SEED, steps=n)
    assert math.isfinite(r['test']) and torch.optim.AdamW is ORIG_ADAMW and M.v.CLIP == 1.0, 'patches not restored'
    M.log('v241 selftest floor (v170 SGD patch and clip %g, both restored after): test %.4f' % (CLIP_G, r['test']))
    M.log('v241 selftest OK (driver %s; LR_G %s, LR_G_FLOOR %s)' % (driver_sha(), LR_G, LR_G_FLOOR))


# ------------------------------------------------------------------------------------------------ pilot (controls only)
def pilot(M, data, items, tag):
    """FULL and FLOOR only, clip CLIP_G, seed PILOT_SEED. No truncated arm is trained."""
    fn = os.path.join(M.v.OUT, 'v241_pilot_%s.jsonl' % tag)
    for it in items.split(','):
        arm, lr = it.split('@'); lr = float(lr)
        assert arm in ('full', 'floor'), 'the pilot runs controls only'
        r = (train_sgd(M, data, 'full', 0, lr, CLIP_G, PILOT_SEED, evaluate=True, probe=True) if arm == 'full' else
             train_floor(M, data, lr, CLIP_G, PILOT_SEED))
        r.update(arm=arm, v241_sha=driver_sha())
        M.log('pilot %s lr %g: val %.4f test %.4f nonfinite %d%s (%ds)' % (
            arm, lr, r['val'], r['test'], r['nonfinite'],
            '; spikes %d, median |gR| %.2f' % (r['spikes'], r['nR_median']) if arm == 'full' else '', r['s']))
        open(fn, 'a').write(json.dumps(r) + '\n')


def pick(dirs):
    """Pilot rule (fixed in the prereg draft before the pilot): per control, the grid rate with the lowest val loss
    among stable runs (finite val and test, <= 10 non-finite steps). If that is a grid end, one more 3x step on that
    side is run (a second pilot round) and the rule re-applied once; the result stands even if it is again an end."""
    import numpy as np
    rows = []
    for d in dirs:
        for f in glob.glob(os.path.join(d, '**', 'v241_pilot_*.jsonl'), recursive=True):
            rows += [json.loads(x) for x in open(f) if x.strip()]
    L = []; P = L.append; out = {}; final = True
    for arm, grid in (('full', PILOT_GRID_FULL), ('floor', PILOT_GRID_FLOOR)):
        rs = {r['lr']: r for r in rows if r['arm'] == arm}
        stable = {lr: r for lr, r in rs.items() if math.isfinite(r['val']) and math.isfinite(r['test'])
                  and r['nonfinite'] <= MAX_NONFINITE}
        for lr in sorted(rs):
            r = rs[lr]
            P('  %s lr %g: val %.4f test %.4f nonfinite %d%s%s' % (arm, lr, r['val'], r['test'], r['nonfinite'],
                '' if lr in stable else ' UNSTABLE', '; spikes %d, median |gR| %.2f' % (r['spikes'], r['nR_median'])
                if arm == 'full' else ''))
        if not stable:
            P('%s: no stable run' % arm); out[arm] = None; continue
        best = min(stable, key=lambda lr: stable[lr]['val'])
        lo, hi = min(rs), max(rs)
        edge = best in (lo, hi) and not (best == lo and lo < min(grid)) and not (best == hi and hi > max(grid))
        out[arm] = best; final = final and not edge
        P('%s: best %g%s' % (arm, best, ' (grid end: run %g next, once)' % (best / 3 if best == lo else best * 3) if edge else ''))
    if out.get('full') and out.get('floor'):
        f, fl = {r['lr']: r for r in rows if r['arm'] == 'full'}[out['full']], {r['lr']: r for r in rows if r['arm'] == 'floor'}[out['floor']]
        P('pilot floor gap (test) %.4f nats at full %g / floor %g' % (fl['test'] - f['test'], out['full'], out['floor']))
        pr = [p for p in f.get('probe', []) if p['step'] >= 100]
        if pr:
            P('pilot shrink sizes (full run, steps >= 100): median |g_1|/|g_R| %.3f, |g_2|/|g_R| %.3f; median |g_R| %.2f; '
              'share of probes with |g_1| < %g: %.2f' % (
                float(np.median([p['n1'] / p['nR'] for p in pr])), float(np.median([p['n2'] / p['nR'] for p in pr])),
                float(np.median([p['nR'] for p in pr])), CLIP_G, float(np.mean([p['n1'] < CLIP_G for p in pr]))))
    if out.get('full'):
        pa = {r['lr']: r for r in rows if r['arm'] == 'full'}[out['full']].get('probe', [])
        pr = [p for p in pa if p['step'] >= 100] or pa          # all probes only on TINY (runs shorter than 100 steps)
        rm = {k: round(float(np.median([min(p['n%d' % k], CLIP_G) / min(p['nR'], CLIP_G) for p in pr])), 3) for k in KS} if pr else None
        P('R_MATCH (median clipped |g_k| / clipped |g_R|, steps >= 100): %s' % rm)
    P('PICK%s: LR_G %s, LR_G_FLOOR %s' % ('' if final else ' (PROVISIONAL: run the grid-end extension first)',
                                          out.get('full'), out.get('floor')))
    return '\n'.join(L)


# ------------------------------------------------------------------------------------------------ analysis
def bad(r):
    return r['nonfinite'] > MAX_NONFINITE or not math.isfinite(r['test'])


def costs(R, cells, base, seeds, P, note=''):
    import numpy as np
    full, floor = base
    phi = {}
    for c in cells:
        ss = [s for s in seeds if (c, s) in R and math.isfinite(R[(c, s)]['test'])]
        ph = [(R[(c, s)]['test'] - R[(full, s)]['test']) / (R[(floor, s)]['test'] - R[(full, s)]['test']) for s in ss]
        phi[c] = float(np.mean(ph)) if ph else float('nan')
        avg = lambda key: float(np.mean([R[(c, s)][key] for s in ss])) if ss else float('nan')
        P('  %-11s cost %.4f +- %.4f (n %d; per seed %s; nonfinite %s; Mbar %.3f; step %.3f; clipped %.2f)%s' % (
            c, phi[c], float(np.std(ph, ddof=1) / math.sqrt(len(ph))) if len(ph) > 1 else float('nan'), len(ph),
            ' '.join('%.3f' % x for x in ph), [R[(c, s)]['nonfinite'] for s in ss], avg('Mbar'), avg('step'),
            avg('frac_clip'), note))
    return phi


def gap_gate(R, base, seeds, P, name):
    import numpy as np
    from scipy import stats
    full, floor = base
    gap = [R[(floor, s)]['test'] - R[(full, s)]['test'] for s in seeds]
    mg, sd = float(np.mean(gap)), float(np.std(gap, ddof=1))
    lo = mg - stats.t.ppf(0.95, len(gap) - 1) * sd / math.sqrt(len(gap))
    ok = mg >= GATE_NATS and lo > 0
    P('%s floor gap %.4f nats (90%% lower %.4f; per seed %s) -> gate %s' % (name, mg, lo, ' '.join('%.4f' % x for x in gap),
                                                                           'PASS' if ok else 'FAIL'))
    return ok


def analyze(dirs):
    import numpy as np
    rows = []
    for d in dirs:
        for f in glob.glob(os.path.join(d, '**', 'v241_train_*.jsonl'), recursive=True):
            rows += [json.loads(x) for x in open(f) if x.strip()]
    R = {(r['arm'], r['seed']): r for r in rows}
    L = []; P = L.append
    order = list(SEEDS) + list(EXTRA_BACKUPS)
    ran = [s for s in order if any((c, s) in R for c in SCORED)]
    ok = lambda s, cells: all((c, s) in R and not bad(R[(c, s)]) for c in cells)
    valid = [s for s in ran if ok(s, SCORED)][:N_USE]
    why = {s: [c for c in SCORED if (c, s) not in R or bad(R[(c, s)])] for s in ran if s not in valid}
    P('v241 analysis: %d runs; scored seeds %s; invalid or unused %s' % (len(rows), valid, why))
    if len(valid) < N_MIN:
        P('VERDICT: INCOMPLETE (fewer than %d valid seeds)' % N_MIN); return '\n'.join(L)
    gate = gap_gate(R, ('full', 'floor'), valid, P, 'G (clip %g):' % CLIP_G)
    phi = costs(R, [c for c in SCORED if c not in ('full', 'floor')], ('full', 'floor'), valid, P)
    lab = {}
    for k in KS:
        t, u, s = phi['trunc%d' % k], phi['turn%d' % k], phi['shrink%d' % k]
        lab[k] = label(t, u, s)
        P('k=%d: trunc %.4f turn %.4f shrink %.4f | margin %.4f | turn share %.2f, shrink share %.2f -> %s' % (
            k, t, u, s, m_k(t), u / t if t else float('nan'), s / t if t else float('nan'), lab[k]))
    P('-- reported, not scored --')
    pm = costs(R, list(REPORTED_G), ('full', 'floor'), [s for s in valid if all((c, s) in R for c in REPORTED_G)], P,
               '  [G, FULL at lr x R_MATCH %s]' % R_MATCH)
    for k in KS:
        s, f = phi['shrink%d' % k], pm['fullm%d' % k]
        rd = ('AS-STEP' if abs(s - f) <= m_k(phi['trunc%d' % k]) else 'BEYOND-STEP' if s > f else 'BELOW-STEP')
        P('k=%d shrink %.4f vs FULL at the matched step %.4f (difference %.4f, margin %.4f) -> %s' % (
            k, s, f, s - f, m_k(phi['trunc%d' % k]), rd))
    v1 = [s for s in ran if ok(s, C1)][:N_USE]
    if not any(r['arm'] in C1 for r in rows):
        P('C1: not run (dropped at the freeze)')
    elif len(v1) >= N_MIN:
        g1 = gap_gate(R, ('full_c1', 'floor_c1'), v1, P, 'C1 (clip 1, S7 rates; seeds %s):' % v1)
        p1 = costs(R, ['trunc1_c1', 'turn1_c1', 'shrink1_c1'], ('full_c1', 'floor_c1'), v1, P, '  [C1]')
        P('C1 k=1 label by the same rule: %s%s; trunc1_c1 vs S7 (v170) 0.263' % (
            label(p1['trunc1_c1'], p1['turn1_c1'], p1['shrink1_c1']), '' if g1 else ' (C1 gate failed)'))
    else:
        P('C1: fewer than %d valid seeds' % N_MIN)
    for c in CELLS:
        rs = [r for r in rows if r['arm'] == c and r['seed'] in order]
        if rs:
            P('  unstable %-11s %d of %d runs; mean nonfinite %.2f' % (c, sum(bad(r) for r in rs), len(rs),
                                                                       float(np.mean([r['nonfinite'] for r in rs]))))
    v = ('INVALID (floor gap gate failed)' if not gate else lab[1] if lab[1] == lab[2] else 'MIXED %s' % lab)
    P('VERDICT: %s' % v)
    return '\n'.join(L)


# ------------------------------------------------------------------------------------------------ main
def main():
    global ORIG_ADAMW
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true'); ap.add_argument('--runs', default=None); ap.add_argument('--tag', default='x')
    ap.add_argument('--pilot', default=None); ap.add_argument('--pick', nargs='*'); ap.add_argument('--analyze', nargs='*')
    A = ap.parse_args()
    if A.analyze is not None or A.pick is not None:
        txt = analyze(A.analyze) if A.analyze is not None else pick(A.pick); print(txt)
        open(os.path.join(os.environ.get('OUT_DIR', '.'), 'v241_%s.txt' % ('analysis' if A.analyze is not None else 'pick')),
             'w').write(txt + '\n')
        return
    import torch
    ORIG_ADAMW = torch.optim.AdamW
    import v174_mix_driver as M
    tok, m = M.v.load_model(); data = M.v.gsm8k_chunks(tok); del m
    M.log('v241 driver %s (v174 %s) dev %s amp %s R %d; G clip %g lr %s / floor %s; C1 clip %g lr %g / floor %g; momentum %g' % (
        driver_sha(), M.driver_sha(), M.v.DEV, M.v.AMP, M.R, CLIP_G, LR_G, LR_G_FLOOR, CLIP_C1, LR_C1, LR_C1_FLOOR, MOM))
    if A.selftest:
        selftest(M, data)
    if A.pilot:
        pilot(M, data, A.pilot, A.tag)
    if A.runs:
        fn = os.path.join(M.v.OUT, 'v241_train_%s.jsonl' % A.tag)
        done = {(json.loads(x)['arm'], json.loads(x)['seed']) for x in open(fn) if x.strip()} if os.path.exists(fn) else set()
        for item in A.runs.split(','):
            cell, seed = item.split('@'); seed = int(seed)
            if (cell, seed) in done:
                M.log('skip %s seed %d (done)' % (cell, seed)); continue
            r = run_one(M, data, cell, seed)
            M.log('scored %s seed %d: test %.4f val %.4f Mbar %s step %s nonfinite %d (%ds)' % (
                cell, seed, r['test'], r['val'], '%.4f' % r['Mbar'] if 'Mbar' in r else '-',
                '%.3f' % r['step'] if 'step' in r else '-', r['nonfinite'], r['s']))
            open(fn, 'a').write(json.dumps(r) + '\n'); done.add((cell, seed))


if __name__ == '__main__':
    main()
