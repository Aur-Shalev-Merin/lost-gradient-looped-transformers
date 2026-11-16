"""v239 (plan in docs/prereg_v239_shrink_turn.md, frozen before any scored run): which part of a truncated gradient
costs training on Parcae-140m, its shrink or its turn?

Recipe and pipeline = v174 (S8), imported unchanged: Parcae-140m at R = 32, GSM8K, 600 steps, batch 8 x 512, AdamW
(0.9, 0.95), lr 3e-4 (full backprop's, as every truncated cell chose in S2), warm-up + cosine, global-norm clip 1.0,
fp16 autocast with manual loss scaling. Every step of a two-gradient arm computes g_R (backprop through all 32 loops)
and g_k (last k loops) on the same batch and state draw, builds the update gradient below, clips it to norm 1 and
steps AdamW:
  trunc  g = g_k                         (plain truncation, through the same pipeline)
  turn   g = g_k * |g_R| / |g_k|         (truncation's direction at the full gradient's length; v174's mix at lam 1)
  shrink g = g_R * |g_k| / |g_R|         (the full gradient's direction at the truncated gradient's length)
FULL is v174's mix arm at lam 0 (g = g_R); FLOOR is the S2/S5 no-loop-gradient baseline (v163 train_cell, k 0, lr 9e-4),
both through v174.run_scored unchanged.

  python v239_shrink_turn.py --selftest                       (real model on GPU in the kernels; TINY=1 on CPU)
  python v239_shrink_turn.py --runs 'trunc1@2391101,full@2391101,...' --tag T
  python v239_shrink_turn.py --analyze DIR [DIR ...]
Env: TINY=1, OUT_DIR, PARCAE_SRC (as v162).
"""
import argparse, glob, hashlib, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
KS = (1, 2, 4)
ARMS2 = ('trunc', 'turn', 'shrink')                   # two-gradient arms, one per k
SEEDS = (2391101, 2391102, 2391103, 2391104, 2391105, 2391106)   # first 5 valid in this order are scored
EXTRA_BACKUPS = (2391107, 2391108)                    # run only if fewer than 5 of SEEDS are valid
N_USE, N_MIN = 5, 4
MAX_NONFINITE = 10                                    # exclusion rule (App. B): full run with > 10 unstable steps
GATE_NATS = 0.03                                      # floor gap gate, as S11 / S18
DECIDE_KS = (1, 2)                                    # k = 4 is reported, not scored (its cost is ~0.02)


def m_k(phi_trunc):
    """Decision margin at one k: max(0.015, 0.2 * the truncated arm's cost)."""
    return max(0.015, 0.2 * phi_trunc)


def sha(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(open(p, 'rb').read().replace(b'\r\n', b'\n'))
    return h.hexdigest()[:12]


HERE = os.path.dirname(os.path.abspath(__file__))


def driver_sha():
    return sha([os.path.join(HERE, f) for f in ('v239_shrink_turn.py', 'v174_mix_driver.py', 'v163_parcae_damped_trunc.py',
                                                 'v162_parcae_trunc_train.py')])


# ------------------------------------------------------------------------------------------------ the three arms
ARM = {'name': 'full'}
CALLS = {}


def build(gR, gk, arm):
    if arm == 'trunc':
        return gk
    if arm == 'turn':
        return gk * (gR.norm() / (gk.norm() + 1e-30))
    if arm == 'shrink':
        return gR * (gk.norm() / (gR.norm() + 1e-30))
    raise ValueError(arm)


def patch(M):
    """Route v174's update-gradient construction through the current arm. 'full' keeps v174's own mix."""
    orig = M.mix

    def mix239(gR, gk, lam):
        CALLS[ARM['name']] = CALLS.get(ARM['name'], 0) + 1
        return orig(gR, gk, lam) if ARM['name'] == 'full' else build(gR, gk, ARM['name'])
    M.mix = mix239
    return orig


def parse(cell):
    """'full' | 'floor' | 'trunc1' | 'turn2' | 'shrink4' ... -> (arm, k)."""
    if cell in ('full', 'floor'):
        return cell, 0
    for a in ARMS2:
        if cell.startswith(a) and cell[len(a):].isdigit():
            return a, int(cell[len(a):])
    raise ValueError(cell)


def run_one(M, data, cell, seed):
    arm, k = parse(cell)
    if arm in ('full', 'floor'):
        ARM['name'] = 'full'
        r = M.run_scored(data, arm, seed)
    else:
        ARM['name'] = arm
        r = M.train_mix(data, k, 1.0, seed, mode='mix', evaluate=True)   # lam != 0: both gradients every step
        ARM['name'] = 'full'
    r.update(arm=cell, k=k, seed=seed, v239_sha=driver_sha())
    return r


# ------------------------------------------------------------------------------------------------ selftest
def selftest(M, data):
    import torch
    v = M.v
    _, m = v.load_model(); m.train()
    ARM['name'] = 'full'
    M.selftest(m, data)                                   # v163 selftest + v174 mix construction (lam 0 / lam 1)
    params = [p for p in m.parameters()]; ids = data['val'][:1].to(v.DEV)
    for k in KS:
        gR, gk = M.grad_pair(m, params, ids, k, 1.0, False)
        t, u, s = build(gR, gk, 'trunc'), build(gR, gk, 'turn'), build(gR, gk, 'shrink')
        assert torch.equal(t, gk)
        assert abs(M.cos(u, gk) - 1) < 1e-5 and abs(float(u.norm() / gR.norm()) - 1) < 1e-5
        assert abs(M.cos(s, gR) - 1) < 1e-5 and abs(float(s.norm() / gk.norm()) - 1) < 1e-5
        M.log('v239 selftest k=%d: |g_k|/|g_R| %.3f cos(g_k, g_R) %.3f; trunc / turn / shrink constructions OK' % (
            k, float(gk.norm() / gR.norm()), M.cos(gk, gR)))
    del m
    steps = 3
    for cell in ('full', 'turn1', 'shrink1', 'trunc1'):
        arm, k = parse(cell); CALLS.clear()
        ARM['name'] = 'full' if arm == 'full' else arm
        r = M.train_mix(data, 1 if arm == 'full' else k, 0.0 if arm == 'full' else 1.0, M.PILOT_SEED, steps=steps)
        ARM['name'] = 'full'
        if arm == 'full':
            assert abs(r['Abar'] - 1) < 1e-4, r['Abar']           # FULL through the pipeline: agreement exactly 1
        else:
            assert CALLS.get(arm, 0) == steps - r['nonfinite'], CALLS   # the arm's construction was used every step
        M.log('v239 selftest %s: %d steps, Abar %.4f, nonfinite %d, constructions used %s' % (
            cell, steps, r['Abar'], r['nonfinite'], dict(CALLS)))
    M.log('v239 selftest OK (driver %s)' % driver_sha())


# ------------------------------------------------------------------------------------------------ analysis
def analyze(dirs):
    import numpy as np
    from scipy import stats
    rows = []
    for d in dirs:
        for f in glob.glob(os.path.join(d, '**', 'v239_train_*.jsonl'), recursive=True):
            rows += [json.loads(x) for x in open(f) if x.strip()]
    R = {(r['arm'], r['seed']): r for r in rows}
    L = []; P = L.append
    order = list(SEEDS) + list(EXTRA_BACKUPS)
    valid = [s for s in order if ('full', s) in R and R[('full', s)]['nonfinite'] <= MAX_NONFINITE
             and math.isfinite(R[('full', s)]['test']) and ('floor', s) in R][:N_USE]
    excl = [s for s in order if ('full', s) in R and s not in valid and order.index(s) < order.index(valid[-1])] if valid else []
    P('v239 analysis: %d runs; scored seeds %s; excluded %s' % (len(rows), valid, excl))
    if len(valid) < N_MIN:
        P('VERDICT: INCOMPLETE (fewer than %d valid seeds)' % N_MIN); return '\n'.join(L)
    gap = [R[('floor', s)]['test'] - R[('full', s)]['test'] for s in valid]
    mg, sd = float(np.mean(gap)), float(np.std(gap, ddof=1))
    lo = mg - stats.t.ppf(0.95, len(gap) - 1) * sd / math.sqrt(len(gap))
    gate = mg >= GATE_NATS and lo > 0
    P('floor gap %.4f nats (90%% lower %.4f; per seed %s) -> gate %s' % (mg, lo, ' '.join('%.4f' % g for g in gap),
                                                                         'PASS' if gate else 'FAIL'))
    phi, se = {}, {}
    for k in KS:
        for a in ARMS2:
            c = '%s%d' % (a, k)
            ss = [s for s in valid if (c, s) in R and math.isfinite(R[(c, s)]['test'])]
            ph = [(R[(c, s)]['test'] - R[('full', s)]['test']) / (R[('floor', s)]['test'] - R[('full', s)]['test']) for s in ss]
            phi[c] = float(np.mean(ph)) if ph else float('nan')
            se[c] = float(np.std(ph, ddof=1) / math.sqrt(len(ph))) if len(ph) > 1 else float('nan')
            nf = [R[(c, s)]['nonfinite'] for s in ss]
            P('  %-8s cost %.4f +- %.4f (n %d; per seed %s; nonfinite %s; Abar %.3f)' % (
                c, phi[c], se[c], len(ph), ' '.join('%.3f' % x for x in ph), nf,
                float(np.mean([R[(c, s)]['Abar'] for s in ss])) if ss else float('nan')))
    lab = {}
    for k in KS:
        t, u, s = phi['trunc%d' % k], phi['turn%d' % k], phi['shrink%d' % k]
        mk = m_k(t)
        lab[k] = ('TURN' if u >= t - mk and s <= mk else 'SHRINK' if s >= 0.5 * t else 'PARTIAL')
        P('k=%d: trunc %.4f turn %.4f shrink %.4f | margin %.4f | turn share %.2f, shrink share %.2f -> %s%s' % (
            k, t, u, s, mk, u / t if t else float('nan'), s / t if t else float('nan'), lab[k],
            '' if k in DECIDE_KS else ' (reported, not scored)'))
    P('reported: trunc vs S1 (v162/v163 pipeline) 0.224 / 0.100 / 0.024 at k = 1 / 2 / 4: %s' % ' / '.join(
        '%.3f' % phi['trunc%d' % k] for k in KS))
    if not gate:
        v = 'INVALID (floor gap gate failed)'
    elif len({lab[k] for k in DECIDE_KS}) == 1:
        v = lab[DECIDE_KS[0]]                         # TURN, SHRINK or PARTIAL at both k
    else:
        v = 'MIXED %s' % {k: lab[k] for k in DECIDE_KS}
    P('VERDICT: %s' % v)
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true'); ap.add_argument('--runs', default=None); ap.add_argument('--tag', default='x')
    ap.add_argument('--analyze', nargs='*')
    A = ap.parse_args()
    if A.analyze is not None:
        txt = analyze(A.analyze); print(txt)
        open(os.path.join(os.environ.get('OUT_DIR', '.'), 'v239_analysis.txt'), 'w').write(txt + '\n')
        return
    import v174_mix_driver as M
    patch(M)
    tok, m = M.v.load_model(); data = M.v.gsm8k_chunks(tok); del m
    M.log('v239 driver %s (v174 %s) dev %s amp %s R %d' % (driver_sha(), M.driver_sha(), M.v.DEV, M.v.AMP, M.R))
    if A.selftest:
        selftest(M, data)
    if A.runs:
        fn = os.path.join(M.v.OUT, 'v239_train_%s.jsonl' % A.tag)
        done = {(json.loads(x)['arm'], json.loads(x)['seed']) for x in open(fn) if x.strip()} if os.path.exists(fn) else set()
        for item in A.runs.split(','):
            cell, seed = item.split('@'); seed = int(seed)
            if (cell, seed) in done:
                M.log('skip %s seed %d (done)' % (cell, seed)); continue
            r = run_one(M, data, cell, seed)
            M.log('scored %s seed %d: test %.4f val %.4f Abar %s nonfinite %d (%ds)' % (
                cell, seed, r['test'], r['val'], '%.4f' % r['Abar'] if 'Abar' in r else '-', r['nonfinite'], r['s']))
            open(fn, 'a').write(json.dumps(r) + '\n'); done.add((cell, seed))


if __name__ == '__main__':
    main()
