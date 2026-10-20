"""v170 analysis (prereg docs/prereg_v170_sgd_vs_adam.md), written before any v170 treatment run.

Question: is the retrofit's insensitivity to truncation error (v167, Adam) caused by the optimiser? v168 measured, on
the frozen retrofit, a plain-SGD first-order progress deficit of 0.21 (a 1, k 1) vs 0.62 (a 0.3, k 1), but an Adam
deficit of 0.024 vs 0.016 (flat). Under Adam the training cost was flat (0.0087 vs 0.0087). If the optimiser explains
it, SGD training should pay MORE at a 0.3 than at a 1 on the retrofit.

  H1 (within v170, primary): retrofit SGD, k 1: phi(a 0.3) - phi(a 1) >= 0.02 with a 90% CI (Welch, over seeds) > 0
  H2 (cross-study, secondary): retrofit a 0.3, k 1: phi_SGD - phi_Adam(v167, 0.0087) >= 0.02
  Reported: k 2 versions of H1/H2; Parcae phi_SGD vs phi_Adam at k 1/2/4 (v163: 0.224 / 0.100 / 0.024);
            per-cell SGD deficit (v168) vs phi_SGD across both models.
Outcome: OPTIMISER-EXPLAINS = H1 and H2 pass; NOT-OPTIMISER = |phi(a 0.3) - phi(a 1)| <= 0.01 at k 1 (SGD also
flat); INCONCLUSIVE otherwise; UNINFORMATIVE if a retrofit variant has < 5 valid seeds. Amendment A1: if a retrofit
variant fails its floor gate, H1 is scored in raw excess loss -> OPTIMISER-DIRECTIONAL / NOT-OPTIMISER-DIRECTIONAL.

  python v170_analyze.py --verdict RUNS...      (Parcae runs keyed R=32; retrofit runs R=8)
  python v170_analyze.py --selftest
"""
import argparse, json, math
import numpy as np
from scipy import stats

SEEDS = list(range(1701101, 1701108))
N_USE, MAX_NONFINITE = 5, 10
ADAM_RETRO = {(1.0, 1): 0.0087, (1.0, 2): -0.0012, (0.3, 1): 0.0087, (0.3, 2): 0.0042}     # v167
ADAM_PARCAE = {1: 0.224, 2: 0.100, 4: 0.024}                                                  # v163 a 1, R 32
DEF_SGD = {('retro', 1.0, 1): 0.209, ('retro', 1.0, 2): 0.057, ('retro', 0.3, 1): 0.620, ('retro', 0.3, 2): 0.413,
           ('parcae', 1.0, 1): 0.722, ('parcae', 1.0, 2): 0.532, ('parcae', 1.0, 4): 0.270}  # v168, 1 - useful_sgd
VARIANTS = {('retro', 1.0): (8, (1, 2)), ('retro', 0.3): (8, (1, 2)), ('parcae', 1.0): (32, (1, 2, 4))}


def load(files):
    runs = {}
    for f in files:
        for line in open(f):
            r = json.loads(line); R = int(r['R']); model = 'parcae' if R == 32 else 'retro'
            key = (model, round(float(r['a']), 2), int(r['k']), r['seed'])
            if key not in runs:
                runs[key] = r
    return runs


def phis(runs, model, a, quiet=False):
    R, ks = VARIANTS[(model, a)]; ok = []
    for s in SEEDS:
        f = runs.get((model, a, R, s))
        if f is None:
            continue
        good = f['nonfinite_steps'] <= MAX_NONFINITE and math.isfinite(f['test']) and not f.get('diverged', False)
        if good and all((model, a, k, s) in runs for k in (0,) + ks):
            ok.append(s)
    ok = ok[:N_USE]
    if len(ok) < N_USE:
        return None, 'UNINFORMATIVE (%d valid seeds)' % len(ok)
    gap = [runs[(model, a, 0, s)]['test'] - runs[(model, a, R, s)]['test'] for s in ok]
    n = len(gap); h = stats.t.ppf(0.95, n - 1) * np.std(gap, ddof=1) / math.sqrt(n)
    if not (np.mean(gap) >= 0.03 and np.mean(gap) - h > 0):
        return None, 'floor gate FAIL (%.4f)' % np.mean(gap)
    out = {k: [(runs[(model, a, k, s)]['test'] - runs[(model, a, R, s)]['test']) / g for s, g in zip(ok, gap)] for k in ks}
    return out, 'floor %.4f [%.4f, %.4f] OK' % (np.mean(gap), np.mean(gap) - h, np.mean(gap) + h)


def welch_ci(x, y):
    x, y = np.asarray(x), np.asarray(y); d = x.mean() - y.mean()
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y); se = math.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1)) if se > 0 else 1
    h = stats.t.ppf(0.95, df) * se
    return d, d - h, d + h


def verdict(runs, quiet=False):
    pr = (lambda *x, **y: None) if quiet else print
    ph = {}
    for (model, a) in VARIANTS:
        p, msg = phis(runs, model, a); pr('%s a=%.1f: %s' % (model, a, msg)); ph[(model, a)] = p
        if p:
            for k, v in p.items():
                adam = ADAM_RETRO.get((a, k)) if model == 'retro' else ADAM_PARCAE.get(k)
                pr('   k=%d: phi_SGD %+.4f (se %.4f) | phi_Adam %+.4f | SGD deficit (v168) %.3f' % (
                    k, np.mean(v), np.std(v, ddof=1) / math.sqrt(len(v)), adam, DEF_SGD[(model, a, k)]))
    if ph[('retro', 1.0)] is None or ph[('retro', 0.3)] is None:
        # AMENDMENT A1 (2026-10-04, before any retrofit treatment run): if a retrofit variant fails the floor gate
        # (small SGD floor gap) but both have >= 5 valid seeds, score H1 in raw loss without normalising by the gap:
        # excess = L_k1 - L_full per seed; d = mean excess(a 0.3) - mean excess(a 1), Welch 90% CI.
        ex = {}
        for a in (1.0, 0.3):
            R = VARIANTS[('retro', a)][0]; ok = []
            for s in SEEDS:
                f_ = runs.get(('retro', a, R, s))
                if f_ is not None and f_['nonfinite_steps'] <= MAX_NONFINITE and math.isfinite(f_['test']) and not f_.get('diverged', False) \
                        and all(('retro', a, k, s) in runs for k in (0, 1, 2)):
                    ok.append(s)
            ok = ok[:N_USE]
            ex[a] = [runs[('retro', a, 1, s)]['test'] - runs[('retro', a, R, s)]['test'] for s in ok] if len(ok) == N_USE else None
        if ex[1.0] is None or ex[0.3] is None:
            pr('OUTCOME: UNINFORMATIVE (valid seeds)'); return dict(outcome='UNINFORMATIVE')
        d, lo, hi = welch_ci(ex[0.3], ex[1.0])
        out = 'OPTIMISER-DIRECTIONAL' if lo > 0 else 'NOT-OPTIMISER-DIRECTIONAL'
        pr('A1 fallback (floor gate failed): excess loss k1, a0.3 - a1 = %+.5f nats [%.5f, %.5f] -> %s (underpowered)' % (d, lo, hi, out))
        return dict(outcome=out, fallback=(d, lo, hi))
    d, lo, hi = welch_ci(ph[('retro', 0.3)][1], ph[('retro', 1.0)][1])
    d2, lo2, hi2 = welch_ci(ph[('retro', 0.3)][2], ph[('retro', 1.0)][2])
    h1 = d >= 0.02 and lo > 0
    sgd31 = float(np.mean(ph[('retro', 0.3)][1])); h2 = sgd31 - ADAM_RETRO[(0.3, 1)] >= 0.02
    pr('\nH1 retro SGD k1: phi(a0.3) - phi(a1) = %+.4f [%.4f, %.4f] -> %s   (k2: %+.4f [%.4f, %.4f])' % (d, lo, hi, 'PASS' if h1 else 'FAIL', d2, lo2, hi2))
    pr('H2 retro a0.3 k1: phi_SGD %.4f - phi_Adam %.4f = %+.4f -> %s' % (sgd31, ADAM_RETRO[(0.3, 1)], sgd31 - ADAM_RETRO[(0.3, 1)], 'PASS' if h2 else 'FAIL'))
    if h1 and h2:
        outcome = 'OPTIMISER-EXPLAINS'
    elif abs(d) <= 0.01:
        outcome = 'NOT-OPTIMISER'
    else:
        outcome = 'INCONCLUSIVE'
    pr('OUTCOME: %s' % outcome)
    return dict(outcome=outcome, h1=(d, lo, hi), h2=sgd31 - ADAM_RETRO[(0.3, 1)],
                phi={'%s_%s' % k: {kk: float(np.mean(v)) for kk, v in p.items()} for k, p in ph.items() if p})


def selftest():
    rng = np.random.default_rng(0)
    for name, eff, want in (('optimiser world', {1.0: 0.02, 0.3: 0.08}, 'OPTIMISER-EXPLAINS'),
                            ('flat world', {1.0: 0.02, 0.3: 0.022}, 'NOT-OPTIMISER'),
                            ('small world', {1.0: 0.02, 0.3: 0.035}, 'INCONCLUSIVE')):
        runs = {}
        for (model, a), (R, ks) in VARIANTS.items():
            for s in SEEDS:
                base = 1.2 + rng.normal(0, 0.001)
                runs[(model, a, R, s)] = dict(test=base, nonfinite_steps=0, R=R); runs[(model, a, 0, s)] = dict(test=base + 0.05, R=R)
                for k in ks:
                    phi = eff.get(a, 0.02) / k if model == 'retro' else 0.2 / k
                    runs[(model, a, k, s)] = dict(test=base + 0.05 * phi + rng.normal(0, 0.0002), R=R)
        got = verdict(runs, quiet=True)['outcome']; print('selftest %s -> %s (expect %s)' % (name, got, want)); assert got == want
    for name, eff, want in (('A1 small-gap optimiser world', {1.0: 0.02, 0.3: 0.2}, 'OPTIMISER-DIRECTIONAL'),
                            ('A1 small-gap flat world', {1.0: 0.02, 0.3: 0.02}, 'NOT-OPTIMISER-DIRECTIONAL')):
        runs = {}
        for (model, a), (R, ks) in VARIANTS.items():
            for s in SEEDS:
                base = 1.2 + rng.normal(0, 0.001); gap = 0.02 if model == 'retro' else 0.05     # retrofit gap below 0.03
                runs[(model, a, R, s)] = dict(test=base, nonfinite_steps=0, R=R); runs[(model, a, 0, s)] = dict(test=base + gap, R=R)
                for k in ks:
                    phi = eff.get(a, 0.02) / k if model == 'retro' else 0.2 / k
                    runs[(model, a, k, s)] = dict(test=base + gap * phi + rng.normal(0, 0.0002), R=R)
        got = verdict(runs, quiet=True)['outcome']; print('selftest %s -> %s (expect %s)' % (name, got, want)); assert got == want
    print('selftest OK')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--verdict', nargs='*'); ap.add_argument('--selftest', action='store_true')
    A = ap.parse_args()
    if A.selftest:
        selftest()
    if A.verdict:
        json.dump(verdict(load(A.verdict)), open('v170_verdict.json', 'w'), indent=1, default=str)
