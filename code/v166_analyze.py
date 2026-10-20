"""v166 analysis, written before any v166 treatment run (prereg: docs/prereg_v166_slam_dunk.md).

Out-of-sample test of "truncation cost is one function of the frozen-model gradient error" on FRESH, discriminating
Parcae-140m cells (block A only, PI decision 2026-10-03). Predictors are refit on all 25 seen truncated cells
(v162 + v163 + v164 damped + v165 undamped) and LOCKED here; test-cell errors are the pre-freeze frozen-model
measurement (kernel astrocyte-v166-pre-a, results_v166_pre/a/v164_measure_pre0.json), LOCKED by value below.

  M_err: phi = c * err^p   (power law; AIC-preferred over the saturating form at the refit, see prereg §1)
  M_k:   phi = mean phi at k over the fit set; unseen k interpolated linearly in log2 k; beyond k 8, last segment's
         slope extrapolated, floored at 0
Scored cells: |M_err - M_k| >= 0.025 -> (64, 0.3, 8), (64, 0.3, 12), (48, 0.4, 4), (48, 0.4, 8); all M_err > M_k.

  python v166_analyze.py --fit                 print locked models and predictions (and check them against the refit)
  python v166_analyze.py --verdict RUNS...     outcome label
  python v166_analyze.py --selftest            synthetic worlds -> expected labels
"""
import argparse, json, math
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

# fit set (err, phi, k): v162/v163 from v164_analyze.FIT; v164 damped cells (docs/v164_parcae_prediction.md);
# v165 undamped cells (docs/v165_undamped_rerun.md)
FIT = [(0.8051, 0.183, 1), (0.6477, 0.077, 2), (0.3860, 0.012, 4), (0.8132, 0.209, 1), (0.6632, 0.090, 2),
       (0.4210, 0.016, 4), (0.1803, -0.005, 8), (0.8201, 0.224, 1), (0.6753, 0.100, 2), (0.4427, 0.024, 4),
       (0.2046, 0.002, 8), (0.8773, 0.212, 1), (0.7911, 0.142, 2), (0.6453, 0.072, 4), (0.4183, 0.020, 8),
       (0.744, 0.127, 2), (0.646, 0.076, 3), (0.558, 0.046, 4), (0.423, 0.015, 6),
       (0.860, 0.169, 2), (0.809, 0.137, 3), (0.763, 0.111, 4), (0.679, 0.073, 6),
       (0.5462, 0.0479, 3), (0.3071, 0.0046, 6)]
LOCKED_C, LOCKED_P = 0.3390, 3.3791          # refit result at freeze (checked by --fit)

VARIANTS = [(64, 0.3), (48, 0.4)]
TEST_ERR = {(64, 0.3, 8): 0.6022, (64, 0.3, 12): 0.4772, (48, 0.4, 4): 0.6992, (48, 0.4, 8): 0.5020}
TEST = list(TEST_ERR)
SEEDS = list(range(1661101, 1661108))        # 5 evaluation + 2 backups per variant
N_USE, MAX_NONFINITE = 5, 10


def by_k():
    ks = sorted(set(k for _, _, k in FIT))
    return {k: float(np.mean([p for _, p, kk in FIT if kk == k])) for k in ks}


def m_k(k):
    b = by_k(); ks = sorted(b); L = math.log2
    if k in b:
        return b[k]
    if k > ks[-1]:
        lo, hi = ks[-2], ks[-1]
        return max(0.0, b[hi] + (b[hi] - b[lo]) / (L(hi) - L(lo)) * (L(k) - L(hi)))
    lo = max(x for x in ks if x < k); hi = min(x for x in ks if x > k); w = (L(k) - L(lo)) / (L(hi) - L(lo))
    return (1 - w) * b[lo] + w * b[hi]


def m_err(e):
    return LOCKED_C * e ** LOCKED_P


def predictions():
    return {c: dict(err=TEST_ERR[c], pred_err=m_err(TEST_ERR[c]), pred_k=m_k(c[2])) for c in TEST}


def ci90(x):
    x = np.asarray(x, float); n = len(x); h = stats.t.ppf(0.95, n - 1) * x.std(ddof=1) / math.sqrt(n)
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def load(files):
    runs = {}
    for f in files:
        for line in open(f):
            r = json.loads(line); key = (int(r['R']), round(float(r['a']), 2), int(r['k']), r['seed'])
            if key in runs:
                print('duplicate ignored', key); continue
            runs[key] = r
    return runs


def valid_seeds(runs, R, a, quiet=False):
    ok = []
    for s in SEEDS:
        f = runs.get((R, a, R, s))
        if f is None:
            continue
        good = f['nonfinite_steps'] <= MAX_NONFINITE and math.isfinite(f['test']) and not f.get('diverged', False)
        if not quiet:
            print('R=%d a=%.1f seed %d: full nonfinite %d test %.4f -> %s' % (R, a, s, f['nonfinite_steps'], f['test'], 'VALID' if good else 'EXCLUDED'))
        if good and all((R, a, k, s) in runs for k in [0] + [c[2] for c in TEST if c[:2] == (R, a)]):
            ok.append(s)
    return ok[:N_USE]


def verdict(runs, quiet=False):
    pr = (lambda *x, **y: None) if quiet else print
    preds = predictions(); rows = []
    for R, a in VARIANTS:
        seeds = valid_seeds(runs, R, a, quiet)
        if len(seeds) < N_USE:
            pr('R=%d a=%.1f: UNINFORMATIVE (%d valid seeds)' % (R, a, len(seeds))); continue
        m, lo, hi = ci90([runs[(R, a, 0, s)]['test'] - runs[(R, a, R, s)]['test'] for s in seeds])
        ok = lo > 0 and m >= 0.03
        pr('floor R=%d a=%.1f: %.4f [%.4f, %.4f] %s' % (R, a, m, lo, hi, 'OK' if ok else 'FAIL'))
        if not ok:
            continue
        for c in [c for c in TEST if c[:2] == (R, a)]:
            ph = [(runs[(R, a, c[2], s)]['test'] - runs[(R, a, R, s)]['test']) / (runs[(R, a, 0, s)]['test'] - runs[(R, a, R, s)]['test']) for s in seeds]
            obs = float(np.mean(ph)); p = preds[c]
            row = dict(cell=c, obs=obs, se=float(np.std(ph, ddof=1) / math.sqrt(len(ph))), pred_err=p['pred_err'], pred_k=p['pred_k'],
                       res_err=abs(obs - p['pred_err']), res_k=abs(obs - p['pred_k']))
            row['calib'] = row['res_err'] <= max(0.02, 0.3 * abs(obs)); rows.append(row)
            pr('R=%d a=%.1f k=%d err %.3f | obs %+.4f (se %.4f) | M_err %.4f (|res| %.4f%s) | M_k %.4f (|res| %.4f)' % (
                R, a, c[2], p['err'], obs, row['se'], p['pred_err'], row['res_err'], ', calibrated' if row['calib'] else '', p['pred_k'], row['res_k']))
    out = dict(rows=rows)
    if len(rows) < 3:
        pr('OUTCOME: UNINFORMATIVE (%d scored cells)' % len(rows)); out['outcome'] = 'UNINFORMATIVE'; return out
    n = len(rows); need = math.ceil(0.75 * n)
    mae_err = float(np.mean([r['res_err'] for r in rows])); mae_k = float(np.mean([r['res_k'] for r in rows]))
    wins = sum(r['res_err'] < r['res_k'] for r in rows); cal = sum(r['calib'] for r in rows)
    pr('\nMAE M_err %.4f  M_k %.4f  ratio %.2f | M_err closer %d/%d | calibrated %d/%d' % (mae_err, mae_k, mae_err / mae_k, wins, n, cal, n))
    if wins >= need and mae_err <= 0.5 * mae_k and cal >= need:
        outcome = 'ERR-PREDICTS'
    elif mae_err >= mae_k:
        outcome = 'K-ALONE-AS-GOOD'
    elif wins >= need and mae_err <= 0.8 * mae_k:
        outcome = 'ERR-ORDERS'
    else:
        outcome = 'INCONCLUSIVE'
    pr('OUTCOME: %s' % outcome)
    out.update(mae_err=mae_err, mae_k=mae_k, wins=wins, calibrated=cal, n=n, outcome=outcome)
    return out


def refit_check():
    e = np.array([x[0] for x in FIT]); y = np.array([x[1] for x in FIT])
    (c, p), _ = curve_fit(lambda x, c, p: c * x ** p, e, y, p0=(0.3, 3.0), bounds=([0, 0.5], [10, 12]))
    print('refit c %.4f p %.4f (locked %.4f %.4f)' % (c, p, LOCKED_C, LOCKED_P))
    assert abs(c - LOCKED_C) < 1e-3 and abs(p - LOCKED_P) < 1e-3


def selftest():
    rng = np.random.default_rng(0); preds = predictions()
    for name, fn, want in (('err world', lambda c: preds[c]['pred_err'], 'ERR-PREDICTS'),
                           ('k world', lambda c: preds[c]['pred_k'], 'K-ALONE-AS-GOOD'),
                           ('half world', lambda c: 0.5 * (preds[c]['pred_err'] + preds[c]['pred_k']), 'INCONCLUSIVE')):
        runs = {}
        for R, a in VARIANTS:
            for s in SEEDS:
                base = 1.4 + rng.normal(0, 0.001)
                runs[(R, a, R, s)] = dict(test=base, nonfinite_steps=0); runs[(R, a, 0, s)] = dict(test=base + 0.18)
                for c in TEST:
                    if c[:2] == (R, a):
                        runs[(R, a, c[2], s)] = dict(test=base + 0.18 * fn(c) + rng.normal(0, 0.0005))
        runs[(64, 0.3, 64, SEEDS[0])]['nonfinite_steps'] = 13          # exclusion rule exercised
        got = verdict(runs, quiet=True)['outcome']; print('selftest %s -> %s (expect %s)' % (name, got, want)); assert got == want
    print('selftest OK')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit', action='store_true'); ap.add_argument('--verdict', nargs='*'); ap.add_argument('--selftest', action='store_true')
    A = ap.parse_args()
    if A.selftest:
        return selftest()
    if A.fit:
        refit_check(); print('M_k by k', {k: round(v, 4) for k, v in by_k().items()})
        for c, p in predictions().items():
            print(c, 'err %.4f -> M_err %.4f, M_k %.4f, diff %+.4f' % (p['err'], p['pred_err'], p['pred_k'], p['pred_err'] - p['pred_k']))
    if A.verdict:
        out = verdict(load(A.verdict))
        json.dump(out, open('v166_verdict.json', 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
