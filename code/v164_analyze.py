"""v164 analysis, written before any v164 training run (prereg: docs/prereg_v164_parcae_prediction.md).

Two predictors of phi (fraction of the loop's training credit lost by backprop through only the last k of R loops),
both FIT ONLY on the 15 truncated cells of v162 + v163 (evaluation-seed means, frozen gradient errors):
  M_err: phi = c * err^p                       (cost is one function of the frozen-model gradient error)
  M_k:   phi = mean phi at that k in the fit set ("k alone"); unseen k (3, 6) interpolated linearly in log2(k)
         between the neighbouring fitted k (2-4 and 4-8)
Test cells (new): (R, a, k) for (32, 1.0, {3, 6}), (32, 0.7, {2, 3, 4, 6}), (64, 0.3, {2, 3, 4, 6}).
Frozen-model err for the test cells is measured pre-freeze (v164 --measure) and LOCKED below at freeze.

  python v164_analyze.py --fit                  print the fitted models and the locked predictions
  python v164_analyze.py --verdict RUNS...      score both predictors on the evaluation runs -> outcome label
  python v164_analyze.py --selftest             synthetic worlds -> expected labels
"""
import argparse, json, math
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

# ---- fit set: (label, err, phi) from v162 (docs/v162_...md; err = v161/v162 pilot measurement) and v163 -----------
FIT = [('v162 R8 k1', 0.8051, 0.183), ('v162 R8 k2', 0.6477, 0.077), ('v162 R8 k4', 0.3860, 0.012),
       ('v162 R16 k1', 0.8132, 0.209), ('v162 R16 k2', 0.6632, 0.090), ('v162 R16 k4', 0.4210, 0.016),
       ('v162 R16 k8', 0.1803, -0.005),
       ('v163 a1 k1', 0.8201, 0.224), ('v163 a1 k2', 0.6753, 0.100), ('v163 a1 k4', 0.4427, 0.024),
       ('v163 a1 k8', 0.2046, 0.002),
       ('v163 a.5 k1', 0.8773, 0.212), ('v163 a.5 k2', 0.7911, 0.142), ('v163 a.5 k4', 0.6453, 0.072),
       ('v163 a.5 k8', 0.4183, 0.020)]
FIT_K = [1, 2, 4, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8]

VARIANTS = [(32, 1.0), (32, 0.7), (64, 0.3)]
TEST = [(32, 1.0, 3), (32, 1.0, 6)] + [(32, 0.7, k) for k in (2, 3, 4, 6)] + [(64, 0.3, k) for k in (2, 3, 4, 6)]
CONTROLS = {(R, a): (0, R) for R, a in VARIANTS}       # floor k = 0, full k = R
LR_TRUNC, LR_FLOOR = 3e-4, 9e-4
EVAL = [1641101, 1641102, 1641103, 1641104, 1641105]
TEST_ERR = None   # set from the measurement kernel's JSON (--measure_json): frozen procedure, locked at freeze


def load_test_err(path):
    """LOCKED PROCEDURE: test-cell errors = err_median of the frozen-model measurement (kernel astrocyte-v164-measure,
    v164_parcae_predict.py --measure, frozen model, fixed seeds), independent of every training run."""
    m = json.load(open(path)); out = {}
    for r in m['res']:
        for k, e in r['err_median'].items():
            out[(int(r['R']), float(r['a']), int(k))] = float(e)
    return {c: out[c] for c in TEST}


def fit_models():
    err = np.array([e for _, e, _ in FIT]); phi = np.array([p for _, _, p in FIT])
    (c, p), _ = curve_fit(lambda x, c, p: c * x ** p, err, phi, p0=(0.3, 3.0), bounds=([0, 0.5], [10, 12]))
    by_k = {k: float(np.mean([phi[i] for i in range(len(FIT)) if FIT_K[i] == k])) for k in sorted(set(FIT_K))}

    def g(k):
        if k in by_k:
            return by_k[k]
        lo = max(x for x in by_k if x < k); hi = min(x for x in by_k if x > k)
        w = (math.log2(k) - math.log2(lo)) / (math.log2(hi) - math.log2(lo))
        return (1 - w) * by_k[lo] + w * by_k[hi]
    return (c, p), by_k, g


def predictions(test_err):
    (c, p), by_k, g = fit_models()
    return {cell: dict(err=test_err[cell], pred_err=c * test_err[cell] ** p, pred_k=g(cell[2])) for cell in TEST}


def ci90(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2:
        return float(np.mean(x)), float('nan'), float('nan')
    h = stats.t.ppf(0.95, n - 1) * x.std(ddof=1) / math.sqrt(n)
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def load(files):
    runs, shas = {}, set()
    for f in files:
        for line in open(f):
            r = json.loads(line)
            key = (int(r['R']), float(r['a']), int(r['k']), r['seed'])
            if key in runs:
                print('duplicate ignored', key); continue
            runs[key] = r; shas.add(r.get('driver_sha'))
    print('runs %d, driver sha %s' % (len(runs), sorted(map(str, shas))))
    return runs


def verdict(runs, test_err, quiet=False):
    pr = print if not quiet else (lambda *a, **k: None)
    preds = predictions(test_err)
    out = {}; ok_variants = []
    for R, a in VARIANTS:
        d = [runs[(R, a, 0, s)]['test'] - runs[(R, a, R, s)]['test'] for s in EVAL if (R, a, 0, s) in runs and (R, a, R, s) in runs]
        m, lo, hi = ci90(d); ok = lo > 0 and m >= 0.03
        out['floor_R%d_a%.1f' % (R, a)] = (m, lo, hi, ok); pr('floor R=%d a=%.1f: %.4f [%.4f, %.4f] %s' % (R, a, m, lo, hi, 'OK' if ok else 'FAIL'))
        if ok:
            ok_variants.append((R, a))
    cells = [c for c in TEST if (c[0], c[1]) in ok_variants]
    rows = []
    for cell in cells:
        R, a, k = cell
        phis = []
        for s in EVAL:
            try:
                x, f, z = runs[(R, a, k, s)]['test'], runs[(R, a, R, s)]['test'], runs[(R, a, 0, s)]['test']
            except KeyError:
                continue
            if all(map(math.isfinite, (x, f, z))):
                phis.append((x - f) / (z - f))
        obs = float(np.mean(phis)); p_ = preds[cell]
        rows.append(dict(cell=cell, obs=obs, se=float(np.std(phis, ddof=1) / math.sqrt(len(phis))) if len(phis) > 1 else float('nan'),
                         pred_err=p_['pred_err'], pred_k=p_['pred_k'], res_err=abs(obs - p_['pred_err']), res_k=abs(obs - p_['pred_k']), n=len(phis)))
        pr('R=%d a=%.1f k=%d  err %.3f | obs phi %+.3f (se %.3f) | M_err %+.3f (|res| %.3f) | M_k %+.3f (|res| %.3f)' % (
            R, a, k, p_['err'], obs, rows[-1]['se'], p_['pred_err'], rows[-1]['res_err'], p_['pred_k'], rows[-1]['res_k']))
    if len(rows) < 6:
        pr('OUTCOME: UNINFORMATIVE (fewer than 6 test cells with a valid floor)'); out['outcome'] = 'UNINFORMATIVE'; return out
    mae_err = float(np.mean([r['res_err'] for r in rows])); mae_k = float(np.mean([r['res_k'] for r in rows]))
    wins = sum(r['res_err'] < r['res_k'] for r in rows)
    # discriminating cells: where the two models' predictions differ by >= 0.03
    disc = [r for r in rows if abs(r['pred_err'] - r['pred_k']) >= 0.03]
    disc_wins = sum(r['res_err'] < r['res_k'] for r in disc)
    pr('\nMAE  M_err %.4f   M_k %.4f   ratio %.2f | M_err closer in %d/%d cells | discriminating cells %d, M_err closer in %d' % (
        mae_err, mae_k, mae_err / mae_k if mae_k > 0 else float('nan'), wins, len(rows), len(disc), disc_wins))
    need = math.ceil(0.7 * len(rows))
    if mae_err <= 0.7 * mae_k and mae_err <= 0.03 and wins >= need and (not disc or disc_wins >= math.ceil(0.7 * len(disc))):
        outcome = 'ERR-PREDICTS'
    elif mae_err >= mae_k:
        outcome = 'K-ALONE-AS-GOOD'
    else:
        outcome = 'INCONCLUSIVE'
    pr('OUTCOME: %s' % outcome)
    out.update(rows=rows, mae_err=mae_err, mae_k=mae_k, wins=wins, n=len(rows), disc=len(disc), disc_wins=disc_wins, outcome=outcome)
    return out


def selftest():
    rng = np.random.default_rng(0)
    (c, p), by_k, g = fit_models()
    te = {cell: 0.4 + 0.08 * i for i, cell in enumerate(TEST)}      # synthetic test errors spanning the fit range
    for name, fn, want in (('err world', lambda cell: c * te[cell] ** p, 'ERR-PREDICTS'),
                           ('k world', lambda cell: g(cell[2]), 'K-ALONE-AS-GOOD')):
        runs = {}
        for R, a in VARIANTS:
            for s in EVAL:
                base = 1.4 + rng.normal(0, 0.001)
                runs[(R, a, R, s)] = dict(test=base); runs[(R, a, 0, s)] = dict(test=base + 0.17)
                for cell in TEST:
                    if cell[:2] == (R, a):
                        runs[(R, a, cell[2], s)] = dict(test=base + 0.17 * fn(cell) + rng.normal(0, 0.0015))
        got = verdict(runs, te, quiet=True)['outcome']; print('selftest %s -> %s (expect %s)' % (name, got, want)); assert got == want
    print('selftest OK')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit', action='store_true'); ap.add_argument('--verdict', nargs='*'); ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--measure_json', default=None)
    A = ap.parse_args()
    if A.selftest:
        return selftest()
    global TEST_ERR
    if A.measure_json:
        TEST_ERR = load_test_err(A.measure_json)
    (c, p), by_k, g = fit_models()
    print('M_err: phi = %.4f * err^%.3f | M_k: %s, k=3 -> %.4f, k=6 -> %.4f' % (c, p, {k: round(x, 4) for k, x in by_k.items()}, g(3), g(6)))
    if A.fit and TEST_ERR:
        for cell, d in predictions(TEST_ERR).items():
            print(cell, 'err %.4f -> M_err %.4f, M_k %.4f' % (d['err'], d['pred_err'], d['pred_k']))
    if A.verdict is not None:
        res = verdict(load(A.verdict), TEST_ERR)
        json.dump(res, open('v164_verdict.json', 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
