"""v167 analysis, written before any v167 treatment run (prereg: docs/prereg_v167_retrofit_transfer.md).

Does the Parcae-fitted cost curve phi = c * err^p transfer, UNCHANGED, to a different model family (the healed looped
SmolLM2 retrofit), and does it beat a fair "k alone" rule fitted on the retrofit itself?

  M_err:     phi = c * err^p, fit on all 29 Parcae truncated cells (v162-v166) and LOCKED (LOCKED_C, LOCKED_P)
  M_k_own:   for a damped cell (a < 1) at k: the observed undamped (a = 1) phi at the same k in THIS study (same seeds)
Test errors: the pre-freeze frozen-model measurement (kernel astrocyte-v167-pilot, v167_measure_floor.json), LOCKED.

  python v167_analyze.py --fit                 locked curve (checked against the refit) and predictions
  python v167_analyze.py --verdict RUNS...     outcome label
  python v167_analyze.py --selftest            synthetic worlds -> expected labels
"""
import argparse, json, math
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

import v166_analyze as V166

# Parcae fit set: v166's 25 cells + v166's 4 scored cells (docs/v166_slam_dunk_result.md), (err, phi)
FIT = [(e, p) for e, p, _ in V166.FIT] + [(0.6022, 0.0578), (0.4772, 0.0268), (0.6992, 0.0879), (0.5020, 0.0316)]
LOCKED_C, LOCKED_P = 0.3418, 3.4335          # refit at freeze (power law); main() asserts the match

R = 8
VARIANTS = (1.0, 0.5, 0.3)
KS = (1, 2)
TEST_ERR = {(1.0, 1): 0.3575, (1.0, 2): 0.1372, (0.5, 1): 0.5658, (0.5, 2): 0.3436, (0.3, 1): 0.6505, (0.3, 2): 0.4418}
CELLS = list(TEST_ERR)
DAMPED = [c for c in CELLS if c[0] < 1.0]
SEEDS = list(range(1671101, 1671108))       # 5 evaluation + 2 backups
N_USE, MAX_NONFINITE = 5, 10


def refit():
    e = np.array([x[0] for x in FIT]); y = np.array([x[1] for x in FIT])
    (c, p), _ = curve_fit(lambda x, c, p: c * x ** p, e, y, p0=(0.3, 3.0), bounds=([0, 0.5], [10, 12]))
    return float(c), float(p)


def m_err(e):
    return LOCKED_C * e ** LOCKED_P


def ci90(x):
    x = np.asarray(x, float); n = len(x); h = stats.t.ppf(0.95, n - 1) * x.std(ddof=1) / math.sqrt(n)
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def load(files):
    runs = {}
    for f in files:
        for line in open(f):
            r = json.loads(line); key = (round(float(r['a']), 2), int(r['k']), r['seed'])
            if key in runs:
                print('duplicate ignored', key); continue
            runs[key] = r
    return runs


def valid_seeds(runs, a, quiet=False):
    ok = []
    for s in SEEDS:
        f = runs.get((a, R, s))
        if f is None:
            continue
        good = f['nonfinite_steps'] <= MAX_NONFINITE and math.isfinite(f['test']) and not f.get('diverged', False)
        if not quiet:
            print('a=%.1f seed %d: full nonfinite %d test %.4f -> %s' % (a, s, f['nonfinite_steps'], f['test'], 'VALID' if good else 'EXCLUDED'))
        if good and all((a, k, s) in runs for k in (0,) + KS):
            ok.append(s)
    return ok[:N_USE]


def verdict(runs, quiet=False):
    pr = (lambda *x, **y: None) if quiet else print
    obs = {}
    for a in VARIANTS:
        seeds = valid_seeds(runs, a, quiet)
        if len(seeds) < N_USE:
            pr('a=%.1f: UNINFORMATIVE (%d valid seeds)' % (a, len(seeds))); continue
        m, lo, hi = ci90([runs[(a, 0, s)]['test'] - runs[(a, R, s)]['test'] for s in seeds])
        ok = lo > 0 and m >= 0.03
        pr('floor a=%.1f: %.4f [%.4f, %.4f] %s' % (a, m, lo, hi, 'OK' if ok else 'FAIL'))
        if not ok:
            continue
        for k in KS:
            ph = [(runs[(a, k, s)]['test'] - runs[(a, R, s)]['test']) / (runs[(a, 0, s)]['test'] - runs[(a, R, s)]['test']) for s in seeds]
            obs[(a, k)] = (float(np.mean(ph)), float(np.std(ph, ddof=1) / math.sqrt(len(ph))))
    out = dict(obs={str(c): v for c, v in obs.items()})
    if 1.0 not in [c[0] for c in obs] or len(obs) < 4:
        pr('OUTCOME: UNINFORMATIVE (undamped variant or too few cells scored)'); out['outcome'] = 'UNINFORMATIVE'; return out
    # (i) TRANSFER: the locked Parcae curve's numbers on all scored cells
    rows = []
    for c in CELLS:
        if c not in obs:
            continue
        o, se = obs[c]; pe = m_err(TEST_ERR[c]); res = abs(o - pe)
        rows.append(dict(cell=c, obs=o, se=se, pred_err=pe, res_err=res, calib=res <= max(0.02, 0.3 * abs(o))))
    n = len(rows); cal = sum(r['calib'] for r in rows); mae = float(np.mean([r['res_err'] for r in rows]))
    transfer_orig = cal >= n - 1 and mae <= 0.015            # as frozen at 880be7a (reported, superseded)
    # AMENDMENT A1 (post-freeze, pre-result; review critique): near-zero undamped cells made (i) easy to pass with a
    # large miss on the one high-cost cell. (i) now also requires: every damped k = 1 cell (the high-cost cells)
    # calibrated, and MAE over the damped cells alone <= 0.015.
    big = [r for r in rows if r['cell'][0] < 1.0 and r['cell'][1] == 1]
    dmae = float(np.mean([r['res_err'] for r in rows if r['cell'][0] < 1.0])) if any(r['cell'][0] < 1.0 for r in rows) else float('inf')
    transfer = transfer_orig and len(big) >= 1 and all(r['calib'] for r in big) and dmae <= 0.015
    # (ii) RIVAL: damped cells, M_err vs M_k_own (= observed undamped phi at the same k)
    drows = []
    for r in rows:
        a, k = r['cell']
        if a < 1.0:
            r['pred_kown'] = obs[(1.0, k)][0]; r['res_kown'] = abs(r['obs'] - r['pred_kown']); drows.append(r)
    for r in rows:
        pr('a=%.1f k=%d err %.3f | obs %+.4f (se %.4f) | M_err %.4f (|res| %.4f%s)%s' % (
            r['cell'][0], r['cell'][1], TEST_ERR[r['cell']], r['obs'], r['se'], r['pred_err'], r['res_err'], ', calibrated' if r['calib'] else '',
            ' | M_k_own %.4f (|res| %.4f)' % (r['pred_kown'], r['res_kown']) if 'pred_kown' in r else ''))
    nd = len(drows)
    if nd < 2:
        pr('OUTCOME: UNINFORMATIVE (fewer than 2 damped cells)'); out['outcome'] = 'UNINFORMATIVE'; return out
    mae_e = float(np.mean([r['res_err'] for r in drows])); mae_k = float(np.mean([r['res_kown'] for r in drows]))
    wins = sum(r['res_err'] < r['res_kown'] for r in drows)
    rival = wins >= math.ceil(0.75 * nd) and mae_e <= 0.5 * mae_k
    pr('\n(i) TRANSFER: calibrated %d/%d, MAE_err %.4f | high-cost damped k=1 cells calibrated %d/%d, damped MAE %.4f -> %s (as frozen: %s)' % (
        cal, n, mae, sum(r['calib'] for r in big), len(big), dmae, 'PASS' if transfer else 'FAIL', 'PASS' if transfer_orig else 'FAIL'))
    pr('(ii) RIVAL (damped cells): M_err closer %d/%d, MAE_err %.4f vs M_k_own %.4f (ratio %.2f) -> %s' % (
        wins, nd, mae_e, mae_k, mae_e / mae_k if mae_k > 0 else float('nan'), 'PASS' if rival else 'FAIL'))
    if transfer and rival:
        outcome = 'ERR-TRANSFERS'
    elif mae_e >= mae_k:
        outcome = 'K-ALONE-AS-GOOD'
    elif rival:
        outcome = 'ERR-ORDERS-ONLY'
    elif transfer:
        outcome = 'CALIBRATED-ONLY'
    else:
        outcome = 'INCONCLUSIVE'
    pr('OUTCOME: %s' % outcome)
    out.update(rows=rows, calibrated=cal, n=n, mae=mae, transfer=transfer, transfer_as_frozen=transfer_orig, damped_mae=dmae, wins=wins, nd=nd, mae_err_damped=mae_e,
               mae_kown=mae_k, rival=rival, outcome=outcome)
    return out


def selftest():
    rng = np.random.default_rng(0)
    worlds = (('err world', lambda c: m_err(TEST_ERR[c]), 'ERR-TRANSFERS'),
              ('k world', lambda c: m_err(TEST_ERR[(1.0, c[1])]), 'K-ALONE-AS-GOOD'),       # damping changes nothing
              ('steep world', lambda c: 1.5 * m_err(TEST_ERR[c]), 'ERR-ORDERS-ONLY'),         # right order, wrong scale
              ('2x world', lambda c: 2.0 * m_err(TEST_ERR[c]), 'INCONCLUSIVE'),               # ratio 0.58 > 0.5
              ('big-miss world', lambda c: 0.03 if c == (0.3, 1) else m_err(TEST_ERR[c]), 'INCONCLUSIVE'))  # A1: passed as frozen
    for name, fn, want in worlds:
        runs = {}
        for a in VARIANTS:
            for s in SEEDS:
                base = 1.18 + rng.normal(0, 0.001)
                runs[(a, R, s)] = dict(test=base, nonfinite_steps=0); runs[(a, 0, s)] = dict(test=base + 0.08)
                for k in KS:
                    runs[(a, k, s)] = dict(test=base + 0.08 * fn((a, k)) + rng.normal(0, 0.0003))
        runs[(0.3, R, SEEDS[0])]['nonfinite_steps'] = 13             # exclusion rule exercised
        got = verdict(runs, quiet=True)['outcome']; print('selftest %s -> %s (expect %s)' % (name, got, want)); assert got == want
    print('selftest OK')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit', action='store_true'); ap.add_argument('--verdict', nargs='*'); ap.add_argument('--selftest', action='store_true')
    A = ap.parse_args()
    c, p = refit()
    print('refit on %d Parcae cells: c %.4f p %.4f (locked %s %s)' % (len(FIT), c, p, LOCKED_C, LOCKED_P))
    assert LOCKED_C is not None and abs(c - LOCKED_C) < 1e-3 and abs(p - LOCKED_P) < 1e-3, 'locked curve != refit'
    if A.selftest:
        return selftest()
    if A.fit:
        for cell in CELLS:
            print(cell, 'err %.4f -> M_err %.4f' % (TEST_ERR[cell], m_err(TEST_ERR[cell])))
    if A.verdict:
        json.dump(verdict(load(A.verdict)), open('v167_verdict.json', 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
