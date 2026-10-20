"""v165 analysis, written before any v165 run (prereg: docs/prereg_v165_undamped_rerun.md).

Re-runs v164's two undamped test cells (R 32, a 1.0, k 3 and 6) with their controls (k 0, k 32) on FRESH seeds, with a
pre-stated instability exclusion, and scores them with v164's FROZEN predictors (v164_analyze: M_err, M_k, locked test
errors from the v164 measurement JSON). No refit.

Seeds 1651101..1651107 are run; a seed is VALID iff its full-backprop control (k 32) has <= 10 non-finite steps and a
finite test loss. The first 5 valid seeds (in seed order) are used; if fewer than 5 are valid, the outcome is
UNINFORMATIVE.

  python v165_analyze.py --verdict V165_RUNS... --v164_runs V164_RUNS... --measure_json v164_measure_measure.json
Primary: v164's criteria over its 10 test cells, with v165's undamped cells replacing v164's (v164's a 0.7 / a 0.3
cells as observed in v164). Secondary: the two undamped cells alone (M_err closer than M_k in both, and
|obs - M_err| <= 0.015 in both).
"""
import argparse, json, math
import numpy as np

import v164_analyze as A64

SEEDS = list(range(1651101, 1651108))
N_USE = 5
MAX_NONFINITE = 10
R, AL = 32, 1.0


def load(files):
    runs = {}
    for f in files:
        for line in open(f):
            r = json.loads(line); runs[(int(r['R']), float(r['a']), int(r['k']), r['seed'])] = r
    return runs


def valid_seeds(runs):
    ok = []
    for s in SEEDS:
        f = runs.get((R, AL, R, s))
        if f is None:
            continue
        good = f['nonfinite_steps'] <= MAX_NONFINITE and math.isfinite(f['test']) and not f['diverged']
        print('seed %d: full control nonfinite %d test %.4f -> %s' % (s, f['nonfinite_steps'], f['test'], 'VALID' if good else 'EXCLUDED'))
        if good and all((R, AL, k, s) in runs for k in (0, 3, 6)):
            ok.append(s)
    return ok[:N_USE]


def phi_cell(runs, seeds, k):
    vals = []
    for s in seeds:
        x, f, z = runs[(R, AL, k, s)]['test'], runs[(R, AL, R, s)]['test'], runs[(R, AL, 0, s)]['test']
        vals.append((x - f) / (z - f))
    return float(np.mean(vals)), float(np.std(vals, ddof=1) / math.sqrt(len(vals)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verdict', nargs='+', required=True); ap.add_argument('--v164_runs', nargs='+', required=True)
    ap.add_argument('--measure_json', required=True)
    a = ap.parse_args()
    test_err = A64.load_test_err(a.measure_json); preds = A64.predictions(test_err)
    r65 = load(a.verdict); seeds = valid_seeds(r65)
    print('valid seeds used: %s' % seeds)
    if len(seeds) < N_USE:
        print('OUTCOME: UNINFORMATIVE (fewer than %d valid seeds)' % N_USE); return
    floor = [r65[(R, AL, 0, s)]['test'] - r65[(R, AL, R, s)]['test'] for s in seeds]
    print('floor gap %.4f (min %.4f)' % (np.mean(floor), min(floor)))
    # v164 runs for the a 0.7 / a 0.3 cells, scored exactly as in v164; then replace the undamped cells
    r64 = A64.load(a.v164_runs)
    base = A64.verdict(r64, test_err, quiet=True)
    rows = [r for r in base['rows'] if r['cell'][1] != 1.0]
    sec = []
    for k in (3, 6):
        obs, se = phi_cell(r65, seeds, k); p = preds[(R, AL, k)]
        row = dict(cell=(R, AL, k), obs=obs, se=se, pred_err=p['pred_err'], pred_k=p['pred_k'],
                   res_err=abs(obs - p['pred_err']), res_k=abs(obs - p['pred_k']))
        rows.append(row); sec.append(row)
        print('v165 R=%d a=%.1f k=%d: obs phi %+.4f (se %.4f) | M_err %.4f (|res| %.4f) | M_k %.4f (|res| %.4f)' % (
            R, AL, k, obs, se, p['pred_err'], row['res_err'], p['pred_k'], row['res_k']))
    mae_err = float(np.mean([r['res_err'] for r in rows])); mae_k = float(np.mean([r['res_k'] for r in rows]))
    wins = sum(r['res_err'] < r['res_k'] for r in rows)
    disc = [r for r in rows if abs(r['pred_err'] - r['pred_k']) >= 0.03]; dw = sum(r['res_err'] < r['res_k'] for r in disc)
    print('\nPRIMARY (v164 criteria, 10 cells): MAE_err %.4f MAE_k %.4f ratio %.2f | closer %d/%d | discriminating %d, closer %d'
          % (mae_err, mae_k, mae_err / mae_k, wins, len(rows), len(disc), dw))
    if mae_err <= 0.7 * mae_k and mae_err <= 0.03 and wins >= math.ceil(0.7 * len(rows)) and (not disc or dw >= math.ceil(0.7 * len(disc))):
        outcome = 'ERR-PREDICTS'
    elif mae_err >= mae_k:
        outcome = 'K-ALONE-AS-GOOD'
    else:
        outcome = 'INCONCLUSIVE'
    sec_ok = all(r['res_err'] < r['res_k'] and r['res_err'] <= 0.015 for r in sec)
    print('PRIMARY OUTCOME: %s' % outcome)
    print('SECONDARY (undamped cells alone: M_err closer and |res| <= 0.015 in both): %s' % ('PASS' if sec_ok else 'FAIL'))
    json.dump(dict(seeds=seeds, rows=rows, mae_err=mae_err, mae_k=mae_k, wins=wins, outcome=outcome, secondary=sec_ok),
              open('v165_verdict.json', 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
