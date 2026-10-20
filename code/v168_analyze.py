"""v168 analysis (EXPLORATORY; written before the v168 measurements came back).

For each candidate metric, turn it into a distance d (0 = truncated gradient as good as full):
  err, adam_err: d = metric;  cos, adam_cos, useful_sgd, useful_adam, sign_agree: d = 1 - metric
Fit phi = c * d^p on the 29 Parcae-140m cells ONLY (observed phi from v162-v166), then predict the 6 retrofit cells
(v167) unchanged. Report per metric: Parcae in-fit R^2 and MAE; retrofit MAE (all 6 and the 4 damped cells); whether
the curve is calibrated on the retrofit damped cells (|res| <= max(0.02, 0.3 phi)).

Decision rule fixed in advance (exploratory, so a "hit" only earns a pre-registered confirmation, not a claim):
  a metric is a CANDIDATE if (a) Parcae in-fit R^2 >= 0.9, (b) retrofit damped-cell MAE <= 0.015, and
  (c) all 4 damped retrofit cells calibrated. With 6 metrics tried, expect false hits; report all of them.
  If none qualifies: the training-cost line closes (gradient law stands; err->cost is Parcae-specific).

  python v168_analyze.py JSON...
"""
import json, sys
import numpy as np
from scipy.optimize import curve_fit

# observed phi: Parcae (model, R, a, k) from v162-v166 docs; retrofit from v167
PARCAE = {(8, 1.0, 1): 0.183, (8, 1.0, 2): 0.077, (8, 1.0, 4): 0.012,
          (16, 1.0, 1): 0.209, (16, 1.0, 2): 0.090, (16, 1.0, 4): 0.016, (16, 1.0, 8): -0.005,
          (32, 1.0, 1): 0.224, (32, 1.0, 2): 0.100, (32, 1.0, 4): 0.024, (32, 1.0, 8): 0.002,
          (32, 0.5, 1): 0.212, (32, 0.5, 2): 0.142, (32, 0.5, 4): 0.072, (32, 0.5, 8): 0.020,
          (32, 0.7, 2): 0.127, (32, 0.7, 3): 0.076, (32, 0.7, 4): 0.046, (32, 0.7, 6): 0.015,
          (64, 0.3, 2): 0.169, (64, 0.3, 3): 0.137, (64, 0.3, 4): 0.111, (64, 0.3, 6): 0.073,
          (32, 1.0, 3): 0.0479, (32, 1.0, 6): 0.0046,
          (64, 0.3, 8): 0.0578, (64, 0.3, 12): 0.0268, (48, 0.4, 4): 0.0879, (48, 0.4, 8): 0.0316}
RETRO = {(8, 1.0, 1): 0.0087, (8, 1.0, 2): -0.0012, (8, 0.5, 1): 0.0113, (8, 0.5, 2): 0.0019,
         (8, 0.3, 1): 0.0087, (8, 0.3, 2): 0.0042}
METRICS = {'err': lambda x: x, 'adam_err': lambda x: x, 'cos': lambda x: 1 - x, 'adam_cos': lambda x: 1 - x,
           'useful_sgd': lambda x: 1 - x, 'useful_adam': lambda x: 1 - x, 'sign_agree': lambda x: 1 - x}


def load(files):
    meas = {}
    for f in files:
        for row in json.load(open(f)):
            for k, mt in row['metrics'].items():
                meas[(row['model'], int(row['R']), round(float(row['a']), 2), int(k))] = mt
    return meas


def main(files):
    meas = load(files)
    miss = [c for c in PARCAE if ('parcae',) + c not in meas] + [c for c in RETRO if ('retro',) + c not in meas]
    print('cells measured: parcae %d/%d, retro %d/%d; missing %s' % (
        sum(('parcae',) + c in meas for c in PARCAE), len(PARCAE), sum(('retro',) + c in meas for c in RETRO), len(RETRO), miss))
    out = {}
    for name, tod in METRICS.items():
        xp = np.array([max(tod(meas[('parcae',) + c][name]), 1e-6) for c in PARCAE if ('parcae',) + c in meas])
        yp = np.array([PARCAE[c] for c in PARCAE if ('parcae',) + c in meas])
        try:
            (c, p), _ = curve_fit(lambda x, c, p: c * x ** p, xp, yp, p0=(0.3, 2.0), bounds=([0, 0.1], [100, 12]), maxfev=20000)
        except RuntimeError:
            print('%s: fit failed' % name); continue
        pred = c * xp ** p
        r2 = 1 - np.sum((yp - pred) ** 2) / np.sum((yp - yp.mean()) ** 2)
        rows = []
        for cell, obs in RETRO.items():
            if ('retro',) + cell not in meas:
                continue
            d = max(tod(meas[('retro',) + cell][name]), 1e-6); pr = c * d ** p
            rows.append((cell, d, obs, pr, abs(obs - pr), abs(obs - pr) <= max(0.02, 0.3 * abs(obs))))
        dmg = [r for r in rows if r[0][1] < 1.0]
        mae_all = float(np.mean([r[4] for r in rows])); mae_d = float(np.mean([r[4] for r in dmg]))
        cand = r2 >= 0.9 and mae_d <= 0.015 and len(dmg) == 4 and all(r[5] for r in dmg)
        out[name] = dict(c=float(c), p=float(p), r2=float(r2), mae_retro=mae_all, mae_retro_damped=mae_d, candidate=bool(cand))
        print('\n%-11s fit c %.3f p %.2f | Parcae R2 %.3f, MAE %.4f | retro MAE %.4f, damped %.4f -> %s' % (
            name, c, p, r2, float(np.mean(np.abs(yp - pred))), mae_all, mae_d, 'CANDIDATE' if cand else '-'))
        for cell, d, obs, pr, res, ok in rows:
            print('   retro a=%.1f k=%d: d %.4f | obs %+.4f pred %.4f |res| %.4f%s' % (cell[1], cell[2], d, obs, pr, res, ' ok' if ok else ''))
    json.dump(out, open('v168_summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1:])
