"""v171 analysis (EXPLORATORY; docs/v171_plan.md), written before any v171 data returned.
Fit phi = c*d^p on the 29 Parcae cells per metric; predict the 6 retrofit cells; candidate rule (primary = adam_def):
(a) Parcae R^2 >= 0.9, (b) retrofit damped MAE <= 0.015, (c) all 4 damped retrofit cells calibrated,
(d) Spearman >= 0.85 between predicted and observed phi over all 35 cells.

  python v171_analyze.py JSON...
"""
import json, sys
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

from v168_analyze import PARCAE, RETRO

METRICS = {'adam_def': lambda x: x, 'sgd_def': lambda x: x, 'err_mean': lambda x: x, 'adam_cos': lambda x: 1 - x}


def main(files):
    meas = {}
    for f in files:
        for row in json.load(open(f)):
            for k, mt in row['metrics'].items():
                meas[(row['model'], int(row['R']), round(float(row['a']), 2), int(k))] = mt
    cells = [('parcae',) + c for c in PARCAE] + [('retro',) + c for c in RETRO]
    obs = {('parcae',) + c: v for c, v in PARCAE.items()}; obs.update({('retro',) + c: v for c, v in RETRO.items()})
    miss = [c for c in cells if c not in meas]
    print('measured %d/%d; missing %s' % (len(cells) - len(miss), len(cells), miss))
    # SANITY GATE (added 2026-10-05 after the first run's apparatus flaw, before v171b data): within each variant,
    # err_mean must be non-increasing in k (tolerance 0.02) and < 0.98 whenever k >= R/4. Otherwise: do not score.
    bad = []
    for c in cells:
        if c not in meas:
            continue
        same = sorted([d for d in meas if d[:3] == c[:3]], key=lambda d: d[3])
        for d0, d1 in zip(same, same[1:]):
            if meas[d1]['err_mean'] > meas[d0]['err_mean'] + 0.02:
                bad.append(('not decreasing in k', d0, d1))
        if c[3] >= c[1] / 4 and meas[c]['err_mean'] >= 0.98:
            bad.append(('err >= 0.98 at k >= R/4', c))
    if bad:
        print('SANITY GATE FAILED (%d issues) -> NOT SCORED. First: %s' % (len(set(map(str, bad))), bad[:4])); return
    print('sanity gate passed')
    for name, tod in METRICS.items():
        pc = [c for c in cells if c[0] == 'parcae' and c in meas]
        x = np.array([max(tod(meas[c][name]), 1e-6) for c in pc]); y = np.array([obs[c] for c in pc])
        (c_, p_), _ = curve_fit(lambda z, c, p: c * z ** p, x, y, p0=(0.5, 1.5), bounds=([0, 0.1], [1000, 12]), maxfev=20000)
        r2 = 1 - np.sum((y - c_ * x ** p_) ** 2) / np.sum((y - y.mean()) ** 2)
        pred = {c: c_ * max(tod(meas[c][name]), 1e-6) ** p_ for c in cells if c in meas}
        dm = [c for c in cells if c[0] == 'retro' and c[2] < 1.0 and c in meas]
        res = {c: abs(obs[c] - pred[c]) for c in pred}
        mae_d = float(np.mean([res[c] for c in dm])); cal = all(res[c] <= max(0.02, 0.3 * abs(obs[c])) for c in dm)
        sp = spearmanr([pred[c] for c in pred], [obs[c] for c in pred]).correlation
        cand = r2 >= 0.9 and mae_d <= 0.015 and len(dm) == 4 and cal and sp >= 0.85
        print('\n%-9s%s c %.3f p %.2f | Parcae R2 %.3f | retro damped MAE %.4f, calibrated %s | Spearman(35) %.3f -> %s' % (
            name, ' (PRIMARY)' if name == 'adam_def' else '', c_, p_, r2, mae_d, cal, sp, 'CANDIDATE' if cand else '-'))
        for c in [c for c in cells if c[0] == 'retro' and c in meas]:
            print('   retro a=%.1f k=%d: d %.4f | obs %+.4f pred %.4f' % (c[2], c[3], tod(meas[c][name]), obs[c], pred[c]))


if __name__ == '__main__':
    main(sys.argv[1:])
