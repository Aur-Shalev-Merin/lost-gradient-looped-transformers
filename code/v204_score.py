"""v204 scorer, written after the freeze but before any 96/128-loop data existed, from the rules in
docs/prereg_v204_drift_prediction.md (with amendment 1). Parcae-140m decides; Parcae-370m is descriptive.
  python v204_score.py V204_OUTPUT_DIR [V196_DIR]
"""
import glob, json, math, os, sys
import numpy as np

D204 = sys.argv[1]; D196 = sys.argv[2] if len(sys.argv) > 2 else 'kaggle-output-v196'
PRED = {'140m': {96: (0.891, 0.875, 0.904), 128: (0.900, 0.875, 0.910)},
        '370m': {96: (0.922, 0.858, 0.951), 128: (0.924, 0.866, 0.954)}}
LOST64 = {'140m': {1: 0.818, 2: 0.677, 4: 0.429, 8: 0.206}, '370m': {1: 0.834, 2: 0.679, 4: 0.462, 8: 0.237}}


def load(d, tag, R):
    rows = []
    for f in glob.glob(os.path.join(d, '**', 'v196_%s_R%d.json' % (tag, R)), recursive=True):
        rows += json.load(open(f))['rows']
    return {r['i']: r for r in rows}


def rho(r, R=None):
    p = np.log(np.array(r['adj'][::-1], dtype=float)); R = R or len(p)
    return math.exp(np.polyfit(np.arange(R), p[:R], 1)[0])


g = np.random.default_rng(0)
verdicts = {}
for tag in ('140m', '370m'):
    P = {96: load(D204, tag, 96), 128: load(D204, tag, 128), 64: load(D196, tag, 64)}
    ids = sorted(set(P[96]) & set(P[128]) & set(P[64]))
    print('==== Parcae-%s: %d sequences (%s)' % (tag, len(ids), 'decides' if tag == '140m' else 'descriptive'))
    inside = True
    for R in (96, 128):
        m = float(np.median([rho(P[R][i]) for i in ids])); pr, lo, hi = PRED[tag][R]
        ok = lo <= m <= hi; inside &= ok
        side = '' if ok else (' ABOVE' if m > hi else ' BELOW')
        print('  R=%d rho_fit median %.3f | predicted %.3f [%.3f, %.3f] -> %s%s' % (R, m, pr, lo, hi, 'inside' if ok else 'outside', side))
        if not ok:
            verdicts.setdefault(tag, 'RHO-ABOVE' if m > hi else 'RHO-BELOW')
    rise = np.array([rho(P[128][i]) - rho(P[64][i]) for i in ids])
    boot = [np.median(g.choice(rise, len(rise))) for _ in range(5000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print('  paired rise rho(128) - rho(64): median %+.3f, bootstrap 95%% [%+.3f, %+.3f] (predicted about +0.03)' % (np.median(rise), lo, hi))
    moves = {}
    for R in (96, 128):
        for k in (1, 2, 4, 8):
            v = float(np.median([P[R][i]['err'][str(k)] for i in ids])); moves[(R, k)] = v - LOST64[tag][k]
    print('  lost gradient minus its 64-loop value (k=1/2/4/8): R=96 %s | R=128 %s' % (
        [round(moves[(96, k)], 3) for k in (1, 2, 4, 8)], [round(moves[(128, k)], 3) for k in (1, 2, 4, 8)]))
    win = np.array([rho(P[128][i], 96) - rho(P[96][i]) for i in ids])
    print('  window check: R=128 profile over j<96 minus the R=96 run, median %+.3f, max |.| %.3f' % (np.median(win), np.max(np.abs(win))))
    if tag not in verdicts:
        if max(abs(x) for x in moves.values()) > 0.03:
            verdicts[tag] = 'LOST-MOVES'
        elif lo <= 0:
            verdicts[tag] = 'RHO-FLAT'
        else:
            verdicts[tag] = 'PREDICTED'
    elif max(abs(x) for x in moves.values()) > 0.03:
        verdicts[tag] += ' + LOST-MOVES'
print('VERDICT (Parcae-140m decides): %s | Parcae-370m (descriptive): %s' % (verdicts.get('140m'), verdicts.get('370m')))
