"""Why the fitted rho drifts up with the loop count while the lost gradient stays put: a derivation, checked AFTER THE
FACT on stored S10 (v196) data. Doc: docs/drift_derivation_2026-10-15.md. CPU only.

Model: the backward signal's size counted back from the output is s_j = A mu_f^j + B mu_s^j (fast near-output mode,
slow far-fade mode; mu_f < mu_s < 1). Its log is close to a hockey stick: slope a_f = log mu_f up to the crossover
j* = log(A/B) / log(mu_s/mu_f), slope a_s = log mu_s after it. A least-squares line over j = 0..R-1 has slope
    a_s + (a_f - a_s) * (3u^2 - 2u^3),   u = j*/R   (0 <= u <= 1; continuous-j limit)
so rho_fit(R) moves from mu_f toward mu_s as R grows. The lost gradient at small k is set by the near-output
contributions and does not see loops beyond R, so it stays put (bound in the doc).

Checks (Parcae-140m and -370m, 16 sequences at R = 32, 48, 64, same sequences):
  1. window: the R = 64 profile's own least-squares slope over j < 32 and j < 48 vs rho fitted in the R = 32 and 48 runs
  2. near-output profile identity across R (log-profile difference over j < 32)
  3. two-mode fit to each R = 64 profile -> mu_f, mu_s, j*; closed form vs exact window slope
  4. predictions for R = 96 and 128 (rho_fit; lost gradient unchanged), for a later out-of-sample run
  python paper_drift_derivation.py [V196_DIR]
"""
import glob, json, math, os, sys
import numpy as np
from scipy.optimize import least_squares

D = sys.argv[1] if len(sys.argv) > 1 else 'kaggle-output-v196'


def load(tag, R):
    rows = []
    for f in glob.glob(os.path.join(D, '**', 'v196_%s_R%d.json' % (tag, R)), recursive=True):
        rows += json.load(open(f))['rows']
    return {r['i']: r for r in rows}


def prof(r):                       # sizes counted back from the output (j = 0 is the last loop)
    return np.array(r['adj'][::-1], dtype=float)


def slope(logs, R):
    return np.polyfit(np.arange(R), logs[:R], 1)[0]


def hockey(a_f, a_s, js, R):
    u = min(max(js / R, 0.0), 1.0)
    return a_s + (a_f - a_s) * (3 * u * u - 2 * u ** 3)


def two_mode(p):
    j = np.arange(len(p)); y = np.log(p / p[0])

    def res(th):
        la, lb, af, as_ = th
        return np.logaddexp(la + af * j, lb + as_ * j) - y
    best = None
    for af0 in (-0.5, -0.3, -0.15):
        for as0 in (-0.02, -0.06, -0.1):
            r = least_squares(res, [0.0, -3.0, af0, as0], bounds=([-20, -30, -3, -1], [5, 5, 0.5, 0.5]))
            if best is None or r.cost < best.cost:
                best = r
    la, lb, af, as_ = best.x
    if af > as_:                   # name the faster-decaying mode "fast"
        la, lb, af, as_ = lb, la, as_, af
    js = (la - lb) / (as_ - af) if as_ != af else float('nan')
    return dict(a_f=af, a_s=as_, js=js, rms=float(np.sqrt(np.mean(best.fun ** 2))))


med = lambda xs: float(np.median(xs))
for tag in ('140m', '370m'):
    P = {R: load(tag, R) for R in (32, 48, 64)}
    ids = sorted(set(P[32]) & set(P[48]) & set(P[64]))
    print('==== Parcae-%s: %d sequences in all three runs' % (tag, len(ids)))
    # 1. window check
    fitted = {R: [math.exp(slope(np.log(prof(P[R][i])), R)) for i in ids] for R in (32, 48, 64)}
    from64 = {R: [math.exp(slope(np.log(prof(P[64][i])), R)) for i in ids] for R in (32, 48, 64)}
    for R in (32, 48, 64):
        d = np.array(from64[R]) - np.array(fitted[R])
        print('  R=%d  rho fitted in its own run %.3f | R=64 profile over j<%d %.3f | per-sequence diff median %+.3f, max |.| %.3f'
              % (R, med(fitted[R]), R, med(from64[R]), med(d), np.max(np.abs(d))))
    # 2. near-output identity
    for R in (32, 48):
        dl = [np.log(prof(P[R][i])[:R] / prof(P[R][i])[0]) - np.log(prof(P[64][i])[:R] / prof(P[64][i])[0]) for i in ids]
        print('  profile over j<%d at R=%d vs R=64 (log, normalised at j=0): median |diff| %.3f, 90th pct %.3f'
              % (R, R, med(np.abs(np.concatenate(dl))), float(np.percentile(np.abs(np.concatenate(dl)), 90))))
    # 3. two-mode fits at R = 64
    fits = [two_mode(prof(P[64][i])) for i in ids]
    mf, ms, js = (np.array([f[k] for f in fits]) for k in ('a_f', 'a_s', 'js'))
    print('  two-mode fit at R=64: mu_f %.3f, mu_s %.3f, crossover j* %.1f (medians), log-fit rms %.3f'
          % (math.exp(med(mf)), math.exp(med(ms)), med(js), med([f['rms'] for f in fits])))
    for R in (32, 48, 64, 96, 128):
        closed = [math.exp(hockey(f['a_f'], f['a_s'], f['js'], R)) for f in fits]
        exact = [math.exp(np.polyfit(np.arange(R), np.logaddexp(0, 0) * 0 + np.log(
                 np.exp(f['a_f'] * np.arange(R)) + np.exp(f['a_s'] * (np.arange(R) - f['js']) + f['a_f'] * f['js'])), 1)[0])
                 for f in fits]
        meas = med(fitted[R]) if R in fitted else float('nan')
        print('  R=%3d  rho_fit: closed form %.3f | exact slope of the fitted two-mode curve %.3f | measured %s'
              % (R, med(closed), med(exact), '%.3f' % meas if R in fitted else '-'))
    # lost gradient vs R
    for k in ('1', '2', '4', '8', '16'):
        v = {R: med([P[R][i]['err'][k] for i in ids]) for R in (32, 48, 64)}
        print('  lost gradient k=%-2s at R=32/48/64: %.3f / %.3f / %.3f; rho_fit^k (measured) %.3f / %.3f / %.3f'
              % (k, v[32], v[48], v[64], *[med(fitted[R]) ** int(k) for R in (32, 48, 64)]))
    tail = [prof(P[64][i])[32:].sum() / prof(P[64][i]).sum() for i in ids]
    print('  share of the R=64 backward signal (summed sizes) beyond loop 32: median %.4f, max %.4f' % (med(tail), max(tail)))


# ---- frozen out-of-sample predictions (R = 96, 128): bootstrap over sequences of the median predicted rho_fit
def pred_rho(f, R):
    j = np.arange(R)
    return math.exp(np.polyfit(j, np.logaddexp(f['a_f'] * j, f['a_s'] * (j - f['js']) + f['a_f'] * f['js']), 1)[0])


g = np.random.default_rng(0)
for tag in ('140m', '370m'):
    P64 = load(tag, 64); ids = sorted(P64)
    fits = [two_mode(prof(P64[i])) for i in ids]
    for R in (96, 128):
        pr = np.array([pred_rho(f, R) for f in fits]); boot = [np.median(g.choice(pr, len(pr))) for _ in range(5000)]
        print('PREDICTION Parcae-%s R=%d: median rho_fit %.3f, bootstrap 95%% [%.3f, %.3f]' % (tag, R, np.median(pr), *np.percentile(boot, [2.5, 97.5])))
