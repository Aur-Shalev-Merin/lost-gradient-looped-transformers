"""v174 analysis, LOCKED at the freeze (prereg: docs/prereg_v174_direction_intervention.md, frozen version).

Gated dose-collapse test on Parcae-140m: at matched realised Adam-update agreement Abar, do k = 1 and k = 4 truncation
mix arms cost the same? Pairs were matched on Abar in the single-seed pilot (seed 1749901) and gated (frozen error
cosine <= 0.75).

  cost phi = (L_arm - L_FULL) / (L_FLOOR - L_FULL), test loss, per seed.

Rules (the prereg's section 3, implemented exactly here):
1. Seeds 1741101-1741108; backup 1741109. A seed whose FULL run has > 10 non-finite (skipped) steps or a non-finite
   test loss is excluded and replaced by the backup in every arm.
2. FLOOR - FULL gap: mean over the used seeds (seeds with a finite FLOOR) < 0.03 -> UNINFORMATIVE.
3. Stability, per pair and seed: both arms and FLOOR have <= 10 non-finite steps and a finite test loss. A pair is
   eligible only with >= 6 such seeds.
4. Drift guard: |mean over the pair's seeds of (Abar_k1 - Abar_k4)| <= 0.01, else the pair is DRIFTED (not eligible).
5. Scoring units: P1, P2, P3 and the high-dose slot H = {P4p, P5}. P5 was added as a substitute for P4p (same dose,
   one step further from k = 4's unstable zone), so H gets ONE verdict: if both pairs are eligible and share >= 6
   seeds, per-seed values are the average of the two pairs; otherwise the eligible pair with more seeds is used.
6. Primary test per unit (Abar-adjusted TOST): regress the per-seed d_phi = phi_k1 - phi_k4 on the per-seed
   d_A = Abar_k1 - Abar_k4; the intercept is the cost difference at equal agreement. 90% CI of the intercept
   (t, n - 2 df). tau = max(0.015, 0.25 * mean phi of the unit's arms).
   same = CI inside [-tau, tau]; different = CI entirely outside; else undecided.
   The unadjusted TOST (90% t-CI of mean d_phi) is reported as a sensitivity analysis; where the two disagree, the
   primary decides and the disagreement is reported.
7. Within-k order: over the arms of eligible pairs, an arm with higher mean Abar must not cost more than an arm with
   lower mean Abar by more than max(0.015, 0.25 * the two arms' mean phi). Each arm's Abar and phi means are taken
   over the same seeds (its pair's stable seeds).
8. Outcome, in this order of precedence:
   UNINFORMATIVE          floor gap < 0.03, or fewer than 3 scored units
   AGREEMENT-INSUFFICIENT any scored unit 'different'
   AGREEMENT-PREDICTS     every scored unit 'same', P3 and H both scored, no within-k inversion
   INCONCLUSIVE           otherwise
9. Reported, not scored: underpowered flag (unadjusted paired SD > 0.011), non-finite steps per arm, the
   phi = c (1 - Abar)^p fit over all mix arms, loop-only Abar, clipped fractions, the FULL pipeline check (v174 FULL
   vs v163 train_cell FULL on seed 1741101).
Duplicate (arm, seed) records are a hard error (resumed Kaggle runs must not leave a choice of record).

  python v174_analyze.py --verdict RUNS.jsonl ...
  python v174_analyze.py --selftest
"""
import argparse, json, math
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

# (name, (k, lam) for k = 1, (k, lam) for k = 4). Pilot |dAbar| / frozen error cosine in comments.
PAIRS = {'P1': ((1, 0.5), (4, 0.94)),        # 0.0005 / 0.578
         'P2': ((1, 0.625), (4, 1.17)),      # 0.0005 / 0.477
         'P3': ((1, 0.82), (4, 1.75)),       # 0.0030 / 0.536
         'P4p': ((1, 0.924), (4, 2.03)),     # 0.0011 / 0.611
         'P5': ((1, 0.91), (4, 1.97))}       # 0.0007 / 0.567
UNITS = [('P1', ['P1']), ('P2', ['P2']), ('P3', ['P3']), ('H', ['P4p', 'P5'])]
REQUIRED = ('P3', 'H')
SEEDS = list(range(1741101, 1741109)); BACKUP = [1741109]
MAX_NONFINITE, MIN_SEEDS, DRIFT, FLOOR_GAP, TAU_MIN, SD_POWER = 10, 6, 0.01, 0.03, 0.015, 0.011


def arm(k, lam):
    return 'k%d_l%.3f' % (k, lam)


def ci90(x):
    x = np.asarray(x, float); n = len(x)
    h = stats.t.ppf(0.95, n - 1) * x.std(ddof=1) / math.sqrt(n)
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def intercept_ci90(y, x):
    """OLS y = a + b x; returns (a, lo, hi) with a t(n - 2) 90% CI. Falls back to the plain mean if x has no spread."""
    y, x = np.asarray(y, float), np.asarray(x, float); n = len(y)
    sxx = float(((x - x.mean()) ** 2).sum())
    if n < 4 or sxx < 1e-12:
        return ci90(y)
    b = float(((x - x.mean()) * (y - y.mean())).sum() / sxx); a = float(y.mean() - b * x.mean())
    s2 = float(((y - a - b * x) ** 2).sum() / (n - 2))
    se = math.sqrt(s2 * (1.0 / n + x.mean() ** 2 / sxx)); h = stats.t.ppf(0.95, n - 2) * se
    return a, a - h, a + h


def load(files):
    runs = {}
    for f in files:
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line); key = (r['arm'], int(r['seed']))
            if key in runs:
                raise SystemExit('DUPLICATE record %s in %s: resolve before analysis (no silent choice)' % (key, f))
            runs[key] = r
    return runs


def ok_run(r):
    return r is not None and r['nonfinite'] <= MAX_NONFINITE and math.isfinite(r['test'])


def used_seeds(runs, pr):
    out, backups = [], list(BACKUP)
    for s in SEEDS:
        f = runs.get(('full', s))
        if ok_run(f):
            out.append(s); continue
        pr('seed %d: FULL %s -> EXCLUDED' % (s, 'missing' if f is None else 'nonfinite %d test %s' % (f['nonfinite'], f['test'])))
        while backups:
            b = backups.pop(0)
            if ok_run(runs.get(('full', b))):
                pr('  replaced by backup %d' % b); out.append(b); break
    return out


def phi(runs, name, s):
    f, z, x = runs[('full', s)]['test'], runs[('floor', s)]['test'], runs[(name, s)]['test']
    return (x - f) / (z - f)


def pair_data(runs, seeds, name):
    n1, n2 = arm(*PAIRS[name][0]), arm(*PAIRS[name][1])
    ss = [s for s in seeds if ok_run(runs.get((n1, s))) and ok_run(runs.get((n2, s))) and ok_run(runs.get(('floor', s)))]
    d = dict(name=name, arms=(n1, n2), seeds=ss)
    if not ss:
        d['status'] = 'UNSTABLE (0 seeds)'; return d
    d['phi1'] = {s: phi(runs, n1, s) for s in ss}; d['phi2'] = {s: phi(runs, n2, s) for s in ss}
    d['A1'] = {s: runs[(n1, s)]['Abar'] for s in ss}; d['A2'] = {s: runs[(n2, s)]['Abar'] for s in ss}
    d['dA'] = float(np.mean([d['A1'][s] - d['A2'][s] for s in ss]))
    d['status'] = ('UNSTABLE (%d seeds)' % len(ss)) if len(ss) < MIN_SEEDS else ('DRIFTED' if abs(d['dA']) > DRIFT else 'eligible')
    return d


def test_unit(pds):
    """pds: eligible pair data (1 or 2). Returns the unit's per-seed vectors and both tests."""
    if len(pds) == 2:
        common = [s for s in pds[0]['seeds'] if s in pds[1]['seeds']]
        if len(common) >= MIN_SEEDS:
            ss = common
            dphi = [np.mean([p['phi1'][s] - p['phi2'][s] for p in pds]) for s in ss]
            dA = [np.mean([p['A1'][s] - p['A2'][s] for p in pds]) for s in ss]
            mphi = float(np.mean([p[k][s] for p in pds for k in ('phi1', 'phi2') for s in ss]))
            return finish(dphi, dA, mphi, '+'.join(p['name'] for p in pds), ss)
        pds = [max(pds, key=lambda p: len(p['seeds']))]
    p = pds[0]; ss = p['seeds']
    dphi = [p['phi1'][s] - p['phi2'][s] for s in ss]; dA = [p['A1'][s] - p['A2'][s] for s in ss]
    mphi = float(np.mean([p[k][s] for k in ('phi1', 'phi2') for s in ss]))
    return finish(dphi, dA, mphi, p['name'], ss)


def finish(dphi, dA, mphi, used, ss):
    tau = max(TAU_MIN, 0.25 * mphi)
    lab = lambda lo, hi: 'same' if (lo >= -tau and hi <= tau) else 'different' if (hi < -tau or lo > tau) else 'undecided'
    a, alo, ahi = intercept_ci90(dphi, dA); m, lo, hi = ci90(dphi)
    return dict(used=used, n=len(ss), dA=float(np.mean(dA)), mean_phi=mphi, tau=tau, adj=(a, alo, ahi), raw=(m, lo, hi),
                status=lab(alo, ahi), raw_status=lab(lo, hi), sd=float(np.std(dphi, ddof=1)))


def verdict(runs, quiet=False):
    pr = (lambda *a, **k: None) if quiet else print
    seeds = used_seeds(runs, pr); out = dict(seeds=seeds, pairs={}, units={})
    gap = [runs[('floor', s)]['test'] - runs[('full', s)]['test'] for s in seeds if ok_run(runs.get(('floor', s)))]
    gm = float(np.mean(gap)) if gap else float('nan'); out['floor_gap'] = gm
    pr('FLOOR - FULL gap: %.4f over %d seeds%s' % (gm, len(gap), '' if gm >= FLOOR_GAP else ' (< 0.03)'))
    pds = {n: pair_data(runs, seeds, n) for n in PAIRS}
    for n, d in pds.items():
        out['pairs'][n] = {k: d[k] for k in ('arms', 'status') if k in d} | dict(n_seeds=len(d['seeds']), dA=d.get('dA'))
        pr('%-4s %s vs %s: %d seeds, dAbar %s -> %s' % (n, d['arms'][0], d['arms'][1], len(d['seeds']),
           '%+.4f' % d['dA'] if 'dA' in d else '-', d['status']))
    scored, sds = [], []
    for u, members in UNITS:
        el = [pds[n] for n in members if pds[n]['status'] == 'eligible']
        if not el:
            out['units'][u] = dict(status='not scored'); pr('unit %-3s not scored' % u); continue
        r = test_unit(el); out['units'][u] = r; scored.append(u); sds.append(r['sd'])
        pr('unit %-3s [%s, n %d] dAbar %+.4f tau %.4f | adjusted %+.4f [%+.4f, %+.4f] %s | raw %+.4f [%+.4f, %+.4f] %s%s' % (
            u, r['used'], r['n'], r['dA'], r['tau'], *r['adj'], r['status'], *r['raw'], r['raw_status'],
            '  (raw and adjusted DISAGREE; adjusted decides)' if r['raw_status'] != r['status'] else ''))
    # within-k order over the arms of eligible pairs (Abar and phi means over the same seeds)
    inv = []
    for idx in (0, 1):
        am = {}
        for n, d in pds.items():
            if d['status'] == 'eligible':
                key = ('phi1', 'A1') if idx == 0 else ('phi2', 'A2')
                am[d['arms'][idx]] = (float(np.mean(list(d[key[1]].values()))), float(np.mean(list(d[key[0]].values()))))
        for a, (Aa, fa) in am.items():
            for b, (Ab, fb) in am.items():
                if Aa > Ab and fa - fb > max(TAU_MIN, 0.25 * (fa + fb) / 2):
                    inv.append((a, b, fa - fb))
    out['inversions'] = inv
    for a, b, d in inv:
        pr('INVERSION: %s (higher Abar) costs %.4f more than %s' % (a, d, b))
    st = [out['units'][u]['status'] for u in scored]
    if len(scored) < 3 or not gm >= FLOOR_GAP:
        outcome = 'UNINFORMATIVE'
    elif 'different' in st:
        outcome = 'AGREEMENT-INSUFFICIENT'
    elif all(x == 'same' for x in st) and all(u in scored for u in REQUIRED) and not inv:
        outcome = 'AGREEMENT-PREDICTS'
    else:
        outcome = 'INCONCLUSIVE'
    out.update(outcome=outcome, scored=scored, max_paired_sd=max(sds) if sds else None,
               underpowered=bool(sds) and max(sds) > SD_POWER)
    pr('scored units %d/%d %s; max paired SD %s%s' % (len(scored), len(UNITS), scored, '%.4f' % max(sds) if sds else '-',
       ' (> 0.011: UNDERPOWERED)' if out['underpowered'] else ''))
    pr('OUTCOME: %s' % outcome)
    return out


def secondary(runs, seeds, pr=print):
    names = sorted({n for n, _ in runs if n.startswith('k')})
    rows = []
    for n in names:
        nf = [runs[(n, s)]['nonfinite'] for s in seeds if (n, s) in runs]
        ss = [s for s in seeds if ok_run(runs.get((n, s))) and ok_run(runs.get(('floor', s)))]
        if not ss:
            pr('  %-12s no stable seeds; non-finite steps %s' % (n, nf)); continue
        rows.append((n, float(np.mean([runs[(n, s)]['Abar'] for s in ss])), float(np.mean([runs[(n, s)]['Abar_loop'] for s in ss])),
                     float(np.mean([phi(runs, n, s) for s in ss])), float(np.mean([runs[(n, s)]['frac_both_clipped'] for s in ss])), nf))
    for n, A, Al, f, c, nf in rows:
        pr('  %-12s Abar %.4f  Abar_loop %.4f  phi %+.4f  both-clipped %.2f  non-finite steps %s' % (n, A, Al, f, c, nf))
    fit = None
    if len(rows) >= 3:
        try:
            x = np.array([1 - r[1] for r in rows]); y = np.array([r[3] for r in rows])
            (c, p), _ = curve_fit(lambda x, c, p: c * x ** p, x, y, p0=(1.0, 1.0), bounds=([0, 0.1], [1e3, 6]))
            fit = dict(c=float(c), p=float(p), mae=float(np.mean(np.abs(c * x ** p - y))))
            pr('  fit phi = %.3f (1 - Abar)^%.2f, MAE %.4f' % (c, p, fit['mae']))
        except RuntimeError:
            pr('  fit failed')
    chk = None
    if ('full163', SEEDS[0]) in runs and ('full', SEEDS[0]) in runs:
        a, b = runs[('full', SEEDS[0])]['test'], runs[('full163', SEEDS[0])]['test']
        chk = dict(v174_full=a, v163_full=b, diff=a - b)
        pr('  FULL pipeline check seed %d: v174 %.4f vs v163 train_cell %.4f (diff %+.4f)' % (SEEDS[0], a, b, a - b))
    return dict(arms=rows, fit=fit, full_pipeline_check=chk)


def curve(A):
    return 3.24 * max(1 - A, 0) ** 1.23          # the reviewer's fit through plain k1 / k4 truncation


def synth(rng, world='collapse', nonfinite_full=None, drift=None, unstable=(), extra=None, sd=0.004, a_noise=0.007):
    """Per seed and arm: Abar = pilot target + N(0, a_noise) (+ drift on the k4 arm); phi = curve(Abar) + N(0, sd)
    (+ extra[pair] on the k4 arm). Cost follows the per-seed Abar, so the adjusted test matters."""
    target = {'P1': 0.9853, 'P2': 0.9748, 'P3': 0.9422, 'P4p': 0.9183, 'P5': 0.9227}
    drift, extra = drift or {}, extra or {}
    runs = {}
    for s in SEEDS + BACKUP:
        base = 1.40 + rng.normal(0, 0.002)
        runs[('full', s)] = dict(test=base, nonfinite=0, Abar=1.0)
        runs[('floor', s)] = dict(test=base + 0.18, nonfinite=0)
        for name, (a1, a2) in PAIRS.items():
            for idx, a in ((0, a1), (1, a2)):
                A = target[name] + rng.normal(0, a_noise) + (drift.get(name, 0) if idx == 1 else 0)
                f = curve(A) + (extra.get(name, 0) if idx == 1 else 0)
                if world == 'noisy':
                    f += rng.normal(0, 0.03)
                nf = 30 if (name in unstable and idx == 1 and s in SEEDS[:3]) else 0
                runs[(arm(*a), s)] = dict(test=base + 0.18 * (f + rng.normal(0, sd)), nonfinite=nf, Abar=A,
                                          Abar_loop=A - 0.05, frac_both_clipped=0.3)
    if nonfinite_full:
        runs[('full', nonfinite_full)]['nonfinite'] = 25
    return runs


def selftest():
    hi = {'P3': 0.06, 'P4p': 0.06, 'P5': 0.06}
    cases = [('collapse', {}, 'AGREEMENT-PREDICTS'),
             ('collapse, seed excluded -> backup', dict(nonfinite_full=SEEDS[2]), 'AGREEMENT-PREDICTS'),
             ('collapse, drift 0.005 in P3 (in guard)', dict(drift={'P3': -0.005}), 'AGREEMENT-PREDICTS'),
             ('k4 costs more at high dose', dict(extra=hi), 'AGREEMENT-INSUFFICIENT'),
             ('noisy costs', dict(world='noisy'), 'INCONCLUSIVE'),
             ('P3, P4p, P5 drift 0.02', dict(drift={'P3': 0.02, 'P4p': 0.02, 'P5': 0.02}), 'UNINFORMATIVE'),
             ('P3 drifted, P4p + P5 scored', dict(drift={'P3': 0.02}), 'INCONCLUSIVE'),
             ('P4p unstable (P5 covers H)', dict(unstable=('P4p',)), 'AGREEMENT-PREDICTS'),
             ('P4p and P5 unstable', dict(unstable=('P4p', 'P5')), 'INCONCLUSIVE'),
             ('P4p/P5 disagree mildly (+0.03 P5)', dict(extra={'P5': 0.03}), 'AGREEMENT-PREDICTS'),
             ('P4p/P5 disagree strongly (+0.12 P5)', dict(extra={'P5': 0.12}), 'AGREEMENT-INSUFFICIENT')]
    for i, (label, kw, want) in enumerate(cases):
        runs = synth(np.random.default_rng(100 + i), **kw); got = verdict(runs, quiet=True)['outcome']
        print('selftest %-40s -> %-24s (expect %s)' % (label, got, want)); assert got == want, label
    # duplicate records are a hard error
    import os, tempfile
    fn = os.path.join(tempfile.mkdtemp(), 'dup.jsonl'); rec = json.dumps(dict(arm='full', seed=1, test=1.0, nonfinite=0))
    open(fn, 'w').write(rec + '\n' + rec + '\n')
    try:
        load([fn]); raise AssertionError('duplicate not caught')
    except SystemExit:
        print('selftest duplicate record -> hard error (expected)')
    # null-world rates for the record (true collapse, drift 0.005 on every pair, Abar noise 0.007)
    rng = np.random.default_rng(7); n = 200
    got = [verdict(synth(rng, drift={p: 0.005 for p in PAIRS}), quiet=True)['outcome'] for _ in range(n)]
    print('null world (drift 0.005 everywhere), %d sims: %s' % (n, {o: got.count(o) / n for o in sorted(set(got))}))
    print('selftest OK')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verdict', nargs='*'); ap.add_argument('--selftest', action='store_true')
    A = ap.parse_args()
    if A.selftest:
        return selftest()
    if A.verdict:
        runs = load(A.verdict); out = verdict(runs)
        print('\nsecondary (reported, not scored):'); out['secondary'] = secondary(runs, out['seeds'])
        json.dump(out, open('v174_verdict.json', 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
