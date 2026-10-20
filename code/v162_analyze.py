"""v162 analysis, written before any truncated-credit run (prereg: docs/prereg_v162_parcae_truncation_training_rerun.md §3).

  python v162_analyze.py --select RUNS...      tuning seeds -> per-cell LR (lowest mean val loss); flags grid-edge cells
                                               (edge rule: extend one 3x step once); writes v162_selection.json
  python v162_analyze.py --verdict RUNS...     evaluation seeds at the selected LRs -> gates, H0-H3, outcome label
  python v162_analyze.py --selftest            synthetic runs where the law holds / fails -> expected labels
The law values are the v161 pilot's frozen-model measurement (kaggle-output-v161/pilot/v161_measure_R*_pilotR8.json,
median over 16 GSM8K-val sequences of rho_seq^k), locked here at freeze.
"""
import argparse, json, math, sys
import numpy as np
from scipy import stats

TUNE = [1621001, 1621002]
EVAL = [1621101, 1621102, 1621103, 1621104, 1621105]
KS = {8: [0, 1, 2, 4, 8], 16: [0, 1, 2, 4, 8, 16]}
GRID = {8: [1e-4, 3e-4, 9e-4], 16: [1e-4, 3e-4, 9e-4]}          # {lr*/3, lr*, 3 lr*}, lr* = 3e-4 (pilot, both R)
TRUNC = [(R, k) for R in (8, 16) for k in KS[R] if 0 < k < R]
LAW = {(8, 1): 0.7726, (8, 2): 0.5969, (8, 4): 0.3563,
       (16, 1): 0.8020, (16, 2): 0.6431, (16, 4): 0.4136, (16, 8): 0.1711}
ERR = {(8, 1): 0.8051, (8, 2): 0.6477, (8, 4): 0.3860,
       (16, 1): 0.8132, (16, 2): 0.6632, (16, 4): 0.4210, (16, 8): 0.1803}
H2_PAIRS = [((16, 1), (8, 1)), ((16, 2), (8, 2)), ((16, 4), (8, 4))]
H3_PAIRS = [((8, 2), (16, 4)), ((8, 4), (16, 8))]
DISC = 0.05                       # a pair is scored only if |law_a - law_b| >= DISC
MIN_PHI, MIN_DL = 0.02, 0.002     # minimum |mean| for a directional label
H1_MIN = 0.75


def load(files):
    runs, shas = {}, set()
    for f in files:
        for line in open(f):
            r = json.loads(line)
            key = (r['R'], r['k'], float('%.3g' % r['lr']), r['seed'])
            if key in runs:
                print('duplicate run ignored (first kept):', key); continue
            runs[key] = r; shas.add(r.get('driver_sha'))
    print('runs %d, driver sha %s' % (len(runs), sorted(map(str, shas))))
    return runs


def val_of(r):
    return float('inf') if r['diverged'] or not math.isfinite(r['val']) else r['val']


def select(runs, write=True):
    sel, edge = {}, []
    for R in (8, 16):
        for k in KS[R]:
            lrs = sorted({lr for (R2, k2, lr, s) in runs if (R2, k2) == (R, k) and s in TUNE})
            lrs = [lr for lr in lrs if all((R, k, lr, s) in runs for s in TUNE)]
            if not lrs:
                print('cell R=%d k=%d: no complete tuning runs' % (R, k)); continue
            mv = {lr: np.mean([val_of(runs[(R, k, lr, s)]) for s in TUNE]) for lr in lrs}
            best = min(lrs, key=lambda lr: mv[lr])
            extended = len(lrs) > len(GRID[R])
            at_edge = best in (lrs[0], lrs[-1])
            sel['%d:%d' % (R, k)] = best
            print('cell R=%-2d k=%-2d  %s  -> %g%s' % (R, k, '  '.join('%g:%.4f' % (lr, mv[lr]) for lr in lrs), best,
                  ' EDGE' + (' (already extended: final)' if extended else ' -> extend') if at_edge else ''))
            if at_edge and not extended:
                edge.append((R, k, float('%.3g' % (best / 3 if best == lrs[0] else best * 3))))
    if write:
        json.dump(dict(selection=sel, extend=edge), open('v162_selection.json', 'w'), indent=1)
        print('wrote v162_selection.json; cells to extend:', edge)
    return sel, edge


def ci90(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2:
        return float(np.mean(x)), float('nan'), float('nan')
    h = stats.t.ppf(0.95, n - 1) * x.std(ddof=1) / math.sqrt(n)
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def label(x, thr):
    m, lo, hi = ci90(x)
    if lo > 0 and abs(m) >= thr:
        return '+', (m, lo, hi)
    if hi < 0 and abs(m) >= thr:
        return '-', (m, lo, hi)
    return '0', (m, lo, hi)


def verdict(runs, sel):
    L = {}       # (R, k) -> {seed: test loss}
    for R in (8, 16):
        for k in KS[R]:
            lr = sel['%d:%d' % (R, k)]
            L[(R, k)] = {s: (float('nan') if runs[(R, k, lr, s)]['diverged'] else runs[(R, k, lr, s)]['test'])
                         for s in EVAL if (R, k, lr, s) in runs}
    out = dict(selection=sel)
    # floor re-check
    floor_ok = True
    for R in (8, 16):
        d = [L[(R, 0)][s] - L[(R, R)][s] for s in EVAL if s in L[(R, 0)] and s in L[(R, R)]]
        d = [x for x in d if math.isfinite(x)]
        lab, c = label(d, 0.0)
        out['floor_R%d' % R] = dict(n=len(d), mean_ci=c, ok=lab == '+')
        floor_ok &= lab == '+'
        print('floor R=%d: L0 - Lfull = %.4f [%.4f, %.4f] (n=%d) %s' % (R, c[0], c[1], c[2], len(d), 'OK' if lab == '+' else 'FAIL'))
    # phi and dL per seed
    phi, dl = {}, {}
    for (R, k) in TRUNC:
        phi[(R, k)], dl[(R, k)] = {}, {}
        for s in EVAL:
            try:
                a, f, z = L[(R, k)][s], L[(R, R)][s], L[(R, 0)][s]
            except KeyError:
                continue
            if all(map(math.isfinite, (a, f, z))):
                dl[(R, k)][s] = a - f; phi[(R, k)][s] = (a - f) / (z - f)
    print('\ncell     law    err    phi mean [90%% CI]           dL mean')
    for c in TRUNC:
        m, lo, hi = ci90(list(phi[c].values()))
        print('R=%-2d k=%-2d %.3f  %.3f  %+.3f [%+.3f, %+.3f] n=%d   %+.4f' % (c[0], c[1], LAW[c], ERR[c], m, lo, hi,
              len(phi[c]), np.mean(list(dl[c].values()))))
    out['phi'] = {'%d:%d' % c: ci90(list(phi[c].values())) for c in TRUNC}
    out['dL'] = {'%d:%d' % c: ci90(list(dl[c].values())) for c in TRUNC}
    # H0
    h0 = all(label(list(phi[(R, 1)].values()), MIN_PHI)[0] == '+' for R in (8, 16))
    # H1
    rho_s = stats.spearmanr([LAW[c] for c in TRUNC], [np.mean(list(phi[c].values())) for c in TRUNC])[0]
    h1 = rho_s >= H1_MIN
    rho_err = stats.spearmanr([ERR[c] for c in TRUNC], [np.mean(list(phi[c].values())) for c in TRUNC])[0]
    rho_dl = stats.spearmanr([LAW[c] for c in TRUNC], [np.mean(list(dl[c].values())) for c in TRUNC])[0]
    print('\nH0 (phi_{R,1} > 0 at both R): %s' % h0)
    print('H1 Spearman(law, phi) over 7 cells: %.3f (>= %.2f: %s) | reported: err predictor %.3f, raw dL %.3f'
          % (rho_s, H1_MIN, h1, rho_err, rho_dl))

    def contrasts(pairs, name, metric, thr):
        res = []
        for a, b in pairs:
            dlaw = LAW[a] - LAW[b]; disc = abs(dlaw) >= DISC
            seeds = [s for s in EVAL if s in metric[a] and s in metric[b]]
            lab, c = label([metric[a][s] - metric[b][s] for s in seeds], thr)
            pred = '+' if dlaw > 0 else '-'
            status = ('n/s' if not disc else 'LAW' if lab == pred else 'TIE' if lab == '0' else 'OPPOSITE')
            res.append(dict(a=a, b=b, law_diff=dlaw, scored=disc, label=lab, pred=pred, status=status, ci=c, n=len(seeds)))
            print('  %s %s - %s: law diff %+.3f (%s) observed %+.4f [%+.4f, %+.4f] n=%d -> %s' % (name, a, b, dlaw,
                  'scored' if disc else 'not scored', c[0], c[1], c[2], len(seeds), status))
        return res
    print('\nH2 (law vs k alone), metric phi:'); h2 = contrasts(H2_PAIRS, 'H2', phi, MIN_PHI)
    print('H3 (law vs k/R), metric phi:'); h3 = contrasts(H3_PAIRS, 'H3', phi, MIN_PHI)
    print('reported: same contrasts on raw dL:'); contrasts(H2_PAIRS, 'H2dL', dl, MIN_DL); contrasts(H3_PAIRS, 'H3dL', dl, MIN_DL)
    scored = [c for c in h2 + h3 if c['scored']]
    opposite = any(c['status'] == 'OPPOSITE' for c in scored)
    fam_ok = all(any(c['status'] == 'LAW' for c in fam if c['scored']) for fam in (h2, h3) if any(c['scored'] for c in fam))
    any_dir = any(c['status'] == 'LAW' for c in scored)
    if not floor_ok or not h0:
        outcome = 'UNINFORMATIVE'
    elif not h1 or opposite:
        outcome = 'AGAINST'
    elif fam_ok and scored:
        outcome = 'LAW'
    else:
        outcome = 'LAW-CONSISTENT'
    # reported fit phi ~ c * law
    for R in (8, 16, None):
        cs = [c for c in TRUNC if R is None or c[0] == R]
        x = np.array([LAW[c] for c in cs]); y = np.array([np.mean(list(phi[c].values())) for c in cs])
        cfit = float(x @ y / (x @ x)); r2 = 1 - ((y - cfit * x) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
        print('reported fit phi = c*law (%s): c %.3f R^2 %.3f' % ('pooled' if R is None else 'R=%d' % R, cfit, r2))
    print('\nOUTCOME: %s' % outcome)
    out.update(h0=h0, h1=dict(spearman=rho_s, ok=h1), h2=h2, h3=h3, outcome=outcome)
    return out


def selftest():
    """Synthetic: test loss = base - credit * (1 - g(R,k)) + noise, g = phi; law-true world uses phi = 0.5*law;
    'k/R' world uses phi depending only on k/R; check labels."""
    rng = np.random.default_rng(0)

    def world(phi_fn, noise=0.002):
        runs = {}
        for R in (8, 16):
            for k in KS[R]:
                for s in TUNE + EVAL:
                    for lr in GRID[R]:
                        p = 1.0 if k == 0 else 0.0 if k == R else phi_fn(R, k)
                        lr_pen = 0.0 if lr == 3e-4 else 0.05
                        t = 1.4 + 0.2 * p + lr_pen + rng.normal(0, noise)
                        runs[(R, k, lr, s)] = dict(R=R, k=k, lr=lr, seed=s, val=t, test=t, diverged=False)
        return runs
    for name, fn, want in (('law world', lambda R, k: 0.5 * LAW[(R, k)], 'LAW'),
                           ('k/R world (H3 ties: law not separated)', lambda R, k: 0.5 * (1 - k / R), 'LAW-CONSISTENT'),
                           ('anti-law world', lambda R, k: 0.5 * (1 - LAW[(R, k)]), 'AGAINST'),
                           ('no-cost world', lambda R, k: 0.0, 'UNINFORMATIVE')):
        print('\n=== selftest: %s (expect %s)' % (name, want))
        runs = world(fn); sel, edge = select(runs, write=False)
        assert not edge and all(abs(v - 3e-4) < 1e-12 for v in sel.values())
        got = verdict(runs, sel)['outcome']
        assert got == want, (name, got, want)
    print('\nselftest OK')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--select', nargs='*'); ap.add_argument('--verdict', nargs='*'); ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--selection', default='v162_selection.json')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.select is not None:
        return select(load(a.select))
    if a.verdict is not None:
        sel = {k: v for k, v in json.load(open(a.selection))['selection'].items()}
        res = verdict(load(a.verdict), sel)
        json.dump(res, open('v162_verdict.json', 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
