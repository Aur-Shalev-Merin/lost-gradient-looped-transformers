"""v163 analysis, written before any truncated-credit run (prereg: docs/prereg_v163_parcae_damped_truncation.md §3).

  python v163_analyze.py --select RUNS...    tuning seed -> per-cell LR (lowest val loss); grid-edge cells flagged for
                                             one 3x extension; writes v163_selection.json
  python v163_analyze.py --verdict RUNS...   evaluation seeds at the selected LRs -> gates, H0-H2, outcome label
  python v163_analyze.py --selftest          synthetic worlds (law / k-alone / anti-law / no-cost) -> expected labels
Cells: (a, k), a in {1.0, 0.5} (damping), k in {0, 1, 2, 4, 8, 32} loops of backprop out of R = 32.
LAW = median over sequences of rho_seq^k from the pilot's frozen-model measurement at each a (v163 driver --measure);
the values below are LOCKED at freeze (pilot measurement).
"""
import argparse, json, math
import numpy as np
from scipy import stats

TUNE = [1631001]
EVAL = [1631101, 1631102, 1631103, 1631104]
ALPHAS = (1.0, 0.5)
KS = [0, 1, 2, 4, 8, 32]
FULL = 32
GRID = [1e-4, 3e-4, 9e-4]
TRUNC = [(a, k) for a in ALPHAS for k in KS if 0 < k < FULL]
# LOCKED at freeze: pilot frozen-model measurement (kaggle-output-v163/pilot-b, R 32, 16 GSM8K-val sequences),
# law = median over sequences of rho_seq^k; rho medians 0.8250 (a=1.0) / 0.9025 (a=0.5)
RHO = {1.0: 0.8250, 0.5: 0.9025}
LAW = {(1.0, 1): 0.8250, (1.0, 2): 0.6807, (1.0, 4): 0.4634, (1.0, 8): 0.2149,
       (0.5, 1): 0.9025, (0.5, 2): 0.8145, (0.5, 4): 0.6635, (0.5, 8): 0.4402}
assert set(LAW) == set(TRUNC)
PAIRS = [((0.5, k), (1.0, k)) for k in (1, 2, 4, 8)]   # law predicts phi(0.5, k) > phi(1.0, k); "k alone" predicts a tie
DISC = 0.05
MIN_PHI, MIN_DL = 0.02, 0.002
SPEARMAN_MIN = 0.75


def load(files):
    runs, shas = {}, set()
    for f in files:
        for line in open(f):
            r = json.loads(line)
            key = (float(r['a']), r['k'], float('%.3g' % r['lr']), r['seed'])
            if key in runs:
                print('duplicate run ignored (first kept):', key); continue
            runs[key] = r; shas.add(r.get('driver_sha'))
    print('runs %d, driver sha %s' % (len(runs), sorted(map(str, shas))))
    return runs


def val_of(r):
    return float('inf') if r['diverged'] or not math.isfinite(r['val']) else r['val']


def select(runs, write=True):
    sel, edge = {}, []
    for a in ALPHAS:
        for k in KS:
            lrs = sorted({lr for (a2, k2, lr, s) in runs if (a2, k2) == (a, k) and s in TUNE})
            lrs = [lr for lr in lrs if all((a, k, lr, s) in runs for s in TUNE)]
            if not lrs:
                print('cell a=%.1f k=%d: no tuning runs' % (a, k)); continue
            mv = {lr: np.mean([val_of(runs[(a, k, lr, s)]) for s in TUNE]) for lr in lrs}
            best = min(lrs, key=lambda lr: mv[lr]); extended = len(lrs) > len(GRID); at_edge = best in (lrs[0], lrs[-1])
            sel['%.1f:%d' % (a, k)] = best
            print('cell a=%.1f k=%-2d %s -> %g%s' % (a, k, '  '.join('%g:%.4f' % (lr, mv[lr]) for lr in lrs), best,
                  (' EDGE' + (' (already extended: final)' if extended else ' -> extend')) if at_edge else ''))
            if at_edge and not extended:
                edge.append((a, k, float('%.3g' % (best / 3 if best == lrs[0] else best * 3))))
    if write:
        json.dump(dict(selection=sel, extend=edge), open('v163_selection.json', 'w'), indent=1)
        print('wrote v163_selection.json; extend:', edge)
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


def verdict(runs, sel, law=None):
    law = law or LAW
    L = {}
    for a in ALPHAS:
        for k in KS:
            lr = sel['%.1f:%d' % (a, k)]
            L[(a, k)] = {s: (float('nan') if runs[(a, k, lr, s)]['diverged'] else runs[(a, k, lr, s)]['test'])
                         for s in EVAL if (a, k, lr, s) in runs}
    out = dict(selection=sel); floor_ok = True
    for a in ALPHAS:
        d = [L[(a, 0)][s] - L[(a, FULL)][s] for s in EVAL if s in L[(a, 0)] and s in L[(a, FULL)]]
        d = [x for x in d if math.isfinite(x)]
        lab, c = label(d, 0.0); floor_ok &= lab == '+'
        out['floor_a%.1f' % a] = dict(n=len(d), mean_ci=c, ok=lab == '+')
        print('floor a=%.1f: L0 - Lfull = %.4f [%.4f, %.4f] (n=%d) %s' % (a, c[0], c[1], c[2], len(d), 'OK' if lab == '+' else 'FAIL'))
    phi, dl = {}, {}
    for (a, k) in TRUNC:
        phi[(a, k)], dl[(a, k)] = {}, {}
        for s in EVAL:
            try:
                x, f, z = L[(a, k)][s], L[(a, FULL)][s], L[(a, 0)][s]
            except KeyError:
                continue
            if all(map(math.isfinite, (x, f, z))):
                dl[(a, k)][s] = x - f; phi[(a, k)][s] = (x - f) / (z - f)
    print('\ncell        law     phi mean [90% CI]            dL mean')
    for c in TRUNC:
        m, lo, hi = ci90(list(phi[c].values()))
        print('a=%.1f k=%-2d  %.3f  %+.3f [%+.3f, %+.3f] n=%d   %+.4f' % (c[0], c[1], law[c], m, lo, hi, len(phi[c]),
              np.mean(list(dl[c].values()))))
    out['phi'] = {'%.1f:%d' % c: ci90(list(phi[c].values())) for c in TRUNC}
    out['dL'] = {'%.1f:%d' % c: ci90(list(dl[c].values())) for c in TRUNC}
    h0 = all(label(list(phi[(a, 1)].values()), MIN_PHI)[0] == '+' for a in ALPHAS)
    sp = stats.spearmanr([law[c] for c in TRUNC], [np.mean(list(phi[c].values())) for c in TRUNC])[0]
    h2 = sp >= SPEARMAN_MIN
    print('\nH0 (phi at k=1 > 0, both a): %s' % h0)
    print('H2 Spearman(law, phi) over %d cells: %.3f (>= %.2f: %s)' % (len(TRUNC), sp, SPEARMAN_MIN, h2))

    def contrasts(metric, thr, name):
        res = []
        for a_, b_ in PAIRS:
            dlaw = law[a_] - law[b_]; disc = abs(dlaw) >= DISC
            seeds = [s for s in EVAL if s in metric[a_] and s in metric[b_]]
            lab, c = label([metric[a_][s] - metric[b_][s] for s in seeds], thr)
            pred = '+' if dlaw > 0 else '-'
            st = 'n/s' if not disc else 'LAW' if lab == pred else 'TIE' if lab == '0' else 'OPPOSITE'
            res.append(dict(a=a_, b=b_, law_diff=dlaw, scored=disc, status=st, ci=c, n=len(seeds)))
            print('  %s %s - %s: law diff %+.3f (%s) observed %+.4f [%+.4f, %+.4f] n=%d -> %s' % (name, a_, b_, dlaw,
                  'scored' if disc else 'not scored', c[0], c[1], c[2], len(seeds), st))
        return res
    print('\nH1 (matched k: damped a=0.5 vs a=1.0), metric phi:'); h1 = contrasts(phi, MIN_PHI, 'H1')
    print('reported: H1 on raw dL:'); h1dl = contrasts(dl, MIN_DL, 'H1dL')
    scored = [c for c in h1 if c['scored']]
    n_law = sum(c['status'] == 'LAW' for c in scored); n_opp = sum(c['status'] == 'OPPOSITE' for c in scored)
    need = min(3, len(scored))
    if not floor_ok or not h0:
        outcome = 'UNINFORMATIVE'
    elif n_opp > 0 or not h2:
        outcome = 'AGAINST'
    elif scored and n_law >= need:
        outcome = 'LAW'
    elif n_law == 0:
        outcome = 'K-ALONE'
    else:
        outcome = 'MIXED'
    print('\nOUTCOME: %s  (scored pairs %d, law-direction %d, opposite %d)' % (outcome, len(scored), n_law, n_opp))
    # REPORTED ONLY (added before freeze): does truncated training reshape the loop? end-of-run rho per cell
    rend = {}
    for a in ALPHAS:
        for k in KS:
            lr = sel['%.1f:%d' % (a, k)]
            xs = [runs[(a, k, lr, s)].get('rho_end') for s in EVAL if (a, k, lr, s) in runs]
            xs = [x for x in xs if x is not None and math.isfinite(x)]
            if xs:
                rend[(a, k)] = float(np.mean(xs))
    if rend:
        print('reported: end-of-run rho (mean over eval seeds): '
              + ' '.join('a=%.1f k=%d %.3f' % (a, k, r) for (a, k), r in sorted(rend.items())))
        for a in ALPHAS:
            ks_ = [k for k in KS if (a, k) in rend and k > 0]
            if len(ks_) >= 3:
                print('  a=%.1f: Spearman(k, end rho) over k>0 = %.3f (prediction: positive, smaller k -> smaller rho)'
                      % (a, stats.spearmanr(ks_, [rend[(a, k)] for k in ks_])[0]))
    out['rho_end'] = {'%.1f:%d' % c: r for c, r in rend.items()}
    out.update(h0=h0, h2=dict(spearman=sp, ok=h2), h1=h1, h1_dL=h1dl, outcome=outcome)
    return out


def selftest():
    rng = np.random.default_rng(0)

    def world(phi_fn, noise=0.002):
        runs = {}
        for a in ALPHAS:
            for k in KS:
                for s in TUNE + EVAL:
                    for lr in GRID:
                        p = 1.0 if k == 0 else 0.0 if k == FULL else phi_fn(a, k)
                        t = 1.4 + 0.2 * p + (0.0 if lr == 3e-4 else 0.05) + rng.normal(0, noise)
                        runs[(a, k, lr, s)] = dict(a=a, k=k, lr=lr, seed=s, val=t, test=t, diverged=False)
        return runs
    for name, fn, want in (('law world', lambda a, k: 0.5 * LAW[(a, k)], 'LAW'),
                           ('k-alone world', lambda a, k: 0.5 * LAW[(1.0, k)], 'K-ALONE'),
                           ('anti-law world', lambda a, k: 0.5 * LAW[(1.0, k)] - 0.5 * (LAW[(a, k)] - LAW[(1.0, k)]), 'AGAINST'),
                           ('no-cost world', lambda a, k: 0.0, 'UNINFORMATIVE')):
        print('\n=== selftest: %s (expect %s)' % (name, want))
        runs = world(fn); sel, edge = select(runs, write=False)
        assert not edge and all(abs(x - 3e-4) < 1e-12 for x in sel.values())
        got = verdict(runs, sel)['outcome']; assert got == want, (name, got, want)
    print('\nselftest OK')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--select', nargs='*'); ap.add_argument('--verdict', nargs='*'); ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--selection', default='v163_selection.json')
    A = ap.parse_args()
    if A.selftest:
        return selftest()
    if A.select is not None:
        return select(load(A.select))
    if A.verdict is not None:
        res = verdict(load(A.verdict), json.load(open(A.selection))['selection'])
        json.dump(res, open('v163_verdict.json', 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
