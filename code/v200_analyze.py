"""v200 analysis: does a cost curve fitted on Parcae predict truncation cost on two new looped models?
(prereg docs/prereg_v200_ifm_transfer.md; this file is frozen with it, before any scored run.)

Cells: model in {pln5, r5} (IFM/LoopedLM-P2-huginn-s-fixed-*), k in {1, 2, 4} of R = 5; controls k = 0 (floor, no loop
gradient) and k = 5 (full backprop). Per seed, phi = (L_k - L_full) / (L_floor - L_full) on the test split.

Seeds: per model, a seed is valid if its full-backprop control has <= 10 non-finite steps, finite val and test losses
and did not diverge (v165 rule); the first N_SEEDS valid seeds in the pre-listed order are used, backups replace
excluded ones. A model is scored only if its floor gate passes: mean(L_floor - L_full) >= 0.03 nats with a 90%
t-interval excluding 0, over its scored seeds; otherwise the model is UNINFORMATIVE and its cells are dropped.

Predictors (LOCKED; values from v198 on the frozen models, curves fitted on Parcae before v198 existed):
  L   lost-gradient curve, phi = 0.3418 * err^3.4335 (S6 / prereg v167)
  A   direction-after-Adam curve, phi = 2.403 * (1 - Abar)^1.58 (v171b, Appendix E, D2)
  KP  "k alone" from Parcae-140m: its own unblended cost at the same k (R = 32)
  KS  "k alone" from SmolLM2 (A): its own unblended cost at the same k (R = 8; k = 4 not measured -> 0)
Score: MAE over the scored cells; calibrated = |phi - pred| <= max(0.02, 0.3 * phi).

Verdicts (applied in order; a scored cell count n < 4 -> UNINFORMATIVE):
  TRANSFER of L (vs the measurement-free rules):
    L-TRANSFERS     MAE(L) <= 0.5 * min(MAE(KP), MAE(KS)) and L calibrated in >= n - 1 cells
    L-BEATS-K       MAE(L) <= 0.5 * min(MAE(KP), MAE(KS)) (calibration fails)
    K-AS-GOOD       min(MAE(KP), MAE(KS)) <= MAE(L)
    INCONCLUSIVE    otherwise
  WHICH CURVE (L vs A):
    A-BETTER        MAE(A) <= 0.5 * MAE(L)
    L-BETTER        MAE(L) <= 0.5 * MAE(A)
    TIE             otherwise
  BEST overall: the predictor with the lowest MAE (reported).
  python v200_analyze.py DIR [DIR ...]          (every v200_*train*.json under the dirs)
  python v200_analyze.py --selftest             (synthetic worlds)
"""
import glob, json, math, os, sys
import numpy as np

N_SEEDS = int(os.environ.get('V200_N_SEEDS', 5))   # set by the prereg's seed rule from the spread pilot (5, 8 or 10)
KS = (1, 2, 4)
R = 5
# frozen-model inputs with the embedding frozen, as in every v200 training arm (v198 re-run, V198_FREEZE_EMBED=1:
# kaggle-output-v198-{pln5,r5}-pred); medians over 16 sequences (lost gradient), 24 clipped batches (after Adam)
ERR = {'pln5': {1: 0.7673, 2: 0.5954, 4: 0.2923}, 'r5': {1: 0.8317, 2: 0.6934, 4: 0.3668}}
ABAR = {'pln5': {1: 0.7567, 2: 0.8646, 4: 0.9721}, 'r5': {1: 0.6722, 2: 0.7946, 4: 0.9571}}
KP = {1: 0.224, 2: 0.100, 4: 0.024}                                                                # Table 5, R = 32
KS_ = {1: 0.009, 2: -0.001, 4: 0.0}                                                                # Table 5 / S6
FIT_L = (0.180, 0.877)       # lost gradient of the 29 Parcae cells the L curve was fitted on (v167_analyze.FIT)
FIT_A = (0.7778, 0.9671)     # direction after Adam of the 29 Parcae cells the A curve was fitted on (results_v171b)
IN_RANGE = {(m, k): FIT_L[0] <= ERR[m][k] <= FIT_L[1] and FIT_A[0] <= ABAR[m][k] <= FIT_A[1] for m in ERR for k in KS}
PRED = {
    'L': {m: {k: 0.3418 * ERR[m][k] ** 3.4335 for k in KS} for m in ERR},
    'A': {m: {k: 2.403 * (1 - ABAR[m][k]) ** 1.58 for k in KS} for m in ABAR},
    'KP': {m: dict(KP) for m in ERR},
    'KS': {m: dict(KS_) for m in ERR},
}
T90 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812,
       11: 1.796}                                                       # two-sided 90% t critical values, by df


def seed_rule(sd_max):
    """Prereg seed rule, applied to the spread pilot BEFORE any scored run: sd_max = the larger, over the two models, of
    the sample s.d. of phi at k = 1 over the pilot seeds 9001-9003 (test). Only this s.d. is read from the pilot."""
    return 5 if sd_max <= 0.045 else 8 if sd_max <= 0.06 else 10


def short(v):
    return 'pln5' if 'pln5' in v else 'r5' if 'r5' in v else v


def load(dirs):
    """Every run record in the driver's v200_runs_<model>_<tag>.jsonl files (one JSON object per line)."""
    runs = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, '**', 'v200_runs_*.jsonl'), recursive=True)):
            for line in open(f):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if isinstance(r, dict) and {'variant', 'k', 'seed', 'test', 'val'} <= set(r):
                    runs.append(r)
    seen = {}
    for r in runs:                                  # one run per (model, k, seed): later files win
        seen[(short(r['variant']), int(r['k']), int(r['seed']))] = r
    return seen


def valid_full(r):
    return (r is not None and r.get('nonfinite_steps', 0) <= 10 and not r.get('diverged', False)
            and math.isfinite(r['val']) and math.isfinite(r['test']))


def score(runs, seed_order, verbose=True):
    out = dict(models={}, cells={}, mae={}, calibrated={})
    cells = []
    for m in ('pln5', 'r5'):
        seeds = [s for s in seed_order.get(m, []) if valid_full(runs.get((m, R, s)))
                 and all((m, k, s) in runs for k in (0,) + KS)][:N_SEEDS]
        gap = [runs[(m, 0, s)]['test'] - runs[(m, R, s)]['test'] for s in seeds]
        n = len(gap)
        if n < N_SEEDS:
            out['models'][m] = dict(status='UNINFORMATIVE (fewer than %d valid seeds: %d)' % (N_SEEDS, n), seeds=seeds)
            continue
        mu, sd = float(np.mean(gap)), float(np.std(gap, ddof=1))
        lo = mu - T90[n - 1] * sd / math.sqrt(n)
        ok = mu >= 0.03 and lo > 0
        out['models'][m] = dict(status='scored' if ok else 'UNINFORMATIVE (floor gate)', seeds=seeds,
                                gap_mean=mu, gap_lo90=lo)
        if not ok:
            continue
        for k in KS:
            phis = [(runs[(m, k, s)]['test'] - runs[(m, R, s)]['test']) / g for s, g in zip(seeds, gap)]
            out['cells'][(m, k)] = dict(phi=float(np.mean(phis)), se=float(np.std(phis, ddof=1) / math.sqrt(n)),
                                        per_seed=[round(p, 4) for p in phis])
            cells.append((m, k))
    n = len(cells)
    for p in PRED:
        res = [out['cells'][c]['phi'] - PRED[p][c[0]][c[1]] for c in cells]
        out['mae'][p] = float(np.mean(np.abs(res))) if res else float('nan')
        out['calibrated'][p] = sum(abs(out['cells'][c]['phi'] - PRED[p][c[0]][c[1]]) <= max(0.02, 0.3 * abs(out['cells'][c]['phi']))
                                   for c in cells)
    if n < 4:
        out['transfer'] = out['which'] = 'UNINFORMATIVE'
    else:
        mae, kmin = out['mae'], min(out['mae']['KP'], out['mae']['KS'])
        if mae['L'] <= 0.5 * kmin and out['calibrated']['L'] >= n - 1:
            out['transfer'] = 'L-TRANSFERS'
        elif mae['L'] <= 0.5 * kmin:
            out['transfer'] = 'L-BEATS-K'
        elif kmin <= mae['L']:
            out['transfer'] = 'K-AS-GOOD'
        else:
            out['transfer'] = 'INCONCLUSIVE'
        out['which_all'] = ('A-BETTER' if mae['A'] <= 0.5 * mae['L'] else 'L-BETTER' if mae['L'] <= 0.5 * mae['A']
                            else 'TIE')                       # secondary: includes A's extrapolated cells
    # primary which-curve verdict: only cells where both curves' inputs lie inside their Parcae fitting ranges
    inr = [c for c in cells if IN_RANGE[c]]
    out['in_range_cells'] = inr
    if len(inr) >= 3:
        mi = {p: float(np.mean([abs(out['cells'][c]['phi'] - PRED[p][c[0]][c[1]]) for c in inr])) for p in ('L', 'A')}
        out['mae_in_range'] = mi
        out['which'] = 'A-BETTER' if mi['A'] <= 0.5 * mi['L'] else 'L-BETTER' if mi['L'] <= 0.5 * mi['A'] else 'TIE'
    else:
        out['which'] = 'UNINFORMATIVE (fewer than 3 in-range cells)'
    out.setdefault('which_all', 'UNINFORMATIVE')
    out['best'] = min(out['mae'], key=lambda p: out['mae'][p]) if n else None
    out['n_cells'] = n
    if verbose:
        for m, s in out['models'].items():
            print('%-5s %s | seeds %s | floor gap %s' % (m, s['status'], s['seeds'], (
                '%.4f (90%% lower %.4f)' % (s['gap_mean'], s['gap_lo90'])) if 'gap_mean' in s else '-'))
        for c, v in out['cells'].items():
            print('  %s k=%d phi %.4f +- %.4f %s | L %.3f A %.3f KP %.3f KS %.3f' % (
                c[0], c[1], v['phi'], v['se'], v['per_seed'], *[PRED[p][c[0]][c[1]] for p in ('L', 'A', 'KP', 'KS')]))
        print('MAE ' + ' '.join('%s %.4f (calibrated %d/%d)' % (p, out['mae'][p], out['calibrated'][p], n) for p in PRED))
        print('in-range cells for L vs A: %s%s' % (out['in_range_cells'], (' | MAE in range L %.4f A %.4f' % (
            out['mae_in_range']['L'], out['mae_in_range']['A'])) if 'mae_in_range' in out else ''))
        print('TRANSFER: %s | WHICH CURVE (in range, primary): %s | all cells, A extrapolated (secondary): %s | best '
              'predictor: %s' % (out['transfer'], out['which'], out['which_all'], out['best']))
    return out


def synthetic(world, seed_order, noise=0.001, gap=0.08):
    """Runs where the true cost of each cell is the named predictor (or a fixed table), plus seed noise."""
    g = np.random.default_rng(0)
    runs = {}
    for m in ('pln5', 'r5'):
        for s in seed_order[m]:
            full = 1.30 + g.normal(0, 0.003)
            runs[(m, R, s)] = dict(variant=m, k=R, seed=s, val=full, test=full, nonfinite_steps=0)
            runs[(m, 0, s)] = dict(variant=m, k=0, seed=s, val=full + gap, test=full + gap + g.normal(0, 0.002))
            for k in KS:
                phi = world[m][k] if isinstance(world, dict) else PRED[world][m][k]
                runs[(m, k, s)] = dict(variant=m, k=k, seed=s, val=0, test=full + gap * phi + g.normal(0, noise))
    return runs


def selftest():
    order = {m: list(range(1, 8)) for m in ('pln5', 'r5')}
    want = {'L': ('L-TRANSFERS', 'L-BETTER'), 'A': ('K-AS-GOOD', 'A-BETTER'), 'KP': ('K-AS-GOOD', 'TIE'),
            'KS': ('K-AS-GOOD', None)}                 # with no cost at all, L vs A is not a meaningful contrast
    ok = True
    for w, (t, wh) in want.items():
        o = score(synthetic(w, order), order, verbose=False)
        good = o['transfer'] == t and o['best'] == w and (wh is None or o['which'] == wh)
        print('world %-2s -> transfer %-11s which %-8s best %-2s %s' % (w, o['transfer'], o['which'], o['best'],
                                                                     'ok' if good else 'UNEXPECTED (want %s, best %s)' % (t, w)))
        ok &= good
    o = score(synthetic('L', order, gap=0.01), order, verbose=False)       # small loop benefit: floor gate fails
    print('small floor gap -> %s' % o['transfer']); ok &= o['transfer'] == 'UNINFORMATIVE'
    bad = synthetic('L', order); bad[('r5', R, 1)]['nonfinite_steps'] = 20   # an excluded seed is replaced by a backup
    o = score(bad, order, verbose=False)
    print('excluded seed -> r5 seeds %s, transfer %s' % (o['models']['r5']['seeds'], o['transfer']))
    ok &= o['models']['r5']['seeds'] == [2, 3, 4, 5, 6] and o['transfer'] == 'L-TRANSFERS'
    print('SELFTEST', 'PASS' if ok else 'FAIL')
    return ok


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        sys.exit(0 if selftest() else 1)
    order = json.loads(os.environ.get('V200_SEED_ORDER', '{}')) or None
    runs = load(sys.argv[1:])
    if order is None:
        order = {m: sorted({s for (mm, k, s) in runs if mm == m}) for m in ('pln5', 'r5')}
    for m in PRED:
        print('%s predictions: ' % m + ' | '.join('%s ' % mm + ' '.join('k%d %.3f' % (k, PRED[m][mm][k]) for k in KS)
                                                 for mm in ('pln5', 'r5')))
    score(runs, order)
