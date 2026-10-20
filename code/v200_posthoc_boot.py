"""v200 POST HOC (not in the prereg): seed bootstrap of the v200 MAEs and verdicts, resampling each model's 5 scored seeds
with replacement (paired across cells, since each seed's phi uses its own full and floor runs)."""
import collections, importlib.util, os, sys
import numpy as np

spec = importlib.util.spec_from_file_location('v200a', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                   'v200_analyze.py'))
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
runs = A.load(sys.argv[1:])
order = {m: sorted({s for (mm, k, s) in runs if mm == m}) for m in ('pln5', 'r5')}
out = A.score(runs, order, verbose=False)
seeds = {m: out['models'][m]['seeds'] for m in ('pln5', 'r5')}
phi = {(m, k): np.array([(runs[(m, k, s)]['test'] - runs[(m, 5, s)]['test']) /
                         (runs[(m, 0, s)]['test'] - runs[(m, 5, s)]['test']) for s in seeds[m]])
       for m in ('pln5', 'r5') for k in A.KS}
cells = list(phi)
inr = [c for c in cells if A.IN_RANGE[c]]
g = np.random.default_rng(0)
B = 20000
cnt, lt, maes = collections.Counter(), collections.Counter(), collections.defaultdict(list)
for _ in range(B):
    idx = {m: g.integers(0, 5, 5) for m in ('pln5', 'r5')}
    mean = {c: phi[c][idx[c[0]]].mean() for c in cells}
    mae = {p: np.mean([abs(mean[c] - A.PRED[p][c[0]][c[1]]) for c in cells]) for p in A.PRED}
    mi = {p: np.mean([abs(mean[c] - A.PRED[p][c[0]][c[1]]) for c in inr]) for p in ('L', 'A')}
    cal = sum(abs(mean[c] - A.PRED['L'][c[0]][c[1]]) <= max(0.02, 0.3 * abs(mean[c])) for c in cells)
    kmin = min(mae['KP'], mae['KS'])
    t = ('L-TRANSFERS' if mae['L'] <= 0.5 * kmin and cal >= 5 else 'L-BEATS-K' if mae['L'] <= 0.5 * kmin
         else 'K-AS-GOOD' if kmin <= mae['L'] else 'INCONCLUSIVE')
    w = 'A-BETTER' if mi['A'] <= 0.5 * mi['L'] else 'L-BETTER' if mi['L'] <= 0.5 * mi['A'] else 'TIE'
    cnt[(t, w)] += 1
    lt['L<KP'] += mae['L'] < mae['KP']
    lt['L<A in range'] += mi['L'] < mi['A']
    for p in A.PRED:
        maes[p].append(mae[p])
print('verdict pairs:', {k: round(v / B, 4) for k, v in cnt.most_common()})
print({k: round(float(v) / B, 4) for k, v in lt.items()})
for p in A.PRED:
    q = np.percentile(maes[p], [5, 50, 95])
    print('MAE %-2s 5/50/95%%: %.4f %.4f %.4f' % (p, *q))
d = np.array(maes['L']) - np.array(maes['KP'])
print('MAE(L) - MAE(KP) 5/50/95%%: %.4f %.4f %.4f' % tuple(np.percentile(d, [5, 50, 95])))
# between-model cost ratio at each k (r5 / pln5), observed vs each curve
for k in A.KS:
    o = phi[('r5', k)].mean() / phi[('pln5', k)].mean()
    print('k=%d r5/pln5 cost ratio: observed %.2f | L %.2f | A %.2f | KP 1.00' % (
        k, o, A.PRED['L']['r5'][k] / A.PRED['L']['pln5'][k], A.PRED['A']['r5'][k] / A.PRED['A']['pln5'][k]))
