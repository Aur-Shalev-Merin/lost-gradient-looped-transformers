"""v191 analysis (plan docs/v191_parcae_contrib_plan.md; written before data). Index j = loops before the output
(j = 0: last loop). Per sequence: s_j = backward signal at loop j, |c_j| = its contribution, gain_j = |c_j| / s_j
relative to j = 0, step_j = forward step of loop j. Lost gradient over the loop part only:
err_loop_k = |g_R - g_k| / |g_R - g_0| (the paper's err_k also has the coda/head gradient g_0 in the denominator).
  P0: contribution sizes proportional to the signal, even angles -> between the orthogonal and aligned forms on s_j
  P1: the measured sizes |c_j|, even angles                        -> the same two forms on |c_j|
Decision (plan): SIZES if err_loop is within P1's band +-0.02 at k = 1, 2, 4, 8; ANGLES if err_loop exceeds P1's
aligned form by more than 0.02 at two or more of those k; SIZES-OVERSHOOT if it falls more than 0.02 below P1's
orthogonal form at two or more; MIXED otherwise.
"""
import glob, json, os, re, sys
import numpy as np

TAG = {'Parcae-370m': '370m', 'Parcae-140m': '140m', 'Huginn-3.5B': 'huginn'}
NAME = {v: k for k, v in TAG.items()}
REF = {('Parcae-370m', 32): {1: .8273, 2: .6739, 4: .4513, 8: .2382, 16: .074},       # v190
       ('Parcae-140m', 32): {1: .8201, 2: .6753, 4: .4427, 8: .2046, 16: .0425},      # v190
       ('Parcae-370m', 16): {1: .8215, 2: .6652, 4: .4453, 8: .2099},                 # v189
       ('Parcae-140m', 16): {1: .8132, 2: .6632, 4: .4210, 8: .1803}}                 # v189 / Table 1
KS = (1, 2, 4, 8)
D = sys.argv[1] if len(sys.argv) > 1 else 'kaggle-output-v191'
# files: v191_<tag>_R<R>.json (Parcae) or v192_<tag>_part<n>_R<R>.json (Huginn, rows merged over parts)
FILES = {}
for f in sorted(glob.glob(os.path.join(D, '**', 'v19[12]_*_R*.json'), recursive=True)):
    mt = re.match(r'v19[12]_([a-z0-9]+?)(?:_part\d+)?_R(\d+)\.json$', os.path.basename(f))
    if mt and mt.group(1) in NAME:
        FILES.setdefault((NAME[mt.group(1)], int(mt.group(2))), []).append(f)
RUNS = sorted(FILES, key=lambda x: (x[0] != 'Huginn-3.5B', -x[1], x[0]))


def huginn_ref(rows, R):
    """Sep-30 probe values for the same sequence indices (per-sequence reproduction check)."""
    P = {s['i']: s for s in json.load(open('kaggle-output-loopgrad/huginn/loop_grad_probe_huginn.json'))['seqs'] if s['R'] == R}
    return {k: [P[r['i']]['trunc'][str(k)]['rel_err'] for r in rows] for k in (1, 2, 4, 8, 16) if k < R}


def forms(w, k):
    w = np.asarray(w, float); return np.sqrt((w[k:] ** 2).sum() / (w ** 2).sum()), w[k:].sum() / w.sum()


for name, R in RUNS:
    rows = [r for f in FILES[(name, R)] for r in json.load(open(f))['rows']]; n = len(rows)
    print('\n=== %s R=%d (n=%d) ===' % (name, R, n))
    if name == 'Huginn-3.5B':
        hr = huginn_ref(rows, R)
        print('reproduction vs Sep-30 probe, same sequences (max per-sequence |diff|): %.4f' % max(
            abs(r['err'][str(k)] - hr[k][q]) for k in hr for q, r in enumerate(rows)))
    else:
        ref = REF.get((name, R), {}); err = {k: np.median([r['err'][str(k)] for r in rows]) for k in ref if str(k) in rows[0]['err']}
        print('reproduction vs earlier runs (max |diff|): %s' % ('%.4f' % max(abs(err[k] - ref[k]) for k in err) if err else 'n/a'))
    print('sketch checks (max over sequences): norm rel %.1e, adjacent cos %.1e, err %.1e' % (
        max(r['val_norm_maxrel'] for r in rows), max(r['val_cos_adjacent_maxabs'] for r in rows),
        max(r['val_err_maxabs'] for r in rows)))
    print('|g_0| / |g_R| median %.3f' % np.median([r['g0_norm'] / r['gR_norm'] for r in rows]))
    if name == 'Huginn-3.5B':
        print('(Huginn: adjoint/residual from the probe; step = residual ||x_{t+1}-x_t||/||x_t||)')
    S = np.array([np.array(r['adj'])[::-1] for r in rows]); C = np.array([r['cnorm'] for r in rows])
    ST = np.array([np.array(r['step'], float)[::-1] for r in rows])
    gain = (C / S) / (C[:, :1] / S[:, :1])
    J = [j for j in (0, 1, 2, 4, 8, 16, R - 1) if j < R]
    print('gain |c_j|/s_j rel. to last loop, median: ' + '  '.join('j%d %.2f' % (j, np.median(gain[:, j])) for j in J))
    print('signal s_j rel. to last loop, median:     ' + '  '.join('j%d %.3f' % (j, np.median(S[:, j] / S[:, 0])) for j in J))
    print('size |c_j| rel. to last loop, median:     ' + '  '.join('j%d %.3f' % (j, np.median(C[:, j] / C[:, 0])) for j in J))
    print('forward step of loop j, median:           ' + '  '.join('j%d %.3f' % (j, np.median(ST[:, j])) for j in J))
    rs = []
    for r in range(n):
        a, b = np.argsort(np.argsort(np.log(gain[r]))), np.argsort(np.argsort(np.log(ST[r])))
        rs.append(np.corrcoef(a, b)[0, 1])
    print('Spearman(gain_j, step_j) across loops, median over sequences: %.2f' % np.median(rs))
    # loop-only lost gradient and the two predictions
    el = {k: np.array([r['err'][str(k)] * r['gR_norm'] / r['gR_minus_g0_norm'] for r in rows]) for k in KS if k < R}
    verdict = {'in': 0, 'above': 0, 'below': 0}
    for k in el:
        p0 = np.median([forms(S[r], k) for r in range(n)], axis=0); p1 = np.median([forms(C[r], k) for r in range(n)], axis=0)
        m = np.median(el[k]); lo, hi = min(p1), max(p1)
        tag = 'in' if lo - 0.02 <= m <= hi + 0.02 else ('above' if m > hi + 0.02 else 'below'); verdict[tag] += 1
        closed = (max(p1) - max(p0)) / (m - max(p0)) if m != max(p0) else float('nan')
        print('k=%d err_loop %.3f | P0 (signal sizes) %.3f-%.3f | P1 (measured sizes) %.3f-%.3f -> %s | gap closed by sizes %.0f%%'
              % (k, m, min(p0), max(p0), lo, hi, tag, 100 * closed))
    v = ('SIZES' if verdict['in'] == len(el) else 'ANGLES' if verdict['above'] >= 2 else
         'SIZES-OVERSHOOT' if verdict['below'] >= 2 else 'MIXED')
    print('VERDICT (Q1): %s %s' % (v, verdict))
    # angle structure from the sketched Gram (contributions only)
    cs = []
    for r in rows:
        G = np.array(r['gram_sketch'])[1:, 1:]; dg = np.sqrt(np.clip(np.diag(G), 1e-300, None)); cs.append(G / np.outer(dg, dg))
    cs = np.array(cs); iu = lambda a, b: [(i, j) for i in range(a, b) for j in range(i + 1, b)]
    mean = lambda pairs: np.median([np.mean([c[i, j] for i, j in pairs]) for c in cs])
    h = min(4, R // 2)
    print('mean cos among kept loops j<%d: %.3f | among dropped j>=%d: %.3f | kept-dropped: %.3f' % (
        h, mean(iu(0, h)), h, mean(iu(h, R)), np.median([np.mean([c[i, j] for i in range(h) for j in range(h, R)]) for c in cs])))
    adjc = np.median(np.array([r['cos_adjacent'] for r in rows]), 0)
    bands = [(a, min(b, R - 1)) for a, b in [(0, 3), (3, 7), (7, 15), (15, R - 1)] if a < R - 1]
    print('adjacent-loop cos (exact), median by position: ' + '  '.join('j%d-%d %.3f' % (a, b, np.mean(adjc[a:b])) for a, b in bands if a < R - 1))
    print('negative off-diagonal cosines: %.1f%% of pairs; minimum %.3f' % (
        100 * np.mean([np.mean(c[np.triu_indices(R, 1)] < 0) for c in cs]), min(c[np.triu_indices(R, 1)].min() for c in cs)))
