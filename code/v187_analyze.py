"""v187 analysis: Huginn-3.5B lost gradient and rho at R = 8, 16, 32, 64 (plan: docs/v187_huginn_rho_transfer_plan.md).
R = 16 / 32 from the 2026-09-30 probe, R = 8 / 64 from kaggle-output-v187/. Same statistics as make_paper_figs.py probe():
rho = median over sequences of exp(-slope) of a log-linear fit to the adjoint; lost gradient = median rel_err per k.
"""
import json, numpy as np

OLD = 'kaggle-output-loopgrad/huginn/loop_grad_probe_huginn.json'
SRC = {8: 'kaggle-output-v187/R8/loop_grad_probe_huginn.json', 16: OLD, 32: OLD,
       64: 'kaggle-output-v187/R64/loop_grad_probe_huginn.json'}


def load(p, R):
    return [s for s in json.load(open(p))['seqs'] if s['R'] == R and s.get('trunc') and s.get('adjoint')]


def fit(a, lo=0, hi=None):
    a = np.array(a)[lo:hi]; return np.exp(-np.polyfit(np.arange(len(a)), np.log(a + 1e-30), 1)[0])


res = {}
for R, p in SRC.items():
    rows = load(p, R); rho = np.array([fit(s['adjoint']) for s in rows])
    ks = sorted(int(k) for k in rows[0]['trunc'] if int(k) < R)
    err = {k: np.array([s['trunc'][str(k)]['rel_err'] for s in rows]) for k in ks}
    res[R] = err
    print('R=%d  n=%d  rho %.3f  resid_end %.3g' % (R, len(rows), np.median(rho), np.median([s['resid'][-1] for s in rows])))
    print('   lost   ', ' '.join('k%d %.3f' % (k, np.median(err[k])) for k in ks))
    print('   rho^k  ', ' '.join('k%d %.3f' % (k, np.median(rho ** k)) for k in ks))
    print('   aligned', ' '.join('k%d %.3f' % (k, np.median((rho ** k - rho ** R) / (1 - rho ** R))) for k in ks))
    print('   orthog ', ' '.join('k%d %.3f' % (k, np.median(np.sqrt((rho ** (2 * k) - rho ** (2 * R)) / (1 - rho ** (2 * R)))))
                               for k in ks))
    # adjoint index 0 = the loop farthest from the output
    segs = [(lo, hi) for lo, hi in [(0, R // 2), (R // 2, R)]]
    print('   rho over loops', ', '.join('%d-%d: %.3f' % (lo, hi, np.median([fit(s['adjoint'], lo, hi) for s in rows]))
                                         for lo, hi in segs), '(0 = farthest from the output)')

rng = np.random.default_rng(0)
for R in (8, 16, 64):
    for k in (1, 2, 4, 8):
        if k not in res[R] or k not in res[32]: continue
        x, y = res[R][k], res[32][k]           # same 32 sequences in the same order
        bs = [np.median(x[i]) - np.median(y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(5000))]
        print('R%d - R32  k=%d  %+.3f  paired bootstrap 95%% [%+.3f, %+.3f]' % (R, k, np.median(x) - np.median(y),
                                                                              *np.percentile(bs, [2.5, 97.5])))
