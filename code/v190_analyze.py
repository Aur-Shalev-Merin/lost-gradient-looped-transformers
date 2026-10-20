"""v190 analysis (plan docs/v190_parcae_adjoint_plan.md): rho fitted over all loops vs over the half nearest the output
(rho_near) and the far half (rho_far), and the lost gradient against median(rho_i^k) for each fit. Per-sequence fits,
medians over sequences. Also applied to Huginn at R = 32 / 64 (v187 and the Sep-30 probe) for comparison.
Backward-signal index t = 0 is the loop farthest from the output.
"""
import json, os
import numpy as np


def fit(a):
    a = np.asarray(a); return float(np.exp(-np.polyfit(np.arange(len(a)), np.log(a + 1e-30), 1)[0]))


def rows_parcae(path):
    d = json.load(open(path)); return d['R'], [(r['adj'], {int(k): v for k, v in r['err'].items()}) for r in d['rows']]


def rows_probe(path, R):
    out = [(s['adjoint'], {int(k): v['rel_err'] for k, v in s['trunc'].items() if int(k) < R})
           for s in json.load(open(path))['seqs'] if s['R'] == R and s.get('trunc') and s.get('adjoint')]
    return R, out


SRC = [('Parcae-370m', lambda: rows_parcae('kaggle-output-v190/v190_370m_R32.json')),
       ('Parcae-140m', lambda: rows_parcae('kaggle-output-v190/v190_140m_R32.json')),
       ('Huginn-3.5B', lambda: rows_probe('kaggle-output-loopgrad/huginn/loop_grad_probe_huginn.json', 32)),
       ('Huginn-3.5B', lambda: rows_probe('kaggle-output-v187/R64/loop_grad_probe_huginn.json', 64))]
for name, get in SRC:
    try:
        R, rows = get()
    except FileNotFoundError as e:
        print('%s: missing %s' % (name, e.filename)); continue
    h = R // 2
    fits = {'all': np.array([fit(a) for a, _ in rows]), 'near': np.array([fit(a[h:]) for a, _ in rows]),
            'far': np.array([fit(a[:h]) for a, _ in rows])}
    print('%s R=%d  n=%d  rho all %.3f  near (last %d loops) %.3f  far %.3f' % (
        name, R, len(rows), *[np.median(fits[x]) for x in ('all',)], h, np.median(fits['near']), np.median(fits['far'])))
    for k in sorted(rows[0][1]):
        m = np.median([e[k] for _, e in rows])
        print('   k=%-2d measured %.3f | rho_all^k %.3f (%+.0f%%) | rho_near^k %.3f (%+.0f%%)' % (
            k, m, np.median(fits['all'] ** k), 100 * (m / np.median(fits['all'] ** k) - 1),
            np.median(fits['near'] ** k), 100 * (m / np.median(fits['near'] ** k) - 1)))
