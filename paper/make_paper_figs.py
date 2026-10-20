"""Table 1 and Figures 1-2 for the looped-truncation workshop paper (paper_looped/draft.md).
Writes paper_looped/{table1.md, fig1_rule.png/.pdf, fig2_cost.png/.pdf}.

Sources:
  Huginn-3.5B, Parcae-1.3B, Ouro-1.4B   kaggle-output-loopgrad/*/loop_grad_probe_*.json (Sep 30; WikiText + GSM8K, 128
                                         tokens). Gradient over a parameter subset: Huginn / Parcae = injection adapter +
                                         first core layer; Ouro = one middle layer of its looped stack.
  Parcae-140m                            kaggle-output-v162/pilot/v162_measure_R16_pilotmeasure.json (all parameters)
  Parcae-370m                            kaggle-output-v189/v189_370m_R16.json (v164 measure(), all parameters; v189)
  SmolLM2 conversions A (run 1), B       docs/retrofit_rho_probe_2026-10-02.md, R 8 (all parameters; JSON on Drive)
  Cost prediction                        docs/v166_slam_dunk_result.md, docs/v167_retrofit_transfer_result.md (means, s.e.)
rho = exp(-slope) of a log-linear fit of the backward-signal size over all R loops, median over sequences.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

OUT = os.path.dirname(os.path.abspath(__file__)); os.makedirs(OUT, exist_ok=True)


def probe(path, R):
    d = json.load(open(path)); rows = [s for s in d['seqs'] if s['R'] == R and s.get('trunc') and s.get('adjoint')]
    rho = np.array([np.exp(-np.polyfit(np.arange(len(s['adjoint'])), np.log(np.array(s['adjoint']) + 1e-30), 1)[0]) for s in rows])
    ks = sorted(int(k) for k in rows[0]['trunc'] if int(k) < R)
    err = {k: float(np.median([s['trunc'][str(k)]['rel_err'] for s in rows])) for k in ks}
    law = {k: float(np.median(rho ** k)) for k in ks}
    return dict(rho=float(np.median(rho)), err=err, law=law)


def v189(path, R):
    """v189 / v164 measure() JSON (Table 1 protocol): median rho, median err, median rho_i^k."""
    e = [x for x in json.load(open(path))['res'] if x['R'] == R][0]; rho = np.array([r['rho'] for r in e['rows']])
    ks = sorted(int(k) for k in e['rows'][0]['err'])
    return dict(rho=float(np.median(rho)), err={k: float(np.median([r['err'][str(k)] for r in e['rows']])) for k in ks},
                law={k: float(np.median(rho ** k)) for k in ks})


def v162(path):
    s = json.load(open(path))['summary']
    return dict(rho=s['rho_median'], err={int(k): v for k, v in s['err_median'].items()},
                law={int(k): v for k, v in s['law_median'].items()})


def v197():
    """Ouro-1.4B on the Table 1 protocol (v197: all parameters, 16 GSM8K-val sequences, batch 1, fp32), if downloaded;
    otherwise None (Table 1 then falls back to the Sep-30 one-layer probe)."""
    rows = []
    for f in sorted(__import__('glob').glob('kaggle-output-v197/**/v197_ouro_*.json', recursive=True)):
        rows += json.load(open(f))['rows']
    if not rows:
        return None
    rows = sorted({r['i']: r for r in rows}.values(), key=lambda r: r['i'])
    rho = np.array([r['rho'] for r in rows])
    return dict(rho=float(np.median(rho)), err={k: float(np.median([r[str(k)]['all']['err'] for r in rows])) for k in (1, 2, 3)},
                law={k: float(np.median(rho ** k)) for k in (1, 2, 3)}, n=len(rows))


M = [  # name, trained with backprop through, R, measurement, all parameters?
    ('Huginn-3.5B', 'last 8 loops', 32, probe('kaggle-output-loopgrad/huginn/loop_grad_probe_huginn.json', 32), False),
    ('Parcae-1.3B', 'last half of the loops', 16, probe('kaggle-output-loopgrad/parcae/loop_grad_probe_parcae.json', 16), False),
    ('Parcae-370m', 'last half of the loops', 16, v189('kaggle-output-v189/v189_370m_R16.json', 16), True),
    ('Parcae-140m', 'last half of the loops', 16, v162('kaggle-output-v162/pilot/v162_measure_R16_pilotmeasure.json'), True),
    ('SmolLM2 converted (A)', 'last 4 loops', 8, dict(rho=0.347, err={1: 0.304, 2: 0.108, 4: 0.015},
                                                      law={1: 0.347, 2: 0.120, 4: 0.015}), True),
    ('SmolLM2 converted (B)', 'last 4 loops', 8, dict(rho=0.384, err={1: 0.316, 2: 0.122, 4: 0.021},
                                                      law={1: 0.384, 2: 0.147, 4: 0.022}), True),
    ('Ouro-1.4B', 'all 4 loops (no truncation described)', 4, v197() or probe('kaggle-output-loopgrad/ouro/loop_grad_probe_ouro.json', 4),
     v197() is not None),
]
f3 = lambda x: ('%.3f' % x) if x < 0.05 else ('%.2f' % x)

# ---------------------------------------------------------------- Table 1
L = ['| model | trained with backprop through | R | ρ | lost gradient at k = 1 / 2 / 4 | ρᵏ | gradient measured over |',
     '|---|---|---|---|---|---|---|']
for name, tr, R, m, allp in M:
    ks = [k for k in (1, 2, 4) if k in m['err']]
    law = ' / '.join(f3(m['law'][k]) for k in ks) if m['rho'] < 1 else 'n/a (ρ > 1)'
    L.append('| %s | %s | %d | %.2f | %s | %s | %s |' % (name, tr, R, m['rho'], ' / '.join(f3(m['err'][k]) for k in ks), law,
                                                          'all parameters' if allp else 'part of the loop'))
open(os.path.join(OUT, 'table1.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('\n'.join(L))

# ---------------------------------------------------------------- Figure 1: (a) Table 1's loop counts, (b) other loop counts
def blended(files):
    """Blended (dial a < 1) settings measured with the Table 1 protocol (16 GSM8K-val sequences, batch 1, fp32, all
    parameters): per variant, median lost gradient and median rho_i^k over sequences."""
    out = {}
    for f in files:
        res = json.load(open(f))['res']
        for e in (res.values() if isinstance(res, dict) else res):
            s = e.get('summary', e); R, a = s['R'], s['a']
            if a >= 1:
                continue
            rho = np.array([r['rho'] for r in e['rows']]); v = out.setdefault((R, a), dict(rho=float(np.median(rho)), pts={}))
            for k in e['rows'][0]['err']:
                if int(k) < R and int(k) not in v['pts']:
                    v['pts'][int(k)] = (float(np.median(rho ** int(k))), float(np.median([r['err'][k] for r in e['rows']])))
    return out


BLEND = [('Parcae-140m, blended', 3, blended(['kaggle-output-v163/pilot-b/v163_measure_pilotB0.json',
                                                'kaggle-output-v164/measure/v164_measure_measure.json',
                                                'results_v166_pre/a/v164_measure_pre0.json'])),
         ('SmolLM2 (A), blended', 4, blended(['results_v167_pilot/p/gpu0/v167_measure_floor.json']))]
for name, ci, B_ in BLEND:
    print(name, {'R%d a%.1f rho %.3f' % (R, a, v['rho']): {k: tuple(round(x, 3) for x in p) for k, p in sorted(v['pts'].items())}
                 for (R, a), v in sorted(B_.items())})


def rows_of(path, R, kind):
    """Per-sequence (rho_i, {k: err_i}) from the probe JSONs, v162/v164/v189 measure JSONs."""
    d = json.load(open(path))
    if kind == 'probe':
        out = []
        for s in d['seqs']:
            if s['R'] == R and s.get('trunc') and s.get('adjoint'):
                a = np.array(s['adjoint']); rho = np.exp(-np.polyfit(np.arange(len(a)), np.log(a + 1e-30), 1)[0])
                out.append((rho, {int(k): v['rel_err'] for k, v in s['trunc'].items() if int(k) < R}))
        return out
    if kind == 'v162':
        return [(r['rho'], {int(k): v for k, v in r['err'].items()}) for r in d['rows']]
    if kind == 'v196':                                              # v196 rows: per-loop signal sizes, t-order
        out = []
        for r in d['rows']:
            a = np.array(r['adj'], float); rho = np.exp(-np.polyfit(np.arange(len(a)), np.log(a + 1e-30), 1)[0])
            out.append((rho, {int(k): v for k, v in r['err'].items() if int(k) < R}))
        return out
    res = [e for e in d['res'] if e['R'] == R and e['a'] == 1.0][0]                        # v164 / v189 measure()
    return [(r['rho'], {int(k): v for k, v in r['err'].items()}) for r in res['rows']]


def boot_ci(x, n=4000, seed=0):
    g = np.random.default_rng(seed); x = np.asarray(x)
    m = [np.median(x[g.integers(0, len(x), len(x))]) for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


# Every unblended point measured per sequence on a model with a shrinking signal (outside reviews 4-5).
LG = 'kaggle-output-loopgrad/%s/loop_grad_probe_%s.json'
ALL = [('Huginn-3.5B', 32, rows_of(LG % ('huginn', 'huginn'), 32, 'probe')),
       ('Huginn-3.5B', 16, rows_of(LG % ('huginn', 'huginn'), 16, 'probe')),
       ('Parcae-1.3B', 16, rows_of(LG % ('parcae', 'parcae'), 16, 'probe')),
       ('Parcae-140m', 16, rows_of('kaggle-output-v162/pilot/v162_measure_R16_pilotmeasure.json', 16, 'v162')),
       ('Parcae-140m', 8, rows_of('kaggle-output-v162/pilot/v162_measure_R8_pilotmeasure.json', 8, 'v162'))]
for R in (8, 64):
    if os.path.exists('kaggle-output-v187/R%d/loop_grad_probe_huginn.json' % R):
        ALL.append(('Huginn-3.5B', R, rows_of('kaggle-output-v187/R%d/loop_grad_probe_huginn.json' % R, R, 'probe')))
for R in (16, 32):
    if os.path.exists('kaggle-output-v189/v189_370m_R%d.json' % R):
        ALL.append(('Parcae-370m', R, rows_of('kaggle-output-v189/v189_370m_R%d.json' % R, R, 'v164')))
# S10 (v196, pre-registered): Parcae-140m at 32 / 48 / 64 loops, Parcae-370m at 48 / 64, Huginn at 48
for tag, name, Rs in (('140m', 'Parcae-140m', (32, 48, 64)), ('370m', 'Parcae-370m', (48, 64)), ('huginn', 'Huginn-3.5B', (48,))):
    for R in Rs:
        rows = []
        for f in sorted(__import__('glob').glob('kaggle-output-v196/**/v196_%s*_R%d.json' % (tag, R), recursive=True)):
            rows += rows_of(f, R, 'v196')
        if rows:
            ALL.append((name, R, rows))
T1R = {'Huginn-3.5B': 32, 'Parcae-1.3B': 16, 'Parcae-370m': 16, 'Parcae-140m': 16}     # Table 1's loop counts
COL = {'Huginn-3.5B': 'C0', 'Parcae-1.3B': 'C1', 'Parcae-370m': 'C2', 'Parcae-140m': 'C3',
       'SmolLM2 converted (A)': 'C4', 'SmolLM2 converted (B)': 'C5'}
PARTIAL = {'Huginn-3.5B', 'Parcae-1.3B'}                                                 # gradient over part of the loop
pts, ciw = [], []
for name, R, rows in ALL:
    rho = np.array([r[0] for r in rows])
    for k in sorted(rows[0][1]):
        e = np.array([r[1][k] for r in rows]); law = float(np.median(rho ** k)); lo, hi = boot_ci(e)
        pts.append((name, R, k, float(np.median(e)), law, lo <= law <= hi)); ciw.append((hi - lo) / 2)
    print('  all-points: %-12s R %2d  ' % (name, R) + '  '.join('k%d %.3f/%.3f%s' % (p[2], p[3], p[4], '' if p[5] else '*')
                                                         for p in pts if p[0] == name and p[1] == R))


def spread(P, label):
    rel = [(e - l) / l for e, l in P if l > 0.05]
    print('%s: n %d, max |measured - rho^k| %.3f, relative %+.0f%% to %+.0f%% (rho^k > 0.05)' % (
        label, len(P), max(abs(e - l) for e, l in P), 100 * min(rel), 100 * max(rel)))


# unblended points at Table 1's loop counts (per-sequence medians; SmolLM2 from its summary)
spread([(m['err'][k], m['law'][k]) for _, _, _, m, _ in M[:-1] for k in m['err']], 'Table 1 loop counts, unblended, six models')
spread([(p[3], p[4]) for p in pts if T1R.get(p[0]) != p[1]], 'other loop counts, unblended')
spread([p[::-1] for v in BLEND[0][2].values() for p in v['pts'].values()], 'blended Parcae-140m (R 32-64)')
spread([p[::-1] for v in BLEND[1][2].values() for p in v['pts'].values()], 'blended SmolLM2 (A) (R 8)')
rel = [(p[3] - p[4]) / p[4] for p in pts if p[4] > 0.05]
print('all measured points (unblended, shrinking signal, per-sequence data; * = rho^k outside the bootstrap 95%% CI of the '
      'median): n %d, max |diff| %.3f, relative %+.0f%% to %+.0f%%; rho^k inside the CI for %d; CI half-width median %.3f '
      '(range %.3f-%.3f)' % (len(pts), max(abs(p[3] - p[4]) for p in pts), 100 * min(rel), 100 * max(rel),
                             sum(p[5] for p in pts), np.median(ciw), min(ciw), max(ciw)))

fig, axs = plt.subplots(1, 2, figsize=(7.8, 4.7))
ax = axs[0]
for (name, tr, R, m, allp), mk in zip(M[:-1], ['o', 's', 'P', '^', 'D', 'v']):
    ks = sorted(m['err'])
    ax.scatter([m['law'][k] for k in ks], [m['err'][k] for k in ks], marker=mk, s=34,
               label='%s%s, %d loops (ρ %.2f)' % (name.replace(' converted', ''), ', ours' if 'SmolLM2' in name else '',
                                                  R, m['rho']),
               facecolors=COL[name] if allp else 'none', edgecolors=COL[name])
for (name, ci, B_), mk in zip(BLEND, ['x', '+']):
    P_ = [p for v in B_.values() for p in v['pts'].values()]; rhos = [v['rho'] for v in B_.values()]
    Rs = sorted({R for (R, a) in B_}); rs = ('%d–%d' % (Rs[0], Rs[-1])) if len(Rs) > 1 else '%d' % Rs[0]
    ax.scatter([p[0] for p in P_], [p[1] for p in P_], marker=mk, s=30, c='C%d' % ci, alpha=0.85,
               label='%s, %s loops (ρ %.2f–%.2f)' % (name, rs, min(rhos), max(rhos)))
ax.set_title('(a) Table 1 loop counts, and blended loops', fontsize=8.5)
ax = axs[1]
RMK = {8: 'v', 16: 'o', 32: 's', 48: '^', 64: '*'}
for name, R, rows in ALL:
    if T1R.get(name) == R:
        continue
    rho = np.array([r[0] for r in rows]); ks = sorted(rows[0][1])
    ax.scatter([float(np.median(rho ** k)) for k in ks], [float(np.median([r[1][k] for r in rows])) for k in ks],
               marker=RMK[R], s=46 if R == 64 else 30, facecolors='none' if name in PARTIAL else COL[name],
               edgecolors=COL[name], linewidths=0.9)
from matplotlib.lines import Line2D
hm = [Line2D([], [], ls='', marker='o', color=COL[n], mfc='none' if n in PARTIAL else COL[n], label=n)
      for n in ('Huginn-3.5B', 'Parcae-370m', 'Parcae-140m')]
hr = [Line2D([], [], ls='', marker=RMK[R], color='0.3', mfc='0.3', label='%d loops' % R) for R in (8, 16, 32, 48, 64)]
ax.set_title('(b) the same models at other loop counts', fontsize=8.5)
for ax in axs:
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('ρᵏ  (shrink per loop, to the power k)'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
axs[0].set_ylabel('measured lost gradient')
axs[0].legend(fontsize=6.2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
axs[1].legend(handles=hm + hr, fontsize=6.2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
fig.tight_layout(); [fig.savefig(os.path.join(OUT, 'fig1_rule.' + e), dpi=200, bbox_inches='tight') for e in ('png', 'pdf')]

# ---------------------------------------------------------------- Table: the lost gradient across loop counts (Section 2)
LR = ['| model | R | ρ | lost gradient at k = 1 / 2 / 4 / 8 | ρᵏ at k = 1 / 2 / 4 / 8 |', '|---|---|---|---|---|']
for name in ('Parcae-140m', 'Parcae-370m', 'Huginn-3.5B'):
    for nm, R, rows in sorted([x for x in ALL if x[0] == name], key=lambda x: x[1]):
        rho = np.array([r[0] for r in rows]); ks = [k for k in (1, 2, 4, 8) if k in rows[0][1]]
        LR.append('| %s | %d | %.2f | %s | %s |' % (name + ('*' if name in PARTIAL else ''), R, float(np.median(rho)),
                                                   ' / '.join('%.2f' % float(np.median([r[1][k] for r in rows])) for k in ks),
                                                   ' / '.join('%.2f' % float(np.median(rho ** k)) for k in ks)))
open(os.path.join(OUT, 'table_R.md'), 'w', encoding='utf-8').write('\n'.join(LR) + '\n')
print('\n'.join(LR))

# ---------------------------------------------------------------- Figure 2 (means +- 1 s.e. over 5 seeds)
# v166, Parcae-140m (locked predictions): (observed, s.e., measured-gradient curve, k alone pooled over earlier runs)
v166 = [(0.0578, 0.0041, 0.0611, 0.0057), (0.0268, 0.0027, 0.0278, 0.0000),
        (0.0879, 0.0054, 0.1012, 0.0468), (0.0316, 0.0034, 0.0330, 0.0057)]
# v167, SmolLM2 converted (A), mixed cells: (observed, s.e., same curve refit on 29 cells, k alone = this model's unmixed cost)
v167 = [(0.0113, 0.0006, 0.0484, 0.0087), (0.0019, 0.0006, 0.0087, -0.0012),
        (0.0087, 0.0009, 0.0781, 0.0087), (0.0042, 0.0008, 0.0207, -0.0012)]
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.6))
for ax, data, title in ((axs[0], v166, 'Parcae-140m (predictions locked before training)'),
                        (axs[1], v167, 'SmolLM2 converted (A): same curve does not transfer')):
    obs = [d[0] for d in data]; se = [d[1] for d in data]
    ax.errorbar([d[2] for d in data], obs, yerr=se, fmt='o', ms=6, capsize=2, label='from measured lost gradient', zorder=3)
    ax.errorbar([d[3] for d in data], obs, yerr=se, fmt='x', ms=7, capsize=2, label='from k alone', zorder=3)
    ax.plot([0, 0.11], [0, 0.11], 'k--', lw=1)
    ax.set_xlim(-0.005, 0.11); ax.set_ylim(-0.005, 0.11)
    ax.set_xlabel('predicted training cost'); ax.set_ylabel('observed training cost'); ax.set_title(title, fontsize=8.3)
axs[0].legend(fontsize=7, loc='lower right')
h, _ = axs[1].get_legend_handles_labels()       # the two errorbar containers (the diagonal has no label)
axs[1].legend(h, ['from measured lost gradient', 'from k alone (this model, unblended)'], fontsize=6.5,
              loc='upper right')
axs[1].set_ylim(-0.003, 0.016)                   # every observed cost is < 0.012: zoom the panel instead of an inset
axs[1].set_xlim(-0.005, 0.085)
fig.tight_layout(); [fig.savefig(os.path.join(OUT, 'fig2_cost.' + e), dpi=200, bbox_inches='tight') for e in ('png', 'pdf')]
# ---------------------------------------------------------------- Table 2: lost gradient split into size and direction
# v182 float64 recompute of v168's measurement (frozen models, all parameters, 4 batches x 8 GSM8K-val sequences; v168's
# float32 CPU dot products ran ~0.5% high on cos). Per-batch err, r, cos and after-Adam cos (u = g / sqrt(mean_b g^2)),
# then the mean over batches.
import glob
from v168_analyze import PARCAE, RETRO
m182 = {}
for f in glob.glob('results_v182/*/gpu1/v182_*_table2.json'):
    d = json.load(open(f))
    for row in d['table2']:
        for k, x in row['metrics'].items():
            m182[(d['model'], int(row['R']), round(float(row['a']), 2), int(k))] = x['mean']
# direction after Adam: v171b (Appendix E, D2), Adam-style statistics from 24 clipped training batches, float64; the same
# measure in Tables 2 and 3 (v182's 4-batch version reads 0.01-0.03 lower on Parcae and is not used)
m171 = {}
for f in glob.glob('results_v171b/*/gpu*/v171_*.json'):
    for row in json.load(open(f)):
        for k, x in row['metrics'].items():
            m171[(row['model'], int(row['R']), round(float(row['a']), 2), int(k))] = x['adam_cos']
for key in m182:
    m182[key]['adam_cos'] = m171[key]
T2 = [('SmolLM2 (A)', 'retro', 8, 1.0, 1), ('SmolLM2 (A)', 'retro', 8, 1.0, 2), ('SmolLM2 (A)', 'retro', 8, 0.5, 1),
      ('SmolLM2 (A)', 'retro', 8, 0.3, 1), ('SmolLM2 (A)', 'retro', 8, 0.3, 2),
      ('Parcae-140m', 'parcae', 32, 1.0, 1), ('Parcae-140m', 'parcae', 32, 1.0, 2), ('Parcae-140m', 'parcae', 32, 1.0, 4),
      ('Parcae-140m', 'parcae', 48, 0.4, 4), ('Parcae-140m', 'parcae', 64, 0.3, 8)]
L2 = ['| model | loops R | blending a | k | lost gradient | size ratio r | raw direction (cos) | direction after Adam | training cost |',
      '|---|---|---|---|---|---|---|---|---|']
for name, mod, R, a, k in T2:
    x = m182[(mod, R, a, k)]
    cost = (PARCAE if mod == 'parcae' else RETRO)[(R, a, k)]
    L2.append('| %s | %d | %.1f | %d | %.2f | %.2f | %.3f | %.3f | %.3f |' % (name, R, a, k, x['err'], x['r'], x['cos'], x['adam_cos'], cost))
open(os.path.join(OUT, 'table2.md'), 'w', encoding='utf-8').write('\n'.join(L2) + '\n')
print('\n'.join(L2))

# ---------------------------------------------------------------- Figure 3: backward-signal size per loop (log), fit quality
def adj_rows(path, R=None, key='seqs', field='adjoint'):
    d = json.load(open(path)); rows = d[key] if key in d else d['rows']
    return [np.array(s[field], float) for s in rows if (R is None or s.get('R') == R) and s.get(field)]
SRC = [('Huginn-3.5B', adj_rows('kaggle-output-loopgrad/huginn/loop_grad_probe_huginn.json', 32)),
       ('Parcae-1.3B', adj_rows('kaggle-output-loopgrad/parcae/loop_grad_probe_parcae.json', 16)),
       ('Parcae-140m', adj_rows('kaggle-output-v162/pilot/v162_measure_R16_pilotmeasure.json', key='rows', field='adj')),
       ('SmolLM2 (A)', [np.array(r['adj'], float) for r in json.load(open('results_v167_pilot/p/gpu0/v167_measure_floor.json'))['res'][0]['rows']]),
       ('Ouro-1.4B', [np.array(r['adj'], float) for f in sorted(glob.glob('kaggle-output-v197/**/v197_ouro_*.json', recursive=True))
                      for r in json.load(open(f))['rows']]                       # Table 1 protocol (v197)
                     or adj_rows('kaggle-output-loopgrad/ouro/loop_grad_probe_ouro.json', 4))]
fig, ax = plt.subplots(figsize=(4.4, 3.6)); fitq = []
for (name, rows), mk in zip(SRC, ['o', 's', '^', 'D', 'x']):
    A = np.array([r[::-1] / r[-1] for r in rows])            # loops back from the output, normalised to the last loop
    med = np.median(A, 0); j = np.arange(A.shape[1])
    r2s = []
    for r in A:
        y = np.log(r + 1e-30); p = np.polyfit(j, y, 1); r2s.append(1 - np.sum((y - np.polyval(p, j)) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12))
    fitq.append((name, float(np.median(r2s)), len(rows)))
    ax.semilogy(j, med, marker=mk, ms=3.5, lw=1, label='%s (median R² %.2f)' % (name, np.median(r2s)))
ax.set_xlabel('loops back from the output'); ax.set_ylabel('backward-signal size (relative to last loop)')
ax.legend(fontsize=6.5); fig.tight_layout(); [fig.savefig(os.path.join(OUT, 'fig3_fit.' + e), dpi=200) for e in ('png', 'pdf')]
print('fit quality (median R^2 of log-linear fit, n seqs):', fitq)
print('wrote', sorted(os.listdir(OUT)))

# ---------------------------------------------------------------- Figure 4: cost vs lost gradient, every trained setting
# lost gradient as in Table 2 (v168: 4 batches x 8 GSM8K-val seqs, mean over batches; norms only, so float32-safe)
m168 = {}
for f in glob.glob('results_v168/*/gpu*/v168_*.json'):
    for row in json.load(open(f)):
        for k, x in row['metrics'].items():
            m168[(row['model'], int(row['R']), round(float(row['a']), 2), int(k))] = x
fig, ax = plt.subplots(figsize=(4.4, 3.6))
for mod, D, name, ci, mk in (('parcae', PARCAE, 'Parcae-140m', 3, 'o'), ('retro', RETRO, 'SmolLM2 (A)', 4, 's')):
    for one in (True, False):
        pts = [(m168[(mod, R, a, k)]['err'], c) for (R, a, k), c in D.items() if (mod, R, a, k) in m168 and (k == 1) == one]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker=mk, s=30, c='C%d' % ci if one else 'none',
                   edgecolors='C%d' % ci, label='%s, k %s' % (name, '= 1' if one else '≥ 2'))
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('lost gradient (as in Table 6)'); ax.set_ylabel('training cost (fraction of loop benefit)')
ax.legend(fontsize=7, loc='upper left', frameon=False)
fig.tight_layout(); [fig.savefig(os.path.join(OUT, 'fig4_cost_vs_lost.' + e), dpi=200) for e in ('png', 'pdf')]
print('fig4 points', sum(1 for D, mod in ((PARCAE, 'parcae'), (RETRO, 'retro')) for (R, a, k) in D if (mod, R, a, k) in m168))
