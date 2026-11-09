"""v226 (plan frozen in docs/prereg_v226_ft_seeds.md before any measurement): more fine-tune seeds for S15-S17 on the
two Part II models.

S15 (v210), S16 (v211) and S17 (v218) each followed one fine-tune per model: seed 2011, the first pre-listed seed that
passed the guard. v226 repeats those fine-tunes and measurements with seeds 2012 and 2013 on pln5 and r5. It imports
the three drivers unchanged and sets their FT_SEEDS to the one seed being run:
  v218 --do ft     full-backprop fine-tune to step 600, measured at steps 0 / 300 / 600 on GSM8K test and WikiText
                   (the step-300 GSM8K value is S15's full300; D1, D2 of S17)
  v210 A trunc300  fine-tune through the last 2 loops to step 300 of 600, then measured (S15 A2)
  v211 s0,zero300  the k = 0 fine-tune to step 300 of 600, measured, with its loop-weights-fixed check (S16)
A seed that fails a driver's guard is reported as excluded and not replaced.

  python v226_ft_seeds.py --model pln5|r5 --seed 2012|2013 --weights DIR
  python v226_ft_seeds.py --analyze RESULTS_V210 RESULTS_V211 RESULTS_V218 V226_DIR [...]
Env: TINY=1 (pipeline test on a tiny Part II model, CPU), OUT_DIR.
"""
import argparse, glob, hashlib, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
KS = (1, 2, 4)
SEEDS = (2012, 2013)                 # the next pre-listed seeds of v210 / v211 / v218 (FT_SEEDS = 2011, 2012, 2013)
MODELS = ('pln5', 'r5')
FALL_MIN = 0.05                      # "it falls": the all-parameter fall at step 300, mean over k (v226 rule)


def sha(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(open(p, 'rb').read().replace(b'\r\n', b'\n'))
    return h.hexdigest()[:12]


def run(model, seed, weights):
    import v218_robustness as D, v210_followups as A, v211_k0_control as Z
    for mod in (D, A, Z):
        mod.FT_SEEDS = (seed,)
    here = os.path.dirname(os.path.abspath(__file__))
    rec = dict(model=model, seed=seed, driver_sha=sha([os.path.abspath(__file__)] + [os.path.join(here, f) for f in (
        'v218_robustness.py', 'v210_followups.py', 'v211_k0_control.py', 'v201_batch_lost.py')]))
    json.dump(rec, open(os.path.join(D.OUT, 'v226_%s_%d_meta.json' % (model, seed)), 'w'), indent=1)
    print('v226 %s seed %d (driver %s)' % (model, seed, rec['driver_sha']), flush=True)
    D.run(model, ['ft'], weights)                    # writes v218_<model>_ft.json
    A.part_a(model, ['trunc300'], weights)           # writes v210_A_<model>.json
    Z.run_model(model, ['s0', 'zero300'], weights)   # writes v211_<model>_s0_zero300.json


# ---------------------------------------------------------------------------------------------------------- analysis
def load(dirs):
    """{(model, seed): dict(ft=v218 ft record, trunc=v210 trunc300 rec, zero=v211 zero300 rec, s0=v211 s0 rec)}.
    Seed 2011 comes from the original v210 / v211 / v218 outputs; seeds 2012 / 2013 from v226 outputs."""
    out = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, '**', 'v218_*.json'), recursive=True):
            r = json.load(open(f))
            for ft in r['runs'].get('ft', []) + r['runs'].get('excluded', []):
                out.setdefault((r['model'], ft['seed']), {})['ft'] = ft
        for f in glob.glob(os.path.join(d, '**', 'v210_A_*.json'), recursive=True):
            r = json.load(open(f)); t = r['runs'].get('trunc300')
            if t and 'finetune' in t:
                out.setdefault((r['model'], t['finetune']['seed']), {})['trunc'] = t
            for x in r['runs'].get('excluded', []):
                out.setdefault((r['model'], x['seed']), {})['trunc_excluded'] = x
        for f in glob.glob(os.path.join(d, '**', 'v211_*.json'), recursive=True):
            r = json.load(open(f)); z = r['runs'].get('zero300')
            if z and 'finetune' in z:
                out.setdefault((r['model'], z['finetune']['seed']), {}).update(zero=z, s0=r['runs'].get('s0'))
            for x in r['runs'].get('excluded', []):
                out.setdefault((r['model'], x['seed']), {})['zero_excluded'] = x
    return out


def med(rows, key, k):
    return float(np.median([r[key][str(k)] if str(k) in r[key] else r[key][k] for r in rows]))


def mean_k(rows, key):
    return float(np.mean([med(rows, key, k) for k in KS]))


def quantities(c):
    """The per-(model, seed) numbers behind S12/S15/S16/S17's labels, as their drivers compute them (all falls are
    means over k = 1, 2, 4 of per-sequence medians on the first 64 GSM8K test sequences)."""
    ft = c.get('ft')
    if not ft or not ft['finetune']['ok']:
        return None
    ck = {int(s): v for s, v in ft['ckpt'].items()}
    st = sorted(ck)                                  # steps 0, 300, 600 (0, 2, 4 under TINY)
    if len(st) < 3:
        return None
    s0, f300, f600 = ck[st[0]]['gsm'], ck[st[1]]['gsm'], ck[st[2]]['gsm']
    q = dict(fall_all=mean_k(s0, 'all') - mean_k(f300, 'all'), fall_loop=mean_k(s0, 'loop') - mean_k(f300, 'loop'))
    q['a1_ratio'] = q['fall_loop'] / q['fall_all'] if q['fall_all'] > 0 else float('nan')        # S15 A1
    q['d1_change'] = (mean_k(s0, 'all') - mean_k(f600, 'all')) - q['fall_all']                    # S17 D1
    w300 = mean_k(ck[st[0]]['wiki'], 'all') - mean_k(ck[st[1]]['wiki'], 'all')
    q['d2_ratio'] = w300 / q['fall_all'] if q['fall_all'] > 0 else float('nan')                   # S17 D2
    if c.get('trunc') and c['trunc']['finetune']['ok']:
        q['trunc_fall'] = mean_k(s0, 'all') - mean_k(c['trunc']['rows'], 'all')                    # S15 A2
        q['a2_ratio'] = q['trunc_fall'] / q['fall_all'] if q['fall_all'] > 0 else float('nan')
    if c.get('zero') and c['zero']['finetune']['ok']:
        zf = mean_k(s0, 'loop') - mean_k(c['zero']['rows'], 'loop')
        q['s16_ratio'] = zf / q['fall_loop'] if q['fall_loop'] > 0 else float('nan')               # S16
        q['loop_fixed'] = c['zero'].get('check_loop_fixed', {}).get('ok', None)
    return q


def labels(q):
    """Each study's own per-model classification (thresholds as frozen in v210, v211, v218), plus 'falls'."""
    L = dict(falls='FALLS' if q['fall_all'] >= FALL_MIN else 'NO-FALL',
             a1='IN-LOOP' if q['a1_ratio'] >= 0.5 else 'READOUT-SHARE',
             d1='FALLS-FURTHER' if q['d1_change'] >= 0.02 else 'HOLDS' if abs(q['d1_change']) < 0.02 else 'RECOVERS',
             d2='GENERAL' if q['d2_ratio'] >= 0.5 else 'DOMAIN' if q['d2_ratio'] < 0.25 else 'PARTIAL')
    if 'trunc_fall' in q:
        ft, fa = q['trunc_fall'], q['fall_all']
        L['a2'] = 'FALLS' if ft >= 0.5 * fa else 'FLAT' if abs(ft) < 0.03 else 'RISES' if ft <= -0.03 else 'PARTIAL'
    if 's16_ratio' in q:
        r = q['s16_ratio']
        L['s16'] = 'LOOP-NEEDED' if r < 0.25 else 'OUTSIDE-SUFFICES' if r >= 0.75 else 'PARTIAL'
    for key, ratio in (('a1', 'a1_ratio'), ('d2', 'd2_ratio'), ('s16', 's16_ratio')):
        if key in L and not np.isfinite(q[ratio]):       # no fall to divide by
            L[key] = 'UNDEFINED'
    return L


def analyze(dirs):
    data = load(dirs)
    lines = []; P = lines.append
    P('v226: fine-tune seeds for S15-S17 on the Part II models (seed 2011 = the original studies)')
    keys = ('fall_all', 'fall_loop', 'a1_ratio', 'a2_ratio', 's16_ratio', 'd1_change', 'd2_ratio')
    verdict = {}
    for m in MODELS:
        per = {}
        for s in (2011,) + SEEDS:
            c = data.get((m, s), {})
            q = quantities(c)
            if q is None:
                P('  %-5s seed %d: %s' % (m, s, 'EXCLUDED (guard)' if c.get('ft') else 'missing')); continue
            excl = [x for x in ('trunc_excluded', 'zero_excluded') if x in c]
            per[s] = (q, labels(q))
            P('  %-5s seed %d: %s | labels %s%s%s' % (m, s, ' '.join('%s %.3f' % (k, q[k]) for k in keys if k in q),
              per[s][1], '' if q.get('loop_fixed', True) else ' | LOOP-FIXED CHECK FAILED', (' | excluded: %s' % excl) if excl else ''))
        if 2011 not in per:
            P('  %-5s: the seed-2011 reference is missing' % m); continue
        new = [s for s in SEEDS if s in per]
        for key in ('falls', 'a1', 'a2', 's16', 'd1', 'd2'):
            ref = per[2011][1].get(key)
            got = [per[s][1].get(key) for s in new]
            verdict[(m, key)] = ('INCOMPLETE' if len(new) < 2 or None in got or ref is None else
                                 'REPLICATED' if all(g == ref for g in got) else 'NOT-REPLICATED')
            P('  %-5s %-5s seed 2011 %-16s new seeds %s -> %s' % (m, key, ref, got, verdict[(m, key)]))
        for key in keys:
            xs = [per[s][0][key] for s in per if key in per[s][0] and np.isfinite(per[s][0][key])]
            if len(xs) >= 2:
                P('  %-5s %-10s mean over %d seeds %.3f, SD %.3f, range %.3f to %.3f' % (
                    m, key, len(xs), np.mean(xs), np.std(xs, ddof=1), min(xs), max(xs)))
    n_rep = sum(v == 'REPLICATED' for v in verdict.values()); n_all = len(verdict)
    inc = any(v == 'INCOMPLETE' for v in verdict.values()) or n_all < 2 * 6
    P('VERDICT: %s (%d of %d model-labels replicated on both new seeds)%s' % (
        'INCOMPLETE' if inc else 'ALL-REPLICATED' if n_rep == n_all else 'PARTLY-REPLICATED', n_rep, n_all,
        '' if inc or n_rep == n_all else '; not replicated: %s' % sorted('%s %s' % k for k, v in verdict.items()
                                                                        if v != 'REPLICATED')))
    return '\n'.join(lines)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=MODELS); ap.add_argument('--seed', type=int, choices=SEEDS)
    ap.add_argument('--weights', default=None); ap.add_argument('--analyze', nargs='*')
    A_ = ap.parse_args()
    if A_.analyze is not None:
        txt = analyze(A_.analyze); print(txt)
        open(os.path.join(os.environ.get('OUT_DIR', '.'), 'v226_analysis.txt'), 'w').write(txt + '\n')
    else:
        run(A_.model, A_.seed, A_.weights)
