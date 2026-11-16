"""v240 (plan in docs/prereg_v240_k1_cleanup.md, frozen before any run): clean-up of v239 (S22) at k = 1, as S4 was
for S3. The runs use v239_shrink_turn.py unchanged (its --runs entry point, arms trunc1 / turn1 / shrink1 / full /
floor); this file holds only the seed list and the analysis.

New in v240, fixed before any run: a seed is valid only if EVERY run on it (full, floor, trunc1, turn1, shrink1) has at
most 10 non-finite steps and a finite test loss. v239's rule looked at the full run alone, so one diverged size-only
run (seed 2391105) decided its k = 1 label.

  python v240_k1_cleanup.py --analyze DIR [DIR ...]
"""
import argparse, glob, json, math, os

SEEDS = (2401101, 2401102, 2401103, 2401104, 2401105, 2401106, 2401107)   # first 5 valid in this order are scored
EXTRA_BACKUPS = (2401108, 2401109)                                         # run only if fewer than 5 of SEEDS are valid
CELLS = ('full', 'floor', 'trunc1', 'turn1', 'shrink1')
N_USE, N_MIN = 5, 4
MAX_NONFINITE = 10
GATE_NATS = 0.03


def m_k(phi_trunc):
    return max(0.015, 0.2 * phi_trunc)                 # v239's margin, unchanged


def valid_seed(R, s):
    for c in CELLS:
        r = R.get((c, s))
        if r is None or r['nonfinite'] > MAX_NONFINITE or not math.isfinite(r['test']):
            return False
    return True


def analyze(dirs):
    import numpy as np
    from scipy import stats
    rows = []
    for d in dirs:
        for f in glob.glob(os.path.join(d, '**', 'v239_train_*.jsonl'), recursive=True):
            rows += [json.loads(x) for x in open(f) if x.strip()]
    R = {(r['arm'], r['seed']): r for r in rows if r['seed'] in SEEDS + EXTRA_BACKUPS}
    L = []; P = L.append
    order = list(SEEDS) + list(EXTRA_BACKUPS)
    ran = [s for s in order if any((c, s) in R for c in CELLS)]
    valid = [s for s in ran if valid_seed(R, s)][:N_USE]
    bad = {s: [c for c in CELLS if (c, s) in R and (R[(c, s)]['nonfinite'] > MAX_NONFINITE or not math.isfinite(R[(c, s)]['test']))]
           for s in ran if s not in valid}
    P('v240 analysis: %d runs; scored seeds %s; invalid or unused seeds %s' % (len(R), valid, bad))
    if len(valid) < N_MIN:
        P('VERDICT: INCOMPLETE (fewer than %d valid seeds)' % N_MIN); return '\n'.join(L)
    gap = [R[('floor', s)]['test'] - R[('full', s)]['test'] for s in valid]
    mg, sd = float(np.mean(gap)), float(np.std(gap, ddof=1))
    lo = mg - stats.t.ppf(0.95, len(gap) - 1) * sd / math.sqrt(len(gap))
    gate = mg >= GATE_NATS and lo > 0
    P('floor gap %.4f nats (90%% lower %.4f) -> gate %s' % (mg, lo, 'PASS' if gate else 'FAIL'))
    phi = {}
    for c in ('trunc1', 'turn1', 'shrink1'):
        ph = [(R[(c, s)]['test'] - R[('full', s)]['test']) / (R[('floor', s)]['test'] - R[('full', s)]['test']) for s in valid]
        phi[c] = float(np.mean(ph))
        P('  %-8s cost %.4f +- %.4f (per seed %s; nonfinite %s; Abar %.3f)' % (
            c, phi[c], float(np.std(ph, ddof=1) / math.sqrt(len(ph))), ' '.join('%.3f' % x for x in ph),
            [R[(c, s)]['nonfinite'] for s in valid], float(np.mean([R[(c, s)]['Abar'] for s in valid]))))
    t, u, s = phi['trunc1'], phi['turn1'], phi['shrink1']; mk = m_k(t)
    lab = 'TURN' if u >= t - mk and s <= mk else 'SHRINK' if s >= 0.5 * t else 'PARTIAL'
    P('k=1: trunc %.4f turn %.4f shrink %.4f | margin %.4f | turn share %.3f, shrink share %.3f -> %s' % (
        t, u, s, mk, u / t, s / t, lab))
    P('VERDICT: %s' % (lab if gate else 'INVALID (floor gap gate failed)'))
    return '\n'.join(L)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--analyze', nargs='+', required=True)
    A = ap.parse_args()
    txt = analyze(A.analyze); print(txt)
    open(os.path.join(os.environ.get('OUT_DIR', '.'), 'v240_analysis.txt'), 'w').write(txt + '\n')
