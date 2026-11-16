"""v243 (plan in docs/prereg_v243_sgd_k1_cleanup.md, frozen before any run): clean-up of v241 (S24) at k = 1 under SGD,
as v240 was for v239. The runs use v241_sgd_split.py unchanged (its --runs entry point, cells full / floor / trunc1 /
turn1 / shrink1 at v241's frozen rates, SGD momentum 0.9, clip 5); this file holds only the seed list and the analysis.

Primary (fixed before any run): D = cost(trunc1) - cost(turn1), paired by seed. Same truncated direction, two lengths:
does giving it back the full gradient's length recover part of truncation's cost under SGD?
  SIZE-COSTS  mean D >= 0.015 and one-sided 90% t lower bound > 0
  SIZE-FREE   one-sided 90% t upper bound < 0.015
  UNCLEAR     otherwise
Secondary (reported with labels, does not decide the verdict): v239's k = 1 label rule; the shrink-only cost by the
same three-way rule.

  python v243_sgd_k1_cleanup.py --analyze DIR [DIR ...]
  python v243_sgd_k1_cleanup.py --synthetic        # known-answer worlds for the analysis (no model)
"""
import argparse, glob, json, math, os

SEEDS = tuple(range(2431101, 2431113))      # 12 seeds, all run; the first 10 valid in this order are scored
CELLS = ('full', 'floor', 'trunc1', 'turn1', 'shrink1')
N_USE, N_MIN = 10, 8
MAX_NONFINITE = 10                          # S23's rule, applied to every run on a seed
GATE_NATS = 0.03
FLOOR_D = 0.015                             # the smallest cost the paper's margins treat as non-zero (v239's m floor)
V241_SHA = 'e505057ac886'


def m_k(phi_trunc):
    return max(0.015, 0.2 * phi_trunc)      # v239's margin, unchanged


def bad(r):
    return r['nonfinite'] > MAX_NONFINITE or not math.isfinite(r['test'])


def three_way(x, P, name):
    """SIZE-COSTS / SIZE-FREE / UNCLEAR on a list of per-seed values, by the frozen rule."""
    import numpy as np
    from scipy import stats
    n = len(x); m = float(np.mean(x)); se = float(np.std(x, ddof=1) / math.sqrt(n))
    t = stats.t.ppf(0.90, n - 1); lo, hi = m - t * se, m + t * se
    lab = 'COSTS' if m >= FLOOR_D and lo > 0 else 'FREE' if hi < FLOOR_D else 'UNCLEAR'
    P('%s: mean %.4f +- %.4f (n %d; one-sided 90%% bounds %.4f / %.4f; per seed %s) -> %s' % (
        name, m, se, n, lo, hi, ' '.join('%.3f' % v for v in x), lab))
    return lab, m, lo, hi


def analyze_rows(rows):
    import numpy as np
    from scipy import stats
    R = {(r['arm'], r['seed']): r for r in rows if r['seed'] in SEEDS}
    L = []; P = L.append
    shas = sorted({r.get('v241_sha', '?')[:12] for r in R.values()})
    ran = [s for s in SEEDS if any((c, s) in R for c in CELLS)]
    valid = [s for s in ran if all((c, s) in R and not bad(R[(c, s)]) for c in CELLS)][:N_USE]
    why = {s: [c for c in CELLS if (c, s) not in R or bad(R[(c, s)])] for s in ran if s not in valid}
    P('v243 analysis: %d runs (combined v241 driver sha %s); scored seeds %s; invalid or unused %s' % (
        len(R), shas, valid, why))
    if len(valid) < N_MIN:
        P('VERDICT: INCOMPLETE (fewer than %d valid seeds)' % N_MIN); return '\n'.join(L)
    gap = [R[('floor', s)]['test'] - R[('full', s)]['test'] for s in valid]
    mg, sd = float(np.mean(gap)), float(np.std(gap, ddof=1))
    glo = mg - stats.t.ppf(0.95, len(gap) - 1) * sd / math.sqrt(len(gap))
    gate = mg >= GATE_NATS and glo > 0
    P('floor gap %.4f nats (90%% lower %.4f) -> gate %s' % (mg, glo, 'PASS' if gate else 'FAIL'))
    ph = {c: [(R[(c, s)]['test'] - R[('full', s)]['test']) / (R[('floor', s)]['test'] - R[('full', s)]['test'])
              for s in valid] for c in ('trunc1', 'turn1', 'shrink1')}
    phi = {c: float(np.mean(v)) for c, v in ph.items()}
    for c, v in ph.items():
        avg = lambda key: float(np.mean([R[(c, s)].get(key, float('nan')) for s in valid]))
        P('  %-8s cost %.4f +- %.4f (per seed %s; nonfinite %s; Mbar %.3f; step %.3f; clipped %.2f)' % (
            c, phi[c], float(np.std(v, ddof=1) / math.sqrt(len(v))), ' '.join('%.3f' % x for x in v),
            [R[(c, s)]['nonfinite'] for s in valid], avg('Mbar'), avg('step'), avg('frac_clip')))
    D = [a - b for a, b in zip(ph['trunc1'], ph['turn1'])]
    lab, dm, dlo, dhi = three_way(D, P, 'PRIMARY D = trunc1 - turn1')
    P('  share of truncation\'s cost recovered by the full length: %.2f' % (dm / phi['trunc1'] if phi['trunc1'] else float('nan')))
    P('-- secondary (reported, labelled, not deciding) --')
    t, u, s = phi['trunc1'], phi['turn1'], phi['shrink1']; mk = m_k(t)
    l239 = 'TURN' if u >= t - mk and s <= mk else 'SHRINK' if s >= 0.5 * t else 'PARTIAL'
    P('v239 rule at k=1: trunc %.4f turn %.4f shrink %.4f | margin %.4f | turn share %.2f, shrink share %.2f -> %s' % (
        t, u, s, mk, u / t if t else float('nan'), s / t if t else float('nan'), l239))
    three_way(ph['shrink1'], P, 'shrink1 cost')
    P('-- reported --')
    for c in CELLS:
        rs = [R[(c, s)] for s in ran if (c, s) in R]
        P('  unstable %-8s %d of %d runs; mean nonfinite %.2f' % (c, sum(bad(r) for r in rs), len(rs),
                                                                  float(np.mean([r['nonfinite'] for r in rs]))))
    P('VERDICT: %s' % ('SIZE-' + lab if gate else 'INVALID (floor gap gate failed)'))
    return '\n'.join(L)


def analyze(dirs):
    rows = []
    for d in dirs:
        for f in glob.glob(os.path.join(d, '**', 'v241_train_*.jsonl'), recursive=True):
            rows += [json.loads(x) for x in open(f) if x.strip()]
    return analyze_rows(rows)


def synthetic():
    """Known-answer worlds: the verdict each should give is in its name."""
    import numpy as np
    out = []
    def world(name, d, sh, sd=0.04, gap=0.227, broken=(), nseeds=12, seed=0):
        # the turn-only and shrink-only noise alternates +-sd, so every world's mean D is exactly d over any even count
        # of seeds and its label is fixed by the rule, not by a random draw
        g = np.random.default_rng(seed); rows = []
        for i, s in enumerate(SEEDS[:nseeds]):
            full = 1.42 + g.normal(0, 0.003); floor = full + gap + g.normal(0, 0.003)
            sign = 1 if i % 2 == 0 else -1
            tr = 0.25 + g.normal(0, 0.03); tu = tr - d + sign * sd; shv = sh - sign * 0.08
            for c, phi in (('full', 0), ('floor', 1), ('trunc1', tr), ('turn1', tu), ('shrink1', shv)):
                nf = 600 if (c, s) in broken else 1
                test = float('nan') if (c, s) in broken else full + phi * (floor - full)
                rows.append(dict(arm=c, seed=s, test=test, nonfinite=nf, Mbar=1.0, step=1.0, frac_clip=0.02,
                                 v241_sha=V241_SHA))
        v = analyze_rows(rows).splitlines()[-1]
        out.append('%-40s %s' % (name, v))
    world('SIZE-COSTS (d 0.057)', 0.057, 0.04)
    world('SIZE-FREE (d 0, sd 0.01)', 0.0, 0.0, sd=0.01)
    world('UNCLEAR (d 0.01, sd 0.04)', 0.01, 0.0)
    world('UNCLEAR (d 0.03, sd 0.10)', 0.03, 0.0, sd=0.10)
    world('SIZE-COSTS, 2 seeds broken -> 10 used', 0.057, 0.04, broken={('full', SEEDS[0]), ('shrink1', SEEDS[3])})
    world('INCOMPLETE (5 broken)', 0.057, 0.04, broken={('turn1', s) for s in SEEDS[:5]})
    world('INVALID (gap 0.01)', 0.057, 0.04, gap=0.01)
    return '\n'.join(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--analyze', nargs='+'); ap.add_argument('--synthetic', action='store_true')
    A = ap.parse_args()
    txt = synthetic() if A.synthetic else analyze(A.analyze); print(txt)
    if A.analyze:
        open(os.path.join(os.environ.get('OUT_DIR', '.'), 'v243_analysis.txt'), 'w').write(txt + '\n')
