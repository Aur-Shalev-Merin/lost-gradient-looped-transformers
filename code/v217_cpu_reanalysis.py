"""v217 (exploratory, CPU only, no new measurement): two reanalyses of saved data suggested by outside reviews 7-8.

Part 1. A fixed near-output window for rho. The paper fits rho over all R loops, so it drifts up with R (the far loops
fade more slowly) and rho^k overstates the lost gradient at long R. Fitting only the W loops next to the output should
not drift by construction. Per sequence: rho_W = exp(slope of log|lambda_j| over j < W), j counted back from the
output. Rule = median over sequences of rho_W,i^k (as Table 1), compared with the median measured lost gradient.
Data: S10/S14 profiles (v196 at R = 32, 48, 64; v204 at R = 96, 128), Parcae-140m and -370m, the same 16 sequences;
Huginn at R = 48. EXPLORATORY: the long-loop data were seen before this analysis was thought of.

Part 2. Paired intervals for the fine-tuning falls (review 8, minor 9). The paper's falls are differences of medians;
here a paired bootstrap over sequences (the same sequences at both stages) gives 90% intervals for each fall and for the
S15/S16 ratios.

  python v217_cpu_reanalysis.py            (from the repo root; reads kaggle-output-v196, kaggle-output-v204,
                                            results_v203, results_v210, results_v211)
"""
import glob, json, os
import numpy as np

KS = (1, 2, 4, 8)
WS = (4, 8, 16)
RNG = np.random.default_rng(217)
B = 4000


def load_profiles(tag, R):
    rows = []
    for d in ('kaggle-output-v196', 'kaggle-output-v204'):
        for f in glob.glob(os.path.join(d, '**', 'v196_%s_R%d.json' % (tag, R)), recursive=True):
            rows += json.load(open(f))['rows']
    return {r['i']: r for r in rows}


def rho_window(adj, W):
    p = np.asarray(adj[::-1], float)            # j = 0 is the last loop
    W = min(W, len(p))
    return float(np.exp(np.polyfit(np.arange(W), np.log(p[:W]), 1)[0]))


def part1():
    print('PART 1: rho fitted on a fixed window of W loops next to the output (exploratory)')
    print('  rule = median_i rho_i^k; "lost" = median measured lost gradient; cells: lost / rule(all R) / rule(W=4) / rule(W=8) / rule(W=16)')
    summary = {}
    for tag, Rs in (('140m', (32, 48, 64, 96, 128)), ('370m', (32, 48, 64, 96, 128)), ('huginn_part0', (48,))):
        for R in Rs:
            rows = load_profiles(tag, R)
            if tag == 'huginn_part0':
                rows.update(load_profiles('huginn_part1', R))
            if not rows:
                continue
            seqs = sorted(rows)
            lost = {k: np.median([rows[i]['err'][str(k)] for i in seqs]) for k in KS}
            rho = {'all': [rho_window(rows[i]['adj'], R) for i in seqs]}
            for W in WS:
                rho[W] = [rho_window(rows[i]['adj'], W) for i in seqs]
            rule = {w: {k: float(np.median(np.power(rho[w], k))) for k in KS} for w in rho}
            name = 'Huginn' if tag.startswith('huginn') else 'Parcae-' + tag
            line = '  %-12s R=%3d  median rho: all %.3f, W4 %.3f, W8 %.3f, W16 %.3f | ' % (
                name, R, np.median(rho['all']), np.median(rho[4]), np.median(rho[8]), np.median(rho[16]))
            line += '  '.join('k=%d %.2f/%.2f/%.2f/%.2f/%.2f' % (k, lost[k], rule['all'][k], rule[4][k], rule[8][k], rule[16][k]) for k in KS)
            print(line)
            for w in rho:
                rel = [abs(rule[w][k] - lost[k]) / lost[k] for k in KS if lost[k] > 0.05]
                summary.setdefault((name, w), []).append((R, float(np.median(rho[w])), max(rel)))
    print('\n  per model and window: median rho across R (drift) and the worst relative error of the rule over k with lost > 0.05')
    for (name, w), v in summary.items():
        print('  %-12s %-4s rho by R: %s | worst rel. error by R: %s' % (
            name, w, ' '.join('%d:%.3f' % (R, r) for R, r, _ in v), ' '.join('%d:%.0f%%' % (R, 100 * e) for R, _, e in v)))


def paired(a0, a1):
    """Fall = mean over k of median(a0) - median(a1); paired bootstrap over sequences. a0, a1: (n, nk)."""
    n = len(a0)
    f = lambda ix: float(np.mean(np.median(a0[ix], 0) - np.median(a1[ix], 0)))
    bs = [f(RNG.integers(0, n, n)) for _ in range(B)]
    return f(np.arange(n)), np.percentile(bs, [5, 95])


def rows_by_seq(runs, run, key, ks=(1, 2, 4)):
    rr = sorted(runs[run]['rows'], key=lambda r: r['s'])
    return np.array([[r[key][str(k)] if str(k) in r[key] else r[key][k] for k in ks] for r in rr])


def part2():
    print('\nPART 2: paired 90% bootstrap intervals for the fine-tuning falls (mean over k = 1, 2, 4; sequences resampled)')
    A, Z = {}, {}
    for f in glob.glob('results_v210/*/gpu*/v210_A_*.json'):
        r = json.load(open(f)); A.setdefault(r['model'], {}).update(r['runs'])
    for f in glob.glob('results_v211/*/gpu*/v211_*.json'):
        r = json.load(open(f)); Z.setdefault(r['model'], {}).update(r['runs'])
    for m in ('parcae', 'pln5', 'r5'):
        s0a, fa = rows_by_seq(A[m], 's0', 'all'), rows_by_seq(A[m], 'full300', 'all')
        s0l, fl = rows_by_seq(A[m], 's0', 'loop'), rows_by_seq(A[m], 'full300', 'loop')
        ta, tl = rows_by_seq(A[m], 'trunc300', 'all'), rows_by_seq(A[m], 'trunc300', 'loop')
        zl = rows_by_seq(Z[m], 'zero300', 'loop')
        out = []
        for lab, a, b in (('full, all', s0a, fa), ('full, loop', s0l, fl), ('last 2 loops, all', s0a, ta),
                          ('last 2 loops, loop', s0l, tl), ('no loop credit, loop', s0l, zl)):
            v, (lo, hi) = paired(a, b)
            out.append('%s %.3f [%.3f, %.3f]' % (lab, v, lo, hi))
        print('  %-6s held-out (S15/S16, 64 test problems): ' % m + ' | '.join(out))
    def b1(path):                                 # per-sequence (B = 1) lost gradient of v201/v203 runs, k = 1, 2, 4
        rr = sorted(json.load(open(path))['b1'], key=lambda r: r['s'])
        return np.array([[r['raw'][str(k)] if str(k) in r['raw'] else r['raw'][k] for k in (1, 2, 4)] for r in rr])
    for study, d, s300 in (('S12 (v201, training sequences)', 'results_v201', {}),
                           ('S13 (v203, held-out test problems)', 'results_v203',
                            {'parcae': 'results_v203/parcae-rerun/gpu0/v203_parcae_s300_seed2012.json'})):
        out = []
        for m in ('parcae', 'pln5', 'r5', 'retro'):
            p0 = glob.glob('%s/**/v2*_%s_s0.json' % (d, m), recursive=True)
            p3 = [s300[m]] if m in s300 else glob.glob('%s/**/v2*_%s_s300.json' % (d, m), recursive=True)
            if not p0 or not p3:
                continue
            a0, a3 = b1(p0[0]), b1(p3[0])
            n = min(len(a0), len(a3))
            v, (lo, hi) = paired(a0[:n], a3[:n])
            out.append('%s %.3f [%.3f, %.3f]' % (m, v, lo, hi))
        print('  %-36s ' % study + ' | '.join(out))


if __name__ == '__main__':
    part1()
    part2()
