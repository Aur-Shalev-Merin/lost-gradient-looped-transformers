"""v196 analysis (prereg docs/prereg_v196_band.md; frozen with it, before any v196 output exists).
Statistic as in the paper: medians over the 16 sequences; the rule value is the median of rho_i^k.
Index j = loops before the output (j = 0: last loop).

VALIDITY (Parcae): at R = 32, median err_k reproduces v190 within 0.002 at k = 1, 2, 4, 8 and median rho within 0.001;
everywhere, the per-token identity BC x W = cos holds within 1e-6.

ACCOUNT (per Parcae model; R = 32, 48, 64 from this run):
  P1  |median err_k(R) - median err_k(32)| <= 0.03 at k = 1, 2, 4, 8, for R = 48 and 64   (the lost gradient settles)
  P2  median rho(32) < median rho(48) < median rho(64)                                      (the fit follows the far loops)
  P3  q_k(R) = median rho_i^k / median err_k rises: q(32) < q(48) < q(64) at k = 4 and 8, and q_4(64) >= 1.10
  model PASS if P1, P2 and P3 hold. CONFIRMED if both models pass; PARTIAL if one; REFUTED if P3 fails on both;
  otherwise MIXED. Huginn at R = 48 is reported against its measured R = 32 (v195) and R = 64 (v187) values on the
  same sequences (P1'-P3', supportive, no verdict weight).

HEAD-TO-HEAD (new cells: Parcae-140m R48, R64; Parcae-370m R48, R64; Huginn R48; R32 cells reported as reference).
  target: the loop's lost share err_loop_k (Parcae |g_R - g_k| / |g_R - g_0|; Huginn's loop-only subset err_k)
  rule   rho_i^k, rho_i = straight-line fit of log|lam_j| over all R loops (the paper's rule)
  share  |T_k| / |T_0|, T_k = sum_{j>=k} lam_j (state-space tail sums, from the same backward pass)
  law    r_i^k, r_i = exp(straight-line slope of log(|T_k| / |T_0|) over k = 0 .. 8)  (one-number tail rate)
  score = mean over k = 1, 2, 4, 8 of |median predictor - median target|
  H1 share score <= 0.04 in all 5 new cells; H2 law score < rule score in >= 4 of 5; H3 law score <= 0.04 in all 5.
  TAIL LAW BETTER if H2 and H3; TAIL SHARE ACCURATE if H1.

PER-TOKEN (descriptive, reading rule fixed): near-output pairs j = 0, 1, 2; s = log BC / log(BC x W) averaged over
the pairs, median over sequences: TRANSPORT (turning mostly redistribution across positions) if s >= 2/3,
WITHIN-POSITION if s <= 1/3, MIXED otherwise. Per model: Parcae at R = 32 and Huginn at R = 48.
  python v196_analyze.py
"""
import glob, json, os, re, sys
import numpy as np

KS = (1, 2, 4, 8)
V190 = {'140m': ({1: .8201, 2: .6753, 4: .4427, 8: .2046}, 0.8250), '370m': ({1: .8273, 2: .6739, 4: .4513, 8: .2382}, 0.8681)}
D = sys.argv[1] if len(sys.argv) > 1 else 'kaggle-output-v196'


def load(tag):
    """Bug fix after the run (disclosed): the pattern was 'v196_%s_R*.json', which missed the kernel's Huginn files
    v196_huginn_part0_R48.json / _part1_; parts are now merged per R. No decision rule changed."""
    out = {}
    for f in sorted(glob.glob(os.path.join(D, '**', 'v196_%s*_R*.json' % tag), recursive=True)):
        h = json.load(open(f)); out.setdefault(h['R'], []).extend(h['rows'])
    return out


def ols_rate(y):
    return float(np.exp(np.polyfit(np.arange(len(y)), y, 1)[0]))


def seq_stats(r):
    R = r['R']; S = np.array(r['state_gram']); lj = [R - 1 - j for j in range(R)]; Sl = S[np.ix_(lj, lj)]
    adj_j = np.sqrt(np.clip(np.diag(Sl), 1e-300, None))
    T = np.sqrt(np.clip(np.array([Sl[k:, k:].sum() for k in range(R)]), 1e-300, None))
    rho = ols_rate(np.log(adj_j)); rt = ols_rate(np.log(T[:9] / T[0]))
    tok = r['tok']; BC = np.array(tok['BC'])[::-1]; W = np.array(tok['W'])[::-1]            # j-order pairs (j, j+1)
    s = np.mean([np.log(BC[j]) / np.log(BC[j] * W[j]) for j in range(3)])
    return dict(rho=rho, rt=rt, share={k: T[k] / T[0] for k in KS if k < R},
                err={int(k): v for k, v in r['err'].items()}, errl={int(k): v for k, v in r['err_loop'].items()},
                s_tok=s, bcw=tok['check_bcw'], com=np.array(tok['com'])[::-1], erank=np.array(tok['erank'])[::-1],
                BC=BC, W=W)


def med(xs):
    return float(np.median(xs))


def summary(rows):
    st = [seq_stats(r) for r in rows]
    R = rows[0]['R']
    s = dict(R=R, n=len(st), rho=med([x['rho'] for x in st]), st=st,
             err={k: med([x['err'][k] for x in st]) for k in KS if k < R},
             errl={k: med([x['errl'][k] for x in st]) for k in KS if k < R},
             rule={k: med([x['rho'] ** k for x in st]) for k in KS if k < R},
             share={k: med([x['share'][k] for x in st]) for k in KS if k < R},
             law={k: med([x['rt'] ** k for x in st]) for k in KS if k < R},
             s_tok=med([x['s_tok'] for x in st]), bcw=max(x['bcw'] for x in st))
    s['q'] = {k: s['rule'][k] / s['err'][k] for k in s['err']}
    s['score'] = {p: float(np.mean([abs(s[p][k] - s['errl'][k]) for k in KS if k < R])) for p in ('rule', 'share', 'law')}
    return s


def tok_reading(sm):
    return 'TRANSPORT' if sm['s_tok'] >= 2 / 3 else 'WITHIN-POSITION' if sm['s_tok'] <= 1 / 3 else 'MIXED'


def main():
    res = {}; new_cells = []
    for tag, name in (('140m', 'Parcae-140m'), ('370m', 'Parcae-370m')):
        runs = load(tag)
        if not runs:
            print('%s: no data' % name); continue
        S = {R: summary(rows) for R, rows in sorted(runs.items())}
        Rs = sorted(S); R0 = Rs[0]
        print('\n=== %s' % name)
        for R in Rs:
            s = S[R]
            print('R%-3d rho %.4f | err %s | rule/err q %s | scores rule %.3f share %.3f law %.3f | per-token s %.2f (%s) | BCxW %.0e' % (
                R, s['rho'], ' '.join('%.4f' % s['err'][k] for k in KS), ' '.join('%.2f' % s['q'][k] for k in KS),
                s['score']['rule'], s['score']['share'], s['score']['law'], s['s_tok'], tok_reading(s), s['bcw']))
        valid = all(S[R]['bcw'] <= 1e-6 for R in Rs)
        if R0 == 32 and tag in V190:
            ref, rref = V190[tag]
            dv = max(abs(S[32]['err'][k] - ref[k]) for k in KS); dr = abs(S[32]['rho'] - rref)
            valid = valid and dv <= 0.002 and dr <= 0.001
            print('validity: R32 err vs v190 max |diff| %.4f, rho %.4f -> %s' % (dv, dr, 'OK' if valid else 'FAILED'))
        if len(Rs) >= 3:
            a, b, c = Rs[:3]
            p1 = all(abs(S[R]['err'][k] - S[a]['err'][k]) <= 0.03 for R in (b, c) for k in KS)
            p2 = S[a]['rho'] < S[b]['rho'] < S[c]['rho']
            p3 = all(S[a]['q'][k] < S[b]['q'][k] < S[c]['q'][k] for k in (4, 8)) and S[c]['q'][4] >= 1.10
            res[name] = dict(valid=valid, P1=p1, P2=p2, P3=p3, PASS=p1 and p2 and p3)
            print('account: P1 %s P2 %s P3 %s -> %s' % (p1, p2, p3, 'PASS' if p1 and p2 and p3 else 'FAIL'))
            new_cells += [(name, R, S[R]) for R in (b, c)]
        res.setdefault(name, {})['per_token'] = tok_reading(S[R0])
    # Huginn at R = 48 vs its measured R = 32 (v195 rows, same sequences) and R = 64 (v187, same sequence indices)
    hug = load('huginn')
    if hug:
        R = sorted(hug)[0]; sh = summary(hug[R]); ids = [r['i'] for r in hug[R]]; hk = [k for k in KS if k in sh['err']]
        print('\n=== Huginn-3.5B R%d: rho %.4f | err %s | q %s | scores rule %.3f share %.3f law %.3f | per-token s %.2f (%s)' % (
            R, sh['rho'], ' '.join('%.4f' % sh['err'][k] for k in hk), ' '.join('%.2f' % sh['q'][k] for k in hk),
            sh['score']['rule'], sh['score']['share'], sh['score']['law'], sh['s_tok'], tok_reading(sh)))
        new_cells.append(('Huginn-3.5B', R, sh)); res['Huginn-3.5B'] = dict(per_token=tok_reading(sh))
        ref = {}
        for f, RR in (('kaggle-output-v187/R64/loop_grad_probe_huginn.json', 64),):
            if os.path.exists(f):
                ss = [s for s in json.load(open(f))['seqs'] if s['i'] in ids]
                rho = [ols_rate(np.log(np.array(s['adjoint'], float)[::-1])) for s in ss]
                err = {k: med([s['trunc'][str(k)]['rel_err'] for s in ss]) for k in KS}
                ref[RR] = dict(rho=med(rho), q={k: med(np.array(rho) ** k) / err[k] for k in KS}, err=err)
        v195 = [r for f in glob.glob('kaggle-output-v195/**/v195_huginn*_R32.json', recursive=True) for r in json.load(open(f))['rows']]
        if v195:
            sys.path.insert(0, '.')
            import v195_analyze as AN
            rho, err = [], {k: [] for k in KS}
            for r in v195:
                u = AN.unpack(r, 32, 32); Cc = u['Cc']; rho.append(ols_rate(np.log(np.sqrt(np.diag(u['Sl'])))))
                for k in KS:
                    err[k].append(np.sqrt(Cc[k:, k:].sum() / Cc.sum()))
            e = {k: med(err[k]) for k in KS}
            ref[32] = dict(rho=med(rho), q={k: med(np.array(rho) ** k) / e[k] for k in KS}, err=e)
        if 32 in ref and 64 in ref:
            p1 = all(abs(sh['err'][k] - ref[32]['err'][k]) <= 0.03 for k in hk)
            p2 = ref[32]['rho'] < sh['rho'] < ref[64]['rho']
            p3 = all(ref[32]['q'][k] < sh['q'][k] < ref[64]['q'][k] for k in (4, 8) if k in hk)
            print("Huginn supportive: P1' %s P2' %s (rho 32/new/64 %.4f %.4f %.4f) P3' %s (q 32/new/64: %s)" % (
                p1, p2, ref[32]['rho'], sh['rho'], ref[64]['rho'], p3,
                ', '.join('k%d %.2f %.2f %.2f' % (k, ref[32]['q'][k], sh['q'][k], ref[64]['q'][k]) for k in (4, 8) if k in hk)))
            res['Huginn-3.5B'].update(P1=p1, P2=p2, P3=p3)
    # verdicts
    pm = [res[n]['PASS'] for n in ('Parcae-140m', 'Parcae-370m') if n in res and 'PASS' in res[n]]
    p3 = [res[n]['P3'] for n in ('Parcae-140m', 'Parcae-370m') if n in res and 'P3' in res[n]]
    if len(pm) == 2:
        acc = 'CONFIRMED' if all(pm) else 'PARTIAL' if any(pm) else ('REFUTED' if not any(p3) else 'MIXED')
        print('\nACCOUNT: %s' % acc)
    if new_cells:
        sc = [(n, R, s['score']) for n, R, s in new_cells]
        h1 = all(x['share'] <= 0.04 for _, _, x in sc); h2 = sum(x['law'] < x['rule'] for _, _, x in sc) >= min(4, len(sc))
        h3 = all(x['law'] <= 0.04 for _, _, x in sc)
        print('HEAD-TO-HEAD over %d new cells: ' % len(sc) + ' | '.join('%s R%d rule %.3f share %.3f law %.3f' % (n, R, x['rule'], x['share'], x['law']) for n, R, x in sc))
        print('H1 %s H2 %s H3 %s -> %s; %s' % (h1, h2, h3, 'TAIL LAW BETTER' if h2 and h3 else 'tail law not better',
                                               'TAIL SHARE ACCURATE' if h1 else 'tail share not accurate'))
    print('per-token readings: %s' % {n: res[n].get('per_token') for n in res})


if __name__ == '__main__':
    main()
