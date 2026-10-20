"""v195 analysis (prereg docs/prereg_v195_rho_modes.md; frozen with it, before any v195 output exists).
Index j = loops before the output (j = 0: last loop). Per sequence, from v195_rho_modes.py's rows:
  actual (A)        lam_j, c_j = P_j^T lam_j            constant-map model (C)  kap_j = (J^T)^j lam_out, chat_j = P^T kap_j
  err_k = |sum_{j>=k} contributions| / |sum_j contributions| over the loop part (j < R); rho = exp(slope) of a straight-line
  fit of log signal size over j = 0 .. R-1; A0 = aligned form on signal sizes, A1 = aligned form on contribution sizes;
  angle factor = err / A1, fit factor = rho^k / A0. Parameter-space inner products use the sketch's angles with exact norms.
Decisions (per model; Parcae-140m, Parcae-370m, Huginn at R = 32):
  A  where the turning comes from. T_c = mean of 1 - cos(c_j, c_{j+1}) over j = 0, 1, 2 (near band; far band j = 8 .. 14),
     T_p the same for p_j (one fixed vector through each loop's parameter map). r = median T_p / median T_c:
     SIGNAL if r < 0.2, MAP if r >= 0.5, MIXED otherwise.
  B  does one loop's maps capture the dynamics? PASS if |median err_C - median err_A| <= 0.02 at k = 1, 2, 4, 8 and
     |median rho_C - median rho_A| <= 0.02; FAIL otherwise.
  C  (models with B = PASS) mode anatomy. Modal model from the Arnoldi Hessenberg H (m = R): kap_i = |lam_out| Q Y T^i
     Y^-1 e_1, chat_i = (P^T Q) Y T^i Y^-1 e_1 (T = Ritz values). Guards: the full modal model must reproduce |kap_i|
     within 2% and err_C within 0.01 (medians), the median non-normality ratio sum|b_m| / |lam_out| must be <= 10,
     and turning modes must carry >= 5% of sum|b_m|; else UNINFORMATIVE / NO-TURNING-MODES. Knock out the turning
     modes (|arg T| > pi/4); per sequence with both full excesses (factor - 1) >= 0.05 at k = 4 (>= 8 sequences needed),
     d = reduced excess / full excess for each factor; medians:
       SHARED      both d <= 1/3: the turning parts carry both effects, so the match is structural for this model
       NOT-SHARED  otherwise (labelled SEPARATE when one d <= 1/3 and the other >= 2/3); no structural claim
     Descriptive: rho^4 / err_4 when the turning modes' amplitudes are scaled by s = 0, 0.5, 1, 1.5, 2.
  D  (Parcae-140m, if B = PASS) out-of-sample: blended loop x <- (1 - a) x + a F(x) maps J^T -> (1 - a) I + a J^T and
     P^T -> a P^T, so chat^a_i = a sum_l C(i, l) (1 - a)^(i - l) a^l chat_l; with g_0 the all-parameter lost gradient is
     |sum_{i>=k} chat^a_i| / |g_0 + sum_{i<R} chat^a_i|. Predicted medians vs the measured ones (same 16 sequences,
     Table 1 protocol): R 32 a 0.5 (v163), R 32 a 0.7 and R 64 a 0.3 (v164), R 48 a 0.4 (v166 pre).
     PASS if every point is within 0.03.
Validity (a model's results are not used if it fails): read-out loss = model loss (Parcae, 1e-4 relative); signal sizes reproduce
v191 / v192 (median over loops of the relative difference, max over sequences, 1e-3 Parcae / 2e-2 Huginn); err_A
reproduces v191 err_loop / v192 err at k = 1, 2, 4, 8 (0.005 / 0.01); sketch norms within 1%; Krylov step 1 equals the
backward pass (cos >= 0.9999 / 0.995); Arnoldi orthogonality <= 1e-8.
  python v195_analyze.py                (reads kaggle-output-v195, kaggle-output-v191, kaggle-output-v192)
  python v195_analyze.py --synthetic    (checks the decision code on linear toys with known answers)
"""
import glob, json, math, os, re, sys
from math import comb
import numpy as np

KS = (1, 2, 4, 8)
CELLS = [('Parcae-140m', '140m'), ('Parcae-370m', '370m'), ('Huginn-3.5B', 'huginn')]
BLEND_FILES = [('kaggle-output-v163/pilot-b/v163_measure_pilotB0.json', 32, 0.5),
               ('kaggle-output-v164/measure/v164_measure_measure.json', 32, 0.7),
               ('kaggle-output-v164/measure/v164_measure_measure.json', 64, 0.3),
               ('results_v166_pre/a/v164_measure_pre0.json', 64, 0.3),
               ('results_v166_pre/a/v164_measure_pre0.json', 48, 0.4)]


def slope_rho(sizes):
    return float(np.exp(np.polyfit(np.arange(len(sizes)), np.log(np.asarray(sizes, float)), 1)[0]))


def aligned(sizes, k):
    s = np.asarray(sizes, float); return s[k:].sum() / s.sum()


def lost(Gm, k, R=None):
    R = len(Gm) if R is None else R
    tot = Gm[:R, :R].sum()
    return math.sqrt(max(Gm[k:R, k:R].sum(), 0) / tot) if tot > 0 else float('nan')


def corrected(r):
    """Parameter Gram: sketch cosines times exact norms (zero vectors stay zero)."""
    G = np.array(r['param_gram']); n = np.array(r['param_norm']); dg = np.sqrt(np.clip(np.diag(G), 1e-300, None))
    C = G / np.outer(dg, dg); C[n == 0, :] = 0; C[:, n == 0] = 0
    return C * np.outer(n, n)


def unpack(r, R, K):
    L = r['layout']; G = corrected(r)
    cj = [L['c'] + R - 1 - j for j in range(R)]; pj = [L['p'] + R - 1 - j for j in range(R)]
    hj = [L['chat'] + i for i in range(K)]
    S = np.array(r['state_gram']); lj = [R - 1 - j for j in range(R)]
    return dict(G=G, Cc=G[np.ix_(cj, cj)], Cp=G[np.ix_(pj, pj)], Ch=G[np.ix_(hj, hj)], g0=G[0, hj], g00=G[0, 0],
                Sl=S[np.ix_(lj, lj)], Sk=S[R:, R:], L=L)


def cosmat(Gm):
    d = np.sqrt(np.clip(np.diag(Gm), 1e-300, None)); return Gm / np.outer(d, d)


def turning(Gm, j0, j1):
    C = cosmat(Gm); v = [1 - C[j, j + 1] for j in range(j0, j1) if j + 1 < len(C)]
    return float(np.mean(v)) if v else float('nan')


def factors(sig_sizes, Gc, k):
    """angle factor err / A1 and fit factor rho^k / A0 for a signal-size profile and a contribution Gram."""
    R = len(sig_sizes); csz = np.sqrt(np.clip(np.diag(Gc)[:R], 0, None))
    e = lost(Gc, k, R); A0, A1 = aligned(sig_sizes, k), aligned(csz, k); rho = slope_rho(sig_sizes)
    return e / A1, rho ** k / A0, e, rho


def modal(r, R, M, Gpq, weight):
    """Constant-map model from the Arnoldi data, each mode's amplitude multiplied by weight(T) (0 drops a mode; the
    weights are equal for conjugate pairs, so the model stays real)."""
    H = np.array(r['H'])[:M, :M]; th, Y = np.linalg.eig(H)
    e1 = np.zeros(M); e1[0] = r['lam_out']; b = np.linalg.solve(Y, e1)
    w = weight(th)
    Z = np.array([Y @ (th ** i * b * w) for i in range(R)])
    kn = np.linalg.norm(Z, axis=1)
    Gch = np.real(Z.conj() @ Gpq @ Z.T)
    return kn, Gch, th, b


def per_sequence(r, R, K, M):
    u = unpack(r, R, K); out = {}
    sl = np.sqrt(np.clip(np.diag(u['Sl']), 0, None)); sk = np.sqrt(np.clip(np.diag(u['Sk'])[:R], 0, None))
    out['adj_t'] = sl[::-1]
    out['err_A'] = {k: lost(u['Cc'], k) for k in KS}; out['err_C'] = {k: lost(u['Ch'], k, R) for k in KS}
    out['rho_A'], out['rho_C'] = slope_rho(sl), slope_rho(sk)
    for tag, Gm in (('c', u['Cc']), ('p', u['Cp']), ('h', u['Ch'][:R, :R]), ('l', u['Sl'])):
        out['T_near_' + tag] = turning(Gm, 0, 3); out['T_far_' + tag] = turning(Gm, 8, 15)
    out['fac_A'] = {k: factors(sl, u['Cc'], k)[:2] for k in KS}; out['fac_C'] = {k: factors(sk, u['Ch'][:R, :R], k)[:2] for k in KS}
    # cos(kap_j, lam_j) and size ratio by loop (descriptive)
    S = np.array(r['state_gram'])
    out['kap_vs_lam_cos'] = [S[R + j, R - 1 - j] / math.sqrt(S[R + j, R + j] * S[R - 1 - j, R - 1 - j]) for j in range(R)]
    out['kap_vs_lam_size'] = [math.sqrt(S[R + j, R + j] / S[R - 1 - j, R - 1 - j]) for j in range(R)]
    # mode anatomy
    L = u['L']; qi = [L['pq'] + l for l in range(M)]; Gpq = u['G'][np.ix_(qi, qi)]
    kn, Gch, th, b = modal(r, R, M, Gpq, lambda t: np.ones(len(t)))
    out['modal_size_err'] = float(np.max(np.abs(kn[:M] / sk[:M] - 1)))
    out['modal_err_diff'] = max(abs(lost(Gch, k) - out['err_C'][k]) for k in KS)
    out['nonnormal'] = float(np.sum(np.abs(b)) / r['lam_out'])
    turn = np.abs(np.angle(th)) > np.pi / 4
    out['turn_share'] = float(np.abs(b[turn]).sum() / np.abs(b).sum())
    out['ritz'] = th; out['amp'] = np.abs(b)
    fa_full, ff_full = factors(kn, Gch, 4)[:2]
    if turn.all() or not turn.any():
        out['knock'] = None
    else:
        kn2, Gch2, _, _ = modal(r, R, M, Gpq, lambda t: (np.abs(np.angle(t)) <= np.pi / 4).astype(float))
        ok = Gch2.sum() > 0 and np.all(kn2 > 0)
        out['knock'] = (fa_full, ff_full) + (factors(kn2, Gch2, 4)[:2] if ok else (float('nan'), float('nan')))
    out['dial'] = {}
    for s_ in (0.0, 0.5, 1.0, 1.5, 2.0):                 # descriptive: scale the turning modes' amplitudes by s
        kn3, Gch3, _, _ = modal(r, R, M, Gpq, lambda t, s_=s_: np.where(np.abs(np.angle(t)) > np.pi / 4, s_, 1.0))
        e4 = lost(Gch3, 4)
        out['dial'][s_] = slope_rho(kn3) ** 4 / e4 if np.all(kn3 > 0) and e4 > 0 else float('nan')
    out['u'] = u
    return out


def blend_predict(u, Rb, a, ks):
    K = len(u['Ch']); B = np.zeros((Rb, K))
    for i in range(Rb):
        for l in range(i + 1):
            B[i, l] = a * comb(i, l) * (1 - a) ** (i - l) * a ** l
    Cb = B @ u['Ch'] @ B.T; gb = B @ u['g0']; tot = u['g00'] + 2 * gb.sum() + Cb.sum()
    return {k: math.sqrt(max(Cb[k:, k:].sum(), 0) / tot) for k in ks}


def measured_blend():
    out = {}
    for f, Rb, a in BLEND_FILES:
        if not os.path.exists(f):
            continue
        res = json.load(open(f))['res']
        for e in (res.values() if isinstance(res, dict) else res):
            s = e.get('summary', e)
            if s['R'] == Rb and abs(s['a'] - a) < 1e-9:
                for k in e['rows'][0]['err']:
                    if int(k) < Rb:
                        out.setdefault((Rb, a), {})[int(k)] = float(np.median([rr['err'][k] for rr in e['rows']]))
    return out


def decide(name, rows, R, K, M, ref=None, blend=None, tol=None):
    tol = tol or dict(adj=1e-3, err=0.005, k1=0.9999)
    P = [per_sequence(r, R, K, M) for r in rows]; md = lambda f: float(np.median([f(p) for p in P]))
    print('\n=== %s (n = %d, R = %d, K = %d, m = %d) ===' % (name, len(P), R, K, M))
    # validity
    v = dict(sketch=max(r['val_norm_maxrel'] for r in rows) <= 0.01,
             krylov=min(r['krylov_step1_cos'] for r in rows) >= tol['k1'],
             orth=max(r['arnoldi_orth'] for r in rows) <= 1e-8)
    if rows[0].get('loss_ref') is not None:
        v['readout'] = max(abs(r['loss_traj'] - r['loss_ref']) / abs(r['loss_ref']) for r in rows) <= 1e-4
    if ref:
        dA = [np.median(np.abs(np.array(p['adj_t']) / np.array(ref[r['i']]['adj']) - 1)) for p, r in zip(P, rows) if r['i'] in ref]
        dE = [max(abs(p['err_A'][k] - ref[r['i']]['err_loop'][k]) for k in KS) for p, r in zip(P, rows) if r['i'] in ref]
        v['repro_sizes'] = len(dA) == len(P) and max(dA) <= tol['adj']; v['repro_err'] = len(dE) == len(P) and max(dE) <= tol['err']
        print('reproduction: signal sizes max %.1e, err_A max |diff| %.4f (n matched %d)' % (max(dA or [np.nan]), max(dE or [np.nan]), len(dA)))
    valid = all(v.values())
    print('validity: %s -> %s' % (v, 'OK' if valid else 'FAILED (results not used)'))
    res = dict(valid=valid)
    # A
    for band in ('near', 'far'):
        Tc, Tp = md(lambda p: p['T_%s_c' % band]), md(lambda p: p['T_%s_p' % band])
        r_ = Tp / Tc if Tc > 0 else float('nan')
        verdict = 'SIGNAL' if r_ < 0.2 else 'MAP' if r_ >= 0.5 else 'MIXED'
        res['A_' + band] = verdict
        print('A %-4s: turning of contributions %.3f | of the fixed vector through each loop\'s map %.3f -> ratio %.2f: %s | '
              '(signal itself, state space %.3f; constant-map contributions %.3f)' % (
                  band, Tc, Tp, r_, verdict, md(lambda p: p['T_%s_l' % band]), md(lambda p: p['T_%s_h' % band])))
    # B
    eA = {k: md(lambda p, k=k: p['err_A'][k]) for k in KS}; eC = {k: md(lambda p, k=k: p['err_C'][k]) for k in KS}
    rA, rC = md(lambda p: p['rho_A']), md(lambda p: p['rho_C'])
    dk = max(abs(eC[k] - eA[k]) for k in KS)
    res['B'] = 'PASS' if dk <= 0.02 and abs(rC - rA) <= 0.02 else 'FAIL'
    print('B: err actual %s | constant-map %s | max |diff| %.3f; rho %.3f vs %.3f -> %s' % (
        ' / '.join('%.3f' % eA[k] for k in KS), ' / '.join('%.3f' % eC[k] for k in KS), dk, rA, rC, res['B']))
    print('   factors at k = 4, actual: angle %.2f fit %.2f (rho^k / lost %.2f) | constant-map: angle %.2f fit %.2f' % (
        md(lambda p: p['fac_A'][4][0]), md(lambda p: p['fac_A'][4][1]), md(lambda p: p['rho_A'] ** 4 / p['err_A'][4]),
        md(lambda p: p['fac_C'][4][0]), md(lambda p: p['fac_C'][4][1])))
    cosj = np.median([p['kap_vs_lam_cos'] for p in P], 0); szj = np.median([p['kap_vs_lam_size'] for p in P], 0)
    print('   constant-map vs actual signal by loop j (cos / size ratio): ' + '  '.join(
        'j%d %.3f/%.2f' % (j, cosj[j], szj[j]) for j in (1, 2, 4, 8, 16, R - 1) if j < R))
    # C
    if res['B'] == 'PASS':
        g1 = md(lambda p: p['modal_size_err']) <= 0.02 and md(lambda p: p['modal_err_diff']) <= 0.01
        g2 = md(lambda p: p['nonnormal']) <= 10; g3 = md(lambda p: p['turn_share']) >= 0.05
        print('C guards: modal reconstruction size %.1e err %.1e (%s); non-normality %.2f (%s); turning-mode share %.2f (%s)' % (
            md(lambda p: p['modal_size_err']), md(lambda p: p['modal_err_diff']), g1, md(lambda p: p['nonnormal']), g2,
            md(lambda p: p['turn_share']), g3))
        if not (g1 and g2):
            res['C'] = 'UNINFORMATIVE'
        elif not g3:
            res['C'] = 'NO-TURNING-MODES'
        else:
            d = [((ka - 1) / (fa - 1), (kf - 1) / (ff - 1)) for fa, ff, ka, kf in [p['knock'] for p in P if p['knock']]
                 if fa - 1 >= 0.05 and ff - 1 >= 0.05 and np.isfinite(ka) and np.isfinite(kf)]
            if len(d) < 8:
                res['C'] = 'UNINFORMATIVE'
            else:
                da, df = np.median([x[0] for x in d]), np.median([x[1] for x in d])
                res['C'] = ('SHARED' if da <= 1 / 3 and df <= 1 / 3 else
                            'NOT-SHARED (SEPARATE)' if (da <= 1 / 3 and df >= 2 / 3) or (df <= 1 / 3 and da >= 2 / 3)
                            else 'NOT-SHARED')
                print('C: knocking out the turning modes keeps %.2f of the angle excess and %.2f of the fit excess (n = %d) -> %s'
                      % (da, df, len(d), res['C']))
        print('   dial (descriptive): rho^4 / err_4 with the turning modes scaled by s = 0 / 0.5 / 1 / 1.5 / 2: %s' % ' / '.join(
            '%.2f' % np.nanmedian([p['dial'][s_] for p in P]) for s_ in (0.0, 0.5, 1.0, 1.5, 2.0)))
        th = np.concatenate([p['ritz'] for p in P]); am = np.concatenate([p['amp'] / p['amp'].sum() for p in P])
        o = np.argsort(-am)
        print('   largest modes over all sequences (|T|, |arg T|, share of its sequence sum|b|): ' + '  '.join(
            '%.2f/%.0fdeg/%.2f' % (abs(th[i]), np.degrees(abs(np.angle(th[i]))), am[i]) for i in o[:10]))
    # D
    if blend and res['B'] == 'PASS':          # bug fix after the run: was 'blend is not None', so D also ran for Huginn
        meas = measured_blend(); dev = []
        print('D: unblended all-parameter prediction (a = 1, R = %d): %s' % (R, ' / '.join(
            '%.3f' % np.median([blend_predict(p['u'], R, 1.0, KS)[k] for p in P]) for k in KS)))
        for (Rb, a), pts in sorted(meas.items()):
            if Rb > K:
                continue
            ks = sorted(pts); pred = {k: float(np.median([blend_predict(p['u'], Rb, a, ks)[k] for p in P])) for k in ks}
            dev += [abs(pred[k] - pts[k]) for k in ks]
            print('   R %d a %.1f: ' % (Rb, a) + '  '.join('k%d %.3f vs %.3f' % (k, pred[k], pts[k]) for k in ks))
        res['D'] = ('PASS' if dev and max(dev) <= 0.03 else 'FAIL') if dev else 'NO-DATA'
        print('D: max |predicted - measured| %.3f over %d points -> %s' % (max(dev) if dev else float('nan'), len(dev), res['D']))
    return res


def refs(tag):
    out = {}
    if tag == 'huginn':
        for f in glob.glob('kaggle-output-v192/**/v192_huginn_part*_R32.json', recursive=True):
            for r in json.load(open(f))['rows']:
                out[r['i']] = dict(adj=r['adj'], err_loop={k: r['err'][str(k)] for k in KS})
    else:
        f = 'kaggle-output-v191/v191_%s_R32.json' % tag
        if os.path.exists(f):
            for r in json.load(open(f))['rows']:
                s = r['gR_norm'] / r['gR_minus_g0_norm']
                out[r['i']] = dict(adj=r['adj'], err_loop={k: r['err'][str(k)] * s for k in KS})
    return out


# ------------------------------------------------------------------------------------------- synthetic check
def synth_rows(kind, n_seq=16, R=32, K=64, M=32, n=60, p=300, seed=0):
    """Linear toys with known answers, written in v195_rho_modes.py's row format (exact Grams, no sketch noise)."""
    rng = np.random.default_rng(seed); rows = []
    for s in range(n_seq):
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        blocks, amps = [], []
        def real(mu, a): blocks.append(np.array([[mu]])); amps.append(np.array([a]))
        def rot(r_, w, a): blocks.append(r_ * np.array([[np.cos(w), -np.sin(w)], [np.sin(w), np.cos(w)]])); amps.append(np.array([a, 0.0]))
        jit = lambda x: x * (1 + 0.1 * rng.standard_normal())
        if kind in ('shared', 'map', 'bfail'):          # fast turning parts make both effects
            real(0.9, 1.0); rot(0.55, 2.0, jit(1.5)); rot(0.6, 2.6, jit(1.2)); rot(0.5, 1.4, jit(1.0))
        if kind == 'separate':                          # turning parts make the angle effect, a small ultra-slow tail the fit
            real(0.84, 1.0); real(0.97, jit(0.35)); rot(0.55, 2.0, jit(1.2)); rot(0.6, 2.6, jit(1.0))
        for _ in range(4):
            real(rng.uniform(0.05, 0.3), jit(0.2))
        dim = sum(len(a) for a in amps); A = np.zeros((dim, dim)); o = 0
        for bl in blocks:
            A[o:o + len(bl), o:o + len(bl)] = bl; o += len(bl)
        Bm = Q[:, :dim]; JT = Bm @ A @ Bm.T                       # backward map on an n-dim state (rank dim)
        lam_out = Bm @ np.concatenate(amps)
        P0 = rng.standard_normal((p, n)) / math.sqrt(n)
        def JT_t(t):                                              # loop t's backward map
            if kind == 'bfail' and t < R // 2:
                A2 = A.copy(); A2[0, 0] = 0.95; return Bm @ A2 @ Bm.T
            return JT
        def PT_t(t):
            if kind == 'map':
                Rot, _ = np.linalg.qr(np.eye(p) + 0.35 * (R - 1 - t) * rng.standard_normal((p, p)) / math.sqrt(p))
                return Rot @ P0
            return P0
        PT = [PT_t(t) for t in range(R)]
        lam = [None] * R; lam[R - 1] = lam_out
        for t in range(R - 1, 0, -1):
            lam[t - 1] = JT_t(t) @ lam[t]
        c = [PT[t] @ lam[t] for t in range(R)]; u = lam_out / np.linalg.norm(lam_out); pv = [PT[t] @ u for t in range(R)]
        kap = [lam_out]
        for i_ in range(K):
            kap.append(JT_t(R - 1) @ kap[-1])
        chat = [PT[R - 1] @ kap[i_] for i_ in range(K)]
        q = [lam_out / np.linalg.norm(lam_out)]; H = np.zeros((M + 1, M)); pq = []
        for l in range(M):
            pq.append(PT[R - 1] @ q[l]); w = JT_t(R - 1) @ q[l]
            for _ in range(2):
                for jj in range(l + 1):
                    h = q[jj] @ w; H[jj, l] += h; w = w - h * q[jj]
            H[l + 1, l] = np.linalg.norm(w); q.append(w / max(np.linalg.norm(w), 1e-300))
        g0 = rng.standard_normal(p) * 0.3 * np.linalg.norm(sum(c))
        V = np.array([g0] + c + pv + chat + pq); G = V @ V.T; nrm = np.linalg.norm(V, axis=1)
        S = np.array(lam + kap); SG = S @ S.T
        rows.append(dict(i=s, adj=[float(np.linalg.norm(x)) for x in lam], step=[0.0] * R, loss_traj=0.0, loss_ref=None,
                         state_gram=SG.tolist(), H=H.tolist(), lam_out=float(np.linalg.norm(lam_out)), arnoldi_orth=0.0,
                         param_gram=G.tolist(), param_norm=nrm.tolist(),
                         layout=dict(g0=0, c=1, p=1 + R, chat=1 + 2 * R, pq=1 + 2 * R + K, n=len(nrm)),
                         val_norm_maxrel=0.0, krylov_step1_cos=1.0, _g0=g0, _chat=chat))
    return rows


def synthetic():
    print('synthetic check of the decision code (expected: shared -> A SIGNAL, B PASS, C SHARED; separate -> C NOT-SHARED;'
          ' map -> A MAP; bfail -> B FAIL; D on shared -> PASS against a direct blended simulation)')
    R, K, M = 32, 64, 32
    for kind in ('shared', 'separate', 'map', 'bfail'):
        rows = synth_rows(kind, R=R, K=K, M=M, seed={'shared': 1, 'separate': 2, 'map': 3, 'bfail': 4}[kind])
        res = decide('synthetic ' + kind, rows, R, K, M)
        print('   -> %s' % res)
    # D: direct blended simulation of the 'shared' toy vs the binomial prediction
    rows = synth_rows('shared', R=R, K=K, M=M, seed=1); worst = 0.0
    for r in rows:
        u = unpack(r, R, K)
        for Rb, a in ((32, 0.5), (48, 0.4), (64, 0.3)):
            pred = blend_predict(u, Rb, a, (1, 4, 8))
            # direct: v <- (1 - a) v + a J^T v ; contribution a P^T v (P, J^T of the last loop), from the stored chat
            # iterate J_a^T = (1 - a) I + a J^T in the basis kap_l (J^T kap_l = kap_{l+1}); contributions a P^T v
            ch = np.array(r['_chat']); vecs = []
            coef = np.zeros(K); coef[0] = 1.0
            for i in range(Rb):
                vecs.append(a * coef @ ch)
                coef = (1 - a) * coef + a * np.concatenate([[0.0], coef[:-1]])
            vecs = np.array(vecs); tot = r['_g0'] + vecs.sum(0)
            for k in (1, 4, 8):
                worst = max(worst, abs(np.linalg.norm(vecs[k:].sum(0)) / np.linalg.norm(tot) - pred[k]))
    print('D algebra check (binomial Gram prediction vs direct iteration): max |diff| %.1e' % worst)


def main():
    if '--synthetic' in sys.argv:
        synthetic(); return
    summary = {}
    for name, tag in CELLS:
        files = sorted(glob.glob('kaggle-output-v195/**/v195_%s*_R32.json' % tag, recursive=True))
        if not files:
            print('\n%s: no data' % name); continue
        heads = [json.load(open(f)) for f in files]; rows = [r for h in heads for r in h['rows']]
        R, K, M = heads[0]['R'], heads[0]['K'], heads[0]['M']
        tol = dict(adj=2e-2, err=0.01, k1=0.995) if tag == 'huginn' else dict(adj=1e-3, err=0.005, k1=0.9999)
        summary[name] = decide(name, rows, R, K, M, ref=refs(tag), blend=(tag == '140m'), tol=tol)
    print('\nSUMMARY', json.dumps(summary))


if __name__ == '__main__':
    main()
