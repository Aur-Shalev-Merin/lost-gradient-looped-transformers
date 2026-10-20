"""v201 (plan frozen in docs/prereg_v201_batch_lost.md before any measurement): the lost gradient of averaged
gradients against batch size, on the four looped models whose truncation cost was trained, at two stages of
fine-tuning: the trained model (stage 0, the stage every measurement in the paper describes) and the same model after
300 of 600 steps of full-backprop fine-tuning with its study's recipe (stage 300).

Per batch b of 8 packed GSM8K training sequences x 512 tokens (permutation seed 201; the first 64 batches), weights
frozen, fp32, train mode, over the parameters that train in the model's study:
  G_b       gradient of the batch-mean loss with backprop through all R loops
  T_{k,b}   the same batch and state-init draw (seed 201000 + b) with backprop through the last k loops; D = G - T
  raw       lost_k(S) = |sum_S D_{k,b}| / |sum_S G_b|
  clipped   each G_b and T_{k,b} first scaled to norm <= 1.0 as training's clip_grad_norm_ does, then the same ratio
over S = each batch (B = 8), 8 consecutive batches (B = 64), all 64 batches (B = 512); and B = 1: the first 64
sequences, one pass each (seed 301000 + s), raw. Running sums only: 8 parameter-sized fp32 CPU buffers per pass;
B = 64 and B = 512 come from two passes over the same batches (their B = 8 values must agree: a determinism check).

  python v201_batch_lost.py --model parcae|retro|pln5|r5 --stages 0[,300] [--weights DIR]
  python v201_batch_lost.py --analyze DIR [DIR ...]
Env: OUT_DIR; TINY=1: tiny random Part II model on the CPU, a pipeline test that also checks the running sums
against direct sums of stored vectors.
"""
import argparse, glob, hashlib, json, math, os, time
import numpy as np
import torch

TINY = os.environ.get('TINY') == '1'
OUT = os.environ.get('OUT_DIR', '/kaggle/working' if os.path.isdir('/kaggle/working') else '.')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
KS = (1, 2, 4)
BS = 2 if TINY else 8                 # sequences per batch, as in training
NB = 8 if TINY else 64                # batches measured
GROUP = 4 if TINY else 8              # batches per B = 64 group
N1 = 4 if TINY else 64                # sequences for B = 1
CLIP = 1.0
PERM_SEED, SEED0, SEED1 = 201, 201000, 301000
FT_SEED = 2011                        # data order and state-init draws of the stage-300 fine-tune
FT_STOP = 4 if TINY else 300          # steps run, of the 600-step schedule
FT_STEPS, FT_WARMUP = (8, 2) if TINY else (600, 20)
LR = dict(parcae=3e-4, retro=1e-4, pln5=1e-4, r5=1e-4)              # full-backprop LRs of S5 (v166), S6 (v167), S11 (v200)
PARAMS_M = dict(parcae=144.3, retro=163.5, pln5=204.9, r5=204.9)     # trainable parameters, millions (v171; v200's
# run records: 204,882,432 with the embedding frozen. Amendment 1: the frozen code had 205.3, a rounding slip)
IFM = dict(pln5='huginn-s-fixed-pln5', r5='huginn-s-fixed-r5')
DET_TOL = 1e-3                        # repeated full pass, same seed: relative difference (catches unshared draws)


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v201_log.txt'), 'a').write(s + '\n')


def sha(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(open(os.path.abspath(p), 'rb').read().replace(b'\r\n', b'\n'))
    return h.hexdigest()[:12]


def setup(model, weights):
    """The model in train mode, its data and the two losses: meas(ids, k, seed) in fp32 for the measurement, and
    train(ids, k, step) exactly as the model's study trains (fp16 autocast, cache off, on a GPU)."""
    if model == 'parcae':
        import v162_parcae_trunc_train as v, v163_parcae_damped_trunc as d
        tok, m = v.load_model(); data = v.gsm8k_chunks(tok); R, pack, amp = d.R, v.pack, v.AMP

        def meas(ids, k, seed):
            torch.manual_seed(seed)
            return d.damped_loss(m, ids, k, 1.0, amp=False, ckpt=True)

        def train(ids, k, step):          # v163.train_cell: state draws from the global RNG after manual_seed(seed)
            return d.damped_loss(m, ids, k, 1.0, amp=amp)
        mods = [v.__file__, d.__file__]
    elif model == 'retro':
        import v167_retrofit_trunc as r
        m = r.load_model(); data = r.gsm8k_chunks(r.load_tok()); R, pack, amp = r.R, r.pack, r.AMP

        def meas(ids, k, seed):
            torch.manual_seed(seed)
            return r.manual_loss(m, ids, k, amp=False, ckpt=True, a=1.0, r=R)

        def train(ids, k, step):          # v167.train_cell
            return r.manual_loss(m, ids, k, amp=amp, ckpt=k > 0, a=1.0)
        mods = [r.__file__]
    else:
        import v200_ifm_trunc as f        # embedding frozen (V200_FREEZE_EMBED default), as in S11
        tok = None if f.TINY else f.M.Tokenizer(weights)
        data = f.gsm8k_chunks(tok); m, _ = f.load_model(weights, IFM[model]); R, pack, amp = f.R, f.pack, f.AMP

        def meas(ids, k, seed):
            return f.train_loss(m, ids, k, amp=False, ckpt=True, state='randn', seed=seed)

        def train(ids, k, step):          # v200.train_cell
            return f.train_loss(m, ids, k, amp=amp, ckpt=k > 0, seed=FT_SEED * 100_003 + step)
        mods = [f.__file__, f.M.__file__]
    m.train()
    chunks = pack(data['train'], np.random.default_rng(PERM_SEED).permutation(len(data['train'])))
    return m, data, pack, chunks, R, meas, train, amp, mods


def finetune(m, params, data, pack, train, amp, R, lr):
    """Stage 300: the study's recipe (AdamW (0.9, 0.95), wd 0, warm-up + cosine to 10% over 600 steps, clip 1.0,
    GradScaler), full backprop, stopped after FT_STOP steps. In place."""
    g = np.random.default_rng(FT_SEED)
    chunks = torch.cat([pack(data['train'], g.permutation(len(data['train']))) for _ in range(4)])
    torch.manual_seed(FT_SEED)
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    sched = lambda s: min(1.0, (s + 1) / FT_WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / FT_STEPS)))
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    nonfinite, clipped, curve, t0 = 0, 0, [], time.time()
    for s in range(FT_STOP):
        for gp in opt.param_groups:
            gp['lr'] = lr * sched(s)
        ids = chunks[s * BS:(s + 1) * BS].to(DEV)
        loss = train(ids, R, s)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(params, CLIP)
        if not torch.isfinite(gn):
            nonfinite += 1
        elif float(gn) > CLIP:
            clipped += 1
        scaler.step(opt); scaler.update()
        if s % 50 == 0 or s == FT_STOP - 1:
            curve.append((s, round(float(loss.detach()), 4), round(float(gn), 3)))
    opt.zero_grad(set_to_none=True); del opt, scaler
    if DEV == 'cuda':
        torch.cuda.empty_cache()
    ok = nonfinite <= 10 and math.isfinite(curve[-1][1])
    return dict(steps=FT_STOP, schedule_steps=FT_STEPS, lr=lr, seed=FT_SEED, nonfinite=nonfinite,
                clipped_frac=round(clipped / FT_STOP, 4), curve=curve, ok=ok, s=round(time.time() - t0))


def measure(m, params, chunks, R, meas):
    P = sum(p.numel() for p in params)

    def grads(ids, seed, arms):
        out = {}
        for k in arms:
            m.zero_grad(set_to_none=True)
            meas(ids, k, seed).backward()
            out[k] = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1).float()
                                for p in params])
        m.zero_grad(set_to_none=True)
        return out

    # selftest: shared state-init draws (a repeated full pass reproduces itself), and truncation acts
    ids0 = chunks[:BS].to(DEV)
    a, b = grads(ids0, SEED0, (R, 1)), grads(ids0, SEED0, (R,))
    det = float((a[R] - b[R]).norm() / a[R].norm()); lost1 = float((a[R] - a[1]).norm() / a[R].norm())
    st = dict(det_rel=det, lost_k1_batch0=lost1, params_m=round(P / 1e6, 2), ok=det < DET_TOL and lost1 > 1e-3)
    del a, b
    log('  selftest %s | repeat-pass rel diff %.2e (< %g) | k=1 lost on batch 0 %.4f (> 0.001) | params %.1fM' % (
        'PASS' if st['ok'] else 'FAIL', det, DET_TOL, lost1, P / 1e6))
    if not st['ok']:
        return dict(selftest=st)
    ratio = lambda x, y: float(x.norm() / y.norm())

    def one_pass(kind):
        z = lambda: torch.zeros(P)
        accG, accGc, accD, accDc = z(), z(), {k: z() for k in KS}, {k: z() for k in KS}
        b8, groups, store, t0 = [], [], [], time.time()
        for bi in range(NB):
            v = grads(chunks[bi * BS:(bi + 1) * BS].to(DEV), SEED0 + bi, (R,) + KS)
            G = v[R]; gn = float(G.norm()); cG = min(1.0, CLIP / (gn + 1e-6))
            row = dict(b=bi, gnorm=gn, tnorm={}, raw={}, clip={}); Gcpu = G.cpu()
            vecs = dict(G=Gcpu, Gc=Gcpu * cG) if TINY else None
            for k in KS:
                T = v[k]; tn = float(T.norm()); cT = min(1.0, CLIP / (tn + 1e-6))
                D = G - T; Dc = G * cG - T * cT
                row['tnorm'][k] = tn; row['raw'][k] = float(D.norm()) / gn; row['clip'][k] = float(Dc.norm()) / (gn * cG)
                D, Dc = D.cpu(), Dc.cpu(); accD[k] += D; accDc[k] += Dc
                if TINY:
                    vecs['D%d' % k], vecs['Dc%d' % k] = D, Dc
            accG += Gcpu; accGc.add_(Gcpu, alpha=cG); del v, G, Gcpu
            b8.append(row)
            if TINY:
                store.append(vecs)
            if kind == 'groups' and (bi + 1) % GROUP == 0:
                groups.append(dict(g=bi // GROUP, raw={k: ratio(accD[k], accG) for k in KS},
                                   clip={k: ratio(accDc[k], accGc) for k in KS}))
                if TINY:              # running sums == direct sums of the stored vectors
                    grp = store[-GROUP:]; sG = sum(x['G'] for x in grp); sGc = sum(x['Gc'] for x in grp)
                    for k in KS:
                        assert abs(ratio(sum(x['D%d' % k] for x in grp), sG) - groups[-1]['raw'][k]) < 1e-5
                        assert abs(ratio(sum(x['Dc%d' % k] for x in grp), sGc) - groups[-1]['clip'][k]) < 1e-5
                for t in [accG, accGc] + list(accD.values()) + list(accDc.values()):
                    t.zero_()
            if bi % 16 == 15:
                log('    %s pass: %d/%d batches, %ds' % (kind, bi + 1, NB, time.time() - t0))
        res = dict(b8=b8, s=round(time.time() - t0))
        if kind == 'groups':
            res['b64'] = groups
        else:
            res['b512'] = dict(raw={k: ratio(accD[k], accG) for k in KS}, clip={k: ratio(accDc[k], accGc) for k in KS})
            if TINY:
                sG = sum(x['G'] for x in store); sGc = sum(x['Gc'] for x in store)
                for k in KS:
                    assert abs(ratio(sum(x['D%d' % k] for x in store), sG) - res['b512']['raw'][k]) < 1e-5
                    assert abs(ratio(sum(x['Dc%d' % k] for x in store), sGc) - res['b512']['clip'][k]) < 1e-5
                log('    TINY: running sums match direct sums of the stored vectors')
        return res

    out = dict(selftest=st, groups=one_pass('groups'), total=one_pass('total'), b1=[])
    t0 = time.time()
    for s in range(N1):
        v = grads(chunks[s:s + 1].to(DEV), SEED1 + s, (R,) + KS); G = v[R]; gn = float(G.norm())
        out['b1'].append(dict(s=s, gnorm=gn, raw={k: float((G - v[k]).norm()) / gn for k in KS}))
        del v, G
    out['b1_s'] = round(time.time() - t0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=('parcae', 'retro', 'pln5', 'r5')); ap.add_argument('--stages', default='0')
    ap.add_argument('--weights', default=None); ap.add_argument('--analyze', nargs='*')
    A = ap.parse_args()
    if A.analyze is not None:
        return analyze(A.analyze or [OUT])
    T0 = time.time()
    m, data, pack, chunks, R, meas, train, amp, mods = setup(A.model, A.weights)
    params = [p for p in m.parameters() if p.requires_grad]
    dsha = sha([__file__] + mods)
    log('v201 %s | driver %s | R %d | dev %s | train chunks %d | %d x %d sequences, groups of %d, B=1 on %d | k %s' % (
        A.model, dsha, R, DEV, len(chunks), NB, BS, GROUP, N1, KS))
    assert len(chunks) >= NB * BS
    n_params = sum(p.numel() for p in params)
    log('  trainable parameters %d (expected %.1fM)' % (n_params, PARAMS_M[A.model]))
    if not TINY:
        assert abs(n_params / 1e6 - PARAMS_M[A.model]) < 0.2, 'unexpected parameter set'
    for stage in [int(x) for x in A.stages.split(',')]:
        rec = dict(model=A.model, stage=stage, R=R, ks=list(KS), nb=NB, bs=BS, group=GROUP, n1=N1, clip=CLIP,
                   driver_sha=dsha, measured=False)
        if stage == 300:
            rec['finetune'] = finetune(m, params, data, pack, train, amp, R, LR[A.model])
            log('  stage 300 fine-tune: %s' % {k: v for k, v in rec['finetune'].items() if k != 'curve'})
            log('  curve (step, loss, grad norm): %s' % rec['finetune']['curve'])
        if stage == 0 or rec['finetune']['ok']:
            res = measure(m, params, chunks, R, meas)
            rec.update(res); rec['measured'] = 'groups' in res
        rec['s'] = round(time.time() - T0)
        json.dump(rec, open(os.path.join(OUT, 'v201_%s_s%d.json' % (A.model, stage)), 'w'), indent=1)
        log('  stage %d %s after %ds' % (stage, 'measured' if rec['measured'] else 'NOT measured', time.time() - T0))


# ------------------------------------------------------------------------------------------------------------ analysis
def analyze(dirs):
    """Tables and the frozen outcome mapping of docs/prereg_v201_batch_lost.md."""
    recs = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, '**', 'v201_*_s*.json'), recursive=True)):
            r = json.load(open(f)); recs[(r['model'], r['stage'])] = r
    med = lambda xs: float(np.median(xs))
    S = {}
    for (model, stage), r in sorted(recs.items()):
        if not r.get('measured'):
            print('%-6s stage %3d NOT measured: selftest %s fine-tune %s' % (model, stage, r.get('selftest'),
                  {k: v for k, v in r.get('finetune', {}).items() if k != 'curve'}))
            continue
        g8, t8 = r['groups']['b8'], r['total']['b8']
        det = max(abs(a['raw'][str(k)] - b['raw'][str(k)]) for a, b in zip(g8, t8) for k in KS)
        clipped = np.mean([x['gnorm'] > CLIP for x in g8])
        print('%-6s stage %3d | batches clipped %.0f%% (median |G_b| %.2f) | B=8 repeat between passes, max diff %.1e%s' % (
            model, stage, 100 * clipped, med([x['gnorm'] for x in g8]), det,
            ' | fine-tune clipped %.0f%% of steps' % (100 * r['finetune']['clipped_frac']) if 'finetune' in r else ''))
        for k in KS:
            sk = str(k)
            s = dict(b1=med([x['raw'][sk] for x in r['b1']]),
                     b8r=med([x['raw'][sk] for x in g8]), b8c=med([x['clip'][sk] for x in g8]),
                     b64r=med([x['raw'][sk] for x in r['groups']['b64']]), b64c=med([x['clip'][sk] for x in r['groups']['b64']]),
                     b64c_rng=(min(x['clip'][sk] for x in r['groups']['b64']), max(x['clip'][sk] for x in r['groups']['b64'])),
                     b512r=r['total']['b512']['raw'][sk], b512c=r['total']['b512']['clip'][sk])
            S[(model, stage, k)] = s
            print('   k=%d | B=1 %.3f | B=8 raw %.3f clip %.3f | B=64 raw %.3f clip %.3f [%.3f-%.3f] | B=512 raw %.3f clip %.3f' % (
                k, s['b1'], s['b8r'], s['b8c'], s['b64r'], s['b64c'], *s['b64c_rng'], s['b512r'], s['b512c']))
    models = sorted({m for (m, st, k) in S})
    print('\nPRIMARY: gap = clipped B=64 median at stage 300 - per-sequence (B=1) median at stage 0, same k')
    under, over, missing = [], [], []
    for model in models:
        if not all((model, st, k) in S for st in (0, 300) for k in KS):
            missing.append(model); print('  %-6s incomplete (a stage was not measured)' % model); continue
        gaps = {k: S[(model, 300, k)]['b64c'] - S[(model, 0, k)]['b1'] for k in KS}
        print('  %-6s gaps %s' % (model, {k: round(v, 3) for k, v in gaps.items()}))
        if max(gaps.values()) > 0.05:
            under.append(model)
        if min(gaps.values()) < -0.05:
            over.append(model)
    if missing or len(models) < 4:
        verdict = 'INCOMPLETE (%s)' % (missing or 'fewer than 4 models')
    elif not under and not over:
        verdict = 'MEDIAN-FAIR'
    elif under and not over:
        verdict = 'MEDIAN-UNDERSTATES %s' % under
    elif over and not under:
        verdict = 'MEDIAN-OVERSTATES %s' % over
    else:
        verdict = 'MIXED (understates %s, overstates %s)' % (under, over)
    print('VERDICT: %s' % verdict)
    print('\nSECONDARY (no verdict)')
    for model in models:
        for k in KS:
            s0, s3 = S.get((model, 0, k)), S.get((model, 300, k))
            bits = []
            if s0:
                bits.append('stage-0 gap (clip B=64 - B=1) %+.3f' % (s0['b64c'] - s0['b1']))
            if s3:
                bits.append('stage-300 gap (clip B=64 - B=1) %+.3f' % (s3['b64c'] - s3['b1']))
                bits.append('plateau |clip B=512 - B=64| %.3f%s' % (abs(s3['b512c'] - s3['b64c']), '' if abs(s3['b512c'] - s3['b64c']) <= 0.03 else ' (MOVING)'))
            for st, s in ((0, s0), (300, s3)):
                if s and s['b512r'] - s['b512c'] > 0.10:
                    bits.append('stage %d raw B=512 exceeds clipped by %.3f (RARE-SEQUENCES)' % (st, s['b512r'] - s['b512c']))
            if s0 and s3:
                bits.append('training effect on B=1 %+.3f' % (s3['b1'] - s0['b1']))
            print('  %-6s k=%d: %s' % (model, k, ' | '.join(bits)))
    if all((mm, 300, k) in S and (mm, 0, k) in S for mm in ('pln5', 'r5') for k in KS):
        print('  Part II, after the fact relative to S11 (descriptive): r5 - pln5 at k=1/2/4')
        print('    per-sequence (stage 0): %s' % [round(S[('r5', 0, k)]['b1'] - S[('pln5', 0, k)]['b1'], 3) for k in KS])
        print('    clipped B=64 (stage 300): %s' % [round(S[('r5', 300, k)]['b64c'] - S[('pln5', 300, k)]['b64c'], 3) for k in KS])
    return verdict


def selftest_analysis():
    """Synthetic worlds: every stage-0 per-sequence value 0.5; the stage-300 clipped B = 64 values set per world."""
    import tempfile

    def world(gap):           # gap(model, k) -> the stage-300 clipped B = 64 value minus 0.5
        d = tempfile.mkdtemp()
        for model in ('parcae', 'retro', 'pln5', 'r5'):
            for stage in (0, 300):
                v = lambda k: 0.5 + (gap(model, k) if stage == 300 else 0.0)
                b8 = [dict(b=i, gnorm=2.0, raw={str(k): 0.5 for k in KS}, clip={str(k): v(k) for k in KS}) for i in range(4)]
                rec = dict(model=model, stage=stage, measured=True, groups=dict(b8=b8, b64=[
                    dict(g=0, raw={str(k): 0.5 for k in KS}, clip={str(k): v(k) for k in KS})]),
                    total=dict(b8=b8, b512=dict(raw={str(k): 0.5 for k in KS}, clip={str(k): v(k) for k in KS})),
                    b1=[dict(s=0, gnorm=1.0, raw={str(k): 0.5 for k in KS})])
                if stage == 300:
                    rec['finetune'] = dict(clipped_frac=0.2)
                json.dump(rec, open(os.path.join(d, 'v201_%s_s%d.json' % (model, stage)), 'w'))
        return d
    import contextlib, io
    cases = [(lambda mo, k: 0.04, 'MEDIAN-FAIR'), (lambda mo, k: 0.12 if mo == 'parcae' and k == 1 else 0.0,
             "MEDIAN-UNDERSTATES ['parcae']"), (lambda mo, k: -0.08 if mo == 'r5' else 0.0, "MEDIAN-OVERSTATES ['r5']"),
             (lambda mo, k: 0.1 if mo == 'retro' else (-0.1 if mo == 'pln5' else 0.0),
              "MIXED (understates ['retro'], overstates ['pln5'])")]
    ok = True
    for gap, want in cases:
        with contextlib.redirect_stdout(io.StringIO()):
            got = analyze([world(gap)])
        print('world -> %-50s %s' % (got, 'ok' if got == want else 'UNEXPECTED (want %s)' % want)); ok &= got == want
    print('ANALYSIS SELFTEST', 'PASS' if ok else 'FAIL')
    return ok


if __name__ == '__main__':
    import sys
    if '--selftest-analysis' in sys.argv:
        raise SystemExit(0 if selftest_analysis() else 1)
    main()
