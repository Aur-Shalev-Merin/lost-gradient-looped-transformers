"""v218 (plan frozen in docs/prereg_v218_robustness.md before any measurement): three robustness questions about the
paper's fine-tuning and measurement results (outside reviews 7 and 8).

D1  Does the fall continue to step 600? The full-backprop fine-tune of v201/v210 (recipe, LR, data order, seed rule),
    run to the end of its 600-step schedule instead of stopping at 300, measured at steps 0, 300 and 600.
D2  Does it fall on unrelated text? At each checkpoint, also measure on WikiText-103 (test split), not only on GSM8K.
D3  How much does the lost gradient vary across random starting-state draws? At step 0, 16 sequences x 8 draws.

Measurement = v210 Part A: batch 1, fp32, train mode, arms k = R, 0, 1, 2, 4;
    all  = |g - g_k| / |g|          loop = |g - g_k| / |g - g_0|
GSM8K: the first 64 test sequences, state seed 301000 + s (as v210). WikiText: the first 64 packed 512-token chunks of
wikitext-103-raw-v1 test, state seed 501000 + s. Draws: the first 16 GSM8K test sequences, seeds 401000 + 1000 d + s,
d = 0..7.
The fine-tune's global RNG state is saved before and restored after each checkpoint measurement, so measuring does not
change the training trajectory.

  python v218_robustness.py --model parcae|pln5|r5 --do ft|draws|ft,draws [--weights DIR]
  python v218_robustness.py --analyze V210_DIR V218_DIR [...]
Env: TINY=1 (pipeline test on a tiny Part II model, CPU), OUT_DIR.
"""
import argparse, glob, hashlib, json, math, os, time
import numpy as np
import torch

TINY = os.environ.get('TINY') == '1'
OUT = os.environ.get('OUT_DIR', '/kaggle/working' if os.path.isdir('/kaggle/working') else '.')
KS = (1, 2, 4)
N_SEQ = 4 if TINY else 64
N_DRAW_SEQ, N_DRAWS = (2, 3) if TINY else (16, 8)
FT_SEEDS = (2011, 2012, 2013)
CKPTS = (2, 4) if TINY else (300, 600)
SEED_GSM, SEED_WIKI, SEED_DRAW = 301000, 501000, 401000
S0_TOL = 0.005
CAPTURED = {}


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v218_log.txt'), 'a').write(s + '\n')


def sha(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(open(os.path.abspath(p), 'rb').read().replace(b'\r\n', b'\n'))
    return h.hexdigest()[:12]


def setup(model, weights):
    """v201's setup, unchanged, with the tokenizer captured on its way into gsm8k_chunks (for WikiText)."""
    import v201_batch_lost as B
    if model == 'parcae':
        import v162_parcae_trunc_train as mod
    else:
        import v200_ifm_trunc as mod
    orig = mod.gsm8k_chunks
    if not getattr(orig, '_v218', False):
        def wrapped(tok, _o=orig):
            CAPTURED['tok'] = tok
            return _o(tok)
        wrapped._v218 = True
        mod.gsm8k_chunks = wrapped
    return B, B.setup(model, weights)


def wiki_chunks(model, pack):
    """The first 64 packed 512-token chunks of WikiText-103 test, one document per article, tokenised as each model's
    GSM8K text is (Parcae: tok(text).input_ids; Part II: tok.encode(text, bos=True, eos=True))."""
    if TINY:
        g = np.random.default_rng(7)
        return pack([list(g.integers(2, 1000, size=200)) for _ in range(20)])
    from datasets import load_dataset
    try:
        ds = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='test')
    except Exception:
        ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')
    docs, cur = [], []
    for line in ds['text']:
        if line.startswith(' = ') and not line.startswith(' = = ') and cur:
            docs.append(''.join(cur).strip()); cur = []
        cur.append(line)
    if cur:
        docs.append(''.join(cur).strip())
    tok = CAPTURED['tok']
    enc = (lambda t: tok(t).input_ids) if model == 'parcae' else (lambda t: tok.encode(t, bos=True, eos=True))
    ex, n = [], 0
    for d in docs:
        ids = enc(d); ex.append(ids); n += len(ids)
        if n > 600 * 512:                      # enough for 64 chunks
            break
    return pack(ex)[:N_SEQ]


def measure(m, R, meas, chunks, params, seed0, n):
    rows = []
    for s in range(min(n, len(chunks))):
        ids = chunks[s:s + 1]; vec = {}
        for k in (R, 0) + KS:
            m.zero_grad(set_to_none=True)
            meas(ids, k, seed0 + s).backward()
            vec[k] = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1).float()
                                for p in params])
        m.zero_grad(set_to_none=True)
        G = vec[R]; gn = float(G.norm()); ln = float((G - vec[0]).norm())
        rows.append(dict(s=s, all={k: float((G - vec[k]).norm()) / gn for k in KS},
                         loop={k: float((G - vec[k]).norm()) / ln for k in KS}))
        del vec, G
    return rows


def rng_state():
    st = {'cpu': torch.get_rng_state()}
    if torch.cuda.is_available():
        st['cuda'] = torch.cuda.get_rng_state_all()
    return st


def set_rng(st):
    torch.set_rng_state(st['cpu'])
    if 'cuda' in st:
        torch.cuda.set_rng_state_all(st['cuda'])


def checkpoint(m, R, meas, params, gsm, wiki, tag):
    t0 = time.time()
    rec = dict(gsm=measure(m, R, meas, gsm, params, SEED_GSM, N_SEQ), wiki=measure(m, R, meas, wiki, params, SEED_WIKI, N_SEQ))
    med = lambda rows, key: [round(float(np.median([r[key][k] for r in rows])), 3) for k in KS]
    log('  %s: GSM8K all %s loop %s | WikiText all %s loop %s (%ds)' % (
        tag, med(rec['gsm'], 'all'), med(rec['gsm'], 'loop'), med(rec['wiki'], 'all'), med(rec['wiki'], 'loop'), time.time() - t0))
    return rec


def finetune_ckpt(B, m, params, data, pack, train, amp, R, lr, seed, on_ckpt):
    """v201's finetune (recipe, schedule over 600 steps, data order, clip, GradScaler), run to the last checkpoint,
    calling on_ckpt(step) after the listed steps with the global RNG saved and restored around it."""
    import math as _m
    g = np.random.default_rng(seed)
    chunks = torch.cat([pack(data['train'], g.permutation(len(data['train']))) for _ in range(4)])
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    sched = lambda s: min(1.0, (s + 1) / B.FT_WARMUP) * (0.1 + 0.9 * 0.5 * (1 + _m.cos(_m.pi * s / B.FT_STEPS)))
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    nonfinite, nf300, clipped, curve, t0 = 0, 0, 0, [], time.time()
    for s in range(CKPTS[-1]):
        for gp in opt.param_groups:
            gp['lr'] = lr * sched(s)
        ids = chunks[s * B.BS:(s + 1) * B.BS].to(B.DEV)
        loss = train(ids, R, s)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(params, B.CLIP)
        if not torch.isfinite(gn):
            nonfinite += 1
        elif float(gn) > B.CLIP:
            clipped += 1
        scaler.step(opt); scaler.update()
        if s % 50 == 0 or s == CKPTS[-1] - 1:
            curve.append((s, round(float(loss.detach()), 4), round(float(gn), 3)))
        if s + 1 == CKPTS[0]:
            nf300 = nonfinite
            if not (nonfinite <= 10 and math.isfinite(float(loss))):
                break
        if s + 1 in CKPTS:
            opt.zero_grad(set_to_none=True)
            st = rng_state(); m.train()
            on_ckpt(s + 1)
            set_rng(st); m.train()
    opt.zero_grad(set_to_none=True); del opt, scaler
    ok = nf300 <= 10 and nonfinite <= 20 and math.isfinite(curve[-1][1])
    return dict(steps=CKPTS[-1], schedule_steps=B.FT_STEPS, lr=lr, seed=seed, nonfinite_300=nf300, nonfinite=nonfinite,
                clipped_frac=round(clipped / max(1, s + 1), 4), curve=curve, ok=ok, s=round(time.time() - t0))


def run(model, todo, weights):
    out = dict(model=model, runs={})
    fn = os.path.join(OUT, 'v218_%s_%s.json' % (model, '_'.join(todo)))
    if 'ft' in todo:
        for seed in FT_SEEDS:
            B, (m, data, pack, _, R, meas, train, amp, mods) = setup(model, weights)
            params = [p for p in m.parameters() if p.requires_grad]
            gsm = data['test'].to(B.DEV)
            if B.TINY:
                gsm = torch.cat([gsm] * 6)
            wiki = wiki_chunks(model, pack).to(B.DEV)
            rec = dict(seed=seed, R=R, n_wiki=int(len(wiki)), ckpt={})
            rec['ckpt'][0] = checkpoint(m, R, meas, params, gsm, wiki, 'step 0')

            def on_ckpt(step):
                rec['ckpt'][step] = checkpoint(m, R, meas, params, gsm, wiki, 'step %d' % step)
            rec['finetune'] = finetune_ckpt(B, m, params, data, pack, train, amp, R, B.LR[model], seed, on_ckpt)
            log('  %s fine-tune seed %d: %s' % (model, seed, {k: v for k, v in rec['finetune'].items() if k != 'curve'}))
            out['driver_sha'] = sha([__file__] + mods)
            out['runs'].setdefault('excluded' if not rec['finetune']['ok'] else 'ft', []).append(rec)
            json.dump(out, open(fn, 'w'), indent=1)
            del m
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if rec['finetune']['ok']:
                break
    if 'draws' in todo:
        B, (m, data, pack, _, R, meas, train, amp, mods) = setup(model, weights)
        params = [p for p in m.parameters() if p.requires_grad]
        gsm = data['test'].to(B.DEV)
        if B.TINY:
            gsm = torch.cat([gsm] * 6)
        t0 = time.time(); draws = []
        for d in range(N_DRAWS):
            draws.append(measure(m, R, meas, gsm, params, SEED_DRAW + 1000 * d, N_DRAW_SEQ))
        out['runs']['draws'] = draws; out['driver_sha'] = sha([__file__] + mods)
        sd = [float(np.median([np.std([draws[d][s]['all'][k] for d in range(N_DRAWS)], ddof=1)
                               for s in range(len(draws[0]))])) for k in KS]
        log('  %s draws: median within-sequence SD of all over %d draws, k = 1/2/4: %s (%ds)' % (
            model, N_DRAWS, [round(x, 4) for x in sd], time.time() - t0))
        json.dump(out, open(fn, 'w'), indent=1)


def analyze(dirs):
    """Frozen outcome mapping of docs/prereg_v218_robustness.md."""
    V210, V218 = {}, {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, '**', 'v210_A_*.json'), recursive=True):
            r = json.load(open(f)); V210.setdefault(r['model'], {}).update(r['runs'])
        for f in glob.glob(os.path.join(d, '**', 'v218_*.json'), recursive=True):
            r = json.load(open(f)); V218.setdefault(r['model'], {}).update(r['runs'])
    g = lambda rows, key, k: float(np.median([r[key][str(k)] if str(k) in r[key] else r[key][k] for r in rows]))
    mean_k = lambda rows, key: float(np.mean([g(rows, key, k) for k in KS]))
    d1, d2, d3, valid = {}, {}, {}, True
    print('v218 robustness (medians over sequences; falls are means over k = 1, 2, 4 of the all-parameter statistic)')
    for m in ('parcae', 'pln5', 'r5'):
        runs = V218.get(m, {})
        ft = (runs.get('ft') or [None])[0]
        if ft:
            ck = {int(s): v for s, v in ft['ckpt'].items()}
            if not all(s in ck for s in (0,) + CKPTS):
                print('  %-6s incomplete checkpoints %s' % (m, sorted(ck))); continue
            for s in (0,) + CKPTS:
                print('  %-6s step %3d  GSM8K all %s | WikiText all %s' % (m, s,
                      [round(g(ck[s]['gsm'], 'all', k), 3) for k in KS], [round(g(ck[s]['wiki'], 'all', k), 3) for k in KS]))
            f300 = mean_k(ck[0]['gsm'], 'all') - mean_k(ck[CKPTS[0]]['gsm'], 'all')
            f600 = mean_k(ck[0]['gsm'], 'all') - mean_k(ck[CKPTS[1]]['gsm'], 'all')
            w300 = mean_k(ck[0]['wiki'], 'all') - mean_k(ck[CKPTS[0]]['wiki'], 'all')
            d1[m] = f600 - f300; d2[m] = w300 / f300 if f300 > 0 else float('nan')
            s0 = V210.get(m, {}).get('s0')
            dev = max(abs(g(ck[0]['gsm'], key, k) - g(s0['rows'], key, k)) for key in ('all', 'loop') for k in KS) if s0 else float('nan')
            ok = dev <= S0_TOL and ft['finetune']['ok']
            valid &= ok
            print('  %-6s fall at 300 %.3f, at 600 %.3f (change %+.3f) | WikiText fall at 300 %.3f (ratio %.2f) | seed %s, '
                  'non-finite %s/%s | step-0 agreement with v210 %.4f (<= %g) %s' % (
                      m, f300, f600, d1[m], w300, d2[m], ft['finetune']['seed'], ft['finetune']['nonfinite_300'],
                      ft['finetune']['nonfinite'], dev, S0_TOL, 'VALID' if ok else 'INVALID'))
        if runs.get('draws'):
            dr = runs['draws']; n = len(dr[0])
            sd = [float(np.median([np.std([dr[d][s]['all'][str(k)] if str(k) in dr[d][s]['all'] else dr[d][s]['all'][k]
                                           for d in range(len(dr))], ddof=1) for s in range(n)])) for k in KS]
            d3[m] = max(sd)
            print('  %-6s state draws: median within-sequence SD over %d draws, k = 1/2/4: %s (max %.4f)' % (
                m, len(dr), [round(x, 4) for x in sd], d3[m]))
    def v(dct, rules, n=3):
        if len(dct) < n:
            return 'INCOMPLETE'
        for name, test in rules:
            if all(test(x) for x in dct.values()):
                return name
        return 'MIXED %s' % {k: round(x, 3) for k, x in dct.items()}
    if not valid:
        print('VERDICTS INVALID: a validity check failed (see above)')
    print('VERDICT D1 (step 600): %s' % v(d1, [('FALLS-FURTHER', lambda x: x >= 0.02), ('HOLDS', lambda x: abs(x) < 0.02),
                                              ('RECOVERS', lambda x: x <= -0.02)]))
    print('VERDICT D2 (WikiText): %s' % v(d2, [('GENERAL', lambda x: x >= 0.5), ('DOMAIN', lambda x: x < 0.25)]).replace('MIXED', 'PARTIAL'))
    print('VERDICT D3 (state draws): %s' % ('INCOMPLETE' if len(d3) < 3 else 'SMALL' if all(x <= 0.02 for x in d3.values())
                                            else 'LARGE' if any(x > 0.05 for x in d3.values()) else 'MODERATE'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model'); ap.add_argument('--do', default='ft,draws'); ap.add_argument('--weights', default=None)
    ap.add_argument('--analyze', nargs='*')
    A_ = ap.parse_args()
    if A_.analyze is not None:
        analyze(A_.analyze or [OUT])
    else:
        run(A_.model, A_.do.split(','), A_.weights)
