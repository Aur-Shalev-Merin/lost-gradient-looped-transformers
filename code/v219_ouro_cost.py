"""v219: what does truncated backprop cost when fine-tuning Ouro-1.4B, the model that loses the most gradient?
Prereg: docs/prereg_v219_ouro_cost.md. Pilot (feasibility, seed spread): v219_ouro_cost_pilot.py.

Recipe (Section 5; v162-v167, v200): GSM8K question + answer text; the last 473 train examples = val, the test split =
test. 600 steps, batch 8 x 512 (micro-batches of 2, gradient accumulation); AdamW (0.9, 0.95), weight decay 0; 20-step
warm-up, then cosine to 10%; clip 1.0; fp16 autocast with the weight cache off, plus GradScaler. Training loss: the
final loop's output. The token embedding is frozen in every arm (as v200: Ouro's embedding is untied and reaches the
loss only through the loop, so only full backprop could train it).
Cells: k = 4 (full backprop), 2, 1 (backprop through the last k of the 4 loops), 0 (the floor: no loop gradient; only
the output head trains). Held-out loss (all 4 loops) under the final output and under Ouro's exit-weighted objective
without its entropy term.

Ouro is split across the kernel's two T4s (v219 pilot): the embedding and layers 0-11 on GPU 0; layers 12-23, the final
norm, the exit gate and the output head on GPU 1. One GPU computes at a time; the maths is unchanged.

  python v219_ouro_cost.py --phase selftest
  python v219_ouro_cost.py --phase measure                 step-0 lost gradient, per sequence, k = 1, 2, 3
  python v219_ouro_cost.py --phase train --cells k:lr:seed,...   appended to v219_runs.json; cells present are skipped
  python v219_ouro_cost.py --analyze DIR [DIR ...]          the locked analysis
Env: TINY=1 (tiny random Ouro on CPU), OUT_DIR.
"""
import argparse, glob, hashlib, json, math, os, sys, time
os.environ.setdefault('LOOP_MODEL', 'ouro')
import numpy as np
import torch
import torch.nn.functional as Fn
from torch.utils.checkpoint import checkpoint

TINY = os.environ.get('TINY') == '1'
OUT = os.environ.get('OUT_DIR', '/kaggle/working' if os.path.isdir('/kaggle/working') else '.')
SEQ = 32 if TINY else 512
BATCH = 4 if TINY else 8
MICRO = 2
STEPS = 6 if TINY else 600
WARMUP = 2 if TINY else 20
CLIP = 1.0
R = 4
N_VAL = 473
N_MEASURE = 2 if TINY else 32
AMP = torch.cuda.is_available()

# ------------------------------------------------------------------ locked at the freeze (prereg sections 2 and 3)
FULL_LR = 1e-4              # full backprop and truncated cells (LR pilot, seed 9001: lowest test loss, interior)
FLOOR_LR = 1e-4             # the floor (same rule; interior)
SEEDS = [2191001, 2191002, 2191003, 2191004, 2191005, 2191006, 2191007]   # pre-listed; valid ones used in order
N_SEEDS, N_SEEDS_MAX, SE_MAX = 3, 5, 0.02        # seed rule (prereg section 1)
ERR = {1: 0.9214538403379658, 2: 0.8391116390706126}   # step-0 lost gradient, median of 32 (results_v219/lr-a)
L_PRED = {k: 0.3418 * ERR[k] ** 3.4335 for k in (1, 2)}  # S6 curve (prereg v167): 0.2581 / 0.1872
MODEL_REV = '574fa66cb8bf5abdc979642d01cf2b79b16bfab1'   # ByteDance/Ouro-1.4B revision of every pilot run
FIT_MAX = 0.877                                  # largest lost gradient the S6 curve was fitted on (v167_analyze.FIT)
KP = {1: 0.224, 2: 0.100}                        # Parcae-140m's own cost at the same k (R = 32; Table 9)
OTHERS = {1: dict(parcae=0.224, pln5=0.162, r5=0.291, smollm2=0.009),     # every trained model's cost at the same k
          2: dict(parcae=0.100, pln5=0.071, r5=0.149, smollm2=-0.001)}
T90 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015}                 # t quantile 0.95, by degrees of freedom


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v219_log.txt'), 'a').write(s + '\n')


def driver_sha():
    return hashlib.sha256(open(os.path.abspath(__file__), 'rb').read().replace(b'\r\n', b'\n')).hexdigest()[:12]


def devices():
    if torch.cuda.device_count() >= 2:
        return torch.device('cuda:0'), torch.device('cuda:1')
    d = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return d, d


def ce(logits, ids):
    return Fn.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), ids[:, 1:].reshape(-1))


# ---------------------------------------------------------------------------------------------------------- model
def load(own_check=None):
    """fp32 Ouro split across the two GPUs, embedding frozen. own_check: token ids on which, before the split, the
    model's own forward (the final loop's output, exit_at_step = R - 1) is scored for the selftest."""
    import loop_grad_probe as P
    tok, model = P.load()
    model = model.float()
    own = None
    if own_check is not None:
        with torch.no_grad():
            ids = own_check.to(next(model.parameters()).device)
            own = float(ce(model(input_ids=ids, exit_at_step=R - 1, use_cache=False).logits, ids))
    d0, d1 = devices(); b = model.model; L = b.config.num_hidden_layers
    b.embed_tokens.to(d0); b.rotary_emb.to(d0)
    for i, layer in enumerate(b.layers[:L]):
        layer.to(d0 if i < L // 2 else d1)
    b.norm.to(d1); model.lm_head.to(d1); b.early_exit_gate.to(d1)
    for p in model.parameters():
        p.requires_grad_(True)
    b.embed_tokens.weight.requires_grad_(False)
    model.train()
    rev = getattr(model.config, '_commit_hash', None)
    return tok, model, own, rev


def make_loss(model):
    b = model.model
    mod = sys.modules[type(b).__module__]
    d0, d1 = devices()
    mv = lambda x, d: x.to(d) if torch.is_tensor(x) else (tuple(mv(y, d) for y in x) if isinstance(x, tuple) else x)

    def prepare(ids):
        h = b.embed_tokens(ids)
        cp = torch.arange(ids.shape[1], device=ids.device); pos = cp.unsqueeze(0)
        mask = mod.create_causal_mask(config=b.config, input_embeds=h, attention_mask=None, cache_position=cp,
                                      past_key_values=None, position_ids=pos)
        c0 = dict(mask=mask, pos=pos, cp=cp, pe=b.rotary_emb(h, pos))
        return h, {d0: c0, d1: {k: mv(v, d1) for k, v in c0.items()}}

    def layers(h, t, ctxs, ckpt):
        for layer in b.layers[: b.config.num_hidden_layers]:
            dev = next(layer.parameters()).device; h = h.to(dev); ctx = ctxs[dev]
            def run(x, layer=layer, ctx=ctx):
                y = layer(x, attention_mask=ctx['mask'], position_ids=ctx['pos'], past_key_value=None, use_cache=False,
                          cache_position=ctx['cp'], position_embeddings=ctx['pe'], current_ut=t)
                return y[0] if isinstance(y, tuple) else y
            h = checkpoint(run, h, use_reentrant=False) if ckpt else run(h)
        return h

    def loss_fn(ids, k, objective='final', amp=True, ckpt=True):
        """Loss after R loops with backprop through the last k (k = 0: no loop gradient). One fp16 autocast region with
        the weight-cast cache off (v161). 'final': CE of the last loop's output. 'exit': Ouro's objective without the
        entropy term, per token sum_t p_t L_t with p_t from the exit gate (as v210 Part B). 'both': (final, exit)."""
        labels = torch.cat([ids[:, 1:], torch.full_like(ids[:, :1], -100)], 1).view(-1).to(d1)
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp and AMP, cache_enabled=False):
            h, ctx = prepare(ids); exits = []
            for t in range(R):
                on = t >= R - k
                with torch.set_grad_enabled(on and torch.is_grad_enabled()):
                    h = b.norm(layers(h, t, ctx, ckpt=on and ckpt))
                if not on:
                    h = h.detach()
                exits.append(h)
            out = {}
            if objective in ('final', 'both'):
                logits = model.lm_head(exits[-1]).float()
                out['final'] = Fn.cross_entropy(logits.view(-1, logits.shape[-1]), labels, ignore_index=-100)
            if objective in ('exit', 'both'):
                valid = (labels != -100).float(); remaining = None; total = 0.0
                for t, hn in enumerate(exits):
                    lam = torch.sigmoid(b.early_exit_gate(hn).squeeze(-1)).float().view(-1)
                    remaining = torch.ones_like(lam) if remaining is None else remaining
                    p_t = lam * remaining if t < R - 1 else remaining
                    remaining = remaining * (1 - lam)
                    logits = model.lm_head(hn).float()
                    l_t = Fn.cross_entropy(logits.view(-1, logits.shape[-1]), labels.clamp(min=0), reduction='none') * valid
                    total = total + (p_t * l_t).sum()
                out['exit'] = total / valid.sum()
        return (out['final'], out['exit']) if objective == 'both' else out[objective]
    return loss_fn


def groups(model):
    b = model.model
    return dict(loop=[p for l in b.layers for p in l.parameters()] + list(b.norm.parameters()),
                head=list(model.lm_head.parameters()), gate=list(b.early_exit_gate.parameters()),
                embed=list(b.embed_tokens.parameters()))


# ----------------------------------------------------------------------------------------------------------- data
def gsm8k(tok):
    """dict(train=[token-id list per example], val=LongTensor chunks, test=LongTensor chunks)."""
    if TINY:
        g = np.random.default_rng(0)
        mk = lambda n: [list(g.integers(1, 1000, size=g.integers(20, 60))) for _ in range(n)]
        tr, va, te = mk(60), mk(10), mk(10)
    else:
        from datasets import load_dataset
        enc = lambda ds: [tok(ex['question'] + '\n' + ex['answer']).input_ids for ex in ds]
        all_tr = enc(load_dataset('openai/gsm8k', 'main', split='train'))
        tr, va = all_tr[:-N_VAL], all_tr[-N_VAL:]
        te = enc(load_dataset('openai/gsm8k', 'main', split='test'))
    return dict(train=tr, val=pack(va), test=pack(te))


def pack(examples, order=None):
    flat = [t for i in (order if order is not None else range(len(examples))) for t in examples[i]]
    n = len(flat) // SEQ
    return torch.tensor(flat[: n * SEQ], dtype=torch.long).view(n, SEQ)


@torch.no_grad()
def evaluate(model, loss_fn, chunks):
    """Held-out loss, all 4 loops, under both objectives (token-weighted over equal-length chunks)."""
    model.eval(); d0 = devices()[0]; f = e = 0.0; n = 0
    for i in range(0, len(chunks), BATCH):
        ids = chunks[i:i + BATCH].to(d0)
        lf, le = loss_fn(ids, 0, 'both')
        f += float(lf) * ids.shape[0]; e += float(le) * ids.shape[0]; n += ids.shape[0]
    model.train()
    return dict(final=f / n, exit=e / n)


# ---------------------------------------------------------------------------------------------------------- train
def train_cell(data, k, lr, seed):
    t0 = time.time()
    _, model, _, rev = load(); loss_fn = make_loss(model); d0 = devices()[0]
    g = np.random.default_rng(seed)
    chunks = torch.cat([pack(data['train'], g.permutation(len(data['train']))) for _ in range(4)])   # 4 shuffled epochs
    assert len(chunks) >= STEPS * BATCH, (len(chunks), STEPS * BATCH)
    torch.manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    sched = lambda s: min(1.0, (s + 1) / WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / STEPS)))
    scaler = torch.amp.GradScaler('cuda', enabled=AMP)
    curve, nonfinite = [], 0
    for s in range(STEPS):
        for gp in opt.param_groups:
            gp['lr'] = lr * sched(s)
        opt.zero_grad(set_to_none=True); tot = 0.0
        ids_s = chunks[s * BATCH:(s + 1) * BATCH]
        for mb in range(BATCH // MICRO):
            loss = loss_fn(ids_s[mb * MICRO:(mb + 1) * MICRO].to(d0), k) / (BATCH // MICRO)
            scaler.scale(loss).backward(); tot += float(loss)
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(params, CLIP)
        if not torch.isfinite(gn):
            nonfinite += 1
        scaler.step(opt); scaler.update()
        if s % 50 == 0 or s == STEPS - 1:
            curve.append((s, round(tot, 4), round(float(gn), 3)))
            log('    k=%d lr=%g seed=%d step %d loss %.4f gn %.3f | %ds' % (k, lr, seed, s, tot, float(gn), time.time() - t0))
    s_train = time.time() - t0
    peak = [round(torch.cuda.max_memory_allocated(i) / 1e9, 2) for i in range(torch.cuda.device_count())]
    del opt; model.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    val, test = evaluate(model, loss_fn, data['val']), evaluate(model, loss_fn, data['test'])
    rec = dict(k=k, lr=lr, seed=seed, val=val, test=test, curve=curve, nonfinite_steps=nonfinite,
               diverged=not all(math.isfinite(x) for x in list(val.values()) + list(test.values())),
               s_train=round(s_train), s=round(time.time() - t0), steps=STEPS, batch=BATCH, seq=SEQ, amp=AMP,
               driver_sha=driver_sha(), model_rev=rev,
               peak_gb=peak)
    del model, params
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); [torch.cuda.reset_peak_memory_stats(i) for i in range(torch.cuda.device_count())]
    return rec


def train(data, cells):
    f = os.path.join(OUT, 'v219_runs.json')
    out = json.load(open(f)) if os.path.exists(f) else dict(runs=[])
    for c in cells:
        k, lr, seed = c.split(':'); k, lr, seed = int(k), float(lr), int(seed)
        if any((r['k'], r['lr'], r['seed']) == (k, lr, seed) for r in out['runs']):
            log('  skip k=%d lr=%g seed=%d (present)' % (k, lr, seed)); continue
        rec = train_cell(data, k, lr, seed)
        log('  RUN k=%d lr=%g seed=%d: val %.4f / %.4f, test %.4f / %.4f (final / exit) | non-finite %d | %.1f s/step | '
            'peak %s GB' % (k, lr, seed, rec['val']['final'], rec['val']['exit'], rec['test']['final'],
                            rec['test']['exit'], rec['nonfinite_steps'], rec['s_train'] / STEPS, rec['peak_gb']))
        out['runs'].append(rec)
        json.dump(out, open(f, 'w'), indent=1)


# -------------------------------------------------------------------------------------------------------- measure
def measure(data):
    """Step-0 lost gradient |g - g_k| / |g| per sequence (batch 1, fp32, train mode, final-output loss), over the
    parameters that train (embedding frozen): the input of the S6 curve, as v200 measured its models."""
    _, model, _, rev = load(); loss_fn = make_loss(model); d0 = devices()[0]
    params = [p for p in model.parameters() if p.requires_grad]
    rows, t0 = [], time.time()
    for i in range(min(N_MEASURE, len(data['test']))):
        ids = data['test'][i:i + 1].to(d0)
        model.zero_grad(set_to_none=True); loss_fn(ids, R, amp=False).backward()
        gR = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        nR = math.sqrt(sum(float(g.double().pow(2).sum()) for g in gR))
        row = dict(i=i)
        for k in (1, 2, 3):
            model.zero_grad(set_to_none=True); loss_fn(ids, k, amp=False).backward()
            d2 = sum(float((g - (p.grad if p.grad is not None else 0.0)).double().pow(2).sum()) for g, p in zip(gR, params))
            row[k] = math.sqrt(d2) / nR
        rows.append(row); del gR
        log('  measure seq %d: %s | %ds' % (i, {k: round(row[k], 4) for k in (1, 2, 3)}, time.time() - t0))
    med = {k: float(np.median([r[k] for r in rows])) for k in (1, 2, 3)}
    pred = {k: 0.3418 * med[k] ** 3.4335 for k in (1, 2)}
    json.dump(dict(rows=rows, median=med, s6_curve=pred, n=len(rows), driver_sha=driver_sha(), model_rev=rev),
              open(os.path.join(OUT, 'v219_measure.json'), 'w'), indent=1)
    log('measure: median lost gradient %s; S6 curve %s' % ({k: round(v, 4) for k, v in med.items()},
                                                         {k: round(v, 4) for k, v in pred.items()}))


# ------------------------------------------------------------------------------------------------------- selftest
def selftest(data):
    """Before any training, in the training precision. Fails loudly (SystemExit) if any check fails."""
    ids = data['test'][:1, : (SEQ if TINY else 128)]
    _, model, own, rev = load(own_check=ids)
    loss_fn = make_loss(model); d0 = devices()[0]; ids = ids.to(d0)
    G = groups(model)
    rep, fails = dict(driver_sha=driver_sha(), model_rev=rev, amp=AMP, torch=torch.__version__), []
    if not TINY and rev != MODEL_REV:
        fails.append('model_revision')
    tol = 1e-5 if not AMP else 1e-4

    def grads(k, amp, ckpt, scale=1024.0):
        model.zero_grad(set_to_none=True)
        loss = loss_fn(ids, k, amp=amp, ckpt=ckpt)
        (loss * (scale if amp else 1.0)).backward()
        out = {g: [(p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) / (scale if amp else 1.0)
                   for p in ps] for g, ps in G.items()}
        model.zero_grad(set_to_none=True)
        return out, float(loss)

    nrm = lambda xs: math.sqrt(sum(float(x.double().pow(2).sum()) for x in xs))

    def rel(h, f):
        return {g: math.sqrt(sum(float((a - b).double().pow(2).sum()) for a, b in zip(h[g], f[g]))) / (nrm(f[g]) + 1e-30)
                for g in f if g != 'embed'}
    # 1. the training forward at k = R (fp32, no checkpointing, split) == the model's own forward (unsplit)
    with torch.no_grad():
        ours = float(loss_fn(ids, R, amp=False, ckpt=False))
    rep['own_forward'] = (ours, own)
    if not abs(ours - own) <= 1e-4 * max(1.0, abs(own)):
        fails.append('own_forward')
    # 2. per-layer checkpointing changes nothing (fp32)
    for k in (2, R):
        d = rel(grads(k, False, True)[0], grads(k, False, False)[0]); rep['ckpt_vs_plain_k%d' % k] = d
        if max(d.values()) >= tol:
            fails.append('ckpt_k%d' % k)
    # 3. fp16 autocast (cache off) gradients vs fp32 (v200's rule), and a synthetic negative control: the comparison
    #    must flag the looped weights getting no gradient (the v161 failure)
    if AMP:
        for k in (1, 2, R):
            (h, lh), (f, lf) = grads(k, True, True), grads(k, False, False)
            d = {g: v for g, v in rel(h, f).items() if g != 'gate'}; rep['fp16_vs_fp32_k%d' % k] = dict(d, loss=(lh, lf))
            if not (d['loop'] < 0.5 and d['loop'] <= 3 * d['head'] + 0.02 and abs(lh - lf) < 0.01):
                fails.append('fp16_k%d' % k)
            del h, f
        f, _ = grads(2, False, False)
        for p in G['loop']:
            p.requires_grad_(False)
        try:
            hs, _ = grads(2, True, True)
        finally:
            for p in G['loop']:
                p.requires_grad_(True)
        rep['negative_control_synthetic_k2'] = ds = rel(hs, f)
        if not ds['loop'] > 0.9:
            fails.append('negative_control')
        del hs, f
    # 4. the floor gives the loop exactly zero gradient and trains the head; k = 1 reaches every looped weight;
    #    the embedding never gets a gradient
    g0, _ = grads(0, False, False); g1, _ = grads(1, False, False)
    rep['floor_norms_k0'] = {g: nrm(v) for g, v in g0.items()}
    rep['norms_k1'] = {g: nrm(v) for g, v in g1.items()}
    rep['k1_loop_tensors_with_zero_grad'] = sum(int(float(x.abs().sum()) == 0.0) for x in g1['loop'])
    if rep['floor_norms_k0']['loop'] != 0.0 or rep['floor_norms_k0']['head'] == 0.0 or \
            rep['k1_loop_tensors_with_zero_grad'] > 0 or rep['norms_k1']['embed'] != 0.0 or \
            G['embed'][0].requires_grad:
        fails.append('floor')
    rep['trainable'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rep['trainable_at_floor'] = sum(p.numel() for p in G['head'])
    rep['fails'], rep['status'] = fails, 'PASS' if not fails else 'FAIL'
    json.dump(rep, open(os.path.join(OUT, 'v219_selftest.json'), 'w'), indent=1, default=str)
    log('selftest %s %s | own forward %.6f / %.6f | %s' % (rep['status'], fails, ours, own,
        {k: {g: round(x, 5) for g, x in v.items() if isinstance(x, float)} for k, v in rep.items()
         if k.startswith(('fp16', 'negative', 'ckpt'))}))
    if fails:
        raise SystemExit('selftest FAILED: %s' % fails)


# -------------------------------------------------------------------------------------------------------- analyze
def analyze(dirs):
    """The locked analysis (prereg section 3). Reads every v219_runs.json under the given directories."""
    runs = []
    for d in dirs:
        for f in glob.glob(os.path.join(d, '**', 'v219_runs.json'), recursive=True):
            runs += json.load(open(f))['runs']
    lines = []; P = lambda s='': lines.append(s)
    cell = lambda k, seed: [r for r in runs if r['seed'] == seed and r['k'] == k and
                            math.isclose(r['lr'], FLOOR_LR if k == 0 else FULL_LR, rel_tol=1e-9)]
    P('v219 analysis: driver %s; full LR %g, floor LR %g; %d runs read' % (driver_sha(), FULL_LR, FLOOR_LR, len(runs)))
    valid = []
    for seed in SEEDS:
        cs = {k: cell(k, seed) for k in (R, 2, 1, 0)}
        if any(len(v) != 1 for v in cs.values()):
            P('  seed %d: cells present %s -> not scored' % (seed, {k: len(v) for k, v in cs.items()})); continue
        full = cs[R][0]
        ok = full['nonfinite_steps'] <= 10 and not full['diverged'] and \
            all(not cs[k][0]['diverged'] for k in (2, 1, 0))
        P('  seed %d: full non-finite %d, diverged %s -> %s' % (seed, full['nonfinite_steps'], full['diverged'],
                                                               'valid' if ok else 'EXCLUDED'))
        if ok:
            valid.append({k: v[0] for k, v in cs.items()})
    if len(valid) < N_SEEDS:
        P('FEWER THAN %d VALID SEEDS (%d): INCOMPLETE' % (N_SEEDS, len(valid)))
        return '\n'.join(lines)
    # seed rule: N_SEEDS, raised to N_SEEDS_MAX if the s.e. of the k = 1 cost over the first N_SEEDS exceeds SE_MAX
    phi1 = [(v[1]['test']['final'] - v[R]['test']['final']) / (v[0]['test']['final'] - v[R]['test']['final'])
            for v in valid[:N_SEEDS]]
    se1 = float(np.std(phi1, ddof=1) / math.sqrt(N_SEEDS))
    n = N_SEEDS if se1 <= SE_MAX else N_SEEDS_MAX
    P('seed rule: s.e. of the k=1 cost over the first %d valid seeds %.4f -> N = %d' % (N_SEEDS, se1, n))
    if len(valid) < n:
        P('FEWER THAN %d VALID SEEDS (%d): INCOMPLETE (run the next pre-listed seeds)' % (n, len(valid)))
        return '\n'.join(lines)
    valid = valid[:n]

    def ci(x):
        x = np.asarray(x, float); m = float(x.mean()); se = float(x.std(ddof=1) / math.sqrt(len(x)))
        t = T90[len(x) - 1]
        return m, se, m - t * se, m + t * se
    res = {}
    for obj in ('final', 'exit'):
        gap = [v[0]['test'][obj] - v[R]['test'][obj] for v in valid]
        res[obj] = dict(gap=ci(gap))
        for k in (1, 2):
            res[obj][k] = ci([(v[k]['test'][obj] - v[R]['test'][obj]) / (v[0]['test'][obj] - v[R]['test'][obj])
                              for v in valid])
        res[obj]['per_seed'] = {k: [round((v[k]['test'][obj] - v[R]['test'][obj]) / (v[0]['test'][obj] - v[R]['test'][obj]), 4)
                                    for v in valid] for k in (1, 2)}
    F = res['final']
    P('test loss, final output, per seed (full / k=2 / k=1 / floor):')
    for v in valid:
        P('  seed %d: %.4f / %.4f / %.4f / %.4f' % (v[R]['seed'], *[v[k]['test']['final'] for k in (R, 2, 1, 0)]))
    P('floor gap (final output): %.4f (s.e. %.4f; 90%% %.4f to %.4f)' % F['gap'])
    gate = F['gap'][0] >= 0.03 and F['gap'][2] > 0
    P('floor gate (>= 0.03 nats, 90%% lower bound > 0): %s' % ('PASS' if gate else 'FAIL'))
    for k in (1, 2):
        P('cost k=%d (final output): %.4f (s.e. %.4f; 90%% %.4f to %.4f) per seed %s | L %.3f%s, KP %.3f, max of the '
          'trained models %.3f' % (k, *F[k], F['per_seed'][k], L_PRED[k], ' (extrapolated)' if ERR[k] > FIT_MAX else '',
                                   KP[k], max(OTHERS[k].values())))
    if not gate:
        P('VERDICT: UNINFORMATIVE (floor gate failed)')
        return '\n'.join(lines)
    lo_pred = {k: min(L_PRED[k], KP[k]) for k in (1, 2)}; hi_pred = {k: max(L_PRED[k], KP[k]) for k in (1, 2)}
    mae_l = float(np.mean([abs(F[k][0] - L_PRED[k]) for k in (1, 2)]))
    mae_k = float(np.mean([abs(F[k][0] - KP[k]) for k in (1, 2)]))
    if all(F[k][3] < lo_pred[k] for k in (1, 2)):
        verdict = 'BELOW-BOTH'
    elif all(F[k][2] > hi_pred[k] for k in (1, 2)):
        verdict = 'ABOVE-BOTH'
    elif mae_l <= 0.5 * mae_k:
        verdict = 'L-CLOSER'
    elif mae_k <= 0.5 * mae_l:
        verdict = 'KP-CLOSER'
    else:
        verdict = 'TIE'
    P('VERDICT (primary): %s (MAE L %.4f, KP %.4f)' % (verdict, mae_l, mae_k))
    for k in (1, 2):
        allc = dict(OTHERS[k], ouro=F[k][0])
        P('  k=%d: rank of Ouro among the trained models (descriptive): %s' % (
            k, ' > '.join('%s %.3f' % (m, c) for m, c in sorted(allc.items(), key=lambda x: -x[1]))))
    E = res['exit']
    P('exit-weighted objective (descriptive): floor gap %.4f; cost k=1 %.4f (90%% %.4f to %.4f), k=2 %.4f (90%% %.4f '
      'to %.4f)' % (E['gap'][0], E[1][0], E[1][2], E[1][3], E[2][0], E[2][2], E[2][3]))
    return '\n'.join(lines)


# -------------------------------------------------------------------------------------------------------------- main
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase'); ap.add_argument('--cells', default='')
    ap.add_argument('--analyze', nargs='+')
    A = ap.parse_args()
    if A.analyze:
        txt = analyze(A.analyze); print(txt)
        open(os.path.join(OUT, 'v219_analysis.txt'), 'w').write(txt + '\n')
        sys.exit(0)
    log('v219 driver %s, phase %s, cells %s' % (driver_sha(), A.phase, A.cells))
    import loop_grad_probe as P
    tok = P.load()[0] if TINY else __import__('transformers').AutoTokenizer.from_pretrained(P.REPO, trust_remote_code=True)
    data = gsm8k(tok)
    log('data: %d train examples, %d val chunks, %d test chunks' % (len(data['train']), len(data['val']), len(data['test'])))
    if A.phase == 'selftest':
        selftest(data)
    elif A.phase == 'measure':
        measure(data)
    elif A.phase == 'eval0':                          # the base model's held-out loss (step 0)
        _, model, _, rev = load(); loss_fn = make_loss(model)
        rec = dict(val=evaluate(model, loss_fn, data['val']), test=evaluate(model, loss_fn, data['test']), model_rev=rev,
                   driver_sha=driver_sha())
        json.dump(rec, open(os.path.join(OUT, 'v219_eval0.json'), 'w'), indent=1)
        log('eval0: val %s, test %s' % (rec['val'], rec['test']))
    elif A.phase == 'train':
        train(data, [c for c in A.cells.split(',') if c])
