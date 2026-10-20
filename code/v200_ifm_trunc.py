"""v200: truncated-loop fine-tuning of Part II looped LMs (Huang et al. 2026, arXiv 2610.06833) at R = 5 on GSM8K: the
training cost of backprop through only the last k loops. Models: IFM/LoopedLM-P2-huginn-s-fixed-pln5 and -fixed-r5
(the two v198 measured; same pinned revisions). Plan: docs/v200_plan_notes.md (the prereg with the locked predictions
is separate). Port: ifm_looped.py, unchanged since v198, which checked it against their xLLM code on both models.

Recipe = v162-v167: GSM8K question + '\\n' + answer, each example in the model's training format (BOS + text + EOS),
attention within documents; the last 473 train examples = val, the test split = test, the rest = training (4
shuffled epochs, rng(seed)); 600 steps of batch 8 x 512 tokens; AdamW (0.9, 0.95), eps 1e-8, weight decay 0;
20-step warm-up + cosine to 10%; global grad clip 1.0; fp16 autocast with cache_enabled=False + GradScaler on a GPU.
R = 5 (the trained mean and eval depth), a = 1. Backprop through the last k loops: the first 5 - k loops run under
no_grad and the state is detached, as in their training; kept loops are checkpointed per block.
  k = 0 floor: no loop credit. Gradient reaches only the coda block, the final norm and the head (about 125M of
        303.6M parameters). The two loop blocks and the input injection get none, and neither do the prelude and the
        embedding: their only path to the loss runs through the injection inside the loops. (As v167's retrofit.)
  k = 5 full backprop. Cost phi_k = (L_k - L_full) / (L_floor - L_full), per seed, on val and test.
Training draws the initial state at random, N(0, 1/d) as in their training, seeded per (seed, step), so all arms of a
seed see the same data and states. Evaluation (val, test; all 5 loops, no grad, fp16) uses the model's deterministic
eval-mode state.

  python v200_ifm_trunc.py --variant V --weights DIR --selftest                 every job runs it before training
  python v200_ifm_trunc.py --variant V --weights DIR --selftest --pilot-lr --part 0/2 --tag T
  python v200_ifm_trunc.py --variant V --weights DIR --selftest --train 'k:lr:seed,...' --tag T
  python v200_ifm_trunc.py --analyze PATH...   [--lrs FLOOR:TRUNC]
Env: TINY=1 (tiny random model, CPU pipeline test), OUT_DIR, V200_STEPS (600), V200_FP32=1 (no autocast),
V200_SELFTEST_BATCH (2), GSM8K_DIR (local dir with train-/test-00000-of-00001.parquet instead of `datasets`),
V200_ALLOW_PORT_CHANGE=1 (run with an ifm_looped.py other than the v198-validated one).
"""
import argparse, glob, hashlib, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # our modules, also under python -I
import numpy as np
import torch

import ifm_looped as M

TINY = os.environ.get('TINY') == '1'
OUT = os.environ.get('OUT_DIR', '/kaggle/working' if os.path.isdir('/kaggle/working') else '.')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
AMP = DEV == 'cuda' and os.environ.get('V200_FP32') != '1'
FREEZE_EMBED = os.environ.get('V200_FREEZE_EMBED', '1') == '1'   # the embedding trains in no arm (see load_model)
R = 5
SEQ = 64 if TINY else 512
BATCH = 2 if TINY else 8
STEPS = int(os.environ.get('V200_STEPS', 6 if TINY else 600))
WARMUP = 2 if TINY else 20
CLIP = 1.0
N_VAL = 473
KS = (0, 1, 2, 4, 5)                     # 0 floor, 5 full
LR_GRID = (1e-4, 3e-4, 9e-4)
PILOT_SEED = 9001
MAX_NONFINITE, N_USE = 10, 5             # v165 exclusion rule; first 5 valid seeds
VARIANTS = {'huginn-s-fixed-pln5': 'c581f369846e74b40072cae77ef19fa51437dfa7',
            'huginn-s-fixed-r5': '17cdf78e0d9fd8d19bba07ed800875e1de5675fd'}
PORT_SHA = '502739cdccc8'                # ifm_looped.py as validated in v198 (LF sha256 prefix; driver 95b158362555)
EST = {0: 450, 1: 585, 2: 685, 4: 885, 5: 990}   # T4 seconds per run (FLOP model calibrated on v167; as the builder)


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v200_log.txt'), 'a').write(s + '\n')


def _sha(path):
    return hashlib.sha256(open(path, 'rb').read().replace(b'\r\n', b'\n')).hexdigest()[:12]


def driver_sha():
    h = hashlib.sha256()
    for f in (__file__, M.__file__):
        h.update(open(os.path.abspath(f), 'rb').read().replace(b'\r\n', b'\n'))
    return h.hexdigest()[:12]


def short(variant):
    return variant.replace('huginn-s-', '').replace('fixed-', '')            # pln5, r5


# --------------------------------------------------------------------------------------------------- model + data
_VERIFIED = set()


def load_model(weights, variant):
    if TINY:
        m = M.tiny_model(seed=0, device=DEV)
        info = dict(checkpoint_id='tiny-random', manifest_sha256=None)
    else:
        m, info = M.load_model(weights, dtype=torch.float32, device=DEV, verify=weights not in _VERIFIED)
        _VERIFIED.add(weights)
        if info['checkpoint_id'] != variant:
            raise SystemExit('weights are %s, not the requested %s' % (info['checkpoint_id'], variant))
    for p in m.parameters():
        p.requires_grad_(True)
    if FREEZE_EMBED:            # untied, and reaches the loss only through the loop: the floor could not train it,
        for p in m.embed.parameters():   # so it is frozen in every arm; the cost then compares loop credit alone
            p.requires_grad_(False)
    return m.train(), info


def gsm8k_rows(split):
    d = os.environ.get('GSM8K_DIR')
    if d:
        import pandas as pd
        df = pd.read_parquet(os.path.join(d, '%s-00000-of-00001.parquet' % split))
        return list(zip(df['question'], df['answer']))
    from datasets import load_dataset
    return [(ex['question'], ex['answer']) for ex in load_dataset('openai/gsm8k', 'main', split=split)]


def pack(examples, order=None):
    flat = [t for i in (order if order is not None else range(len(examples))) for t in examples[i]]
    n = len(flat) // SEQ
    return torch.tensor(flat[:n * SEQ], dtype=torch.long).view(n, SEQ)


def gsm8k_chunks(tok):
    """dict(train=[token lists, one per example], val=, test= packed LongTensor chunks); split as v162 / v167."""
    if TINY:
        rng = np.random.default_rng(0)
        mk = lambda n: [[0] + [int(x) for x in rng.integers(2, 1000, size=int(rng.integers(20, 60)))] + [1] for _ in range(n)]
        tr, va, te = mk(60), mk(10), mk(10)
    else:
        enc = lambda rows: [tok.encode(q + '\n' + a, bos=True, eos=True) for q, a in rows]
        all_tr = enc(gsm8k_rows('train'))
        tr, va = all_tr[:-N_VAL], all_tr[-N_VAL:]
        te = enc(gsm8k_rows('test'))
    return dict(train=tr, val=pack(va), test=pack(te))


# --------------------------------------------------------------------------------------------------- forward
def ce_shifted(logits, ids):
    """= M.ce (mean next-token CE over positions 0..L-2), via shifted labels with ignore_index as v162 / v167: the
    logits are viewed, not copied (M.ce's slice-then-reshape copies 1 GB of fp32 logits at batch 8)."""
    labels = torch.cat([ids[:, 1:], torch.full_like(ids[:, :1], -100)], 1)
    return torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100)


def train_loss(m, ids, k, amp=False, ckpt=False, state='randn', seed=0, cache=False):
    """Next-token loss after R loops with backprop through the last k (k = 0: no loop credit). One autocast region
    (fp16) with the weight-cast cache OFF: with it on, weights first cast inside the no-grad loops lose their grad
    connection in the later loops (v161). state='randn': a seeded N(0, 1/d) draw (training); 'hashed': the model's
    deterministic eval-mode state."""
    with torch.autocast('cuda', dtype=torch.float16, enabled=amp and DEV == 'cuda', cache_enabled=cache):
        h, ctx = m.prepare(ids, True, state=state, seed=seed)
        with torch.no_grad():
            for t in range(R - k):
                h = m.F(h, t, ctx)
        h = h.detach()
        for t in range(R - k, R):
            h = m.step(h, t, ctx, 1.0, ckpt=ckpt)
        logits = m.readout(h, ctx)
    return ce_shifted(logits, ids)


@torch.no_grad()
def eval_loss(m, chunks):
    tot, n = 0.0, 0
    for i in range(0, len(chunks), BATCH):
        ids = chunks[i:i + BATCH].to(DEV)
        tot += float(train_loss(m, ids, 0, amp=AMP, state='hashed')) * ids.shape[0]; n += ids.shape[0]
    return tot / n


def rho_of(m, chunks, n=4):
    """Reported only (as v167's rho_end): median over n val chunks of exp(-slope) of log |dL/dh_t|, fp32, batch 1."""
    rhos = []
    for i in range(min(n, len(chunks))):
        ids = chunks[i:i + 1].to(DEV); adj = {}
        m.zero_grad(set_to_none=True)
        h, ctx = m.prepare(ids, True)
        for t in range(R):
            h = m.step(h, t, ctx, 1.0, ckpt=True)
            h.register_hook(lambda g, t=t: adj.__setitem__(t, float(g.float().norm())))
        M.ce(m.readout(h, ctx), ids).backward()
        a = np.array([adj[t] for t in range(R)])
        rhos.append(float(np.exp(-np.polyfit(np.arange(R), np.log(a + 1e-30), 1)[0])))
    m.zero_grad(set_to_none=True)
    return float(np.median(rhos)), rhos


# --------------------------------------------------------------------------------------------------- selftest
def param_groups(m):
    g = dict(embed=list(m.embed.parameters()), prelude=[p for b in m.prelude_layers() for p in b.parameters()],
             injection=list(m.input_injection.parameters()), core=[p for b in m.core_layers() for p in b.parameters()],
             coda=[p for b in m.coda_layers() for p in b.parameters()], head=list(m.output.parameters()))
    if hasattr(m, 'huginn_prelude_norm_module'):
        g['prelude'] += list(m.huginn_prelude_norm_module.parameters())
    return g


def selftest(m, data, info, tag):
    """Before any training, in the training precision. Fails loudly (SystemExit) if any check fails."""
    rep, fails = dict(driver_sha=driver_sha(), port_sha=_sha(M.__file__), device=DEV, amp=AMP,
                      checkpoint=info.get('checkpoint_id')), []
    if rep['port_sha'] != PORT_SHA and not TINY and os.environ.get('V200_ALLOW_PORT_CHANGE') != '1':
        fails.append('port_changed_since_v198')
    nb = int(os.environ.get('V200_SELFTEST_BATCH', 2))
    ids = data['val'][:nb].to(DEV)
    G = param_groups(m)
    tol = 1e-5 if DEV == 'cpu' else 1e-4

    def grads(k, amp, ckpt, cache=False, scale=1024.0):
        m.zero_grad(set_to_none=True)
        loss = train_loss(m, ids, k, amp=amp, ckpt=ckpt, seed=7, cache=cache)
        (loss * (scale if amp else 1.0)).backward()
        out = {g: torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().float().flatten().cpu()
                             / (scale if amp else 1.0) for p in ps]) for g, ps in G.items()}
        m.zero_grad(set_to_none=True)
        return out, float(loss)

    def rel(h, f):
        return {g: float((h[g].double() - f[g].double()).norm() / (f[g].double().norm() + 1e-30)) for g in f}
    # 1. the training forward at k = R (fp32, eval state, no checkpointing) == the model's own forward
    with torch.no_grad():
        l1 = float(train_loss(m, ids, R, state='hashed')); l2 = float(M.ce(m(ids, R=R, docmask=True), ids))
    rep['decomposition'] = (l1, l2)
    if not abs(l1 - l2) <= 1e-5 * max(1.0, abs(l2)):
        fails.append('decomposition')
    # 2. per-block checkpointing changes nothing (fp32)
    for k in (2, R):
        d = rel(grads(k, False, True)[0], grads(k, False, False)[0]); rep['ckpt_vs_plain_k%d' % k] = d
        if max(d.values()) >= tol:
            fails.append('ckpt_k%d' % k)
    # 3. fp16 autocast (cache off) truncated gradients vs fp32, per group (v162 / v167); and the v161 negative control
    if AMP:
        for k in (1, 2, 4, R):
            (h, lh), (f, lf) = grads(k, True, True), grads(k, False, False)
            d = rel(h, f); rep['fp16_vs_fp32_k%d' % k] = dict(d, loss=(lh, lf))
            if not (d['core'] < 0.5 and d['core'] <= 3 * d['coda'] + 0.02 and d['injection'] < 0.5 and abs(lh - lf) < 0.01):
                fails.append('fp16_k%d' % k)
        h, _ = grads(2, True, False, cache=True); f, _ = grads(2, False, False)   # plain path, as v162's control
        rep['negative_control_cache_on_k2'] = d = rel(h, f)
        rep['torch'] = torch.__version__
        # The v161 bug needs autocast to cache weight casts made under no_grad. Kaggle's PyTorch (2026-10-14 pilot) does
        # not, so cache-on gradients came out correct there and this control could not fire. Then the check's
        # sensitivity is shown on a synthetic version of the same failure (the looped weights get no gradient, as v161's
        # did): the comparison must flag it. Where the bug does reproduce, that already shows the sensitivity.
        rep['negative_control_bug_reproduces'] = bool(d['core'] > 0.5)
        if not rep['negative_control_bug_reproduces'] or os.environ.get('V200_FORCE_SYNTHETIC_NEG') == '1':
            loop_ps = [p for g in ('injection', 'core') for p in G[g]]
            for p in loop_ps:
                p.requires_grad_(False)
            try:
                hs, _ = grads(2, True, True)
            finally:
                for p in loop_ps:
                    p.requires_grad_(True)
            rep['negative_control_synthetic_k2'] = ds = rel(hs, f)
            if not (ds['core'] > 0.9 and ds['injection'] > 0.9):   # the check must see missing loop gradients
                fails.append('negative_control')
    # 4. the floor: k = 0 gives everything upstream of the loop's output no gradient; k = 1 reaches all of it
    g0, _ = grads(0, False, False); g1, _ = grads(1, False, False)
    rep['floor_norms_k0'] = {g: float(v.norm()) for g, v in g0.items()}
    rep['floor_norms_k1'] = {g: float(v.norm()) for g, v in g1.items()}
    if any(rep['floor_norms_k0'][g] != 0.0 for g in ('embed', 'prelude', 'injection', 'core')) or \
            any(rep['floor_norms_k0'][g] == 0.0 for g in ('coda', 'head')) or \
            any(rep['floor_norms_k1'][g] == 0.0 for g in G if not (FREEZE_EMBED and g == 'embed')) or \
            (FREEZE_EMBED and rep['floor_norms_k1']['embed'] != 0.0):
        fails.append('floor')
    # 5. the training state draw is reproducible (same states for every arm of a seed)
    with torch.no_grad():
        a = m.prepare(ids, True, state='randn', seed=123)[0]; b = m.prepare(ids, True, state='randn', seed=123)[0]
    if not torch.equal(a, b):
        fails.append('state_draw')
    rep['trainable_at_floor'] = sum(p.numel() for g in ('coda', 'head') for p in G[g])
    rep['freeze_embed'] = FREEZE_EMBED
    rep['trainable'] = sum(p.numel() for p in m.parameters() if p.requires_grad)
    rep['fails'], rep['status'] = fails, 'PASS' if not fails else 'FAIL'
    json.dump(rep, open(os.path.join(OUT, 'v200_selftest_%s.json' % tag), 'w'), indent=1, default=str)
    show = {k: v for k, v in rep.items() if k.startswith(('fp16', 'negative', 'ckpt')) and isinstance(v, dict)}
    log('selftest %s %s | decomposition %.6f / %.6f | %s | floor k0 %s' % (
        rep['status'], fails, l1, l2, {k: {g: round(x, 4) for g, x in v.items() if g in ('core', 'coda', 'injection', 'prelude', 'embed', 'head')}
                                     for k, v in show.items()}, {g: '%.2e' % v for g, v in rep['floor_norms_k0'].items()}))
    if fails:
        raise SystemExit('selftest FAILED: %s' % fails)


# ------------------------------------------------------------------------------------------------------ train
def train_cell(data, k, lr, seed, weights, variant):
    t0 = time.time()
    m, info = load_model(weights, variant)
    g = np.random.default_rng(seed)
    chunks = torch.cat([pack(data['train'], g.permutation(len(data['train']))) for _ in range(4)])
    assert len(chunks) >= STEPS * BATCH, (len(chunks), STEPS * BATCH)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr, betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=0.0)
    sched = lambda s: min(1.0, (s + 1) / WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / STEPS)))
    scaler = torch.amp.GradScaler('cuda', enabled=AMP)
    curve, nonfinite = [], 0
    if DEV == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    for s in range(STEPS):
        for gp in opt.param_groups:
            gp['lr'] = lr * sched(s)
        ids = chunks[s * BATCH:(s + 1) * BATCH].to(DEV)
        loss = train_loss(m, ids, k, amp=AMP, ckpt=k > 0, seed=seed * 100_003 + s)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP)
        if not torch.isfinite(gn):
            nonfinite += 1
        scaler.step(opt); scaler.update()
        if s % 50 == 0 or s == STEPS - 1:
            curve.append((s, round(float(loss.detach()), 4), round(float(gn), 3)))
    t_train = time.time() - t0
    m.eval()
    val, test = eval_loss(m, data['val']), eval_loss(m, data['test'])
    rho_end = rho_of(m, data['val'])[0] if math.isfinite(val) else float('nan')
    return dict(variant=variant, R=R, a=1.0, k=k, lr=lr, seed=seed, val=val, test=test, rho_end=rho_end, curve=curve,
                freeze_embed=FREEZE_EMBED,
                nonfinite_steps=nonfinite, diverged=not (math.isfinite(val) and math.isfinite(test)), s=round(time.time() - t0),
                s_train=round(t_train), steps=STEPS, batch=BATCH, seq=SEQ, amp=AMP, docmask=True,
                peak_gb=torch.cuda.max_memory_allocated() / 1e9 if DEV == 'cuda' else 0.0, driver_sha=driver_sha(),
                port_sha=_sha(M.__file__), checkpoint=info.get('checkpoint_id'), manifest=info.get('manifest_sha256'))


def est_seconds(k):
    return EST.get(k, EST[1] + (EST[R] - EST[1]) * (k - 1) / (R - 1))


def split_cells(cells, part):
    """Greedy balance of cells over n parts by estimated time; returns part i's cells (in their original order)."""
    i, n = (int(x) for x in part.split('/'))
    bins = [[0.0, []] for _ in range(n)]
    for c in sorted(cells, key=lambda c: -est_seconds(int(c.split(':')[0]))):
        b = min(bins, key=lambda b: b[0]); b[0] += est_seconds(int(c.split(':')[0])); b[1].append(c)
    return [c for c in cells if c in bins[i][1]]


def pilot_lr_cells():
    return ['%d:%g:%d' % (k, lr, PILOT_SEED) for k in (R, 0) for lr in LR_GRID]


# --------------------------------------------------------------------------------------------------- analysis
def load_runs(paths):
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, '**', 'v200_runs_*.jsonl'), recursive=True)) if os.path.isdir(p) else [p]
    runs = {}
    for f in files:
        for line in open(f):
            r = json.loads(line); key = (r['variant'], int(r['k']), float(r['lr']), int(r['seed']))
            if key in runs:
                print('duplicate ignored', key); continue
            runs[key] = r
    return runs, files


def analyze(paths, lrs=None):
    from scipy import stats
    runs, files = load_runs(paths)
    print('%d runs from %d files' % (len(runs), len(files)))
    for v in sorted({key[0] for key in runs}):
        rs = {key[1:]: r for key, r in runs.items() if key[0] == v}
        print('\n=== %s: %d runs (drivers %s)' % (v, len(rs), sorted({r['driver_sha'] for r in rs.values()})))
        for (k, lr, s), r in sorted(rs.items()):
            print('  k=%d lr=%-7g seed=%-8d val %.4f test %.4f | nonfinite %3d | rho_end %.3f | %4ds (train %4ds) peak %.1f GB%s' % (
                k, lr, s, r['val'], r['test'], r['nonfinite_steps'], r['rho_end'], r['s'], r.get('s_train', 0), r['peak_gb'],
                ' DIVERGED' if r['diverged'] else ''))
        for k in (R, 0):                           # LR pilot table: pick by val loss (v167)
            grid = sorted({lr for (kk, lr, s) in rs if kk == k and s == PILOT_SEED})
            if len(grid) > 1:
                best = min(grid, key=lambda lr: rs[(k, lr, PILOT_SEED)]['val'])
                print('LR pilot %s (seed %d): %s -> best by val %g%s' % (
                    'full' if k == R else 'floor', PILOT_SEED, '  '.join('%g: val %.4f test %.4f' % (
                        lr, rs[(k, lr, PILOT_SEED)]['val'], rs[(k, lr, PILOT_SEED)]['test']) for lr in grid), best,
                    ' (EDGE of the grid: extend it before fixing the LR, as v167 pilot 2)' if best in (grid[0], grid[-1]) else ''))
        if lrs:
            lr_f, lr_t = (float(x) for x in lrs.split(':'))
        else:                                      # the most common LR per arm
            pick = lambda ks: max({lr for (kk, lr, s) in rs if kk in ks}, default=None,
                                  key=lambda lr: sum(1 for (kk, l2, s) in rs if kk in ks and l2 == lr))
            lr_f, lr_t = pick((0,)), pick((1, 2, 4, R))
        if lr_f is None or lr_t is None:
            continue
        seeds = sorted({s for (kk, lr, s) in rs if kk == R and lr == lr_t})
        valid = []
        for s in seeds:
            f = rs[(R, lr_t, s)]
            ok = f['nonfinite_steps'] <= MAX_NONFINITE and math.isfinite(f['test']) and not f['diverged']
            print('seed %d: full nonfinite %d test %.4f -> %s' % (s, f['nonfinite_steps'], f['test'], 'VALID' if ok else 'EXCLUDED'))
            if ok and (0, lr_f, s) in rs:
                valid.append(s)
        valid = valid[:N_USE]
        print('floor lr %g, truncated/full lr %g; seeds used: %s' % (lr_f, lr_t, valid))
        if len(valid) < 2:
            continue
        for split in ('val', 'test'):
            gap = [rs[(0, lr_f, s)][split] - rs[(R, lr_t, s)][split] for s in valid]
            mg, sd = float(np.mean(gap)), float(np.std(gap, ddof=1))
            h = stats.t.ppf(0.95, len(gap) - 1) * sd / math.sqrt(len(gap))
            print('%-4s floor gap L_floor - L_full: %.4f, 90%% CI [%.4f, %.4f] (n %d; per seed %s)' % (
                split, mg, mg - h, mg + h, len(gap), ' '.join('%.4f' % x for x in gap)))
            for k in (1, 2, 4):
                ss = [s for s in valid if (k, lr_t, s) in rs]
                if not ss:
                    continue
                ph = [(rs[(k, lr_t, s)][split] - rs[(R, lr_t, s)][split]) /
                      (rs[(0, lr_f, s)][split] - rs[(R, lr_t, s)][split]) for s in ss]
                se = float(np.std(ph, ddof=1) / math.sqrt(len(ph))) if len(ph) > 1 else float('nan')
                sd = float(np.std(ph, ddof=1)) if len(ph) > 1 else float('nan')
                print('  %s phi k=%d: %.4f +- %.4f s.e. (n %d; sd %.4f -> s.e. with 5 seeds %.4f) per seed %s' % (
                    split, k, float(np.mean(ph)), se, len(ph), sd, sd / math.sqrt(5), ' '.join('%.4f' % x for x in ph)))


# ------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='huginn-s-fixed-pln5', choices=sorted(VARIANTS))
    ap.add_argument('--weights', default=None); ap.add_argument('--tag', default='x')
    ap.add_argument('--selftest', action='store_true'); ap.add_argument('--pilot-lr', action='store_true')
    ap.add_argument('--part', default='0/1'); ap.add_argument('--train', default=None)
    ap.add_argument('--analyze', nargs='*'); ap.add_argument('--lrs', default=None)
    A = ap.parse_args()
    if A.analyze is not None:
        return analyze(A.analyze or [OUT], A.lrs)
    if not TINY and not A.weights:
        raise SystemExit('--weights DIR is required (or TINY=1)')
    tok = None if TINY else M.Tokenizer(A.weights)
    data = gsm8k_chunks(tok)
    log('v200 driver %s (port %s) | %s @ %s | dev %s amp %s | R %d | train ex %d, val/test chunks %d/%d, %d steps x %d x %d' % (
        driver_sha(), _sha(M.__file__), A.variant, VARIANTS[A.variant][:8], DEV, AMP, R, len(data['train']),
        len(data['val']), len(data['test']), STEPS, BATCH, SEQ))
    if A.selftest:
        m, info = load_model(A.weights, A.variant)
        selftest(m, data, info, '%s_%s' % (short(A.variant), A.tag))
        if A.tag.startswith('step0') or A.pilot_lr:          # the frozen model's val / test loss (reference)
            m.eval(); v0, t0 = eval_loss(m, data['val']), eval_loss(m, data['test'])
            log('step-0 (frozen model) val %.4f test %.4f' % (v0, t0))
            json.dump(dict(variant=A.variant, val=v0, test=t0), open(os.path.join(OUT, 'v200_step0_%s.json' % short(A.variant)), 'w'))
        del m
        if DEV == 'cuda':
            torch.cuda.empty_cache()
    cells = split_cells(pilot_lr_cells(), A.part) if A.pilot_lr else (A.train.split(',') if A.train else [])
    for cell in cells:
        k, lr, seed = cell.split(':'); k, lr, seed = int(k), float(lr), int(seed)
        r = train_cell(data, k, lr, seed, A.weights, A.variant)
        log('cell %s k=%d lr=%g seed=%d: val %.4f test %.4f rho_end %.3f nonfinite %d (%ds, train %ds, peak %.1f GB)' % (
            short(A.variant), k, lr, seed, r['val'], r['test'], r['rho_end'], r['nonfinite_steps'], r['s'], r['s_train'], r['peak_gb']))
        open(os.path.join(OUT, 'v200_runs_%s_%s.jsonl' % (short(A.variant), A.tag)), 'a').write(json.dumps(r) + '\n')
        if DEV == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
