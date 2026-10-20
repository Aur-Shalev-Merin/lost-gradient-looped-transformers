"""v167: truncated-loop training on the healed looped SmolLM2 retrofit (raven/Huginn architecture, split (11, 8, 11),
healing run 1; docs/retrofit_heal_nodrop_2026-10-02.md). Opposite-direction test of "truncation cost is one function
of the frozen-model gradient error": this model's loop forgets fast (rho ~ 0.35), so the Parcae-fitted error curve
predicts ~0 cost at k = 1-2 where "k alone" (fitted on Parcae) predicts 0.21 / 0.12.

Recipe = v162-v166 (GSM8K question+answer text, 600 steps, batch 8 x 512 tokens, AdamW betas (0.9, 0.95), 20-step
warm-up + cosine to 10%, clip 1.0, fp16 autocast with cache_enabled=False + GradScaler on T4). Fixed loop count R = 8
(the healing mean); backprop through the last k loops (k = 0: floor, the loop gets no gradient; k = R: full).

The forward is the probe's (retrofit_rho_probe.manual_loss: prelude -> state init -> R x core_block_forward -> coda ->
ln_f -> head), checked against the model's own forward in --selftest, with optional per-loop gradient checkpointing.

  --selftest            decomposition check, fp16 vs fp32 truncated grads per group, checkpoint == plain, k 0 floor
  --measure             frozen-model err_k (k 1, 2, 4) and rho at R 8 for a in MEASURE_A: 16 GSM8K-val sequences, fp32,
                        all parameters; step-0 loss at R 8 and 16 (does the damped loop reach the same function?)
  --train CELLS --tag T cells 'a:k:lr:seed'
Env: RETRO_DIR (healed model dir; copied to a writable place first), TINY=1 (CPU pipeline test), OUT_DIR.
"""
import argparse, hashlib, json, math, os, shutil, time
import numpy as np
import torch
import torch.nn.functional as Fn
from torch.utils.checkpoint import checkpoint

TINY = os.environ.get('TINY') == '1'
SRC_DIR = os.environ.get('RETRO_DIR', 'retro_healed')
R = 8
SEQ = 64 if TINY else 512
BATCH = 2 if TINY else 8
STEPS = int(os.environ.get('V167_STEPS', 3 if TINY else 600))
WARMUP = 2 if TINY else 20
CLIP = 1.0
N_VAL = 473                      # last 473 GSM8K train examples = validation; the rest = training (as v162)
N_MEASURE = 2 if TINY else 16
KS = (1, 2, 4)
MEASURE_A = (1.0, 0.5, 0.3, 0.2)       # damping values measured pre-freeze
OUT = os.environ.get('OUT_DIR', '/kaggle/working' if os.path.isdir('/kaggle/working') else '.')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
AMP = DEV == 'cuda' and os.environ.get('V167_FP32') != '1'


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v167_log.txt'), 'a').write(s + '\n')


def driver_sha():
    return hashlib.sha256(open(os.path.abspath(__file__), 'rb').read().replace(b'\r\n', b'\n')).hexdigest()[:12]


# ------------------------------------------------------------------------------------------------------------ model
_LOCAL = None


def load_model():
    global _LOCAL
    from transformers import AutoModelForCausalLM
    if _LOCAL is None:                    # the input dir may be read-only; config.json needs the duplicate key removed
        _LOCAL = os.path.join(OUT, 'retro_model'); shutil.copytree(SRC_DIR, _LOCAL, dirs_exist_ok=True)
        cp = os.path.join(_LOCAL, 'config.json'); c = json.load(open(cp))
        if 'tie_word_embeddings' in c:    # save_pretrained re-adds it; raven's config passes it itself (retrofit_heal.py)
            assert not c['tie_word_embeddings'] and not c.get('tie_embeddings', False)
            c.pop('tie_word_embeddings'); json.dump(c, open(cp, 'w'), indent=2)
    m = AutoModelForCausalLM.from_pretrained(_LOCAL, trust_remote_code=True, torch_dtype=torch.float32).to(DEV)
    m.config.test_time_noise = 0
    return m


def load_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M')


def labels_of(ids):
    return torch.cat([ids[:, 1:], torch.full_like(ids[:, :1], -100)], 1)


def manual_loss(m, ids, k, amp=False, ckpt=False, adj=None, r=None, a=1.0):
    """Loss with backprop through the last k of R loops (retrofit_rho_probe.manual_loss + autocast + checkpointing).
    Damped loop (a < 1): x <- (1 - a) x + a F(x), as v163 on Parcae; a = 1 is the model's own loop."""
    r = R if r is None else r
    with torch.autocast('cuda', dtype=torch.float16, enabled=amp and DEV == 'cuda', cache_enabled=False):
        pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        e = m.transformer.wte(ids)
        if m.emb_scale != 1:
            e = e * m.emb_scale
        freqs = m.rotary_emb(e, pos)
        bi = torch.tensor(-1, device=torch.device('cpu'), dtype=torch.long)
        for block in m.transformer.prelude:
            bi += 1
            e = block(e, freqs, bi, None, None)
        x = m.initialize_state(e, scale=1.0)
        with torch.no_grad():
            for t in range(r - k):
                y, bi = m.core_block_forward(x, e, freqs, None, None, bi, t)
                x = y if a == 1.0 else (1 - a) * x + a * y
        x = x.detach()
        for t in range(r - k, r):
            if ckpt:      # block_idx is mutated in place: give each (re)computation its own copy
                b0 = bi.clone()
                y = checkpoint(lambda x_, e_, b0=b0, t=t: m.core_block_forward(x_, e_, freqs, None, None, b0.clone(), t)[0],
                               x, e, use_reentrant=False)
                bi = bi + len(m.transformer.core_block)
            else:
                y, bi = m.core_block_forward(x, e, freqs, None, None, bi, t)
            x = y if a == 1.0 else (1 - a) * x + a * y
            if adj is not None:
                x.register_hook(lambda g, t=t: adj.__setitem__(t, float(g.float().norm())))
        bi = torch.tensor(0, device=torch.device('cpu'), dtype=torch.long)
        for block in m.transformer.coda:
            bi -= 1
            x = block(x, freqs, bi, None, None)
        logits = m.lm_head(m.transformer.ln_f(x)).float()
    return Fn.cross_entropy(logits.view(-1, logits.shape[-1]), labels_of(ids).view(-1), ignore_index=-100)


# ------------------------------------------------------------------------------------------------------------- data
def gsm8k_chunks(tok):
    if TINY:
        g = np.random.default_rng(0)
        mk = lambda n: [list(g.integers(1, 1000, size=g.integers(20, 60))) for _ in range(n)]
        tr, va, te = mk(60), mk(10), mk(10)
    else:
        from datasets import load_dataset
        enc = lambda ds: [tok(ex['question'] + '\n' + ex['answer']).input_ids + [tok.eos_token_id] for ex in ds]
        all_tr = enc(load_dataset('openai/gsm8k', 'main', split='train'))
        tr, va = all_tr[:-N_VAL], all_tr[-N_VAL:]
        te = enc(load_dataset('openai/gsm8k', 'main', split='test'))
    return dict(train=tr, val=pack(va), test=pack(te))


def pack(examples, order=None):
    flat = [t for i in (order if order is not None else range(len(examples))) for t in examples[i]]
    n = len(flat) // SEQ
    return torch.tensor(flat[: n * SEQ], dtype=torch.long).view(n, SEQ)


@torch.no_grad()
def eval_loss(m, chunks, a=1.0, r=None):
    m.eval(); tot, n = 0.0, 0
    for i in range(0, len(chunks), BATCH):
        ids = chunks[i:i + BATCH].to(DEV); torch.manual_seed(12345 + i)        # state-init noise fixed across arms
        tot += float(manual_loss(m, ids, 0, amp=AMP, a=a, r=r)) * ids.shape[0]; n += ids.shape[0]
    m.train()
    return tot / n


# ---------------------------------------------------------------------------------------------------------- measure
def measure(m, chunks, ks=KS, n=None, a=1.0):
    params = [p for p in m.parameters() if p.requires_grad]
    grad = lambda: torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in params])
    m.eval(); rows = []
    for i in range(min(n or N_MEASURE, len(chunks))):
        ids = chunks[i:i + 1].to(DEV); adj = {}
        torch.manual_seed(777 + i); m.zero_grad(set_to_none=True)
        manual_loss(m, ids, R, adj=adj, a=a).backward(); gR = grad()
        adjn = np.array([adj[t] for t in range(R)])
        row = dict(rho=float(np.exp(-np.polyfit(np.arange(R), np.log(adjn + 1e-30), 1)[0])), adj=adjn.tolist(), err={})
        for k in ks:
            torch.manual_seed(777 + i); m.zero_grad(set_to_none=True)
            manual_loss(m, ids, k, a=a).backward()
            row['err'][k] = float((grad() - gR).norm() / gR.norm())
        rows.append(row)
    m.zero_grad(set_to_none=True); m.train()
    return dict(R=R, a=a, rho_median=float(np.median([r['rho'] for r in rows])),
                err_median={k: float(np.median([r['err'][k] for r in rows])) for k in ks}, rows=rows)


# ------------------------------------------------------------------------------------------------------------ train
def train_cell(data, a, k, lr, seed):
    t0 = time.time(); m = load_model(); m.train()
    g = np.random.default_rng(seed)
    chunks = torch.cat([pack(data['train'], g.permutation(len(data['train']))) for _ in range(4)])
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    sched = lambda s: min(1.0, (s + 1) / WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / STEPS)))
    scaler = torch.amp.GradScaler('cuda', enabled=AMP)
    curve, nonfinite = [], 0
    if DEV == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    for s in range(STEPS):
        for gp in opt.param_groups:
            gp['lr'] = lr * sched(s)
        ids = chunks[s * BATCH:(s + 1) * BATCH].to(DEV)
        loss = manual_loss(m, ids, k, amp=AMP, ckpt=k > 0, a=a)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP)
        if not torch.isfinite(gn):
            nonfinite += 1
        scaler.step(opt); scaler.update()
        if s % 50 == 0 or s == STEPS - 1:
            curve.append((s, round(float(loss.detach()), 4), round(float(gn), 3)))
    val, test = eval_loss(m, data['val'], a), eval_loss(m, data['test'], a)
    rho_end = measure(m, data['val'], ks=(), n=4, a=a)['rho_median'] if math.isfinite(val) else float('nan')
    return dict(R=R, a=a, k=k, lr=lr, seed=seed, val=val, test=test, rho_end=rho_end, curve=curve, nonfinite_steps=nonfinite,
                diverged=not (math.isfinite(val) and math.isfinite(test)), s=round(time.time() - t0), steps=STEPS,
                batch=BATCH, seq=SEQ, amp=AMP, peak_gb=torch.cuda.max_memory_allocated() / 1e9 if DEV == 'cuda' else 0.0,
                driver_sha=driver_sha())


# --------------------------------------------------------------------------------------------------------- selftest
def selftest(m, data):
    ids = data['val'][:1].to(DEV)
    for r in (1, 4, R):           # 1. written-out forward == the model's own forward (fp32)
        torch.manual_seed(0); a = float(m(ids, labels=labels_of(ids), num_steps=r).loss)
        torch.manual_seed(0); b = float(manual_loss(m, ids, 0, r=r))
        log('selftest decomposition R=%d: model %.6f manual %.6f' % (r, a, b))
        assert abs(a - b) <= 1e-4 * max(1.0, abs(a)), 'decomposition mismatch'
    groups = {}
    for n, p in m.named_parameters():
        groups.setdefault(next((x for x in ('adapter', 'core_block', 'prelude', 'coda') if x in n), 'other'), []).append(p)

    def grads(k, amp, ckpt, a=1.0):
        m.zero_grad(set_to_none=True); torch.manual_seed(0)
        manual_loss(m, data['val'][:2].to(DEV), k, amp=amp, ckpt=ckpt, a=a).backward()
        return {g: torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten().double().cpu() for p in ps])
                for g, ps in groups.items()}
    rel = lambda h, f: {g: float((h[g] - f[g]).norm() / (f[g].norm() + 1e-30)) for g in f}
    # 2. per-loop checkpointing changes nothing (fp32)
    d = rel(grads(4, False, True), grads(4, False, False)); log('selftest checkpointed vs plain k=4: %s' % {g: '%.1e' % v for g, v in d.items()})
    assert max(d.values()) < 1e-4, 'checkpointing changes the gradient'
    d = rel(grads(4, False, True, 0.5), grads(4, False, False, 0.5)); log('selftest damped a=0.5 checkpointed vs plain: %s' % {g: '%.1e' % v for g, v in d.items()})
    assert max(d.values()) < 1e-4, 'checkpointing changes the damped gradient'
    # 3. fp16 autocast truncated grads vs fp32 (the v161 cache bug puts the looped core near 1.0)
    if AMP:
        for k in (1, 4):
            d = rel(grads(k, True, True), grads(k, False, False)); log('selftest fp16 vs fp32 k=%d: %s' % (k, {g: round(v, 4) for g, v in d.items()}))
            assert d['core_block'] < 0.5 and d['core_block'] <= 3 * d['coda'] + 0.02, 'looped-core gradient lost under fp16 autocast'
    # 4. floor: k = 0 gives the loop no gradient; k = 1 does
    g0, g1 = grads(0, False, False), grads(1, False, False)
    log('selftest floor: core grad norm k=0 %.3e, k=1 %.3e' % (g0['core_block'].norm(), g1['core_block'].norm()))
    assert g0['core_block'].norm() == 0 and g1['core_block'].norm() > 0
    m.zero_grad(set_to_none=True)
    log('selftest OK')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true'); ap.add_argument('--measure', action='store_true')
    ap.add_argument('--train', default=None); ap.add_argument('--tag', default='x')
    A = ap.parse_args()
    tok = load_tok(); data = gsm8k_chunks(tok); m = load_model()
    log('v167 driver %s dev %s amp %s | params %.1fM | train ex %d, val/test chunks %d/%d' % (
        driver_sha(), DEV, AMP, sum(p.numel() for p in m.parameters()) / 1e6, len(data['train']), len(data['val']), len(data['test'])))
    if A.selftest:
        selftest(m, data)
    if A.measure:
        res = []
        for a in MEASURE_A:
            r = measure(m, data['val'], a=a)
            r['step0'] = {rr: (eval_loss(m, data['val'], a, rr), eval_loss(m, data['test'], a, rr)) for rr in (R, 2 * R)}
            log('measure R=%d a=%.2f: rho %.4f err %s | step-0 val/test R%d %.4f / %.4f, R%d %.4f / %.4f' % (
                R, a, r['rho_median'], {k: round(x, 4) for k, x in r['err_median'].items()}, R, *r['step0'][R], 2 * R, *r['step0'][2 * R]))
            res.append(r)
        json.dump(dict(driver_sha=driver_sha(), res=res), open(os.path.join(OUT, 'v167_measure_%s.json' % A.tag), 'w'))
    if A.train:
        del m
        for cell in A.train.split(','):
            a, k, lr, seed = cell.split(':'); a, k, lr, seed = float(a), int(k), float(lr), int(seed)
            r = train_cell(data, a, k, lr, seed)
            log('cell R=%d a=%.2f k=%d lr=%g seed=%d: val %.4f test %.4f rho_end %.3f nonfinite %d (%ds, peak %.1f GB)' % (
                R, a, k, lr, seed, r['val'], r['test'], r['rho_end'], r['nonfinite_steps'], r['s'], r['peak_gb']))
            open(os.path.join(OUT, 'v167_runs_%s.jsonl' % A.tag), 'a').write(json.dumps(r) + '\n')
            if DEV == 'cuda':
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
