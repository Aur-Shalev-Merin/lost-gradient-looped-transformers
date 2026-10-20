"""v162: does the per-loop adjoint decay rho, measured on the FROZEN model, predict what truncated backprop through
the loops costs in TRAINING? (prereg: docs/prereg_v162_parcae_truncation_training_rerun.md)

RE-RUN OF v161 (INVALID). v161 trained under torch.autocast with the default weight-cast cache. Parcae runs the no-grad
loops first, so the fp16 copy of each looped-core weight was cached inside no_grad without a grad connection, and the
later grad loops reused it: in every truncated arm the looped core got almost no gradient (docs/v161_...md). The only
changes from the v161 driver: autocast(cache_enabled=False) in loss_model, and a selftest that compares the truncated
gradient (training path, fp16 autocast) with fp32 at several (R, k) and aborts if they disagree.

Model: Parcae-140m (SandyResearch/parcae-140m; prelude 2 + looped core 2 + coda 2 layers, d 768, diagonal injection,
pretrained with ~Poisson(8) loops and backprop through the last <= 4). Continued training on GSM8K (question + answer,
packed into 512-token chunks) with the loop run R times and backprop through the last k loops only:
    num_steps_pair = (R - k, k)   -- Parcae's own forward: the first R-k loops under no_grad, the last k with grad.
k = 0 is the no-loop-credit floor (loop, prelude and injection get no gradient; C, coda, head, embeddings still train),
k = R is full backprop. Cells: R = 8: k in {0,1,2,4,8}; R = 16: k in {0,1,2,4,8,16}. Evaluation always runs all R
loops forward; only the backward pass is truncated.

Modes
  --selftest            manual decomposition == model forward; fp16-autocast loss vs fp32; FD-free gradient sanity
                        (k = R equals the gradient of the plain forward). TINY=1 for a random tiny model on CPU.
  --measure R,..        FROZEN model: per sequence (16 GSM8K-val chunks) gradient of all params with backprop through
                        the last k loops vs all R; relative error; adjoint norms ||dL/dh_t||; rho = exp(-slope) of a
                        log-linear fit over t. Pre-freeze (frozen probe) and, post-hoc, on the full arm's end state.
  --train CELLS         CELLS = 'R:k:lr:seed,...'; one JSON line per run.
  --report FILES        merges run files (analysis per the prereg is in v162_analyze.py, written before the study).
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as Fn

TINY = os.environ.get('TINY') == '1'
REPO = 'SandyResearch/parcae-140m'
SEQ = 64 if TINY else 512
BATCH = 2 if TINY else 8
STEPS = int(os.environ.get('V162_STEPS', 6 if TINY else 600))
WARMUP = 2 if TINY else 20
CLIP = 1.0
N_VAL = 473                      # last 473 GSM8K train examples = validation (LR selection); first 7000 = training
N_MEASURE = 4 if TINY else 16    # sequences for the frozen-model measurement
OUT = os.environ.get('OUT_DIR', '/kaggle/working' if os.path.isdir('/kaggle/working') else '.')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
AMP = DEV == 'cuda' and os.environ.get('V162_FP32') != '1'   # fp16 autocast on T4 (no fast bf16); checked in selftest


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v162_log.txt'), 'a').write(s + '\n')


def driver_sha():
    import hashlib
    return hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()[:12]


# ----------------------------------------------------------------------------------------------------------- model
def load_model():
    sys.path.insert(0, os.environ.get('PARCAE_SRC', 'parcae_src'))
    from huggingface_hub import hf_hub_download
    from transformers import PreTrainedTokenizerFast
    from tokenizers.processors import TemplateProcessing
    from parcae_lm.models.config import RoPESettings
    from parcae_lm.models.parcae import Parcae as ParcaeModel, ParcaeConfig
    tok = PreTrainedTokenizerFast(tokenizer_file=hf_hub_download('SandyResearch/parcae-tokenizer', 'tokenizer.json'))
    tok._tokenizer.post_processor = TemplateProcessing(single='<|bos|> $A', special_tokens=[('<|bos|>', 0)])
    cfg = json.load(open(hf_hub_download(REPO, 'config.json')))
    for key in ('_class_name', 'init'):
        cfg.pop(key, None)
    cfg['rope_settings'] = RoPESettings(**cfg['rope_settings'])
    if TINY:
        cfg.update(n_embd=64, intermediate_size=128, num_attention_heads=2, num_key_value_heads=2, n_layers_in_prelude=1,
                   n_layers_in_recurrent_block=1, n_layers_in_coda=1, recurrent_embedding_dimension=64,
                   recurrent_intermediation_embedding_dimension=128, block_size=256)
    model = ParcaeModel(ParcaeConfig(**cfg))
    if not TINY:
        sd = torch.load(hf_hub_download(REPO, 'pytorch_model.bin'), map_location='cpu', weights_only=True)
        sd = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (missing[:5], unexpected[:5])
    freqs = model.freqs_cis
    model = model.to(DEV).float()
    model.freqs_cis = freqs.to(DEV)              # rope cache stays in its original precision
    return tok, model


def labels_of(ids):
    return torch.cat([ids[:, 1:], torch.full_like(ids[:, :1], -100)], 1)


def loss_model(model, ids, R, k):
    """Parcae's own forward with backprop through the last k of R loops."""
    steps = torch.tensor([R - k, k])
    # cache_enabled=False: with the cache, weights first cast inside the no-grad loops lose their grad connection (v161)
    with torch.autocast('cuda', dtype=torch.float16, enabled=AMP, cache_enabled=False):
        return model(ids, labels=labels_of(ids), num_steps_pair=steps)['loss'].float()


def loss_manual(model, ids, R, k, adj=None):
    """Same computation, written out (prelude -> state init -> R loops -> C -> coda -> ln_f -> head), so that the
    adjoint dL/dh_t can be hooked. Must equal loss_model (checked in selftest)."""
    m = model; freqs = m.freqs_cis[:, : ids.shape[1]]; m._current_input_ids = ids
    ve = lambda i: m.value_embeds[str(i)](ids) if str(i) in m.value_embeds else None
    e = m.transformer.wte(ids)
    if m.emb_scale != 1:
        e = e * m.emb_scale
    for i, block in enumerate(m.transformer.prelude):
        e = block(e, freqs, None, ve=ve(i))
    if m.config.prelude_norm:
        e = m.transformer.ln_prelude(e)
    x = m.initialize_state(e); tot = torch.tensor(R)
    with torch.no_grad():
        for t in range(R - k):
            x = m.core_block_forward(x, e, freqs, None, torch.tensor(t), tot)
    for t in range(R - k, R):
        x = m.core_block_forward(x, e, freqs, None, torch.tensor(t), tot)
        if adj is not None and x.requires_grad:
            x.register_hook(lambda g, t=t: adj.__setitem__(t, float(g.float().norm())))
    x = m.transformer.C(x); off = m.config.n_layers_in_prelude + m.config.n_layers_in_recurrent_block
    for i, block in enumerate(m.transformer.coda):
        x = block(x, freqs, None, ve=ve(off + i))
    logits = m.lm_head(m.transformer.ln_f(x)).float() * m.config.init.logit_scale
    if m.config.logit_softcap is not None:
        logits = m.config.logit_softcap * torch.tanh(logits / m.config.logit_softcap)
    return Fn.cross_entropy(logits.view(-1, logits.shape[-1]), labels_of(ids).view(-1), ignore_index=-100)


# ------------------------------------------------------------------------------------------------------------ data
def gsm8k_chunks(tok):
    """Returns dict(train=[list of token-id lists, one per example], val=LongTensor chunks, test=LongTensor chunks)."""
    if TINY:
        g = np.random.default_rng(0)
        mk = lambda n: [list(g.integers(1, 1000, size=g.integers(20, 60))) for _ in range(n)]
        tr, va, te = mk(60), mk(10), mk(10)
    else:
        from datasets import load_dataset
        enc = lambda ds: [tok(ex['question'] + '\n' + ex['answer']).input_ids for ex in ds]
        train = load_dataset('openai/gsm8k', 'main', split='train'); test = load_dataset('openai/gsm8k', 'main', split='test')
        all_tr = enc(train); tr, va = all_tr[:-N_VAL], all_tr[-N_VAL:]; te = enc(test)
    return dict(train=tr, val=pack(va), test=pack(te))


def pack(examples, order=None):
    flat = [t for i in (order if order is not None else range(len(examples))) for t in examples[i]]
    n = len(flat) // SEQ
    return torch.tensor(flat[: n * SEQ], dtype=torch.long).view(n, SEQ)


@torch.no_grad()
def eval_loss(model, chunks, R):
    model.eval(); tot, n = 0.0, 0
    for i in range(0, len(chunks), BATCH):
        ids = chunks[i:i + BATCH].to(DEV)
        torch.manual_seed(12345 + i)                 # state init noise fixed across arms
        tot += float(loss_model(model, ids, R, 0)) * ids.shape[0]; n += ids.shape[0]
    model.train()
    return tot / n


# ----------------------------------------------------------------------------------------------------------- train
def train_cell(tok, data, R, k, lr, seed):
    t0 = time.time()
    _, model = load_model(); model.train()
    g = np.random.default_rng(seed)
    chunks = torch.cat([pack(data['train'], g.permutation(len(data['train']))) for _ in range(4)])   # 4 shuffled epochs
    assert len(chunks) >= STEPS * BATCH, (len(chunks), STEPS * BATCH)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    sched = lambda s: min(1.0, (s + 1) / WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / STEPS)))
    scaler = torch.amp.GradScaler('cuda', enabled=AMP)
    curve, nonfinite = [], 0
    for s in range(STEPS):
        for gp in opt.param_groups:
            gp['lr'] = lr * sched(s)
        ids = chunks[s * BATCH:(s + 1) * BATCH].to(DEV)
        loss = loss_model(model, ids, R, k)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        if not torch.isfinite(gn):
            nonfinite += 1
        scaler.step(opt); scaler.update()
        if s % 50 == 0 or s == STEPS - 1:
            curve.append((s, round(float(loss), 4), round(float(gn), 3)))
    val, test = eval_loss(model, data['val'], R), eval_loss(model, data['test'], R)
    return dict(R=R, k=k, lr=lr, seed=seed, val=val, test=test, curve=curve, nonfinite_steps=nonfinite,
                diverged=not (math.isfinite(val) and math.isfinite(test)), s=round(time.time() - t0),
                steps=STEPS, batch=BATCH, seq=SEQ, amp=AMP)


# --------------------------------------------------------------------------------------------------------- measure
def measure(model, chunks, R, ks):
    """Frozen-model truncation error and adjoint decay, per sequence (batch 1), all trainable parameters, fp32."""
    params = [p for p in model.parameters() if p.requires_grad]
    model.eval(); rows = []
    for i in range(min(N_MEASURE, len(chunks))):
        ids = chunks[i:i + 1].to(DEV); adj = {}
        torch.manual_seed(777 + i)
        model.zero_grad(set_to_none=True)
        loss = loss_manual(model, ids, R, R, adj); loss.backward()
        gR = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in params])
        a = np.array([adj[t] for t in range(R)])
        slope = np.polyfit(np.arange(R), np.log(a + 1e-30), 1)[0]
        row = dict(i=i, loss=float(loss), adj=a.tolist(), rho=float(np.exp(-slope)), err={})
        for k in ks:
            torch.manual_seed(777 + i)
            model.zero_grad(set_to_none=True)
            loss_manual(model, ids, R, k).backward()
            gk = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in params])
            row['err'][k] = float((gk - gR).norm() / gR.norm())
        rows.append(row)
    model.zero_grad(set_to_none=True); model.train()
    rho = np.array([r['rho'] for r in rows])
    summ = dict(R=R, rho_median=float(np.median(rho)),
                err_median={k: float(np.median([r['err'][k] for r in rows])) for k in ks},
                law_median={k: float(np.median(rho ** k)) for k in ks})
    return dict(summary=summ, rows=rows)


# -------------------------------------------------------------------------------------------------------- selftest
def selftest(tok, model, data, check_r=(8, 16)):
    ids = data['val'][:2].to(DEV)
    model.eval()
    for R, k in ((4, 4), (4, 2), (4, 0)):
        torch.manual_seed(0); a = float(model(ids, labels=labels_of(ids), num_steps_pair=torch.tensor([R - k, k]))['loss'])
        torch.manual_seed(0); b = float(loss_manual(model, ids, R, k))
        log('selftest decomposition R=%d k=%d: model %.6f manual %.6f' % (R, k, a, b)); assert abs(a - b) < 1e-4 * max(1, abs(a))
    if AMP:
        with torch.no_grad():
            torch.manual_seed(0); a = float(loss_model(model, ids, 8, 0))
            torch.manual_seed(0); b = float(model(ids, labels=labels_of(ids), num_steps_pair=torch.tensor([8, 0]))['loss'])
        log('selftest fp16-autocast loss %.5f vs fp32 %.5f (|diff| must be < 0.01)' % (a, b)); assert abs(a - b) < 0.01
    # gradient of the manual k=R loss == gradient of the model forward with (0, R)
    p0 = next(model.transformer.core_block[0].parameters())
    model.zero_grad(); torch.manual_seed(0); model(ids, labels=labels_of(ids), num_steps_pair=torch.tensor([0, 4]))['loss'].backward()
    g1 = p0.grad.clone(); model.zero_grad(); torch.manual_seed(0); loss_manual(model, ids, 4, 4).backward()
    rel = float((p0.grad - g1).norm() / (g1.norm() + 1e-30)); model.zero_grad()
    log('selftest gradient manual vs model (k = R = 4): rel diff %.2e' % rel); assert rel < 1e-4
    if AMP:
        truncated_grad_check(model, data, check_r)
    model.train(); log('selftest OK')


def truncated_grad_check(model, data, check_r=(8, 16)):
    """The v161 failure mode: the TRAINING path (loss_model, train mode, fp16 autocast) must give the same truncated
    gradient as fp32, for every parameter group, at k < R. Also checks that the truncation error (g_k vs g_R) is the
    same in fp16 and fp32."""
    model.train()   # gradients are stored on CPU; R = 8 and R = 16 cases can run on different GPUs (--check_r)
    names = [n for n, _ in model.named_parameters()]; params = list(model.parameters())
    groups = lambda n: 'core_block' if 'core_block' in n else 'adapter' if 'adapter' in n else 'prelude' if 'prelude' in n else 'rest'

    def grad(R, k, amp):
        global AMP
        ids = data['val'][:BATCH if R <= 8 else 4].to(DEV)   # fp32 full backprop at R 16, batch 8 OOMs a T4
        old, AMP = AMP, amp
        model.zero_grad(set_to_none=True); torch.manual_seed(0)
        loss = loss_model(model, ids, R, k); (loss * (1024.0 if amp else 1.0)).backward()
        AMP = old
        out = {}
        for n, p in zip(names, params):
            out.setdefault(groups(n), []).append((p.grad if p.grad is not None else torch.zeros_like(p)).flatten().cpu().double() / (1024.0 if amp else 1.0))
        model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        return {g: torch.cat(v) for g, v in out.items()}
    for R, k in [c for c in ((8, 7), (8, 4), (16, 8), (16, 1)) if c[0] in check_r]:
        h, f = grad(R, k, True), grad(R, k, False); hR, fR = grad(R, R, True), grad(R, R, False)
        rel = {g: float((h[g] - f[g]).norm() / (f[g].norm() + 1e-30)) for g in f}
        cat = lambda d: torch.cat([d[g] for g in sorted(d)])
        e16 = float((cat(h) - cat(hR)).norm() / cat(hR).norm()); e32 = float((cat(f) - cat(fR)).norm() / cat(fR).norm())
        log('selftest truncated grad fp16 vs fp32 R=%d k=%d: per-group rel diff %s | trunc err fp16 %.4f fp32 %.4f' % (
            R, k, {g: round(v, 4) for g, v in rel.items()}, e16, e32))
        assert max(rel.values()) < 0.05 and abs(e16 - e32) < 0.02, 'fp16 truncated gradient disagrees with fp32'
    if 8 not in check_r:
        return model.zero_grad(set_to_none=True)
    # negative control: the v161 setting (autocast cache on) must reproduce the bug, or this check proves nothing
    f = grad(8, 4, False); model.zero_grad(set_to_none=True); torch.manual_seed(0)
    ids = data['val'][:BATCH].to(DEV)
    with torch.autocast('cuda', dtype=torch.float16, enabled=True):
        loss = model(ids, labels=labels_of(ids), num_steps_pair=torch.tensor([4, 4]))['loss'].float()
    (loss * 1024.0).backward()
    core = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten().cpu().double() / 1024.0
                      for n, p in zip(names, params) if groups(n) == 'core_block'])
    bug = float((core - f['core_block']).norm() / f['core_block'].norm())
    log('selftest negative control (v161 setting, cache on) R=8 k=4: core_block rel diff vs fp32 %.3f (expect ~1)' % bug)
    assert bug > 0.5, 'cache-on setting did not reproduce the v161 bug'
    model.zero_grad(set_to_none=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true'); ap.add_argument('--measure', default=None)
    ap.add_argument('--train', default=None); ap.add_argument('--tag', default='x')
    ap.add_argument('--report', nargs='*'); ap.add_argument('--check_r', default='8,16')
    a = ap.parse_args()
    if a.report is not None:
        for f in a.report:
            for line in open(f):
                r = json.loads(line)
                if 'R' in r:
                    print('R=%-2d k=%-2d lr=%-7g seed=%d val %.4f test %.4f (%ds)%s' % (r['R'], r['k'], r['lr'], r['seed'],
                          r['val'], r['test'], r['s'], ' DIVERGED' if r['diverged'] else ''))
        return
    tok, model = load_model(); data = gsm8k_chunks(tok)
    log('v162 driver %s dev %s amp %s | train ex %d, val chunks %d, test chunks %d' % (driver_sha(), DEV, AMP,
        len(data['train']), len(data['val']), len(data['test'])))
    if a.selftest:
        selftest(tok, model, data, tuple(int(x) for x in a.check_r.split(',')))
    if a.measure:
        for R in [int(x) for x in a.measure.split(',')]:
            ks = [k for k in (1, 2, 4, 8) if k < R]
            res = measure(model, data['val'], R, ks); res['driver_sha'] = driver_sha()
            log('measure R=%d: %s' % (R, json.dumps(res['summary'])))
            json.dump(res, open(os.path.join(OUT, 'v162_measure_R%d_%s.json' % (R, a.tag)), 'w'))
        for R in (8, 16):
            log('step-0 test loss R=%d: %.4f  val %.4f' % (R, eval_loss(model, data['test'], R), eval_loss(model, data['val'], R)))
    if a.train:
        del model
        for cell in a.train.split(','):
            R, k, lr, seed = cell.split(':'); R, k, lr, seed = int(R), int(k), float(lr), int(seed)
            r = train_cell(tok, data, R, k, lr, seed); r['driver_sha'] = driver_sha()
            log('cell R=%d k=%d lr=%g seed=%d: val %.4f test %.4f nonfinite %d (%ds)' % (R, k, lr, seed, r['val'],
                r['test'], r['nonfinite_steps'], r['s']))
            open(os.path.join(OUT, 'v162_runs_%s.jsonl' % a.tag), 'a').write(json.dumps(r) + '\n')
            torch.cuda.empty_cache() if DEV == 'cuda' else None


if __name__ == '__main__':
    main()
