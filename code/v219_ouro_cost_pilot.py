"""v219 PILOT (before any v219 prereg; only feasibility and seed spread are read): can Ouro-1.4B be fine-tuned with full
backprop through all 4 loops on a 16 GB T4, and how large and how noisy are the training costs of truncation?

The planned study (v219, to be pre-registered after this pilot) is Section 5's cost measurement on Ouro:
    cost = (loss with k loops - loss with all loops) / (loss with no loop gradient - loss with all loops),
fine-tuning on GSM8K text with the final-output loss, k in {4 (full), 2, 0 (no loop credit)}.

Phase "probe": peak memory and seconds per optimiser step (effective batch 8 x 512 tokens, micro-batches with gradient
accumulation, fp16 autocast with fp32 master weights, activation checkpointing per layer in the loops with gradient),
for each optimiser in order: torch AdamW, bitsandbytes AdamW8bit, Adafactor. The first that fits is used.
Phase "runs": short fine-tunes (STEPS steps, constant LR after a short warm-up) for (k, seed) pairs; held-out loss on
the 16 GSM8K validation chunks of v197 (the last 473 training problems), all 4 loops, before and after.

  python v219_ouro_cost_pilot.py --phase probe
  python v219_ouro_cost_pilot.py --phase runs --runs 4:1:final,0:1:exit,... [--opt adamw8] [--lr 1e-5]   (k:seed:objective)
Env: TINY=1 (tiny random Ouro on CPU), OUT_DIR.
"""
import argparse, json, math, os, sys, time
os.environ.setdefault('LOOP_MODEL', 'ouro')
import numpy as np
import torch
import torch.nn.functional as Fn
from torch.utils.checkpoint import checkpoint

TINY = os.environ.get('TINY') == '1'
OUT = os.environ.get('OUT_DIR', '/kaggle/working' if os.path.isdir('/kaggle/working') else '.')
SEQ = 32 if TINY else 512
BATCH = 2 if TINY else 8
STEPS = 4 if TINY else 40
WARMUP = 1 if TINY else 4
R = 4
N_VAL = 473


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v219_pilot_log.txt'), 'a').write(s + '\n')


def devices():
    if torch.cuda.device_count() >= 2:
        return torch.device('cuda:0'), torch.device('cuda:1')
    d = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return d, d


def load():
    """fp32 Ouro split across the kernel's two T4s (naive model parallelism): the embedding and layers 0-11 on GPU 0;
    layers 12-23, the final norm, the exit gate and the output head on GPU 1. Each loop passes the hidden state from one
    GPU to the other. One GPU computes at a time; the maths is unchanged."""
    import loop_grad_probe as P
    tok, model = P.load()
    model = model.float()
    d0, d1 = devices(); b = model.model; L = b.config.num_hidden_layers
    b.embed_tokens.to(d0); b.rotary_emb.to(d0)
    for i, layer in enumerate(b.layers[:L]):
        layer.to(d0 if i < L // 2 else d1)
    b.norm.to(d1); model.lm_head.to(d1)
    gate = b.early_exit_gate if hasattr(b, 'early_exit_gate') else None
    if gate is not None:
        gate.to(d1)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    P.DEV = d0
    return P, tok, model


def make_data(tok, P):
    import v197_ouro_allparam as V
    val = V.val_chunks(tok, P)[:16]
    if TINY:
        g = torch.Generator().manual_seed(1)
        return torch.randint(0, 1000, (400, SEQ), generator=g), val
    from datasets import load_dataset
    tr = load_dataset('openai/gsm8k', 'main', split='train')
    ex = [tok(tr[i]['question'] + '\n' + tr[i]['answer']).input_ids for i in range(len(tr) - N_VAL)]
    return ex, val


def pack(ex, seed):
    if TINY:
        return ex[torch.randperm(len(ex), generator=torch.Generator().manual_seed(seed))]
    order = np.random.default_rng(seed).permutation(len(ex))
    flat = [t for i in order for t in ex[i]]
    n = len(flat) // SEQ
    return torch.tensor(flat[: n * SEQ], dtype=torch.long).view(n, SEQ)


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

    gate = b.early_exit_gate if hasattr(b, 'early_exit_gate') else model.model.early_exit_gate

    def loss_fn(ids, k, objective='final', amp=True):
        """Loss after R loops with backprop through the last k (k = 0: no loop credit). One fp16 autocast region with
        the weight-cast cache off (v161). objective 'final': CE of the last loop's output (Section 5's cost tests).
        'exit': Ouro's training objective without its entropy term, per token sum_t p_t L_t with p_t from the exit
        gate (as v210 Part B); loops before R - k run without gradient, every exit read-out (gate, lm_head) stays
        differentiable."""
        labels = torch.cat([ids[:, 1:], torch.full_like(ids[:, :1], -100)], 1).view(-1).to(d1)
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp and torch.cuda.is_available(), cache_enabled=False):
            h, ctx = prepare(ids); exits = []
            for t in range(R):
                on = t >= R - k
                with torch.set_grad_enabled(on and torch.is_grad_enabled()):
                    h = b.norm(layers(h, t, ctx, ckpt=on))
                if not on:
                    h = h.detach()
                exits.append(h)
            if objective == 'final':
                logits = model.lm_head(exits[-1]).float()
                return Fn.cross_entropy(logits.view(-1, logits.shape[-1]), labels, ignore_index=-100)
            valid = (labels != -100).float(); remaining = None; total = 0.0
            for t, hn in enumerate(exits):
                lam = torch.sigmoid(gate(hn).squeeze(-1)).float().view(-1)
                remaining = torch.ones_like(lam) if remaining is None else remaining
                p_t = lam * remaining if t < R - 1 else remaining
                remaining = remaining * (1 - lam)
                logits = model.lm_head(hn).float()
                ce = Fn.cross_entropy(logits.view(-1, logits.shape[-1]), labels.clamp(min=0), reduction='none') * valid
                total = total + (p_t * ce).sum()
            return total / valid.sum()
    return loss_fn


def make_opt(name, params, lr):
    if name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    if name == 'adamw8':
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(params, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    from transformers.optimization import Adafactor
    return Adafactor(params, lr=lr, scale_parameter=False, relative_step=False, warmup_init=False)


def train(model, loss_fn, chunks, k, opt, micro, steps, lr, dev, objective='final'):
    params = [p for p in model.parameters() if p.requires_grad]
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    hist, nonfinite, t0 = [], 0, time.time()
    for s in range(steps):
        for g in opt.param_groups:
            g['lr'] = lr * min(1.0, (s + 1) / WARMUP)
        opt.zero_grad(set_to_none=True); tot = 0.0
        for mb in range(BATCH // micro):
            ids = chunks[(s * BATCH + mb * micro) % len(chunks):][:micro].to(dev)
            loss = loss_fn(ids, k, objective) / (BATCH // micro)
            if k == 0 and not loss.requires_grad:     # TINY guard: nothing to train without a grad path
                break
            scaler.scale(loss).backward(); tot += float(loss)
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(gn):
            nonfinite += 1
        scaler.step(opt); scaler.update()
        hist.append(round(tot, 4))
    return dict(loss_hist=hist, nonfinite=nonfinite, s_per_step=round((time.time() - t0) / steps, 2))


@torch.no_grad()
def evaluate(model, loss_fn, val, dev):
    """Held-out loss under both objectives (all 4 loops, no gradient)."""
    model.eval()
    v = {o: float(np.mean([float(loss_fn(val[i:i + 1].to(dev), 0, o)) for i in range(len(val))])) for o in ('final', 'exit')}
    model.train()
    return v


def probe_one(opt_name, micro):
    """One probe configuration in a fresh process: 2 optimiser steps of full backprop (k = 4), final-output objective."""
    P, tok, model = load(); loss_fn = make_loss(model)
    ex, val = make_data(tok, P); chunks = pack(ex, 7)
    params = [p for p in model.parameters() if p.requires_grad]
    rec = dict(opt=opt_name, micro=micro, n_params=sum(p.numel() for p in params))
    try:
        out = train(model, loss_fn, chunks, R, make_opt(opt_name, params, 1e-5), micro, 2, 1e-5, P.DEV)
        rec.update(ok=True, s_per_step=out['s_per_step'])
    except Exception as e:                             # OOM or a missing package
        rec.update(ok=False, error=repr(e)[:200])
    rec['peak_gb'] = [round(torch.cuda.max_memory_allocated(i) / 1e9, 2) for i in range(torch.cuda.device_count())]         if torch.cuda.is_available() else None
    json.dump(rec, open(os.path.join(OUT, 'v219_probe_%s_%d.json' % (opt_name, micro)), 'w'))


def probe():
    """Configurations in order of preference, each in its own process; stop at the first that fits."""
    import subprocess
    res = dict(probes=[])
    for opt_name, micro in (('adamw', 2), ('adamw', 1), ('adamw8', 1), ('adafactor', 1)):
        subprocess.call([sys.executable, '-u', os.path.abspath(__file__), '--phase', 'probe1', '--opt', opt_name,
                         '--micro', str(micro)], env=dict(os.environ))
        f = os.path.join(OUT, 'v219_probe_%s_%d.json' % (opt_name, micro))
        rec = json.load(open(f)) if os.path.exists(f) else dict(opt=opt_name, micro=micro, ok=False, error='no output')
        log('  probe %s' % rec); res['probes'].append(rec)
        json.dump(res, open(os.path.join(OUT, 'v219_pilot_probe.json'), 'w'), indent=1)
        if rec['ok']:
            break


def runs(spec, opt_name, lr, micro):
    out = dict(opt=opt_name, lr=lr, micro=micro, steps=STEPS, runs=[])
    for item in spec:
        parts = item.split(':'); k, seed = int(parts[0]), int(parts[1]); obj = parts[2] if len(parts) > 2 else 'final'
        P, tok, model = load(); loss_fn = make_loss(model)
        ex, val = make_data(tok, P); chunks = pack(ex, 1000 + seed)
        torch.manual_seed(seed)
        params = [p for p in model.parameters() if p.requires_grad]
        v0 = evaluate(model, loss_fn, val, P.DEV)
        t0 = time.time()
        tr = train(model, loss_fn, chunks, k, make_opt(opt_name, params, lr), micro, STEPS, lr, P.DEV, obj)
        v1 = evaluate(model, loss_fn, val, P.DEV)
        rec = dict(k=k, seed=seed, objective=obj, val_before=v0, val_after=v1, s=round(time.time() - t0), **tr)
        rec['peak_gb'] = [round(torch.cuda.max_memory_allocated(i) / 1e9, 2) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None
        log('  run k=%d seed=%d obj=%s: held-out final %.4f -> %.4f, exit %.4f -> %.4f | %.1f s/step | non-finite %d | '
            'peak %s GB' % (k, seed, obj, v0['final'], v1['final'], v0['exit'], v1['exit'], tr['s_per_step'],
                            tr['nonfinite'], rec['peak_gb']))
        out['runs'].append(rec)
        json.dump(out, open(os.path.join(OUT, 'v219_pilot_runs_%s.json' % '_'.join(spec).replace(':', '-')), 'w'), indent=1)
        del model, params
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); [torch.cuda.reset_peak_memory_stats(i) for i in range(torch.cuda.device_count())]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True); ap.add_argument('--runs', default='4:1,0:1,2:1')
    ap.add_argument('--opt', default='auto'); ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--micro', type=int, default=1)
    A = ap.parse_args()
    if A.phase == 'probe':
        probe()
    elif A.phase == 'probe1':
        probe_one(A.opt, A.micro)
    else:
        opt = A.opt
        if opt == 'auto':                              # the first optimiser the probe found to fit
            pr = json.load(open(os.path.join(OUT, 'v219_pilot_probe.json')))['probes']
            fit = [p for p in pr if p['ok']]
            opt, A.micro = (fit[0]['opt'], fit[0]['micro']) if fit else ('adafactor', 1)
        runs(A.runs.split(','), opt, A.lr, A.micro)
