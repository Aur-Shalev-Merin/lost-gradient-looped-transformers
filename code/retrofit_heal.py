"""Heal the retrofitted looped SmolLM2-135M (raven/Huginn architecture, split (11, 8, 11), adapter [0 | I]; see
docs/retrofit_convert_2026-10-01.md) so that the looped core learns to use its state across loops.

Recipe (McLeish et al. 2511.07384, scaled down):
  - continued training on FineWeb-Edu (sample-10BT, streamed), sequences packed to SEQ tokens with EOS between docs;
  - per step, total loops n ~ Poisson(lognormal(mean)) + 1 (their sampler), mean ramped linearly from 1 to MEAN_REC over
    the first CURRICULUM fraction of steps; backprop through the last k = min(BACKPROP, n) loops (num_steps = (n-k, k));
  - AdamW (fused; adapter at a higher LR: it is new), warm-up then cosine to 10%, grad clip 1.0;
  - speed (2026-10-02): 32 sequences per pass, no accumulation (same 32k tokens/step); tokenization in a background
    thread with a prefetch queue; eval batch 16;
  - bf16 autocast with cache_enabled=False (the v161 lesson: the autocast weight cache drops the looped weights'
    gradient when no-grad loops run first, on older PyTorch), checked before training against fp32.
Gate (printed at the end): held-out loss at R = 8 must be at least 0.02 nats below R = 1.
Control (--fixed_loops 1): the same cut-down model healed on the same data/steps/LRs but never looping; its R = 1 loss
is the fair "no looping" baseline for a looped run's R = 8 loss.

  python retrofit_heal.py --model_dir DIR --out_dir DIR [--smoke] [--tokens 150e6] [--max_hours H]
Checkpoints (model, optimizer, step, docs consumed) go to out_dir every --ckpt_every steps; a re-run resumes.
"""
import argparse, json, math, os, time
import torch
import torch.nn.functional as Fn

ap = argparse.ArgumentParser()
ap.add_argument('--model_dir', required=True); ap.add_argument('--out_dir', required=True)
ap.add_argument('--smoke', action='store_true'); ap.add_argument('--tokens', type=float, default=150e6)
ap.add_argument('--seq', type=int, default=1024); ap.add_argument('--micro_bs', type=int, default=32)
ap.add_argument('--accum', type=int, default=1); ap.add_argument('--lr', type=float, default=2e-4)
ap.add_argument('--adapter_lr', type=float, default=1e-3); ap.add_argument('--warmup', type=int, default=200)
ap.add_argument('--mean_rec', type=float, default=8.0); ap.add_argument('--curriculum', type=float, default=0.25)
ap.add_argument('--backprop', type=int, default=4); ap.add_argument('--max_loops', type=int, default=32)
ap.add_argument('--fixed_loops', type=int, default=None)   # control: always exactly this many loops, no sampler or curriculum
ap.add_argument('--eval_every', type=int, default=500); ap.add_argument('--ckpt_every', type=int, default=500)
ap.add_argument('--n_val', type=int, default=128); ap.add_argument('--max_hours', type=float, default=None)
ap.add_argument('--seed', type=int, default=0)
A = ap.parse_args()
if A.micro_bs * A.accum == 32 and A.micro_bs == 32 and torch.cuda.get_device_properties(0).total_memory < 60e9:
    A.micro_bs, A.accum = 16, 2          # batch 32 needs ~50-55 GB with 4 backprop loops; same tokens/step on smaller GPUs
if A.smoke:
    A.tokens, A.eval_every, A.ckpt_every, A.n_val, A.warmup = 30 * A.seq * A.micro_bs * A.accum, 15, 10 ** 9, 16, 5
    A.out_dir = A.out_dir.rstrip('/') + '_smoke'
os.makedirs(A.out_dir, exist_ok=True)
DEV = 'cuda'
EVAL_RS = (1, 2, 4, 8, 16)
STEPS = int(A.tokens // (A.seq * A.micro_bs * A.accum))


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(A.out_dir, 'heal_log.txt'), 'a').write(s + '\n')


# ------------------------------------------------------------------------------------------------------------- data
def doc_stream(skip_docs):
    from datasets import load_dataset
    ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', streaming=True)
    return iter(ds.skip(skip_docs)) if skip_docs else iter(ds)


class Packer:
    """Packs streamed docs into SEQ-token chunks (EOS id 0 between docs); counts docs consumed for resuming."""
    def __init__(self, tok, skip_docs):
        self.tok, self.it, self.buf, self.docs = tok, doc_stream(skip_docs), [], skip_docs

    def chunk(self):
        while len(self.buf) < A.seq:
            texts = [next(self.it)['text'] for _ in range(64)]
            self.docs += len(texts)
            for ids in self.tok(texts).input_ids:
                self.buf += ids + [0]
        out, self.buf = self.buf[:A.seq], self.buf[A.seq:]
        return out

    def batch(self, n):
        return torch.tensor([self.chunk() for _ in range(n)], dtype=torch.long)


class Prefetcher:
    """Tokenizes and packs in a background thread so the GPU does not wait (speed-up 2026-10-02). Each queued batch
    carries the Packer's doc count at the time it was made; .docs is the count for the last batch CONSUMED, so a
    resume skips exactly the documents already trained on (up to the partial-chunk remainder, as before)."""
    def __init__(self, packer, n, depth=8):
        import threading, queue
        self.q, self.docs, self.err = queue.Queue(maxsize=depth), packer.docs, None

        def work():
            try:
                while True:
                    b = packer.batch(n)
                    self.q.put((b.pin_memory(), packer.docs))
            except Exception as e:      # surfaced in .next()
                self.err = e; self.q.put((None, None))
        threading.Thread(target=work, daemon=True).start()

    def next(self):
        b, d = self.q.get()
        if b is None:
            raise RuntimeError('data thread failed: %r' % self.err)
        self.docs = d
        return b


def labels_of(ids):
    return torch.cat([ids[:, 1:], torch.full_like(ids[:, :1], -100)], 1)


# ------------------------------------------------------------------------------------------------------------ model
def load_model():
    from transformers import AutoModelForCausalLM
    # save_pretrained writes tie_word_embeddings into config.json, and raven's config class passes it again (derived
    # from tie_embeddings) -> "multiple values for keyword argument". Both are False; drop the duplicate.
    cp = os.path.join(A.model_dir, 'config.json'); c = json.load(open(cp))
    if 'tie_word_embeddings' in c:
        assert not c['tie_word_embeddings'] and not c.get('tie_embeddings', False)
        c.pop('tie_word_embeddings'); json.dump(c, open(cp, 'w'), indent=2)
    m = AutoModelForCausalLM.from_pretrained(A.model_dir, trust_remote_code=True, torch_dtype=torch.float32).to(DEV)
    m.config.test_time_noise = 0
    return m


def loss_fn(model, ids, n_nograd, k, amp=True):
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp, cache_enabled=False):
        return model(ids, labels=labels_of(ids), num_steps=(n_nograd, k)).loss.float()


def sample_loops(step):
    if A.fixed_loops:                     # control arm (e.g. 1 = never loop): same data, steps, LR, backprop rule
        n = A.fixed_loops; k = min(A.backprop, n)
        return n - k, k, float(n)
    """McLeish/Huginn sampler: n = Poisson(LogNormal(log(mean) - s^2/2, s)) + 1, s = 0.5; mean ramps 1 -> MEAN_REC."""
    ramp = max(1, int(A.curriculum * STEPS))
    mean = 1.0 + (A.mean_rec - 1.0) * min(1.0, step / ramp)
    sigma = 0.5
    rate = torch.zeros(1).log_normal_(mean=math.log(mean) - sigma ** 2 / 2, std=sigma)
    n = int(torch.poisson(rate).item()) + 1
    n = min(n, A.max_loops)
    k = min(A.backprop, n)
    return n - k, k, mean


@torch.no_grad()
def evaluate(model, val):
    model.eval(); out = {}
    for R in EVAL_RS:
        tot = 0.0
        for i in range(0, len(val), 16):
            b = val[i:i + 16].to(DEV); torch.manual_seed(1234 + i)
            with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                tot += float(model(b, labels=labels_of(b), num_steps=R).loss) * len(b)
        out[R] = tot / len(val)
    model.train()
    return out


def grad_check(model, val):
    """bf16-autocast truncated gradient vs fp32, per parameter group (the v161 failure mode)."""
    ids = val[:2].to(DEV); groups = {}
    for n, p in model.named_parameters():
        g = next((x for x in ('adapter', 'core_block', 'prelude', 'coda') if x in n), 'other')
        groups.setdefault(g, []).append(p)

    def grads(amp):
        model.zero_grad(set_to_none=True); torch.manual_seed(0)
        loss_fn(model, ids, 4, 4, amp).backward()
        return {g: torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten().double().cpu() for p in ps])
                for g, ps in groups.items()}
    h, f = grads(True), grads(False); model.zero_grad(set_to_none=True)
    rel = {g: float((h[g] - f[g]).norm() / (f[g].norm() + 1e-30)) for g in f}
    log('grad check (n_nograd 4, k 4) bf16 vs fp32 per-group rel diff: %s' % {g: round(v, 4) for g, v in rel.items()})
    # bf16 rounding alone gives ~0.1 relative gradient noise in this 54-layer-deep forward (smoke test: coda 0.11,
    # which the loops and the autocast cache cannot affect). The failure we guard against (v161: weights first cast
    # inside no_grad lose their gradient) puts the looped core near 1.0. Test that signature, with the coda as the
    # unaffected baseline.
    assert rel['core_block'] < 0.5 and rel['core_block'] <= 3 * rel['coda'] + 0.02,         'looped-core gradient lost under bf16 autocast (v161 failure mode)'


# ------------------------------------------------------------------------------------------------------------- main
def main():
    from transformers import AutoTokenizer
    torch.manual_seed(A.seed)
    tok = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M')
    log('torch %s | GPU %s | steps %d (%.0fM tokens, %d tokens/step) | args %s' % (torch.__version__,
        torch.cuda.get_device_name(0), STEPS, A.tokens / 1e6, A.seq * A.micro_bs * A.accum, vars(A)))
    model = load_model(); model.train()
    ad = [p for n, p in model.named_parameters() if 'adapter' in n]
    rest = [p for n, p in model.named_parameters() if 'adapter' not in n]
    opt = torch.optim.AdamW([dict(params=rest, lr=A.lr), dict(params=ad, lr=A.adapter_lr)], betas=(0.9, 0.95), fused=True,
                            eps=1e-8, weight_decay=0.0)
    base = [A.lr, A.adapter_lr]
    sched = lambda s: min(1.0, (s + 1) / A.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / STEPS))))

    # held-out set = the first N_VAL chunks of the stream; training starts after them
    val_path = os.path.join(A.out_dir, 'val.pt'); ck = os.path.join(A.out_dir, 'ckpt.pt')
    if os.path.exists(val_path):
        v = torch.load(val_path); val, val_docs = v['val'], v['docs']
    else:
        pk = Packer(tok, 0); val = pk.batch(A.n_val); val_docs = pk.docs
        torch.save(dict(val=val, docs=val_docs), val_path)
    step, docs = 0, val_docs
    if os.path.exists(ck):
        c = torch.load(ck, map_location='cpu', weights_only=False)
        model.load_state_dict(c['model']); opt.load_state_dict(c['opt']); step, docs = c['step'], c['docs']
        log('resumed from step %d (docs consumed %d)' % (step, docs))
    else:
        grad_check(model, val)
        e0 = evaluate(model, val)
        log('eval step 0: %s' % json.dumps(e0))
        open(os.path.join(A.out_dir, 'evals.jsonl'), 'a').write(json.dumps(dict(step=0, loss=e0)) + '\n')
    pk = Prefetcher(Packer(tok, docs), A.micro_bs)

    t0 = time.time(); tok_seen = 0; run_loss = None
    while step < STEPS:
        for gi, g in enumerate(opt.param_groups):
            g['lr'] = base[gi] * sched(step)
        n_ng, k, mean = sample_loops(step)
        opt.zero_grad(set_to_none=True); tot = 0.0
        for _ in range(A.accum):
            ids = pk.next().to(DEV, non_blocking=True)
            loss = loss_fn(model, ids, n_ng, k) / A.accum
            loss.backward(); tot += float(loss)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(gn):
            opt.step()
        step += 1; tok_seen += A.seq * A.micro_bs * A.accum
        run_loss = tot if run_loss is None else 0.98 * run_loss + 0.02 * tot
        if step % 25 == 0 or step == 1:
            el = time.time() - t0
            log('step %d/%d | loss %.4f (avg %.4f) | loops %d (no-grad %d, grad %d, mean %.2f) | gnorm %.3f | %.0f tok/s | %.2f h'
                % (step, STEPS, tot, run_loss, n_ng + k, n_ng, k, mean, float(gn), tok_seen / el, el / 3600))
        if step % A.eval_every == 0 or step == STEPS:
            e = evaluate(model, val)
            log('eval step %d: %s' % (step, json.dumps(e)))
            open(os.path.join(A.out_dir, 'evals.jsonl'), 'a').write(json.dumps(dict(step=step, loss=e)) + '\n')
        stop = A.max_hours is not None and (time.time() - t0) / 3600 > A.max_hours
        if step % A.ckpt_every == 0 or step == STEPS or stop:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, docs=pk.docs), ck + '.tmp')
            os.replace(ck + '.tmp', ck)
            log('checkpoint saved at step %d' % step)
        if stop:
            log('max_hours reached; stopping (re-run to resume)'); return

    e = evaluate(model, val)
    gate = e[8] <= e[1] - 0.02
    log('FINAL eval: %s' % json.dumps(e))
    log('GATE (L(R=8) <= L(R=1) - 0.02): %s  [L1 %.4f, L8 %.4f, diff %.4f]' % ('PASS' if gate else 'FAIL', e[1], e[8], e[1] - e[8]))
    json.dump(dict(final=e, gate=gate, steps=STEPS, tokens=A.tokens, args=vars(A)),      # written before the slow save
              open(os.path.join(A.out_dir, 'heal_result.json'), 'w'), indent=1)
    if not A.smoke:
        hd = os.path.join(A.out_dir, 'healed'); model.save_pretrained(hd)
        import shutil
        for fn in os.listdir(A.model_dir):      # remote-code model files (raven, GQA-patched) travel with the weights
            if fn.endswith('.py'):
                shutil.copy(os.path.join(A.model_dir, fn), os.path.join(hd, fn))
    log('done')


if __name__ == '__main__':
    main()
    os._exit(0)   # the datasets streaming iterator can leave threads that keep the process alive after main()
