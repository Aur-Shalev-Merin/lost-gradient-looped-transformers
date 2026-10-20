"""v163: does rho^k predict the training cost of truncated loop backprop better than "k alone"? Manipulate rho on the
SAME model by convex damping of Parcae-140m's loop (prereg: docs/prereg_v163_parcae_damped_truncation.md).

Damped loop: x_{t+1} = (1 - a) x_t + a F(x_t), F = Parcae's core_block_forward (injection + core layers). Feasibility
(docs/parcae_damping_feasibility_2026-10-02.md): rho follows 1 - a + a rho_0 (a = 1: ~0.82, a = 0.5: ~0.90 at R 32); the
function at convergence is unchanged (R 32: 2.612 / 2.616). Cells: a in {1.0, 0.5} x k in {0 (floor), 1, 2, 4, 8, 32 (full)},
R = 32 loops always. Recipe otherwise = v162 (GSM8K, 600 steps, batch 8 x 512, AdamW, warm-up + cosine, clip 1.0, fp16
autocast with cache_enabled=False). Training goes through the hand-written damped forward (Parcae's own forward has no
damping); at a = 1 it equals v162's loss_manual, which equals Parcae's forward (selftest). Grad loops are checkpointed
per loop (memory at R 32; identical math).

Modes: --selftest | --measure (frozen model: rho, err_k per a at R 32) | --train 'a:k:lr:seed,...' --tag T | --report
Reuses v162_parcae_trunc_train.py (frozen 89d8758) for loading, data and helpers; driver sha covers both files.
"""
import argparse, hashlib, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as Fn
from torch.utils.checkpoint import checkpoint

import v162_parcae_trunc_train as v

R = 4 if v.TINY else 32
ALPHAS = (1.0, 0.5)


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(v.OUT, 'v163_log.txt'), 'a').write(s + '\n')


def driver_sha():
    h = hashlib.sha256()
    for f in (__file__, v.__file__):
        h.update(open(os.path.abspath(f), 'rb').read().replace(b'\r\n', b'\n'))
    return h.hexdigest()[:12]


def damped_loss(m, ids, k, a, amp=False, adj=None, ckpt=True):
    """Loss with the convex-damped loop, R loops, backprop through the last k. amp: fp16 autocast, cache OFF."""
    with torch.autocast('cuda', dtype=torch.float16, enabled=amp and v.DEV == 'cuda', cache_enabled=False):
        freqs = m.freqs_cis[:, : ids.shape[1]]; m._current_input_ids = ids
        ve = lambda i: m.value_embeds[str(i)](ids) if str(i) in m.value_embeds else None
        e = m.transformer.wte(ids)
        if m.emb_scale != 1:
            e = e * m.emb_scale
        for i, block in enumerate(m.transformer.prelude):
            e = block(e, freqs, None, ve=ve(i))
        if m.config.prelude_norm:
            e = m.transformer.ln_prelude(e)
        x = m.initialize_state(e); tot = torch.tensor(R)

        def step(x, e, t):
            return (1 - a) * x + a * m.core_block_forward(x, e, freqs, None, torch.tensor(t), tot)
        with torch.no_grad():
            for t in range(R - k):
                x = step(x, e, t)
        x = x.detach()
        for t in range(R - k, R):
            x = checkpoint(step, x, e, t, use_reentrant=False) if (ckpt and adj is None) else step(x, e, t)
            if adj is not None and x.requires_grad:
                x.register_hook(lambda g, t=t: adj.__setitem__(t, float(g.float().norm())))
        x = m.transformer.C(x); off = m.config.n_layers_in_prelude + m.config.n_layers_in_recurrent_block
        for i, block in enumerate(m.transformer.coda):
            x = block(x, freqs, None, ve=ve(off + i))
        logits = m.lm_head(m.transformer.ln_f(x)).float() * m.config.init.logit_scale
        if m.config.logit_softcap is not None:
            logits = m.config.logit_softcap * torch.tanh(logits / m.config.logit_softcap)
        return Fn.cross_entropy(logits.view(-1, logits.shape[-1]), v.labels_of(ids).view(-1), ignore_index=-100)


@torch.no_grad()
def eval_loss(m, chunks, a):
    m.eval(); tot, n = 0.0, 0
    for i in range(0, len(chunks), v.BATCH):
        ids = chunks[i:i + v.BATCH].to(v.DEV); torch.manual_seed(12345 + i)
        tot += float(damped_loss(m, ids, 0, a, amp=v.AMP)) * ids.shape[0]; n += ids.shape[0]
    m.train()
    return tot / n


def end_rho(m, chunks, a, n=8):
    """Reported-only (added before freeze): adjoint decay of the TRAINED model, fp32, batch 1, all R loops with grad,
    first n val chunks. rho = median over sequences of exp(-slope) of the log-linear fit of ||dL/dh_t|| over t."""
    m.eval(); rhos, adjs = [], []
    for i in range(min(n, len(chunks))):
        ids = chunks[i:i + 1].to(v.DEV); adj = {}
        torch.manual_seed(777 + i); m.zero_grad(set_to_none=True)
        damped_loss(m, ids, R, a, adj=adj).backward()
        A_ = np.array([adj[t] for t in range(R)]); adjs.append(A_)
        rhos.append(float(np.exp(-np.polyfit(np.arange(R), np.log(A_ + 1e-30), 1)[0])))
    m.zero_grad(set_to_none=True); m.train()
    return float(np.median(rhos)), np.median(np.stack(adjs), 0).tolist()


def train_cell(data, a, k, lr, seed):
    t0 = time.time()
    _, m = v.load_model(); m.train()
    g = np.random.default_rng(seed)
    chunks = torch.cat([v.pack(data['train'], g.permutation(len(data['train']))) for _ in range(4)])
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    sched = lambda s: min(1.0, (s + 1) / v.WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / v.STEPS)))
    scaler = torch.amp.GradScaler('cuda', enabled=v.AMP)
    curve, nonfinite = [], 0
    for s in range(v.STEPS):
        for gp in opt.param_groups:
            gp['lr'] = lr * sched(s)
        ids = chunks[s * v.BATCH:(s + 1) * v.BATCH].to(v.DEV)
        loss = damped_loss(m, ids, k, a, amp=v.AMP)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), v.CLIP)
        if not torch.isfinite(gn):
            nonfinite += 1
        scaler.step(opt); scaler.update()
        if s % 50 == 0 or s == v.STEPS - 1:
            curve.append((s, round(float(loss.detach()), 4), round(float(gn), 3)))
    val, test = eval_loss(m, data['val'], a), eval_loss(m, data['test'], a)
    rho_end, adj_end = end_rho(m, data['val'], a) if math.isfinite(val) else (float('nan'), None)
    return dict(a=a, k=k, R=R, lr=lr, seed=seed, val=val, test=test, rho_end=rho_end, adj_end=adj_end,
                curve=curve, nonfinite_steps=nonfinite,
                diverged=not (math.isfinite(val) and math.isfinite(test)), s=round(time.time() - t0),
                steps=v.STEPS, batch=v.BATCH, seq=v.SEQ, amp=v.AMP)


def measure(m, chunks, a):
    ks = [k for k in (1, 2, 4, 8, 16) if k < R]
    params = [p for p in m.parameters() if p.requires_grad]
    grad = lambda: torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in params])
    m.eval(); rows = []
    for i in range(min(v.N_MEASURE, len(chunks))):
        ids = chunks[i:i + 1].to(v.DEV); adj = {}
        torch.manual_seed(777 + i); m.zero_grad(set_to_none=True)
        damped_loss(m, ids, R, a, adj=adj).backward(); gR = grad()
        A_ = np.array([adj[t] for t in range(R)])
        rho = float(np.exp(-np.polyfit(np.arange(R), np.log(A_ + 1e-30), 1)[0]))
        row = dict(rho=rho, adj=A_.tolist(), err={})
        for k in ks:
            torch.manual_seed(777 + i); m.zero_grad(set_to_none=True)
            damped_loss(m, ids, k, a, ckpt=False).backward()
            row['err'][k] = float((grad() - gR).norm() / gR.norm())
        rows.append(row)
    m.zero_grad(set_to_none=True); m.train()
    rho = np.array([r['rho'] for r in rows])
    return dict(summary=dict(a=a, R=R, rho_median=float(np.median(rho)),
                             err_median={k: float(np.median([r['err'][k] for r in rows])) for k in ks},
                             law_median={k: float(np.median(rho ** k)) for k in ks}), rows=rows)


def selftest(m, data):
    ids = data['val'][:2].to(v.DEV); m.eval()
    # 1. a = 1 damped forward == v162 loss_manual (== Parcae forward, v162 selftest)
    for k in (R, R // 2, 0):
        torch.manual_seed(0); x1 = float(v.loss_manual(m, ids, R, k))
        torch.manual_seed(0); x2 = float(damped_loss(m, ids, k, 1.0, ckpt=False))
        log('selftest a=1 vs v162 loss_manual R=%d k=%d: %.6f %.6f' % (R, k, x1, x2)); assert abs(x1 - x2) < 1e-4 * max(1, abs(x1))
    # 2. checkpointed vs plain gradient (same math)
    p0 = next(m.transformer.core_block[0].parameters())
    m.zero_grad(); torch.manual_seed(0); damped_loss(m, ids, 4, 0.5, ckpt=True).backward(); g1 = p0.grad.clone()
    m.zero_grad(); torch.manual_seed(0); damped_loss(m, ids, 4, 0.5, ckpt=False).backward()
    rel = float((p0.grad - g1).norm() / (g1.norm() + 1e-30)); m.zero_grad()
    log('selftest checkpointed vs plain grad (a=0.5, k=4): rel diff %.2e' % rel); assert rel < 1e-4
    # 3. truncated gradient fp16 (training path) vs fp32, per group, both alphas
    if v.AMP:
        names = [n for n, _ in m.named_parameters()]; params = list(m.parameters())
        grp = lambda n: 'core_block' if 'core_block' in n else 'adapter' if 'adapter' in n else 'prelude' if 'prelude' in n else 'rest'

        def grads(a, k, amp):
            m.zero_grad(set_to_none=True); torch.manual_seed(0)
            (damped_loss(m, ids, k, a, amp=amp) * (1024.0 if amp else 1.0)).backward()
            out = {}
            for n, p in zip(names, params):
                out.setdefault(grp(n), []).append((p.grad if p.grad is not None else torch.zeros_like(p)).flatten().double().cpu()
                                                  / (1024.0 if amp else 1.0))
            m.zero_grad(set_to_none=True)
            return {g_: torch.cat(x) for g_, x in out.items()}
        for a in ALPHAS:
            for k in (1, 4):
                h, f = grads(a, k, True), grads(a, k, False)
                rel = {g_: float((h[g_] - f[g_]).norm() / (f[g_].norm() + 1e-30)) for g_ in f}
                log('selftest truncated grad fp16 vs fp32 a=%.1f k=%d: %s' % (a, k, {g_: round(x, 4) for g_, x in rel.items()}))
                assert max(rel.values()) < 0.05, 'fp16 truncated gradient disagrees with fp32'
    m.train(); log('selftest OK')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true'); ap.add_argument('--measure', action='store_true')
    ap.add_argument('--train', default=None); ap.add_argument('--tag', default='x')
    ap.add_argument('--report', nargs='*')
    A = ap.parse_args()
    if A.report is not None:
        for f in A.report:
            for line in open(f):
                r = json.loads(line)
                print('a=%.1f k=%-2d lr=%-7g seed=%d val %.4f test %.4f (%ds)%s' % (r['a'], r['k'], r['lr'], r['seed'],
                      r['val'], r['test'], r['s'], ' DIVERGED' if r['diverged'] else ''))
        return
    tok, m = v.load_model(); data = v.gsm8k_chunks(tok)
    log('v163 driver %s (v162 module %s) dev %s amp %s R %d' % (driver_sha(), v.driver_sha(), v.DEV, v.AMP, R))
    if A.selftest:
        selftest(m, data)
    if A.measure:
        res = {}
        for a in ALPHAS:
            r = measure(m, data['val'], a); res[a] = r
            log('measure a=%.1f: %s' % (a, json.dumps(r['summary'])))
            log('step-0 val/test loss a=%.1f R=%d: %.4f / %.4f' % (a, R, eval_loss(m, data['val'], a), eval_loss(m, data['test'], a)))
        json.dump(dict(driver_sha=driver_sha(), res=res), open(os.path.join(v.OUT, 'v163_measure_%s.json' % A.tag), 'w'), default=str)
    if A.train:
        del m
        for cell in A.train.split(','):
            a, k, lr, seed = cell.split(':'); a, k, lr, seed = float(a), int(k), float(lr), int(seed)
            r = train_cell(data, a, k, lr, seed); r['driver_sha'] = driver_sha()
            log('cell a=%.1f k=%d lr=%g seed=%d: val %.4f test %.4f rho_end %.3f nonfinite %d (%ds)' % (a, k, lr, seed,
                r['val'], r['test'], r['rho_end'], r['nonfinite_steps'], r['s']))
            open(os.path.join(v.OUT, 'v163_runs_%s.jsonl' % A.tag), 'a').write(json.dumps(r) + '\n')
            torch.cuda.empty_cache() if v.DEV == 'cuda' else None


if __name__ == '__main__':
    main()
