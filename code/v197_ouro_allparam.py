"""v197: Ouro-1.4B lost gradient over ALL parameters with the Table 1 protocol (frozen model; no training).

Why: Table 1's Ouro row was measured on one middle layer with the probe's 128-token texts, unlike every other
unstarred row (all parameters, 16 GSM8K-val sequences of 512 tokens, batch 1, fp32: v164 measure()). Appendix E's
all-parameter number (0.996 / 0.893) used batch-averaged, clipped gradients, a third protocol. This run puts Ouro on
Table 1's protocol so the row is comparable (paper v15; outside reviews 4-5).

Protocol (mirrors v164 measure() / v189):
  - text: GSM8K train, last 473 examples (= v162's val split), question + '\\n' + answer, packed in order into 512-token
    chunks; the first 16 chunks.
  - per sequence, batch 1, fp32 (TF32 off): full gradient g_R (R = 4, Ouro's trained loop count) and truncated
    gradients g_k (backprop through the last k loops; earlier loops under no_grad), k = 1, 2, 3.
  - err_k = |g_k - g_R| / |g_R| over all parameters (also over the loop's parameters only, and over the one middle
    layer that Table 1's old row used, from the same backward passes).
  - backward-signal size at each loop's output, adj[t] = |dL/dh_{t+1}|, t = 0..R-1; rho_i = exp(-slope) of a
    straight-line fit to log adj[t] (as v164).
  - loss: next-token cross-entropy of the final output only (Ouro's exit gate is not used), as Table 1's Ouro row.
  - noise floor: on the first sequence the full pass is run twice and the two gradients compared.

  python v197_ouro_allparam.py --seqs 0:8 --tag g0        (Kaggle: one process per T4)
  python v197_ouro_allparam.py --analyze DIR              (reads every v197_ouro_*.json under DIR)
Env: TINY=1 (tiny random Ouro, CPU pipeline test), OUT_DIR.
"""
import argparse, glob, hashlib, json, os, time
os.environ.setdefault('LOOP_MODEL', 'ouro')
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

R = 4
KS = (1, 2, 3)
N_VAL = 473
CHUNK = 8_000_000                                   # elements per float64 reduction chunk (GPU memory)


def log(*a):
    import loop_grad_probe as P
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(P.OUT, 'v197_log.txt'), 'a').write(s + '\n')


def driver_sha():
    return hashlib.sha256(open(os.path.abspath(__file__), 'rb').read().replace(b'\r\n', b'\n')).hexdigest()[:12]


def val_chunks(tok, P):
    seq = 32 if P.TINY else 512
    if P.TINY:
        g = torch.Generator().manual_seed(0)
        return torch.randint(0, 1000, (2, seq), generator=g)
    from datasets import load_dataset
    tr = load_dataset('openai/gsm8k', 'main', split='train')
    va = [tok(tr[i]['question'] + '\n' + tr[i]['answer']).input_ids for i in range(len(tr) - N_VAL, len(tr))]
    flat = [t for ex in va for t in ex]
    n = len(flat) // seq
    return torch.tensor(flat[: n * seq], dtype=torch.long).view(n, seq)


def measure(seqs, tag):
    import loop_grad_probe as P
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)                                # TINY builds a random model: identical across processes
    tok, model = P.load(); ad = P.Ouro(model); b = ad.b
    params = list(model.parameters())                   # unique tensors (a tied embedding appears once)
    for p in params:
        p.requires_grad_(True)
    sizes = [p.numel() for p in params]; offs = np.cumsum([0] + sizes); ntot = int(offs[-1])
    layer_ids = {id(p) for p in ad.subset}
    loop_ids = {id(p) for p in b.layers[: b.config.num_hidden_layers].parameters()} | {id(p) for p in b.norm.parameters()}
    groups = [('all', lambda p: True), ('loop', lambda p: id(p) in loop_ids), ('layer', lambda p: id(p) in layer_ids)]
    data = val_chunks(tok, P)
    log('v197 driver %s | dev %s dtype %s | params %.1fM (loop %.1fM, %s %.1fM) | tied %s | %d chunks of %d tokens' % (
        driver_sha(), P.DEV, P.DTYPE, ntot / 1e6, sum(p.numel() for p in params if id(p) in loop_ids) / 1e6,
        ad.subset_name, sum(p.numel() for p in ad.subset) / 1e6,
        model.model.embed_tokens.weight is model.lm_head.weight, data.shape[0], data.shape[1]))

    def prepare(ids):                                   # as P.Ouro.prepare, with gradients into the embedding
        h = b.embed_tokens(ids)
        cp = torch.arange(ids.shape[1], device=ids.device); pos = cp.unsqueeze(0)
        mask = ad.mod.create_causal_mask(config=b.config, input_embeds=h, attention_mask=None, cache_position=cp,
                                         past_key_values=None, position_ids=pos)
        return h, dict(mask=mask, pos=pos, cp=cp, pe=b.rotary_emb(h, pos))

    def F_ckpt(h, t, ctx):                              # = P.Ouro.F, checkpointed per layer (T4 memory; same math)
        for layer in b.layers[: b.config.num_hidden_layers]:
            def run(x, layer=layer):
                y = layer(x, attention_mask=ctx['mask'], position_ids=ctx['pos'], past_key_value=None, use_cache=False,
                          cache_position=ctx['cp'], position_embeddings=ctx['pe'], current_ut=t)
                return y[0] if isinstance(y, tuple) else y
            h = checkpoint(run, h, use_reentrant=False)
        return b.norm(h)

    def backward(ids, k, adj=None, ckpt=True):
        for p in params:
            p.grad = None
        h, ctx = prepare(ids)
        with torch.no_grad():
            for t in range(R - k):
                h = ad.F(h, t, ctx)
        if k < R:
            h = h.detach()
        for t in range(R - k, R):
            h = F_ckpt(h, t, ctx) if ckpt else ad.F(h, t, ctx)
            if adj is not None:
                h.register_hook(lambda g, t=t: adj.__setitem__(t, float(g.float().norm())))
        loss = P.ce(ad.readout(h, ctx), ids)
        loss.backward()
        return float(loss)

    def to_cpu(buf):
        for p, o, n in zip(params, offs[:-1], sizes):
            if p.grad is None:
                buf[o:o + n].zero_()
            else:
                buf[o:o + n].copy_(p.grad.detach().float().flatten())

    def compare(gR):
        """For each group: |g - gR|^2, |g|^2, g.gR, |gR|^2 (float64), g = the current p.grad."""
        acc = {name: np.zeros(4) for name, _ in groups}
        for p, o, n in zip(params, offs[:-1], sizes):
            g = p.grad.detach().float().flatten() if p.grad is not None else None
            part = np.zeros(4)
            for s in range(0, n, CHUNK):
                e = min(n, s + CHUNK)
                r = gR[o + s:o + e].to(P.DEV).double()
                x = g[s:e].double() if g is not None else torch.zeros_like(r)
                part += [float(((x - r) ** 2).sum()), float((x * x).sum()), float((x * r).sum()), float((r * r).sum())]
            for name, sel in groups:
                if sel(p):
                    acc[name] += part
        return {name: dict(err=float(np.sqrt(a[0] / a[3])), ratio=float(np.sqrt(a[1] / a[3])),
                           cos=float(a[2] / np.sqrt(a[1] * a[3] + 1e-300))) for name, a in acc.items()}

    gR = torch.empty(ntot, dtype=torch.float32)
    out = dict(driver_sha=driver_sha(), R=R, ks=list(KS), n_params=ntot, rows=[], checks={})
    path = os.path.join(P.OUT, 'v197_ouro_%s.json' % tag)
    lo, hi = seqs
    for i in range(lo, min(hi, data.shape[0])):
        t0 = time.time(); ids = data[i:i + 1].to(P.DEV); adj = {}
        loss = backward(ids, R, adj=adj); to_cpu(gR)
        A_ = np.array([adj[t] for t in range(R)])
        row = dict(i=i, loss=loss, adj=A_.tolist(), rho=float(np.exp(-np.polyfit(np.arange(R), np.log(A_ + 1e-30), 1)[0])))
        if i == lo:                                     # noise floor: the identical full pass, run again
            backward(ids, R); out['checks']['repeat_full'] = compare(gR)
            log('  check: repeated full pass vs first, all-param err %.2e' % out['checks']['repeat_full']['all']['err'])
            if P.TINY:                                  # checkpointed vs plain loop (CPU only: same math)
                backward(ids, R, ckpt=False); out['checks']['plain_vs_ckpt'] = compare(gR)
                log('  check: plain vs checkpointed, all-param err %.2e' % out['checks']['plain_vs_ckpt']['all']['err'])
        for k in KS:
            backward(ids, k); row[str(k)] = compare(gR)
        row['seconds'] = round(time.time() - t0, 1); out['rows'].append(row)
        log('seq %d: loss %.3f rho %.3f adj %s | err all %s | loop %s | layer %s (%.0fs)' % (
            i, loss, row['rho'], ' '.join('%.3g' % a for a in A_),
            ' / '.join('%.3f' % row[str(k)]['all']['err'] for k in KS),
            ' / '.join('%.3f' % row[str(k)]['loop']['err'] for k in KS),
            ' / '.join('%.3f' % row[str(k)]['layer']['err'] for k in KS), row['seconds']))
        json.dump(out, open(path, 'w'), indent=1)
    for p in params:
        p.grad = None


def analyze(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, '**', 'v197_ouro_*.json'), recursive=True)):
        h = json.load(open(f)); rows += h['rows']
        print('%s: %d rows, driver %s, checks %s' % (os.path.relpath(f, d), len(h['rows']), h['driver_sha'],
              {c: round(v['all']['err'], 8) for c, v in h['checks'].items()}))
    rows = sorted({r['i']: r for r in rows}.values(), key=lambda r: r['i'])
    n = len(rows); rng = np.random.default_rng(197)
    print('n = %d sequences: %s' % (n, [r['i'] for r in rows]))

    def med_ci(x):
        x = np.asarray(x); bs = [np.median(x[rng.integers(0, n, n)]) for _ in range(4000)]
        return float(np.median(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    rho = np.array([r['rho'] for r in rows])
    print('rho: median %.3f [%.3f, %.3f]; per-sequence range %.3f - %.3f; rho_i > 1 in %d of %d' % (
        *med_ci(rho), rho.min(), rho.max(), int((rho > 1).sum()), n))
    for g in ('all', 'loop', 'layer'):
        print('%-5s err   ' % g + '  '.join('k=%d %.3f [%.3f, %.3f]' % (k, *med_ci([r[str(k)][g]['err'] for r in rows]))
                                         for k in KS))
    for q in ('ratio', 'cos'):
        print('all   %-5s ' % q + '  '.join('k=%d %.3f' % (k, np.median([r[str(k)]['all'][q] for r in rows])) for k in KS))
    al = lambda p, k: (p ** k - p ** R) / (1 - p ** R)
    ort = lambda p, k: np.sqrt((p ** (2 * k) - p ** (2 * R)) / (1 - p ** (2 * R)))
    print('geometric forms (median over sequences of each rho_i form):')
    for k in KS:
        print('  k=%d aligned %.3f orthogonal %.3f' % (k, np.median([al(p, k) for p in rho]), np.median([ort(p, k) for p in rho])))
    print('measured-profile forms (median over sequences):')
    for k in KS:
        a_, o_ = [], []
        for r in rows:
            s = np.array(r['adj'][::-1])                # s_j = signal j loops back from the output
            a_.append(s[k:].sum() / s.sum()); o_.append(np.sqrt((s[k:] ** 2).sum() / (s ** 2).sum()))
        print('  k=%d aligned %.3f orthogonal %.3f' % (k, np.median(a_), np.median(o_)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seqs', default='0:16'); ap.add_argument('--tag', default='x')
    ap.add_argument('--analyze', default=None); A = ap.parse_args()
    if A.analyze:
        return analyze(A.analyze)
    lo, hi = (int(x) for x in A.seqs.split(':'))
    measure((lo, hi), A.tag)


if __name__ == '__main__':
    main()
