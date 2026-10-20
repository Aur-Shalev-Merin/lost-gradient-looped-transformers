"""v182 (EXPLORATORY, measurement only, no training): checks requested by the external review of the looped-transformer
paper (draft v6, paper_looped/draft.md). Frozen models, fp32 forward/backward, all reductions in float64.

  --subset R:K,K,...   Table 1 protocol (as v162.measure / v167.measure: first N_MEASURE GSM8K-val sequences, batch 1,
                       state-init seed 777+i, the plain truncated loss). From the SAME backward passes, the relative
                       gradient error over parameter groups:
                         all            every trainable parameter (Table 1 rows for Parcae-140m and SmolLM2)
                         adapter+core0  input-injection adapter + first loop layer (Table 1's Huginn / Parcae-1.3B choice)
                         coreI          loop layer I alone (Table 1's Ouro choice is one middle loop layer)
                       Question: does the part-of-loop measurement agree with the all-parameter one on the same model?
  --table2 R:A:K,...   v168's Table 2 measurement (B=4 batches x 8 GSM8K-val sequences, seeds 4242+i, the exact training
                       losses: v163 damped_loss / v167 manual_loss), with every dot product in float64. v168 used float32
                       dot products over ~1.35e8 weights on CPU, which run ~1% high; its adam_cos reached 1.017 on SmolLM2.
                       Per batch: err, r = |g_k|/|g_R|, cos (err^2 = 1 + r^2 - 2 r cos holds per batch, not for the
                       batch means), cos_fp32 (v168's computation, for comparison); adam_cos as v168
                       (u = g / (sqrt(mean_b g^2) + 1e-8), v per arm).

  --probe subset|loop|subset_bf16  (Parcae-140m only) run loop_grad_probe.py, the script behind Table 1's starred rows
                       (128-token WikiText-103 + GSM8K-test sequences, its own loop code), unchanged except: the model is
                       Parcae-140m, and the gradient is taken over its usual subset (adapter + core0), over the whole loop
                       (adapter + every core layer; the probe computes the prelude without gradient, so 'all parameters'
                       is not available there), or over the subset in bfloat16 (Huginn's precision). Set OUT_DIR per run.
                       With --subset (the Table 1 protocol) this splits the starred-row question in two: the effect of the
                       parameter subset, and the effect of everything else in the probe's setup.

  python v182_review_checks.py --model parcae --subset 16:1,2,4,8 --tag T
  python v182_review_checks.py --model retro  --table2 8:1.0:1,2 8:0.3:1,2 --tag T        (RETRO_DIR set)
"""
import argparse, hashlib, json, os, re, time
import numpy as np
import torch

B = int(os.environ.get('V182_B', 4))
BS = int(os.environ.get('V182_BS', 8))
CH = 1 << 24


def driver_sha():
    return hashlib.sha256(open(os.path.abspath(__file__), 'rb').read().replace(b'\r\n', b'\n')).hexdigest()[:12]


def setup(model):
    if model == 'parcae':
        import v162_parcae_trunc_train as v
        import v163_parcae_damped_trunc as d
        tok, m = v.load_model(); val = v.gsm8k_chunks(tok)['val']

        def loss1(ids, R, k, adj=None):
            return v.loss_manual(m, ids, R, k, adj)

        def loss2(ids, R, k, a):
            d.R = R
            return d.damped_loss(m, ids, k, a, ckpt=True)
        return m, val, loss1, loss2, v.OUT, v.DEV, v.N_MEASURE
    import v167_retrofit_trunc as r
    m = r.load_model(); val = r.gsm8k_chunks(r.load_tok())['val']

    def loss1(ids, R, k, adj=None):
        return r.manual_loss(m, ids, k, adj=adj, r=R)

    def loss2(ids, R, k, a):
        return r.manual_loss(m, ids, k, ckpt=k > 0, a=a, r=R)
    return m, val, loss1, loss2, r.OUT, r.DEV, r.N_MEASURE


def groups_of(names):
    core = sorted({int(x) for n in names for x in re.findall(r'transformer\.core_block\.(\d+)\.', n)})
    g = {'all': list(range(len(names))),
         'adapter+core0': [i for i, n in enumerate(names) if 'transformer.adapter' in n or 'transformer.core_block.0.' in n],
         'loop': [i for i, n in enumerate(names) if 'transformer.adapter' in n or 'transformer.core_block.' in n]}
    for c in core:
        g['core%d' % c] = [i for i, n in enumerate(names) if 'transformer.core_block.%d.' % c in n]
    assert core and all(g.values()), {k: len(v) for k, v in g.items()}
    return g


def dot64(a, b):
    return sum(float(torch.dot(a[i:i + CH].double(), b[i:i + CH].double())) for i in range(0, a.numel(), CH))


def run_probe(variant):
    os.environ['LOOP_MODEL'] = 'parcae'
    import loop_grad_probe as L
    L.REPO = 'SandyResearch/parcae-140m'
    if variant == 'subset_bf16':
        L.DTYPE = torch.bfloat16
    init = L.Parcae.__init__

    def patched(self, m):
        init(self, m)
        if variant == 'loop':
            uses_ve = str(m.config.n_layers_in_prelude) in m.value_embeds
            self.subset = list(m.transformer.adapter.parameters()) + [
                p for blk in m.transformer.core_block for n, p in blk.named_parameters() if uses_ve or 've_gate' not in n]
            self.subset_name = 'adapter + every core_block layer (the whole loop)'
        self.subset_name += ' | v182 probe variant %s, driver %s' % (variant, driver_sha())
    L.Parcae.__init__ = patched
    L.main()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--tag', default='x')
    ap.add_argument('--subset', nargs='*', default=[]); ap.add_argument('--table2', nargs='*', default=[])
    ap.add_argument('--probe', choices=['subset', 'loop', 'subset_bf16'])
    A = ap.parse_args()
    if A.probe:
        assert A.model == 'parcae'
        return run_probe(A.probe)
    m, val, loss1, loss2, out, dev, nmeas = setup(A.model); m.eval()
    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    names, params = [n for n, _ in named], [p for _, p in named]
    logf = os.path.join(out, 'v182_log_%s.txt' % A.tag)

    def log(*x):
        s = ' '.join(str(y) for y in x); print(s, flush=True); open(logf, 'a').write(s + '\n')

    def grads():
        return [(p.grad if p.grad is not None else torch.zeros_like(p)).detach().float() for p in params]
    res = dict(model=A.model, driver_sha=driver_sha(), subset=[], table2=[])
    log('v182 %s sha %s | params %.1fM | val chunks %d' % (A.model, driver_sha(), sum(p.numel() for p in params) / 1e6, len(val)))

    for spec in A.subset:
        R, ks = spec.split(':'); R, ks = int(R), [int(x) for x in ks.split(',')]
        G = groups_of(names); rows = []; t0 = time.time()
        log('subset R=%d groups %s' % (R, {k: sum(params[i].numel() for i in v) for k, v in G.items()}))
        for i in range(min(nmeas, len(val))):
            ids = val[i:i + 1].to(dev); adj = {}
            torch.manual_seed(777 + i); m.zero_grad(set_to_none=True)
            loss1(ids, R, R, adj).backward(); gR = grads()
            a = np.array([adj[t] for t in range(R)])
            row = dict(i=i, rho=float(np.exp(-np.polyfit(np.arange(R), np.log(a + 1e-30), 1)[0])), err={})
            n2 = [float((g.double() ** 2).sum()) for g in gR]
            for k in ks:
                torch.manual_seed(777 + i); m.zero_grad(set_to_none=True)
                loss1(ids, R, k).backward(); gk = grads()
                d2 = [float(((x.double() - y.double()) ** 2).sum()) for x, y in zip(gk, gR)]
                row['err'][k] = {gn: (sum(d2[j] for j in ix) / sum(n2[j] for j in ix)) ** 0.5 for gn, ix in G.items()}
            rows.append(row)
        m.zero_grad(set_to_none=True)
        summ = dict(R=R, n=len(rows), rho_median=float(np.median([r['rho'] for r in rows])),
                    err_median={k: {gn: float(np.median([r['err'][k][gn] for r in rows])) for gn in G} for k in ks})
        res['subset'].append(dict(summary=summ, rows=rows))
        log('subset R=%d rho %.3f (%ds)' % (R, summ['rho_median'], time.time() - t0))
        for k in ks:
            log('  k=%d ' % k + ' '.join('%s %.4f' % (gn, v) for gn, v in summ['err_median'][k].items()))
        json.dump(res, open(os.path.join(out, 'v182_%s.json' % A.tag), 'w'), indent=1)

    def flat(gs):
        return torch.cat([g.flatten() for g in gs]).cpu()
    for spec in A.table2:
        R, a, ks = spec.split(':'); R, a, ks = int(R), float(a), [int(x) for x in ks.split(',')]
        t0 = time.time(); batches = [(val[i * BS:(i + 1) * BS].to(dev), 4242 + i) for i in range(B)]
        assert len(val) >= B * BS, (len(val), B, BS)

        def bgrad(k, ids, sd):
            m.zero_grad(set_to_none=True); torch.manual_seed(sd)
            loss2(ids, R, k, a).backward(); g = flat(grads()); m.zero_grad(set_to_none=True)
            return g
        gR = [bgrad(R, ids, sd) for ids, sd in batches]
        sR = (sum(g.double() ** 2 for g in gR) / B).sqrt().add(1e-8).float()
        row = dict(R=R, a=a, metrics={})
        for k in ks:
            gk = [bgrad(k, ids, sd) for ids, sd in batches]
            sk = (sum(g.double() ** 2 for g in gk) / B).sqrt().add(1e-8).float()
            per = []
            for b in range(B):
                g1, g0 = gk[b], gR[b]; u1, u0 = g1 / sk, g0 / sR
                n1, n0, d = dot64(g1, g1) ** 0.5, dot64(g0, g0) ** 0.5, dot64(g1, g0)
                r_, c_ = n1 / n0, d / (n1 * n0)
                per.append(dict(err=max(n1 * n1 + n0 * n0 - 2 * d, 0.0) ** 0.5 / n0, r=r_, cos=c_,
                                adam_cos=dot64(u1, u0) / (dot64(u1, u1) * dot64(u0, u0)) ** 0.5,
                                cos_fp32=float(g1 @ g0 / (g1.norm() * g0.norm()))))
            row['metrics'][k] = dict(per_batch=per, mean={n: float(np.mean([p[n] for p in per])) for n in per[0]})
            log('table2 R=%d a=%.2f k=%d: %s' % (R, a, k, {n: round(v, 4) for n, v in row['metrics'][k]['mean'].items()}))
        res['table2'].append(row)
        log('  variant R=%d a=%.2f done in %ds' % (R, a, time.time() - t0))
        json.dump(res, open(os.path.join(out, 'v182_%s.json' % A.tag), 'w'), indent=1)
    json.dump(res, open(os.path.join(out, 'v182_%s.json' % A.tag), 'w'), indent=1)
    log('v182 done')


if __name__ == '__main__':
    main()
