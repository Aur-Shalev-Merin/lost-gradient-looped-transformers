"""v168 (EXPLORATORY, not pre-registered): does the DIRECTION of the truncated gradient, rather than the size of its
error, explain truncation cost on both Parcae-140m (where err predicted cost, v162-v166) and the SmolLM2 retrofit
(where it did not, v167)?

Frozen models, fp32, the exact losses used in training (v163 damped_loss for Parcae; v167 manual_loss for the
retrofit). Per variant (R, a): B batches of 8 GSM8K-val sequences x 512 tokens (the training batch shape), state-init
noise seeded identically across k. For each k, from batch gradients g_k (truncated) and g_R (full):
  err          |g_k - g_R| / |g_R|                       (the v162-v167 metric, here on batch gradients)
  cos          cos(g_k, g_R)
  useful_sgd   g_R . g_k / |g_R|^2                       (first-order loss decrease of an SGD step, relative to full)
  adam_cos     cos(u_k, u_R), u = g / (sqrt(v) + 1e-8), v = mean over the B batches of g^2 FOR THAT ARM
               (in truncated training Adam's second moment is built from the truncated gradients themselves)
  adam_err     |u_k - u_R| / |u_R|
  useful_adam  (g_R . u_k) / (g_R . u_R)                 (first-order loss decrease of an Adam-like step, rel. to full)
  sign_agree   sum |u_R| [sign g_k == sign g_R] / sum |u_R|
Per-batch values, then the mean over batches. All parameters.

  python v168_direction_probe.py --model parcae --variants 8:1.0:1,2,4 32:0.5:1,2,4,8 ... --tag T
  python v168_direction_probe.py --model retro  --variants 8:1.0:1,2 ... --tag T        (RETRO_DIR set)
"""
import argparse, json, os, time
import numpy as np
import torch

B = int(os.environ.get('V168_B', 4))
BS = 8


def setup(model):
    if model == 'parcae':
        import v162_parcae_trunc_train as v
        import v163_parcae_damped_trunc as d
        tok, m = v.load_model(); data = v.gsm8k_chunks(tok)

        def lossfn(ids, R, k, a):
            d.R = R
            return d.damped_loss(m, ids, k, a, ckpt=True)
        return m, data['val'], lossfn, v.OUT, v.DEV
    import v167_retrofit_trunc as r
    m = r.load_model(); data = r.gsm8k_chunks(r.load_tok())

    def lossfn(ids, R, k, a):
        return r.manual_loss(m, ids, k, ckpt=k > 0, a=a, r=R)
    return m, data['val'], lossfn, r.OUT, r.DEV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--variants', nargs='+', required=True); ap.add_argument('--tag', default='x')
    A = ap.parse_args()
    m, val, lossfn, out, dev = setup(A.model); m.eval()
    params = [p for p in m.parameters() if p.requires_grad]
    logf = os.path.join(out, 'v168_log.txt')

    def log(*x):
        s = ' '.join(str(y) for y in x); print(s, flush=True); open(logf, 'a').write(s + '\n')

    def grad(ids, R, k, a, seed):
        m.zero_grad(set_to_none=True); torch.manual_seed(seed)
        lossfn(ids, R, k, a).backward()
        g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().flatten() for p in params])
        m.zero_grad(set_to_none=True)
        return g
    log('v168 %s | params %.1fM | B %d x %d seqs | val chunks %d' % (A.model, sum(p.numel() for p in params) / 1e6, B, BS, len(val)))
    res = []
    for spec in A.variants:
        R, a, ks = spec.split(':'); R, a, ks = int(R), float(a), [int(x) for x in ks.split(',')]
        t0 = time.time()
        batches = [(val[i * BS:(i + 1) * BS].to(dev), 4242 + i) for i in range(B)]
        gR = [grad(ids, R, R, a, sd).cpu() for ids, sd in batches]
        sR = sum(g ** 2 for g in gR).div_(B).sqrt_().add_(1e-8)          # sqrt(v) of the full arm
        row = dict(model=A.model, R=R, a=a, metrics={})
        for k in ks:
            gk = [grad(ids, R, k, a, sd).cpu() for ids, sd in batches]
            sk = sum(g ** 2 for g in gk).div_(B).sqrt_().add_(1e-8)      # sqrt(v) of the truncated arm
            mt = {n: [] for n in ('err', 'cos', 'useful_sgd', 'adam_cos', 'adam_err', 'useful_adam', 'sign_agree')}
            for b in range(B):
                g1, g0 = gk[b], gR[b]; u1, u0 = g1 / sk, g0 / sR
                mt['err'].append(float((g1 - g0).norm() / g0.norm()))
                mt['cos'].append(float(g1 @ g0 / (g1.norm() * g0.norm())))
                mt['useful_sgd'].append(float(g0 @ g1 / (g0 @ g0)))
                mt['adam_cos'].append(float(u1 @ u0 / (u1.norm() * u0.norm())))
                mt['adam_err'].append(float((u1 - u0).norm() / u0.norm()))
                mt['useful_adam'].append(float(g0 @ u1 / (g0 @ u0)))
                w = u0.abs(); mt['sign_agree'].append(float((w * (torch.sign(g1) == torch.sign(g0)).float()).sum() / w.sum()))
            row['metrics'][k] = {n: float(np.mean(x)) for n, x in mt.items()}
            log('%s R=%d a=%.2f k=%d: %s' % (A.model, R, a, k, {n: round(x, 4) for n, x in row['metrics'][k].items()}))
        res.append(row)
        log('  variant R=%d a=%.2f done in %ds' % (R, a, time.time() - t0))
        json.dump(res, open(os.path.join(out, 'v168_%s.json' % A.tag), 'w'), indent=1)


if __name__ == '__main__':
    main()
