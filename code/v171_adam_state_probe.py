"""v171 (EXPLORATORY; plan fixed before data in docs/v171_plan.md): Adam's steady-state step direction at the frozen
point, per arm, from many batches, to test whether an optimiser-aware deficit puts Parcae-140m and the SmolLM2 retrofit
on one cost curve (v168 used only 4 single batches).

Per variant (R, a), B batches of 8 TRAINING sequences x 512 tokens (the data Adam sees in training), state-init noise
seeded identically across arms. For the full arm (k = R) and each truncated arm k:
  gbar = mean_b g_b,  s = sqrt(mean_b g_b^2) + 1e-8,  u = gbar / s        (Adam's update direction with m ~ gbar,
                                                                          v ~ mean g^2 of THAT arm's own gradients)
Metrics (all parameters):
  adam_def  = 1 - (gbar_R . u_k) / (gbar_R . u_R)     PRIMARY: first-order loss-decrease deficit of the Adam step
  sgd_def   = 1 - (gbar_R . gbar_k) / |gbar_R|^2       SGD analogue on mean gradients
  err_mean  = |gbar_k - gbar_R| / |gbar_R|
  adam_cos  = cos(u_k, u_R)
Running sums only (no per-batch storage).
v171b (2026-10-05): each batch gradient is clipped to global norm <= 1.0 as in training (the first run's unclipped means
were dominated by outlier batches); pre-clip norms are logged.

  python v171_adam_state_probe.py --model parcae|retro --variants R:a:k1,k2,... ... --tag T
"""
import argparse, json, os, time
import numpy as np
import torch

B = int(os.environ.get('V171_B', 24))
CLIP = os.environ.get('V171_CLIP', '1') == '1'   # v171b default: clip each batch gradient to norm 1 (training-matched)
BS = 8


def setup(model):
    if model == 'parcae':
        import v162_parcae_trunc_train as v
        import v163_parcae_damped_trunc as d
        tok, m = v.load_model(); data = v.gsm8k_chunks(tok)
        train = v.pack(data['train'], np.random.default_rng(171).permutation(len(data['train'])))

        def lossfn(ids, R, k, a):
            d.R = R
            return d.damped_loss(m, ids, k, a, ckpt=True)
        return m, train, lossfn, v.OUT, v.DEV
    import v167_retrofit_trunc as r
    m = r.load_model(); data = r.gsm8k_chunks(r.load_tok())
    train = r.pack(data['train'], np.random.default_rng(171).permutation(len(data['train'])))

    def lossfn(ids, R, k, a):
        return r.manual_loss(m, ids, k, ckpt=k > 0, a=a, r=R)
    return m, train, lossfn, r.OUT, r.DEV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--variants', nargs='+', required=True); ap.add_argument('--tag', default='x')
    A = ap.parse_args()
    m, train, lossfn, out, dev = setup(A.model); m.eval()
    params = [p for p in m.parameters() if p.requires_grad]
    logf = os.path.join(out, 'v171_log.txt')

    def log(*x):
        s = ' '.join(str(y) for y in x); print(s, flush=True); open(logf, 'a').write(s + '\n')

    def sums(R, k, a):
        s1, s2 = None, None; norms = []
        for i in range(B):
            ids = train[i * BS:(i + 1) * BS].to(dev)
            m.zero_grad(set_to_none=True); torch.manual_seed(4242 + i)
            lossfn(ids, R, k, a).backward()
            g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().flatten() for p in params]).double().cpu()
            nrm = float(g.norm()); norms.append(round(nrm, 4))
            if CLIP:            # v171b: exactly as training's clip_grad_norm_(params, 1.0) before AdamW
                g.mul_(min(1.0, 1.0 / (nrm + 1e-6)))
            s1 = g.clone() if s1 is None else s1.add_(g)
            s2 = g * g if s2 is None else s2.add_(g * g)
        m.zero_grad(set_to_none=True)
        log('    R=%d a=%.2f k=%d batch grad norms (pre-clip): median %.3f max %.3f' % (R, a, k, float(np.median(norms)), max(norms)))
        return s1 / B, (s2 / B).sqrt_().add_(1e-8)
    log('v171b clip=%s | ' % CLIP + 'v171 %s | params %.1fM | B %d x %d training seqs (train chunks %d)' % (A.model, sum(p.numel() for p in params) / 1e6, B, BS, len(train)))
    res = []
    for spec in A.variants:
        R, a, ks = spec.split(':'); R, a, ks = int(R), float(a), [int(x) for x in ks.split(',')]
        t0 = time.time()
        gR, sR = sums(R, R, a); uR = gR / sR; base = float(gR @ uR)
        row = dict(model=A.model, R=R, a=a, metrics={})
        for k in ks:
            gk, sk = sums(R, k, a); uk = gk / sk
            mt = dict(adam_def=1 - float(gR @ uk) / base, sgd_def=1 - float(gR @ gk) / float(gR @ gR),
                      err_mean=float((gk - gR).norm() / gR.norm()), adam_cos=float(uk @ uR / (uk.norm() * uR.norm())))
            row['metrics'][k] = mt
            log('%s R=%d a=%.2f k=%d: %s' % (A.model, R, a, k, {n: round(x, 4) for n, x in mt.items()}))
        res.append(row)
        log('  variant R=%d a=%.2f done in %ds' % (R, a, time.time() - t0))
        json.dump(res, open(os.path.join(out, 'v171_%s.json' % A.tag), 'w'), indent=1)


if __name__ == '__main__':
    main()
