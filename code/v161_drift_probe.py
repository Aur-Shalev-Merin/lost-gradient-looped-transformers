"""v161 EXPLORATORY (post-freeze, not part of the verdict): does training move rho and the truncation error?

The v161 tuning wave suggested that the cost of truncation hardly depends on k, which the step-0 law does not
predict. One candidate: training changes the loop dynamics, so the frozen-model rho stops describing the gradients
seen during training. This script reuses the frozen driver unchanged (imported as a module), trains one cell with
the same recipe (seed, data order, LR 3e-4, 600 steps), and re-runs the frozen-model measurement at steps 0, 150,
300 and 600.

  python v161_drift_probe.py R:k[,R:k...] --tag T     (seed 1611101, LR 3e-4; OUT_DIR as in the driver)
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v161_parcae_trunc_train as v

SEED, LR, CHECK = 1611101, 3e-4, (0, 150, 300, 600)


def run(tok, data, R, k):
    _, model = v.load_model(); model.train()
    g = np.random.default_rng(SEED)
    chunks = torch.cat([v.pack(data['train'], g.permutation(len(data['train']))) for _ in range(4)])
    torch.manual_seed(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    sched = lambda s: min(1.0, (s + 1) / v.WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / v.STEPS)))
    scaler = torch.amp.GradScaler('cuda', enabled=v.AMP)
    ks = [j for j in (1, 2, 4, 8) if j < R]; out = []
    for s in range(v.STEPS + 1):
        if s in CHECK:
            rng_state = torch.get_rng_state(); cuda_state = torch.cuda.get_rng_state_all() if v.DEV == 'cuda' else None
            m = v.measure(model, data['val'], R, ks)
            torch.set_rng_state(rng_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            row = dict(step=s, **m['summary'], val=v.eval_loss(model, data['val'], R))
            v.log('drift R=%d k=%d step %d: rho %.3f err %s law %s val %.4f' % (R, k, s, row['rho_median'],
                  {a: round(b, 3) for a, b in row['err_median'].items()}, {a: round(b, 3) for a, b in row['law_median'].items()}, row['val']))
            out.append(row)
        if s == v.STEPS:
            break
        for gp in opt.param_groups:
            gp['lr'] = LR * sched(s)
        ids = chunks[s * v.BATCH:(s + 1) * v.BATCH].to(v.DEV)
        loss = v.loss_model(model, ids, R, k)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), v.CLIP); scaler.step(opt); scaler.update()
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cells'); ap.add_argument('--tag', default='x'); a = ap.parse_args()
    tok, _ = v.load_model(); data = v.gsm8k_chunks(tok)
    res = {}
    for c in a.cells.split(','):
        R, k = map(int, c.split(':')); t0 = time.time()
        res[c] = run(tok, data, R, k)
        v.log('drift cell %s done (%ds)' % (c, time.time() - t0))
        json.dump(dict(driver_sha=v.driver_sha(), seed=SEED, lr=LR, res=res),
                  open(os.path.join(v.OUT, 'v161_drift_%s.json' % a.tag), 'w'), indent=1)


if __name__ == '__main__':
    main()
