"""v166 pre-freeze pilot, block B (prereg draft docs/prereg_v166_slam_dunk.md): is Parcae-370m usable for the
cross-model transfer test on a Kaggle T4?

Reuses the frozen v162/v163/v164 machinery unchanged; only the checkpoint repo is switched (v162's REPO global is read
at load time). Steps:
  --check   v163 selftest at R 32 (a = 1 vs v162 forward, checkpoint, fp16 vs fp32 truncated grads), step-0 val/test,
            frozen-model rho / err_k (k in 2, 4, 8) at a in {1.0, 0.5}, peak GPU memory
  --train   cells 'a:k:lr:seed' at R 32 (control LR checks); logs time and peak memory per run
Gates (prereg §3): loads with no missing/unexpected keys; selftest passes; full run fits in memory and trains
(>= 0.10 nats below step 0); floor gap >= 0.03.
"""
import argparse, json, os, time
import torch

import v162_parcae_trunc_train as v
import v163_parcae_damped_trunc as d
import v164_parcae_predict as p

v.REPO = 'SandyResearch/parcae-370m'


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(v.OUT, 'v166_pilotB_log.txt'), 'a').write(s + '\n')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--check', action='store_true'); ap.add_argument('--train', default=None)
    ap.add_argument('--tag', default='x'); A = ap.parse_args()
    tok, m = v.load_model(); data = v.gsm8k_chunks(tok)
    log('pilot B: repo %s | params %.1fM | driver %s' % (v.REPO, sum(x.numel() for x in m.parameters()) / 1e6, p.driver_sha()))
    if A.check:
        d.R = 32; d.selftest(m, data)
        p.KS = (2, 4, 8); res = {}
        for a in (1.0, 0.5):
            r = p.measure(m, data['val'], 32, a)
            ev = (d.eval_loss(m, data['val'], a), d.eval_loss(m, data['test'], a))
            res[a] = dict(rho=r['rho_median'], err=r['err_median'], step0=ev)
            log('a=%.1f R=32: rho %.4f err %s | step-0 val/test %.4f / %.4f' % (a, r['rho_median'],
                {k: round(x, 4) for k, x in r['err_median'].items()}, ev[0], ev[1]))
        log('peak GPU memory after check: %.2f GB' % (torch.cuda.max_memory_allocated() / 1e9))
        json.dump({str(k): x for k, x in res.items()}, open(os.path.join(v.OUT, 'v166_pilotB_check.json'), 'w'), default=str)
    if A.train:
        del m; torch.cuda.empty_cache()
        for cell in A.train.split(','):
            a, k, lr, seed = cell.split(':'); a, k, lr, seed = float(a), int(k), float(lr), int(seed)
            d.R = 32; torch.cuda.reset_peak_memory_stats(); t0 = time.time()
            try:
                r = d.train_cell(data, a, k, lr, seed)
            except torch.OutOfMemoryError as e:
                log('cell a=%.1f k=%d lr=%g: OUT OF MEMORY (%s)' % (a, k, lr, str(e)[:120])); torch.cuda.empty_cache(); continue
            r['peak_gb'] = torch.cuda.max_memory_allocated() / 1e9; r['repo'] = v.REPO
            log('cell a=%.1f k=%d lr=%g seed=%d: val %.4f test %.4f nonfinite %d | %ds | peak %.2f GB' % (
                a, k, lr, seed, r['val'], r['test'], r['nonfinite_steps'], time.time() - t0, r['peak_gb']))
            open(os.path.join(v.OUT, 'v166_pilotB_runs_%s.jsonl' % A.tag), 'a').write(json.dumps(r) + '\n')


if __name__ == '__main__':
    main()
