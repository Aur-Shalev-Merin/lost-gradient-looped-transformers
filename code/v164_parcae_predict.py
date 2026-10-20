"""v164: out-of-sample prediction of truncation cost (prereg: docs/prereg_v164_parcae_prediction.md).

Reuses the frozen v163 driver (v163_parcae_damped_trunc.py, which reuses v162's) for the damped loop, training recipe,
evaluation and end-of-run rho. The only addition is a per-cell loop count R (v163 had R = 32 fixed): the v163 module's
global R is set before each cell; every v163 function reads it at call time.

Cells: 'R:a:k:lr:seed'. Modes:
  --measure   frozen-model gradient error err_k (and rho) for the new test variants, k in {2, 3, 4, 6}
  --selftest  v163 selftest at R = 64 (a = 1 vs v162 forward, checkpoint, fp16 vs fp32) for the new loop count
  --train CELLS --tag T
"""
import argparse, hashlib, json, os
import numpy as np
import torch

import v163_parcae_damped_trunc as d
import v162_parcae_trunc_train as v

MEASURE = [(32, 1.0), (32, 0.7), (64, 0.3)]     # (R, a) test variants
KS = (2, 3, 4, 6)


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(v.OUT, 'v164_log.txt'), 'a').write(s + '\n')


def driver_sha():
    h = hashlib.sha256()
    for f in (__file__, d.__file__, v.__file__):
        h.update(open(os.path.abspath(f), 'rb').read().replace(b'\r\n', b'\n'))
    return h.hexdigest()[:12]


def measure(m, chunks, R, a):
    d.R = R
    params = [p for p in m.parameters() if p.requires_grad]
    grad = lambda: torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in params])
    m.eval(); rows = []
    for i in range(min(v.N_MEASURE, len(chunks))):
        ids = chunks[i:i + 1].to(v.DEV); adj = {}
        torch.manual_seed(777 + i); m.zero_grad(set_to_none=True)
        d.damped_loss(m, ids, R, a, adj=adj).backward(); gR = grad()
        A_ = np.array([adj[t] for t in range(R)])
        row = dict(rho=float(np.exp(-np.polyfit(np.arange(R), np.log(A_ + 1e-30), 1)[0])), err={})
        for k in KS:
            torch.manual_seed(777 + i); m.zero_grad(set_to_none=True)
            d.damped_loss(m, ids, k, a, ckpt=False).backward()
            row['err'][k] = float((grad() - gR).norm() / gR.norm())
        rows.append(row)
    m.zero_grad(set_to_none=True); m.train()
    return dict(R=R, a=a, rho_median=float(np.median([r['rho'] for r in rows])),
                err_median={k: float(np.median([r['err'][k] for r in rows])) for k in KS}, rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--measure', action='store_true'); ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--train', default=None); ap.add_argument('--tag', default='x')
    A = ap.parse_args()
    tok, m = v.load_model(); data = v.gsm8k_chunks(tok)
    log('v164 driver %s (v163 module %s) dev %s amp %s' % (driver_sha(), d.driver_sha(), v.DEV, v.AMP))
    if A.selftest:
        d.R = 64; d.selftest(m, data)
    if A.measure:
        res = []
        for R, a in MEASURE:
            r = measure(m, data['val'], R, a); res.append(r)
            log('measure R=%d a=%.1f: rho %.4f err %s' % (R, a, r['rho_median'], {k: round(x, 4) for k, x in r['err_median'].items()}))
            d.R = R
            log('  step-0 val/test R=%d a=%.1f: %.4f / %.4f' % (R, a, d.eval_loss(m, data['val'], a), d.eval_loss(m, data['test'], a)))
        json.dump(dict(driver_sha=driver_sha(), res=res), open(os.path.join(v.OUT, 'v164_measure_%s.json' % A.tag), 'w'))
    if A.train:
        del m
        for cell in A.train.split(','):
            R, a, k, lr, seed = cell.split(':'); R, a, k, lr, seed = int(R), float(a), int(k), float(lr), int(seed)
            d.R = R
            r = d.train_cell(data, a, k, lr, seed); r['driver_sha'] = driver_sha()
            log('cell R=%d a=%.1f k=%d lr=%g seed=%d: val %.4f test %.4f rho_end %.3f nonfinite %d (%ds)' % (
                R, a, k, lr, seed, r['val'], r['test'], r['rho_end'], r['nonfinite_steps'], r['s']))
            open(os.path.join(v.OUT, 'v164_runs_%s.jsonl' % A.tag), 'a').write(json.dumps(r) + '\n')
            torch.cuda.empty_cache() if v.DEV == 'cuda' else None


if __name__ == '__main__':
    main()
