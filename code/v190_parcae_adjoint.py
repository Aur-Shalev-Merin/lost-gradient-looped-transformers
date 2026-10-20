"""v190: v189 with the per-loop backward-signal sizes saved (plan: docs/v190_parcae_adjoint_plan.md).
Same computation as v164 measure() (16 GSM8K-val sequences, batch 1, fp32, all parameters, a = 1, same seeds), but each
row keeps adj[t] = ||dL/dx_t|| for t = 0 .. R-1 (t = R-1 is the loop nearest the output).
  python v190_parcae_adjoint.py --repo SandyResearch/parcae-370m --R 32 --tag 370m
"""
import argparse, json, os, time
import numpy as np
import torch

import v162_parcae_trunc_train as v
import v163_parcae_damped_trunc as d
import v164_parcae_predict as p


def measure_adj(m, chunks, R, ks):
    d.R = R
    params = [q for q in m.parameters() if q.requires_grad]
    grad = lambda: torch.cat([(q.grad if q.grad is not None else torch.zeros_like(q)).flatten() for q in params])
    m.eval(); rows = []
    for i in range(min(v.N_MEASURE, len(chunks))):
        ids = chunks[i:i + 1].to(v.DEV); adj = {}
        torch.manual_seed(777 + i); m.zero_grad(set_to_none=True)
        d.damped_loss(m, ids, R, 1.0, adj=adj).backward(); gR = grad()
        A_ = np.array([adj[t] for t in range(R)])
        row = dict(rho=float(np.exp(-np.polyfit(np.arange(R), np.log(A_ + 1e-30), 1)[0])), adj=A_.tolist(), err={})
        for k in ks:
            torch.manual_seed(777 + i); m.zero_grad(set_to_none=True)
            d.damped_loss(m, ids, k, 1.0, ckpt=False).backward()
            row['err'][k] = float((grad() - gR).norm() / gR.norm())
        rows.append(row)
    m.zero_grad(set_to_none=True)
    return dict(R=R, a=1.0, rho_median=float(np.median([r['rho'] for r in rows])),
                err_median={k: float(np.median([r['err'][k] for r in rows])) for k in ks}, rows=rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--repo', required=True); ap.add_argument('--R', type=int, default=32)
    ap.add_argument('--tag', default='x'); A = ap.parse_args()
    v.REPO = A.repo
    tok, m = v.load_model(); data = v.gsm8k_chunks(tok)
    print('v190: repo %s | v164 driver %s' % (v.REPO, p.driver_sha()), flush=True)
    t0 = time.time(); r = measure_adj(m, data['val'], A.R, tuple(k for k in (1, 2, 4, 8, 16) if k < A.R))
    r['seconds'] = round(time.time() - t0); r['repo'] = v.REPO
    print('R=%d: rho %.4f err %s (%ds)' % (A.R, r['rho_median'], {k: round(x, 4) for k, x in r['err_median'].items()},
                                           r['seconds']), flush=True)
    json.dump(r, open(os.path.join(v.OUT, 'v190_%s.json' % A.tag), 'w'))


if __name__ == '__main__':
    main()
