"""v166 pre-freeze pilot, block A (frozen model only; prereg draft docs/prereg_v166_slam_dunk.md).

Over-relaxation feasibility: x <- (1 - a) x + a F(x) with a > 1. For a in {1.0 (reference), 1.15, 1.3}:
  - held-out GSM8K val/test loss at R in {32, 64} (no grad): does the over-relaxed loop reach the SAME converged function
    as the undamped model (|loss(a) - loss(1.0)| <= 0.01 at R 32), or does it oscillate / diverge?
  - frozen-model rho and gradient error err_k for k in {1, 2, 3, 4} at R 32 (v164's measurement code).
Gate (prereg §2): a = 1.3 usable iff converged loss within 0.01 of a = 1.0 at R 32 AND rho < the a = 1.0 rho; else
a = 1.15 under the same test; else block A's over-relaxation arm is dropped before the freeze.
Training LR checks for the over-relaxed controls are run with v164_parcae_predict.py --train (cells R:a:k:lr:seed).
"""
import json, os
import torch

import v164_parcae_predict as p
import v163_parcae_damped_trunc as d
import v162_parcae_trunc_train as v


def main():
    tok, m = v.load_model(); data = v.gsm8k_chunks(tok)
    p.log('v166 pilot A (driver %s)' % p.driver_sha())
    p.KS = (1, 2, 3, 4)
    res = {}
    for a in (1.0, 1.15, 1.3):
        ev = {}
        for R in (32, 64):
            d.R = R
            ev[R] = (d.eval_loss(m, data['val'], a), d.eval_loss(m, data['test'], a))
        meas = p.measure(m, data['val'], 32, a)
        res[a] = dict(eval=ev, rho=meas['rho_median'], err=meas['err_median'])
        p.log('a=%.2f: val/test R32 %.4f/%.4f R64 %.4f/%.4f | rho %.4f | err %s' % (
            a, ev[32][0], ev[32][1], ev[64][0], ev[64][1], meas['rho_median'], {k: round(x, 4) for k, x in meas['err_median'].items()}))
    base = res[1.0]
    for a in (1.3, 1.15):
        same = abs(res[a]['eval'][32][1] - base['eval'][32][1]) <= 0.01
        lower = res[a]['rho'] < base['rho']
        p.log('gate a=%.2f: same converged test loss at R32 (|diff| %.4f <= 0.01): %s | rho %.4f < %.4f: %s -> %s' % (
            a, abs(res[a]['eval'][32][1] - base['eval'][32][1]), same, res[a]['rho'], base['rho'], lower, 'USABLE' if same and lower else 'NOT USABLE'))
    json.dump({str(k): x for k, x in res.items()}, open(os.path.join(v.OUT, 'v166_pilotA.json'), 'w'), default=str)


if __name__ == '__main__':
    main()
