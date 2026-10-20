"""v189: frozen-model lost gradient vs rho^k for a Parcae checkpoint at several loop counts (plan:
docs/v189_parcae370_rho_plan.md). Reuses the frozen v162/v163/v164 machinery unchanged: v164 measure() is the Table 1
protocol (16 GSM8K-val sequences, batch 1, fp32, all parameters, a = 1). Unlike the v166 pilot it saves the
per-sequence rows, so the paper can use the median of rho_i^k and bootstrap intervals.
  python v189_parcae_rho.py --repo SandyResearch/parcae-370m --R 16,32 --tag 370m
"""
import argparse, json, os, time

import v162_parcae_trunc_train as v
import v163_parcae_damped_trunc as d
import v164_parcae_predict as p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--repo', required=True); ap.add_argument('--R', default='16')
    ap.add_argument('--tag', default='x'); A = ap.parse_args()
    v.REPO = A.repo
    tok, m = v.load_model(); data = v.gsm8k_chunks(tok)
    print('v189: repo %s | params %.1fM | v164 driver %s' % (v.REPO, sum(x.numel() for x in m.parameters()) / 1e6,
                                                             p.driver_sha()), flush=True)
    out = dict(repo=v.REPO, driver_sha=p.driver_sha(), res=[])
    for R in [int(r) for r in A.R.split(',')]:
        p.KS = tuple(k for k in (1, 2, 4, 8, 16) if k < R); t0 = time.time()
        r = p.measure(m, data['val'], R, 1.0); r['seconds'] = round(time.time() - t0); out['res'].append(r)
        print('R=%d: rho %.4f err %s (%ds)' % (R, r['rho_median'], {k: round(x, 4) for k, x in r['err_median'].items()},
                                               r['seconds']), flush=True)
        json.dump(out, open(os.path.join(v.OUT, 'v189_%s.json' % A.tag), 'w'))


if __name__ == '__main__':
    main()
