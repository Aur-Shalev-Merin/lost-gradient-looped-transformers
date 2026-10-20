"""Frozen-model loop-gradient probe for the healed looped SmolLM2 retrofits (raven/Huginn architecture).
Same measurement as loop_grad_probe.py / v162's --measure: per held-out sequence, the gradient of the next-token loss
w.r.t. ALL parameters with backprop through only the last k of R loops vs all R; relative error err_k; the adjoint norms
||dL/dh_t|| along the loop; rho = exp(-slope) of a log-linear fit over t; law value rho^k.

The forward is written out from the model's own modules (prelude -> state init -> R x core_block_forward -> coda ->
ln_f -> head), mirroring RavenForCausalLM.forward in the GQA-patched raven_modeling_minimal.py, and checked against the
model's own forward before measuring (abort on a mismatch > 1e-4 relative).

  python retrofit_rho_probe.py --model_dir DIR [--model_dir DIR2 ...] --val PATH/val.pt --out PATH.json
fp32, batch 1, first --n sequences of the healing run's held-out set, truncated to --seq tokens.
"""
import argparse, json, math, os
import numpy as np
import torch
import torch.nn.functional as Fn

ap = argparse.ArgumentParser()
ap.add_argument('--model_dir', action='append', required=True); ap.add_argument('--val', required=True)
ap.add_argument('--out', required=True); ap.add_argument('--n', type=int, default=16); ap.add_argument('--seq', type=int, default=512)
ap.add_argument('--Rs', default='8,16')
A = ap.parse_args()
DEV = 'cuda'


def load(model_dir):
    from transformers import AutoModelForCausalLM
    cp = os.path.join(model_dir, 'config.json'); c = json.load(open(cp))
    if 'tie_word_embeddings' in c:          # save_pretrained re-adds it; raven's config passes it itself (see retrofit_heal.py)
        assert not c['tie_word_embeddings'] and not c.get('tie_embeddings', False)
        c.pop('tie_word_embeddings'); json.dump(c, open(cp, 'w'), indent=2)
    m = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.float32).to(DEV).eval()
    m.config.test_time_noise = 0
    return m


def labels_of(ids):
    return torch.cat([ids[:, 1:], torch.full_like(ids[:, :1], -100)], 1)


def manual_loss(m, ids, R, k, adj=None):
    """Loss with backprop through the last k of R loops; adjoint hooks on the state after each grad loop."""
    pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
    e = m.transformer.wte(ids)
    if m.emb_scale != 1:
        e = e * m.emb_scale
    freqs = m.rotary_emb(e, pos)
    bi = torch.tensor(-1, device=torch.device('cpu'), dtype=torch.long)
    for block in m.transformer.prelude:
        bi += 1
        e = block(e, freqs, bi, None, None)
    x = m.initialize_state(e, scale=1.0)
    with torch.no_grad():
        for t in range(R - k):
            x, bi = m.core_block_forward(x, e, freqs, None, None, bi, t)
    x = x.detach()
    for t in range(R - k, R):
        x, bi = m.core_block_forward(x, e, freqs, None, None, bi, t)
        if adj is not None:
            x.register_hook(lambda g, t=t: adj.__setitem__(t, float(g.float().norm())))
    bi = torch.tensor(0, device=torch.device('cpu'), dtype=torch.long)
    for block in m.transformer.coda:
        bi -= 1
        x = block(x, freqs, bi, None, None)
    logits = m.lm_head(m.transformer.ln_f(x)).float()
    return Fn.cross_entropy(logits.view(-1, logits.shape[-1]), labels_of(ids).view(-1), ignore_index=-100)


def check(m, ids):
    out = {}
    for R in (1, 4):
        torch.manual_seed(0); a = float(m(ids, labels=labels_of(ids), num_steps=R).loss)
        torch.manual_seed(0); b = float(manual_loss(m, ids, R, 0))
        out[R] = (a, b)
        assert abs(a - b) <= 1e-4 * max(1.0, abs(a)), ('decomposition mismatch', R, a, b)
    return out


def flat_grad(params):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().flatten() for p in params])


def measure(m, val, R):
    ks = [k for k in (1, 2, 4, 8) if k < R]
    params = list(m.parameters()); core = [p for n, p in m.named_parameters() if 'core_block' in n or 'adapter' in n]
    rows = []
    for i in range(min(A.n, len(val))):
        ids = val[i:i + 1, :A.seq].to(DEV); adj = {}
        torch.manual_seed(777 + i); m.zero_grad(set_to_none=True)
        loss = manual_loss(m, ids, R, R, adj); loss.backward()
        gR = flat_grad(params); gRc = flat_grad(core)
        a = np.array([adj[t] for t in range(R)])
        slope = np.polyfit(np.arange(R), np.log(a + 1e-30), 1)[0]
        row = dict(i=i, loss=float(loss), adj=a.tolist(), rho=float(np.exp(-slope)), err={}, err_loop={})
        for k in ks:
            torch.manual_seed(777 + i); m.zero_grad(set_to_none=True)
            manual_loss(m, ids, R, k).backward()
            row['err'][k] = float((flat_grad(params) - gR).norm() / gR.norm())
            row['err_loop'][k] = float((flat_grad(core) - gRc).norm() / gRc.norm())
        rows.append(row)
    m.zero_grad(set_to_none=True)
    rho = np.array([r['rho'] for r in rows])
    summ = dict(R=R, rho_median=float(np.median(rho)),
                err_median={k: float(np.median([r['err'][k] for r in rows])) for k in ks},
                err_loop_median={k: float(np.median([r['err_loop'][k] for r in rows])) for k in ks},
                law_median={k: float(np.median(rho ** k)) for k in ks},
                law_abs_dev_median={k: float(np.median([abs(r['err'][k] - r['rho'] ** k) for r in rows])) for k in ks},
                adj_median=[float(np.median([r['adj'][t] for r in rows])) for t in range(R)])
    return dict(summary=summ, rows=rows)


def main():
    v = torch.load(A.val); val = v['val'] if isinstance(v, dict) else v
    res = {}
    for md in A.model_dir:
        m = load(md); ids = val[:1, :A.seq].to(DEV)
        res[md] = dict(check=check(m, ids), measure={})
        print('%s\n  decomposition check (model vs manual loss): %s' % (md, res[md]['check']), flush=True)
        for R in (int(x) for x in A.Rs.split(',')):
            r = measure(m, val, R); res[md]['measure'][R] = r
            s = r['summary']
            print('  R=%d rho %.3f | k: err (law rho^k, |err-law| median, loop-params-only err)' % (R, s['rho_median']), flush=True)
            for k in s['err_median']:
                print('    k=%d: %.3f (%.3f, %.3f, %.3f)' % (k, s['err_median'][k], s['law_median'][k],
                      s['law_abs_dev_median'][k], s['err_loop_median'][k]), flush=True)
            print('    adjoint norms over loops (median): %s' % [round(x, 4) for x in s['adj_median']], flush=True)
        del m; torch.cuda.empty_cache()
        json.dump(res, open(A.out, 'w'), indent=1, default=str)
    print('wrote', A.out)


if __name__ == '__main__':
    main()
