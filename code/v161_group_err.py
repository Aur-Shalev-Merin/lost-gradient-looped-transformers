"""v161 EXPLORATORY: truncation error per parameter group on the frozen Parcae-140m.

v161 measured the relative error of the truncated gradient over ALL parameters. Embeddings/head and the coda get the
same gradient with or without truncation and have large gradients, so they can make the total error look small
while a small group (prelude, injection, core) is mostly wrong. Adam rescales each coordinate, so a group's own error
is what reaches its update. This prints per-group relative error and, per group, the error after Adam-like
normalisation (g / |g| elementwise, i.e. sign disagreement weighted equally per coordinate).

  python v161_group_err.py R k1,k2,..  [N sequences, default 8]
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v161_parcae_trunc_train as v


def group_of(name):
    for g in ('prelude', 'adapter', 'core_block', 'coda', 'wte', 'lm_head', 'value_embeds', 'ln_prelude', 'ln_f'):
        if g in name:
            return g
    return 'other'


def main():
    R = int(sys.argv[1]); ks = [int(x) for x in sys.argv[2].split(',')]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    tok, model = v.load_model(); data = v.gsm8k_chunks(tok); model.eval()
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    groups = sorted({group_of(n) for n in names})

    def grads(ids, k, i):
        torch.manual_seed(777 + i); model.zero_grad(set_to_none=True)
        v.loss_manual(model, ids, R, k).backward()
        out = {}
        for n, p in zip(names, params):
            g = group_of(n); gr = (p.grad if p.grad is not None else torch.zeros_like(p)).detach().flatten().double()
            out.setdefault(g, []).append(gr)
        return {g: torch.cat(x) for g, x in out.items()}

    rel = {k: {g: [] for g in groups + ['ALL']} for k in ks}
    sgn = {k: {g: [] for g in groups} for k in ks}
    norm_share = {g: [] for g in groups}
    for i in range(N):
        ids = data['val'][i:i + 1].to(v.DEV)
        gR = grads(ids, R, i); tot = torch.cat([gR[g] for g in groups])
        for g in groups:
            norm_share[g].append(float(gR[g].norm() ** 2 / tot.norm() ** 2))
        for k in ks:
            gk = grads(ids, k, i)
            for g in groups:
                rel[k][g].append(float((gk[g] - gR[g]).norm() / (gR[g].norm() + 1e-30)))
                nz = gR[g].abs() > 1e-12
                sgn[k][g].append(float((torch.sign(gk[g][nz]) != torch.sign(gR[g][nz])).double().mean()) if nz.any() else float('nan'))
            allk = torch.cat([gk[g] for g in groups])
            rel[k]['ALL'].append(float((allk - tot).norm() / tot.norm()))
        print('seq %d done' % i, flush=True)
    print('\nR=%d, %d sequences. Share of full-gradient squared norm per group, and per-group truncation error (median):' % (R, N))
    print('%-13s %8s ' % ('group', 'share') + ' '.join('  k=%d rel / sign-flip' % k for k in ks))
    for g in groups + ['ALL']:
        sh = '%8.4f' % np.median(norm_share[g]) if g != 'ALL' else '%8s' % '1'
        print('%-13s %s ' % (g, sh) + ' '.join('     %6.3f / %5.3f   ' % (np.median(rel[k][g]), np.nanmedian(sgn[k][g]) if g != 'ALL' else float('nan')) for k in ks))


if __name__ == '__main__':
    main()
