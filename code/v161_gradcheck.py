"""v161 EXPLORATORY: why does adding loop 0 to the backprop raise the training gradient norm by ~35% (k = R vs R-1),
when the frozen-model measurement says the first loop contributes a few percent?

On the frozen Parcae-140m and a GSM8K training batch, compute the gradient (all parameters) for k = R and k = R - 1
under each combination of the differences between the v161 training path and the v161 measurement path:
  path     'model'  = Parcae's own forward (loss_model, as in training)   | 'manual' = loss_manual (as in measurement)
  mode     train / eval
  amp      fp16 autocast (training) / fp32 (measurement)
  batch    8 (training) / 1 (measurement)
Reports |g_R|, |g_{R-1}|, relative error |g_R - g_{R-1}| / |g_R|, and the norm of the loop-0 contribution per
parameter group. Same seed for the state init in each pair.
  python v161_gradcheck.py [R]
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v161_parcae_trunc_train as v


def group_of(name):
    for g in ('prelude', 'adapter', 'core_block', 'coda', 'wte', 'value_embeds', 'ln_prelude', 'ln_f'):
        if g in name:
            return g
    return 'other'


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    tok, model = v.load_model(); data = v.gsm8k_chunks(tok)
    g = np.random.default_rng(1611101)
    chunks = v.pack(data['train'], g.permutation(len(data['train'])))
    names = [n for n, p in model.named_parameters()]; params = [p for p in model.parameters()]

    def grad(path, mode, amp, ids, k):
        model.train() if mode == 'train' else model.eval()
        model.zero_grad(set_to_none=True); torch.manual_seed(0)
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            if path == 'model':
                loss = model(ids, labels=v.labels_of(ids), num_steps_pair=torch.tensor([R - k, k]))['loss'].float()
            else:
                loss = v.loss_manual(model, ids, R, k)
        (loss * (1024.0 if amp else 1.0)).backward()
        out = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten().double() for p in params])
        return out / (1024.0 if amp else 1.0), float(loss)

    v.log('gradcheck R=%d: path mode amp batch | loss | |g_R| |g_R-1| relerr' % R)
    for path, mode, amp, B in (('model', 'train', True, 8), ('model', 'train', False, 8), ('model', 'eval', False, 8),
                               ('manual', 'eval', False, 8), ('manual', 'eval', False, 1), ('model', 'train', True, 1)):
        ids = chunks[:B].to(v.DEV)
        gR, lR = grad(path, mode, amp, ids, R); g1, l1 = grad(path, mode, amp, ids, R - 1)
        d = gR - g1
        v.log('  %-6s %-5s amp=%-5s B=%d | loss %.4f/%.4f | %.4f %.4f %.4f' % (path, mode, amp, B, lR, l1,
              float(gR.norm()), float(g1.norm()), float(d.norm() / gR.norm())))
        if path == 'model' and B == 8:
            off = 0; per = {}
            for n, p in zip(names, params):
                n_ = p.numel(); grp = group_of(n)
                a = per.setdefault(grp, [0.0, 0.0]); a[0] += float(d[off:off + n_].norm() ** 2); a[1] += float(gR[off:off + n_].norm() ** 2)
                off += n_
            v.log('    loop-0 contribution by group (|d|, |g_R|): ' + ', '.join('%s %.4f/%.4f' % (k_, a[0] ** .5, a[1] ** .5) for k_, a in sorted(per.items())))


if __name__ == '__main__':
    main()
