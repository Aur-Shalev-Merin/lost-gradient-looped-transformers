"""v170: run the frozen truncated-loop training drivers with SGD + momentum instead of AdamW.

The drivers (v164_parcae_predict.py -> v163/v162 for Parcae-140m; v167_retrofit_trunc.py for the SmolLM2 retrofit)
are used UNCHANGED; this wrapper replaces torch.optim.AdamW, looked up at call time inside their train_cell, with
  SGD(lr, momentum 0.9, weight_decay as passed)   (Adam's betas / eps are ignored)
Everything else (data, steps, warm-up + cosine schedule, clip 1.0, fp16 + GradScaler, damping, truncation) is identical.

  python v170_sgd.py parcae --train R:a:k:lr:seed,... --tag T [--selftest]
  python v170_sgd.py retro  --train a:k:lr:seed,...   --tag T [--selftest]
"""
import sys
import torch

_SGD = torch.optim.SGD


def sgd_momentum(params, lr, betas=None, eps=None, weight_decay=0.0, **kw):
    print('v170: optimizer = SGD(lr %g, momentum 0.9, wd %g)' % (lr, weight_decay), flush=True)
    return _SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)


torch.optim.AdamW = sgd_momentum

if __name__ == '__main__':
    which = sys.argv.pop(1)
    print('v170: AdamW replaced by SGD(momentum 0.9) | driver %s' % which, flush=True)
    if which == 'parcae':
        import v164_parcae_predict as drv
    elif which == 'retro':
        import v167_retrofit_trunc as drv
    else:
        raise SystemExit('parcae | retro')
    drv.main()
