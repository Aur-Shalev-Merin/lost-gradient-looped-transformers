"""v211 (plan frozen in docs/prereg_v211_k0_control.md before any measurement): the k = 0 control for S15's in-loop
reading (outside review 7, M1).

S15 (v210) found that the fall of the lost gradient during full-backprop fine-tuning survives leaving the no-loop-credit
gradient out of the denominator (loop = |G - T_k| / |G - T_0|). That locates the fall in the loop's part of the
gradient, but a change outside the loop (layers after it, read-out) could produce it by changing the backward signal
that enters the loop. Control: the same fine-tune (v201 recipe, LR, data order, 300 of 600 steps) at k = 0, i.e. no
loop credit, so the loop's own weights never get a gradient and stay fixed while everything that gets gradient without
loop credit trains. Then the v210 Part A measurement, unchanged: first 64 GSM8K TEST sequences, batch 1, fp32, train
mode, state draw seed 301000 + s, arms k = R, 0, 1, 2, 4, statistics all and loop.

Runs per model: s0 (re-measured: must agree with v210's s0) and zero300 (fine-tune seeds pre-listed 2011, 2012, 2013;
the first that passes the guard, <= 10 non-finite steps and a finite last loss, is used). On Parcae a second GPU runs
zero300 from seed 2012 as a reported seed check (zero300b), not used by any rule.

Check (validity): after zero300, the parameters that changed are a subset of those with a nonzero k = 0 gradient on
the first measured sequence of s0 (so the loop's weights stayed exactly fixed). Recorded: names, counts and sizes of
changed parameters, and |delta W| per changed parameter group.

  python v211_k0_control.py --model parcae|pln5|r5 --runs s0,zero300 [--weights DIR]
  python v211_k0_control.py --analyze V210_DIR V211_DIR [...]
Env: TINY=1 (pipeline test on a tiny Part II model, CPU), OUT_DIR.
"""
import argparse, glob, hashlib, json, os, time
import numpy as np
import torch

TINY = os.environ.get('TINY') == '1'
OUT = os.environ.get('OUT_DIR', '/kaggle/working' if os.path.isdir('/kaggle/working') else '.')
KS = (1, 2, 4)
N_SEQ = 4 if TINY else 64
FT_SEEDS = (2011, 2012, 2013)
SEED1 = 301000
S0_TOL = 0.005                         # |median(s0, v211) - median(s0, v210)|, every k and statistic


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v211_log.txt'), 'a').write(s + '\n')


def sha(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(open(os.path.abspath(p), 'rb').read().replace(b'\r\n', b'\n'))
    return h.hexdigest()[:12]


def measure(m, R, meas, chunks, params):
    """v210 Part A measurement, unchanged. Also returns which parameters get a nonzero k = 0 gradient (sequence 0)."""
    rows, nz0 = [], None
    for s in range(min(N_SEQ, len(chunks))):
        ids = chunks[s:s + 1]; vec = {}
        for k in (R, 0) + KS:
            m.zero_grad(set_to_none=True)
            meas(ids, k, SEED1 + s).backward()
            if s == 0 and k == 0:
                nz0 = [i for i, p in enumerate(params) if p.grad is not None and bool(p.grad.detach().abs().sum() > 0)]
            vec[k] = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1).float()
                                for p in params])
        m.zero_grad(set_to_none=True)
        G = vec[R]; gn = float(G.norm()); ln = float((G - vec[0]).norm())
        rows.append(dict(s=s, gnorm=gn, loopnorm=ln, all={k: float((G - vec[k]).norm()) / gn for k in KS},
                         loop={k: float((G - vec[k]).norm()) / ln for k in KS}))
        del vec, G
    return rows, nz0


def run_model(model, runs, weights):
    import v201_batch_lost as B
    out = dict(model=model, runs={})
    fn = os.path.join(OUT, 'v211_%s_%s.json' % (model, '_'.join(runs)))
    nz0_names = None
    for run in runs:
        attempts = [None] if run == 's0' else [2012] if run == 'zero300b' else list(FT_SEEDS)
        for seed in attempts:
            m, data, pack, _, R, meas, train, amp, mods = B.setup(model, weights)
            names = [n for n, p in m.named_parameters() if p.requires_grad]
            params = [p for p in m.parameters() if p.requires_grad]
            chunks = data['test'].to(B.DEV)
            if B.TINY:
                chunks = torch.cat([chunks] * 6)
            rec = dict(run=run, R=R, n_params=sum(p.numel() for p in params))
            if seed is not None:
                before = [p.detach().clone().cpu() for p in params]
                B.FT_SEED = seed
                rec['finetune'] = B.finetune(m, params, data, pack, train, amp, 0, B.LR[model])
                rec['finetune']['k'] = 0
                ch = [(i, float((p.detach().cpu() - b).norm())) for i, (p, b) in enumerate(zip(params, before))]
                ch = [(i, d) for i, d in ch if d > 0]
                del before
                rec['changed'] = dict(n=len(ch), numel=sum(params[i].numel() for i, _ in ch),
                                      names=[names[i] for i, _ in ch], dnorm=[d for _, d in ch])
                log('  %s %s fine-tune (k = 0) seed %d: %s | changed %d tensors, %.2fM params' % (
                    model, run, seed, {k: v for k, v in rec['finetune'].items() if k != 'curve'},
                    len(ch), rec['changed']['numel'] / 1e6))
                if not rec['finetune']['ok']:
                    out['runs'].setdefault('excluded', []).append(dict(run=run, seed=seed, nonfinite=rec['finetune']['nonfinite']))
                    del m
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
            t0 = time.time(); rec['rows'], nz0 = measure(m, R, meas, chunks, params); rec['s'] = round(time.time() - t0)
            if run == 's0':
                nz0_names = [names[i] for i in nz0]; rec['nz0_names'] = nz0_names
                rec['nz0_numel'] = sum(params[i].numel() for i in nz0)
                log('  %s: %d tensors (%.2fM params) get a nonzero k = 0 gradient' % (model, len(nz0), rec['nz0_numel'] / 1e6))
            elif nz0_names is not None:
                extra = sorted(set(rec['changed']['names']) - set(nz0_names))
                rec['check_loop_fixed'] = dict(ok=not extra, changed_without_k0_grad=extra)
                log('  CHECK loop weights fixed: %s%s' % ('PASS' if not extra else 'FAIL', '' if not extra else ' %s' % extra[:10]))
            med = lambda key, k: float(np.median([r[key][k] for r in rec['rows']]))
            log('  %s %s: all %s | loop %s (%ds)' % (model, run, [round(med('all', k), 3) for k in KS],
                                                  [round(med('loop', k), 3) for k in KS], rec['s']))
            out['runs'][run] = rec; out['driver_sha'] = sha([__file__] + mods)
            del m
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            break
        json.dump(out, open(fn, 'w'), indent=1)


def analyze(dirs):
    """Frozen outcome mapping of docs/prereg_v211_k0_control.md. Reads v210_A_*.json (s0, full300) and v211_*.json."""
    v210, v211 = {}, {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, '**', 'v210_A_*.json'), recursive=True):
            r = json.load(open(f)); v210.setdefault(r['model'], {}).update(r['runs'])
        for f in glob.glob(os.path.join(d, '**', 'v211_*.json'), recursive=True):
            r = json.load(open(f)); v211.setdefault(r['model'], {}).update(r['runs'])
    med = lambda rows, key, k: float(np.median([r[key][str(k)] if str(k) in r[key] else r[key][k] for r in rows]))
    ratio, valid = {}, True
    print('v211: k = 0 fine-tune control (held-out GSM8K test, per-sequence medians, k = 1 / 2 / 4)')
    for model in ('parcae', 'pln5', 'r5'):
        a, b = v210.get(model, {}), v211.get(model, {})
        if not all(x in a for x in ('s0', 'full300')) or not all(x in b for x in ('s0', 'zero300')):
            print('  %-6s incomplete: v210 %s, v211 %s' % (model, sorted(a), sorted(b))); continue
        g = lambda runs, run, key: [med(runs[run]['rows'], key, k) for k in KS]
        dev = max(abs(x - y) for key in ('all', 'loop') for x, y in zip(g(a, 's0', key), g(b, 's0', key)))
        chk = b['zero300'].get('check_loop_fixed', {}).get('ok', False)
        ok = dev <= S0_TOL and chk
        valid &= ok
        f_full = {key: float(np.mean(np.subtract(g(a, 's0', key), g(a, 'full300', key)))) for key in ('all', 'loop')}
        f_zero = {key: float(np.mean(np.subtract(g(b, 's0', key), g(b, 'zero300', key)))) for key in ('all', 'loop')}
        ratio[model] = f_zero['loop'] / f_full['loop'] if f_full['loop'] > 0 else float('nan')
        for runs, run in ((a, 's0'), (a, 'full300'), (b, 's0'), (b, 'zero300')) + (((b, 'zero300b'),) if 'zero300b' in b else ()):
            print('  %-6s %-14s all %s | loop %s' % (model, ('v210 ' if runs is a else 'v211 ') + run,
                  [round(x, 3) for x in g(runs, run, 'all')], [round(x, 3) for x in g(runs, run, 'loop')]))
        print('  %-6s fall (mean over k): full300 all %.3f loop %.3f | zero300 all %.3f loop %.3f | ratio (loop) %.2f, (all) %.2f'
              % (model, f_full['all'], f_full['loop'], f_zero['all'], f_zero['loop'], ratio[model],
                 f_zero['all'] / f_full['all'] if f_full['all'] > 0 else float('nan')))
        ft = b['zero300'].get('finetune', {})
        print('  %-6s checks: s0 agreement max |diff| %.4f (<= %g) | loop weights fixed %s | seed %s, non-finite %s, clipped %s | %s'
              % (model, dev, S0_TOL, 'PASS' if chk else 'FAIL', ft.get('seed'), ft.get('nonfinite'), ft.get('clipped_frac'),
                 'VALID' if ok else 'INVALID'))
    if len(ratio) < 3:
        v = 'INCOMPLETE'
    elif not valid:
        v = 'INVALID (a check failed; see above)'
    elif all(r < 0.25 for r in ratio.values()):
        v = 'LOOP-NEEDED'
    elif all(r >= 0.75 for r in ratio.values()):
        v = 'OUTSIDE-SUFFICES'
    else:
        v = 'PARTIAL %s' % {k: round(r, 2) for k, r in ratio.items()}
    print('VERDICT: %s' % v)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model'); ap.add_argument('--runs', default='s0,zero300'); ap.add_argument('--weights', default=None)
    ap.add_argument('--analyze', nargs='*')
    A_ = ap.parse_args()
    if A_.analyze is not None:
        analyze(A_.analyze or [OUT])
    else:
        run_model(A_.model, A_.runs.split(','), A_.weights)
