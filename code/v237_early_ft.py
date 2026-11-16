"""v237 (plan frozen in docs/prereg_v237_early_ft.md before any measurement): two follow-ups from the methods review of
the looped-truncation paper (v17.6).

Part A: when does the fine-tuning fall happen? v218's full-backprop fine-tune (recipe, LR, data order, seed 2011, the
seed v218 used), imported unchanged, with its checkpoint list extended to steps 10, 25, 50, 100, 200, 300 and 600. At
step 0 and every checkpoint, v210 Part A's measurement on the first 64 GSM8K test sequences and the first 64
WikiText-103 test chunks (v218's measure, unchanged). The global RNG is saved and restored around every measurement, so
the trajectory is v218's: steps 300 and 600 must reproduce v218's values (validity check).
Part B: does Ouro's backward signal grow on text it has not seen? v210 Part B (growth of the backward signal, scale-free
versions, exit-objective lost gradient) and v197's final-output lost gradient, both imported unchanged, run on the first
16 GSM8K TEST chunks instead of the 16 validation chunks (last 473 GSM8K train problems, which Ouro appears to have seen).

  python v237_early_ft.py A --model parcae|pln5|r5 [--weights DIR]
  python v237_early_ft.py B --seqs 0:8 --tag g0
  python v237_early_ft.py --analyze RESULTS_V218 RESULTS_V210 V237_DIR [...]
Env: TINY=1 (pipeline test on tiny models, CPU), OUT_DIR.
"""
import argparse, glob, hashlib, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
KS = (1, 2, 4)
EARLY = (10, 25, 50, 100, 200)
SEED = 2011
VALID_TOL = 0.01          # steps 300 / 600 must match v218 within this, per k, all and loop, GSM8K
S0_TOL = 0.005            # step 0 must match v218 within this (v218's own S0_TOL)
FAST, SLOW = 0.75, 0.5    # Part A rule thresholds (prereg section 2)


def sha(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(open(p, 'rb').read().replace(b'\r\n', b'\n'))
    return h.hexdigest()[:12]


def here(*f):
    return [os.path.join(os.path.dirname(os.path.abspath(__file__)), x) for x in f]


def part_a(model, weights):
    import v218_robustness as D
    D.CKPTS = (2, 3, 4) if D.TINY else EARLY + (300, 600)
    D.FT_SEEDS = (SEED,)
    json.dump(dict(model=model, seed=SEED, ckpts=list(D.CKPTS), driver_sha=sha(here('v237_early_ft.py', 'v218_robustness.py',
                                                                                    'v201_batch_lost.py'))),
              open(os.path.join(D.OUT, 'v237_A_%s_meta.json' % model), 'w'), indent=1)
    D.run(model, ['ft'], weights)                     # writes v218_<model>_ft.json in OUT_DIR


def test_chunks(tok, P):
    """The first GSM8K TEST problems, packed in order into 512-token chunks (v197's packing, test split)."""
    import torch
    seq = 32 if P.TINY else 512
    if P.TINY:
        g = torch.Generator().manual_seed(1)
        return torch.randint(0, 1000, (2, seq), generator=g)
    from datasets import load_dataset
    te = load_dataset('openai/gsm8k', 'main', split='test')
    ex = [tok(te[i]['question'] + '\n' + te[i]['answer']).input_ids for i in range(len(te))]
    flat = [t for e in ex for t in e]
    n = len(flat) // seq
    return torch.tensor(flat[: n * seq], dtype=torch.long).view(n, seq)


def part_b(seqs, tag):
    os.environ.setdefault('LOOP_MODEL', 'ouro')
    import v197_ouro_allparam as V
    import v210_followups as A
    V.val_chunks = test_chunks                        # both drivers read their text through V.val_chunks
    lo, hi = (int(x) for x in seqs.split(':'))
    A.part_b((lo, hi), 'test_' + tag)                 # writes v210_B_test_<tag>.json
    V.measure((lo, hi), 'test_' + tag)                # writes v197_ouro_test_<tag>.json


# ---------------------------------------------------------------------------------------------------------- analysis
def med(rows, key, k):
    return float(np.median([r[key][str(k)] if str(k) in r[key] else r[key][k] for r in rows]))


def mean_k(rows, key='all'):
    return float(np.mean([med(rows, key, k) for k in KS]))


def analyze(dirs):
    lines = []; P = lines.append
    v218, v237, B, L = {}, {}, [], []
    for d in dirs:
        for f in glob.glob(os.path.join(d, '**', 'v218_*_ft*.json'), recursive=True):
            r = json.load(open(f)); ft = (r['runs'].get('ft') or [None])[0]
            if ft is None or ft.get('seed') != SEED:
                continue
            steps = sorted(int(s) for s in ft['ckpt'])
            (v237 if any(s in steps for s in EARLY) else v218)[r['model']] = ft
        for f in glob.glob(os.path.join(d, '**', 'v210_B_test_*.json'), recursive=True):
            B += json.load(open(f))['rows']
        for f in glob.glob(os.path.join(d, '**', 'v197_ouro_test_*.json'), recursive=True):
            L += json.load(open(f))['rows']
    # ---- Part A
    P('PART A: the fall over fine-tuning steps (GSM8K test, all parameters, mean over k = 1, 2, 4 of per-sequence medians)')
    lab, valid = {}, True
    for m in ('parcae', 'pln5', 'r5'):
        ft, ref = v237.get(m), v218.get(m)
        if not ft or not ref:
            P('  %-6s incomplete (v237 %s, v218 %s)' % (m, bool(ft), bool(ref))); continue
        ck = {int(s): v for s, v in ft['ckpt'].items()}; rk = {int(s): v for s, v in ref['ckpt'].items()}
        dev = {s: max(abs(med(ck[s]['gsm'], key, k) - med(rk[s]['gsm'], key, k)) for key in ('all', 'loop') for k in KS)
               for s in (0, 300, 600) if s in ck and s in rk}
        ok = ft['finetune']['ok'] and dev.get(0, 1) <= S0_TOL and dev.get(300, 1) <= VALID_TOL and dev.get(600, 1) <= VALID_TOL
        valid &= ok
        g0, w0 = mean_k(ck[0]['gsm']), mean_k(ck[0]['wiki'])
        F = {s: g0 - mean_k(ck[s]['gsm']) for s in sorted(ck)}
        W = {s: w0 - mean_k(ck[s]['wiki']) for s in sorted(ck)}
        sh = {s: F[s] / F[300] if F[300] > 0 else float('nan') for s in F}
        lab[m] = ('FAST' if sh[50] >= FAST else 'GRADUAL' if sh[100] <= SLOW else 'INTERMEDIATE')
        P('  %-6s GSM8K fall by step: %s' % (m, '  '.join('%d: %.3f' % (s, F[s]) for s in sorted(F) if s)))
        P('  %-6s   share of the step-300 fall: %s' % (m, '  '.join('%d: %.2f' % (s, sh[s]) for s in sorted(sh) if s)))
        P('  %-6s   WikiText fall by step: %s | step-0 WikiText value %.3f, GSM8K value by step: %s' % (
            m, '  '.join('%d: %.3f' % (s, W[s]) for s in sorted(W) if s), w0,
            '  '.join('%d: %.3f' % (s, mean_k(ck[s]['gsm'])) for s in sorted(ck))))
        P('  %-6s   training loss (curve, every 50 steps): %s' % (m, ft['finetune']['curve'][:8]))
        P('  %-6s   validity: step 0 / 300 / 600 vs v218 max |diff| %s (<= %g / %g), fine-tune ok %s -> %s; label %s' % (
            m, {s: round(v, 4) for s, v in dev.items()}, S0_TOL, VALID_TOL, ft['finetune']['ok'], 'VALID' if ok else 'INVALID', lab[m]))
    if len(lab) < 3:
        va = 'INCOMPLETE'
    elif not valid:
        va = 'INVALID (a validity check failed)'
    elif all(v == 'FAST' for v in lab.values()):
        va = 'FAST'
    elif all(v == 'GRADUAL' for v in lab.values()):
        va = 'GRADUAL'
    else:
        va = 'MIXED %s' % lab
    P('VERDICT A: %s' % va)
    # ---- Part B
    if B:
        import v210_followups as A
        rh = [A.fit_rho(np.multiply(r['adj_h'], r['norm_h'])) for r in B]
        rx = [A.fit_rho(np.multiply(r['adj_x'], r['norm_x'])) for r in B]
        rraw = [A.fit_rho(r['adj_h']) for r in B]
        ee = {k: float(np.median([r['exit_err'][str(k)] if str(k) in r['exit_err'] else r['exit_err'][k] for r in B])) for k in (1, 2, 3)}
        P('PART B: Ouro on %d GSM8K test chunks' % len(B))
        P('  rho of |dL/dh| %.3f (min %.3f) | scale-free on h %.3f (min %.3f), on pre-norm x %.3f (min %.3f) (medians)' % (
            np.median(rraw), min(rraw), np.median(rh), min(rh), np.median(rx), min(rx)))
        P('  exit-weighted objective lost gradient k = 1 / 2 / 3: %.3f / %.3f / %.3f' % (ee[1], ee[2], ee[3]))
        if L:
            fl = {k: float(np.median([r[str(k)]['all']['err'] for r in L if str(k) in r])) for k in (1, 2, 3) if all(str(k) in r for r in L)}
            P('  final-output lost gradient (all parameters) k = %s: %s (%d chunks; Table 1 on validation text: 0.92 / 0.81)' % (
                '/'.join(str(k) for k in fl), ' / '.join('%.3f' % v for v in fl.values()), len(L)))
        vb = ('GROWS' if np.median(rh) > 1 and np.median(rx) > 1 and np.median(rraw) > 1 else
              'NO-GROWTH' if np.median(rh) <= 1 and np.median(rx) <= 1 and np.median(rraw) <= 1 else 'MIXED')
        P('VERDICT B: %s' % vb)
    else:
        P('PART B: no data')
    return '\n'.join(lines)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('part', nargs='?'); ap.add_argument('--model'); ap.add_argument('--weights', default=None)
    ap.add_argument('--seqs', default='0:16'); ap.add_argument('--tag', default='g0'); ap.add_argument('--analyze', nargs='*')
    A_ = ap.parse_args()
    if A_.analyze is not None:
        txt = analyze(A_.analyze); print(txt)
        open(os.path.join(os.environ.get('OUT_DIR', '.'), 'v237_analysis.txt'), 'w').write(txt + '\n')
    elif A_.part == 'A':
        part_a(A_.model, A_.weights)
    elif A_.part == 'B':
        part_b(A_.seqs, A_.tag)
