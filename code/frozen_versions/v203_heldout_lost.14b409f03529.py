"""v203 (plan frozen in docs/prereg_v203_heldout_lost.md before any measurement): is v201's fall of the lost gradient
during fine-tuning a change in the loop, or partly the model fitting the sequences it was measured on?

v201 measured at stage 300 on training sequences that the 300-step fine-tune had just seen about once. v203 repeats the
measurement on GSM8K TEST-split sequences, which no run trains on: the same models, settings, fine-tune (seed 2011,
300 of 600 steps, full backprop, the study's recipe on the training split) and code (v201_batch_lost.measure /
.finetune, imported unchanged). Test chunks are packed in split order; as many full groups of 8 batches of 8 as there
are, up to 64 batches (the count is logged); B = 1 on the first 64 sequences.

  python v203_heldout_lost.py --model parcae|retro|pln5|r5 --stages 0,300 [--weights DIR]
  python v203_heldout_lost.py --analyze V203_DIR [V203_DIR ...] --v201 V201_DIR
"""
import argparse, glob, json, os, time
import numpy as np
import torch
import v201_batch_lost as B

KS = B.KS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=('parcae', 'retro', 'pln5', 'r5')); ap.add_argument('--stages', default='0,300')
    ap.add_argument('--weights', default=None); ap.add_argument('--analyze', nargs='*'); ap.add_argument('--v201', default=None)
    A = ap.parse_args()
    if A.analyze is not None:
        return analyze(A.analyze, A.v201)
    T0 = time.time()
    m, data, pack, _, R, meas, train, amp, mods = B.setup(A.model, A.weights)
    chunks = data['test']
    if B.TINY:                            # the tiny test split has too few chunks for a group; pipeline test only
        chunks = torch.cat([chunks] * 6)
    B.NB = min(64, (len(chunks) // B.BS) // B.GROUP * B.GROUP)
    assert B.NB >= 4 * B.GROUP and len(chunks) >= B.N1, (len(chunks), B.NB)
    params = [p for p in m.parameters() if p.requires_grad]
    dsha = B.sha([__file__, B.__file__] + mods)
    B.log('v203 %s | driver %s | R %d | held-out (test) chunks %d -> %d batches of %d, groups of %d, B=1 on %d' % (
        A.model, dsha, R, len(chunks), B.NB, B.BS, B.GROUP, B.N1))
    for stage in [int(x) for x in A.stages.split(',')]:
        rec = dict(model=A.model, stage=stage, R=R, ks=list(KS), nb=B.NB, bs=B.BS, group=B.GROUP, n1=B.N1,
                   split='test', driver_sha=dsha, measured=False)
        if stage == 300:
            rec['finetune'] = B.finetune(m, params, data, pack, train, amp, R, B.LR[A.model])
            B.log('  stage 300 fine-tune: %s' % {k: v for k, v in rec['finetune'].items() if k != 'curve'})
        if stage == 0 or rec['finetune']['ok']:
            res = B.measure(m, params, chunks, R, meas)
            rec.update(res); rec['measured'] = 'groups' in res
        rec['s'] = round(time.time() - T0)
        json.dump(rec, open(os.path.join(B.OUT, 'v203_%s_s%d.json' % (A.model, stage)), 'w'), indent=1)
        B.log('  stage %d %s after %ds' % (stage, 'measured' if rec['measured'] else 'NOT measured', time.time() - T0))


def b1(r, k):
    return float(np.median([x['raw'][str(k)] for x in r['b1']]))


def c64(r, k):
    return float(np.median([x['clip'][str(k)] for x in r['groups']['b64']]))


def load(dirs, prefix):
    out = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, '**', prefix + '_*_s*.json'), recursive=True):
            r = json.load(open(f))
            if r.get('measured'):
                out[(r['model'], r['stage'])] = r
    return out


def analyze(dirs, v201_dir):
    """Frozen rule: per model with a training-sequence drop (v201), ratio = mean over k of the held-out per-sequence
    drop (stage 0 - stage 300) / mean over k of the training-sequence drop. GENERAL if ratio >= 0.5 on every such
    model; MOSTLY-SEEN-SEQUENCES if < 0.5 on every one; MIXED otherwise. Models: those whose v201 training-sequence
    drop averages > 0.05 (Parcae, pln5, r5 in v201)."""
    h, t = load(dirs, 'v203'), load([v201_dir], 'v201')
    ratios = {}
    for model in ('parcae', 'retro', 'pln5', 'r5'):
        if not all((model, s) in h and (model, s) in t for s in (0, 300)):
            print('%-6s incomplete' % model); continue
        dt = {k: b1(t[(model, 0)], k) - b1(t[(model, 300)], k) for k in KS}
        dh = {k: b1(h[(model, 0)], k) - b1(h[(model, 300)], k) for k in KS}
        print('%-6s per-sequence drop, training seqs %s | held-out %s | held-out stage 0 %s, stage 300 %s | held-out '
              'clip B=64 stage 0 %s, stage 300 %s | batches %d' % (
                  model, [round(dt[k], 3) for k in KS], [round(dh[k], 3) for k in KS],
                  [round(b1(h[(model, 0)], k), 3) for k in KS], [round(b1(h[(model, 300)], k), 3) for k in KS],
                  [round(c64(h[(model, 0)], k), 3) for k in KS], [round(c64(h[(model, 300)], k), 3) for k in KS],
                  h[(model, 0)]['nb']))
        if np.mean(list(dt.values())) > 0.05:
            ratios[model] = float(np.mean(list(dh.values())) / np.mean(list(dt.values())))
    print('ratios (held-out drop / training-sequence drop): %s' % {k: round(v, 2) for k, v in ratios.items()})
    if len(ratios) < 3:
        verdict = 'INCOMPLETE'
    elif all(v >= 0.5 for v in ratios.values()):
        verdict = 'GENERAL'
    elif all(v < 0.5 for v in ratios.values()):
        verdict = 'MOSTLY-SEEN-SEQUENCES'
    else:
        verdict = 'MIXED %s' % {k: ('general' if v >= 0.5 else 'seen') for k, v in ratios.items()}
    print('VERDICT: %s' % verdict)
    return verdict


if __name__ == '__main__':
    main()
