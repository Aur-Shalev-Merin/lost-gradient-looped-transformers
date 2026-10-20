"""v210 (plan frozen in docs/prereg_v210_followups.md before any measurement): two follow-ups from outside review 6.

Part A: is the fall of the lost gradient during fine-tuning a change in the loop? (Parcae-140m R=32, Part II pln5 / r5
R=5; v201_batch_lost setup and fine-tune, imported unchanged.) On the first 64 GSM8K TEST sequences (held out, as
v203), batch 1, fp32, train mode, per sequence and state draw (seed 301000 + s), arms k = R (full), 0 (no loop credit),
1, 2, 4:
    all    = |G - T_k| / |G|            (the paper's statistic)
    loop   = |G - T_k| / |G - T_0|      (denominator without the gradient that reaches the weights with no loop credit)
at three states: the trained model (stage 0), after 300 of 600 steps of full-backprop fine-tuning (full300), and after
300 of 600 steps of fine-tuning with backprop through the last 2 loops (trunc300). Fine-tune seeds pre-listed
2011, 2012, 2013: the first that passes the guard (<= 10 non-finite steps, finite last loss) is used.

Part B: Ouro-1.4B (R = 4), the 16 GSM8K-validation chunks of v197 / Table 1, batch 1, fp32.
    B1  per loop t: the pre-norm residual x_t (output of the 24 layers) and the loop output h_t = norm(x_t); their norms,
        and the backward signal on each (final-output loss, k = R). Scale-free sizes |dL/dx_t| |x_t| and |dL/dh_t| |h_t|.
    B2  lost gradient of Ouro's training objective without its entropy term: per token sum_t p_t L_t, with p_t from
        the model's exit gate (lambda_t = sigmoid(gate(h_t)); p_t = lambda_t prod_{s<t}(1 - lambda_s), last step takes
        the rest), L_t the next-token CE of lm_head(h_t). k = 3, 2, 1: loops before R - k run without gradient; the
        exit read-outs (gate, lm_head) of every loop stay differentiable. Over all parameters.

  python v210_followups.py A --model parcae|pln5|r5 --runs s0,full300,trunc300 [--weights DIR]
  python v210_followups.py B --seqs 0:8 --tag g0
  python v210_followups.py --analyze DIR [DIR ...]
Env: TINY=1 (pipeline test on tiny models, CPU), OUT_DIR.
"""
import argparse, glob, hashlib, json, math, os, sys, time
import numpy as np
import torch

TINY = os.environ.get('TINY') == '1'
OUT = os.environ.get('OUT_DIR', '/kaggle/working' if os.path.isdir('/kaggle/working') else '.')
KS = (1, 2, 4)
N_SEQ = 4 if TINY else 64
FT_SEEDS = (2011, 2012, 2013)
K_TRUNC = 2
SEED1 = 301000


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v210_log.txt'), 'a').write(s + '\n')


def sha(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(open(os.path.abspath(p), 'rb').read().replace(b'\r\n', b'\n'))
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------------------------------------- Part A
def part_a(model, runs, weights):
    import v201_batch_lost as B
    out = dict(model=model, part='A', ks=list(KS), k_trunc=K_TRUNC, runs={})

    def measure(m, R, meas, chunks, params):
        rows = []
        for s in range(min(N_SEQ, len(chunks))):
            ids = chunks[s:s + 1].to(B.DEV); vec = {}
            for k in (R, 0) + KS:
                m.zero_grad(set_to_none=True)
                meas(ids, k, SEED1 + s).backward()
                vec[k] = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1).float()
                                    for p in params])
            m.zero_grad(set_to_none=True)
            G = vec[R]; gn = float(G.norm()); ln = float((G - vec[0]).norm())
            rows.append(dict(s=s, gnorm=gn, loopnorm=ln, all={k: float((G - vec[k]).norm()) / gn for k in KS},
                             loop={k: float((G - vec[k]).norm()) / ln for k in KS}))
            del vec, G
        return rows

    for run in runs:
        attempts = [None] if run == 's0' else list(FT_SEEDS)
        for seed in attempts:
            m, data, pack, _, R, meas, train, amp, mods = B.setup(model, weights)
            params = [p for p in m.parameters() if p.requires_grad]
            chunks = data['test']
            if B.TINY:
                chunks = torch.cat([chunks] * 6)
            rec = dict(run=run, R=R, n_params=sum(p.numel() for p in params))
            if seed is not None:
                B.FT_SEED = seed
                k_ft = R if run == 'full300' else K_TRUNC
                rec['finetune'] = B.finetune(m, params, data, pack, train, amp, k_ft, B.LR[model])
                rec['finetune']['k'] = k_ft
                log('  %s %s fine-tune seed %d: %s' % (model, run, seed, {k: v for k, v in rec['finetune'].items() if k != 'curve'}))
                if not rec['finetune']['ok']:
                    out['runs'].setdefault('excluded', []).append(dict(run=run, seed=seed, nonfinite=rec['finetune']['nonfinite']))
                    del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    continue
            t0 = time.time(); rec['rows'] = measure(m, R, meas, chunks, params); rec['s'] = round(time.time() - t0)
            med = lambda key, k: float(np.median([r[key][k] for r in rec['rows']]))
            log('  %s %s: all %s | loop %s (%ds)' % (model, run, [round(med('all', k), 3) for k in KS],
                                                  [round(med('loop', k), 3) for k in KS], rec['s']))
            out['runs'][run] = rec; out['driver_sha'] = sha([__file__] + mods)
            del m
            break
        json.dump(out, open(os.path.join(OUT, 'v210_A_%s.json' % model), 'w'), indent=1)


# ---------------------------------------------------------------------------------------------------------- Part B
def part_b(seqs, tag):
    os.environ.setdefault('LOOP_MODEL', 'ouro')
    import loop_grad_probe as P
    import v197_ouro_allparam as V
    from torch.utils.checkpoint import checkpoint
    import torch.nn.functional as Fn
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    tok, model = P.load(); ad = P.Ouro(model); b = ad.b; R = 4
    params = list(model.parameters())
    for p in params:
        p.requires_grad_(True)
    sizes = [p.numel() for p in params]; offs = np.cumsum([0] + sizes); ntot = int(offs[-1])
    data = V.val_chunks(tok, P)
    gate = b.early_exit_gate if hasattr(b, 'early_exit_gate') else model.model.early_exit_gate

    def prepare(ids):
        h = b.embed_tokens(ids)
        cp = torch.arange(ids.shape[1], device=ids.device); pos = cp.unsqueeze(0)
        mask = ad.mod.create_causal_mask(config=b.config, input_embeds=h, attention_mask=None, cache_position=cp,
                                         past_key_values=None, position_ids=pos)
        return h, dict(mask=mask, pos=pos, cp=cp, pe=b.rotary_emb(h, pos))

    def layers(h, t, ctx, ckpt):
        for layer in b.layers[: b.config.num_hidden_layers]:
            def run(x, layer=layer):
                y = layer(x, attention_mask=ctx['mask'], position_ids=ctx['pos'], past_key_value=None, use_cache=False,
                          cache_position=ctx['cp'], position_embeddings=ctx['pe'], current_ut=t)
                return y[0] if isinstance(y, tuple) else y
            h = checkpoint(run, h, use_reentrant=False) if ckpt else run(h)
        return h

    def forward(ids, k, objective, hooks=None):
        """objective 'final' (Table 1) or 'exit' (sum_t p_t L_t). Loops t < R - k run without gradient."""
        for p in params:
            p.grad = None
        h, ctx = prepare(ids); exits, stats = [], []
        for t in range(R):
            on = t >= R - k
            with torch.set_grad_enabled(on):
                x = layers(h, t, ctx, ckpt=on)
                hn = b.norm(x)
            if not on:
                x, hn = x.detach(), hn.detach()
            if hooks is not None and on:
                x.register_hook(lambda g, t=t: hooks.__setitem__(('x', t), float(g.float().norm())))
                hn.register_hook(lambda g, t=t: hooks.__setitem__(('h', t), float(g.float().norm())))
            stats.append((float(x.float().norm()), float(hn.float().norm())))
            exits.append(hn); h = hn
        labels = torch.cat([ids[:, 1:], torch.full_like(ids[:, :1], -100)], 1).view(-1)
        if objective == 'final':
            logits = model.lm_head(exits[-1]).float()
            loss = Fn.cross_entropy(logits.view(-1, logits.shape[-1]), labels, ignore_index=-100)
        else:
            remaining = None; total = 0.0; valid = (labels != -100).float()
            for t, hn in enumerate(exits):
                lam = torch.sigmoid(gate(hn).squeeze(-1)).float().view(-1)
                if remaining is None:
                    remaining = torch.ones_like(lam)
                p_t = lam * remaining if t < R - 1 else remaining
                remaining = remaining * (1 - lam)
                logits = model.lm_head(hn).float()
                ce = Fn.cross_entropy(logits.view(-1, logits.shape[-1]), labels.clamp(min=0), reduction='none') * valid
                total = total + (p_t * ce).sum()
            loss = total / valid.sum()
        loss.backward()
        return float(loss), stats

    def to_cpu(buf):
        for p, o, n in zip(params, offs[:-1], sizes):
            buf[o:o + n].copy_(p.grad.detach().float().flatten() if p.grad is not None else torch.zeros(n))

    def err_vs(gR):
        a = np.zeros(2)
        for p, o, n in zip(params, offs[:-1], sizes):
            g = p.grad.detach().float().flatten() if p.grad is not None else None
            for s in range(0, n, V.CHUNK):
                e = min(n, s + V.CHUNK); r = gR[o + s:o + e].to(P.DEV).double()
                x = g[s:e].double() if g is not None else torch.zeros_like(r)
                a += [float(((x - r) ** 2).sum()), float((r * r).sum())]
        return float(np.sqrt(a[0] / a[1]))

    gR = torch.empty(ntot, dtype=torch.float32)
    out = dict(part='B', R=R, n_params=ntot, rows=[], checks={}, driver_sha=sha([__file__, V.__file__]))
    lo, hi = seqs
    for i in range(lo, min(hi, data.shape[0])):
        t0 = time.time(); ids = data[i:i + 1].to(P.DEV); hk = {}
        loss_f, stats = forward(ids, R, 'final', hooks=hk)
        row = dict(i=i, loss_final=loss_f, norm_x=[s[0] for s in stats], norm_h=[s[1] for s in stats],
                   adj_x=[hk[('x', t)] for t in range(R)], adj_h=[hk[('h', t)] for t in range(R)])
        loss_e, _ = forward(ids, R, 'exit'); to_cpu(gR); row['loss_exit'] = loss_e
        if i == lo:
            forward(ids, R, 'exit'); out['checks']['repeat_exit'] = err_vs(gR)
            log('  check: repeated exit-objective pass, rel err %.2e' % out['checks']['repeat_exit'])
        row['exit_err'] = {}
        for k in (3, 2, 1):
            forward(ids, k, 'exit'); row['exit_err'][k] = err_vs(gR)
        row['s'] = round(time.time() - t0, 1); out['rows'].append(row)
        log('seq %d: |x| %s |h| %s | adj_x %s adj_h %s | exit-objective lost k=3/2/1 %s (%.0fs)' % (
            i, ['%.3g' % v for v in row['norm_x']], ['%.3g' % v for v in row['norm_h']], ['%.3g' % v for v in row['adj_x']],
            ['%.3g' % v for v in row['adj_h']], ['%.3f' % row['exit_err'][k] for k in (3, 2, 1)], row['s']))
        json.dump(out, open(os.path.join(P.OUT, 'v210_B_%s.json' % tag), 'w'), indent=1)


# ---------------------------------------------------------------------------------------------------------- analysis
def fit_rho(v):
    v = np.asarray(v, float); return float(np.exp(np.polyfit(np.arange(len(v)), np.log(v[::-1] + 1e-300), 1)[0]))


def analyze(dirs):
    """Frozen outcome mapping of docs/prereg_v210_followups.md."""
    A = {}; Brows = []
    for d in dirs:
        for f in glob.glob(os.path.join(d, '**', 'v210_A_*.json'), recursive=True):
            r = json.load(open(f))
            A.setdefault(r['model'], {}).update(r['runs'])
        for f in glob.glob(os.path.join(d, '**', 'v210_B_*.json'), recursive=True):
            Brows += json.load(open(f))['rows']
    med = lambda rows, key, k: float(np.median([r[key][str(k)] if str(k) in r[key] else r[key][k] for r in rows]))
    a1, a2 = {}, {}
    print('PART A (held-out GSM8K test, per-sequence medians, k = 1 / 2 / 4)')
    for model in ('parcae', 'pln5', 'r5'):
        runs = A.get(model, {})
        if not all(x in runs for x in ('s0', 'full300', 'trunc300')):
            print('  %-6s incomplete: %s' % (model, sorted(runs))); continue
        g = lambda run, key: [med(runs[run]['rows'], key, k) for k in KS]
        fa = np.mean(np.subtract(g('s0', 'all'), g('full300', 'all')))
        fl = np.mean(np.subtract(g('s0', 'loop'), g('full300', 'loop')))
        ft = np.mean(np.subtract(g('s0', 'all'), g('trunc300', 'all')))
        a1[model] = fl / fa if fa > 0 else float('nan'); a2[model] = (ft, fa)
        for run in ('s0', 'full300', 'trunc300'):
            print('  %-6s %-8s all %s | loop %s' % (model, run, [round(x, 3) for x in g(run, 'all')], [round(x, 3) for x in g(run, 'loop')]))
        print('  %-6s mean fall: all %.3f, loop %.3f (ratio %.2f) | truncated fine-tune fall (all) %.3f' % (model, fa, fl, a1[model], ft))
    v1 = ('INCOMPLETE' if len(a1) < 3 else 'IN-LOOP' if all(v >= 0.5 for v in a1.values())
          else 'READOUT-SHARE' if all(v < 0.5 for v in a1.values()) else 'MIXED %s' % {k: round(v, 2) for k, v in a1.items()})
    def tcls(ft, fa):
        return 'FALLS' if ft >= 0.5 * fa else 'FLAT' if abs(ft) < 0.03 else 'RISES' if ft <= -0.03 else 'PARTIAL'
    cl = {m: tcls(*v) for m, v in a2.items()}
    v2 = 'INCOMPLETE' if len(cl) < 3 else ('TRUNC-%s' % next(iter(cl.values())) if len(set(cl.values())) == 1 else 'MIXED %s' % cl)
    print('VERDICT A1 (loop-only): %s | VERDICT A2 (truncated fine-tune): %s' % (v1, v2))
    if Brows:
        R = len(Brows[0]['adj_h'])
        rx = [fit_rho(np.multiply(r['adj_x'], r['norm_x'])) for r in Brows]
        rh = [fit_rho(np.multiply(r['adj_h'], r['norm_h'])) for r in Brows]
        rraw = [fit_rho(r['adj_h']) for r in Brows]
        print('PART B (Ouro, %d sequences)' % len(Brows))
        print('  median |x_t| %s | |h_t| %s' % ([round(float(np.median([r['norm_x'][t] for r in Brows])), 1) for t in range(R)],
                                            [round(float(np.median([r['norm_h'][t] for r in Brows])), 1) for t in range(R)]))
        print('  rho of |dL/dh| %.3f | scale-free rho on h %.3f, on pre-norm x %.3f (medians)' % (
            np.median(rraw), np.median(rh), np.median(rx)))
        b1 = ('GROWS-SCALE-FREE' if np.median(rh) > 1 and np.median(rx) > 1 else
              'NORMALISATION' if np.median(rh) <= 1 and np.median(rx) <= 1 else 'MIXED')
        ee = {k: float(np.median([r['exit_err'][str(k)] if str(k) in r['exit_err'] else r['exit_err'][k] for r in Brows])) for k in (1, 2, 3)}
        print('  exit-weighted objective lost gradient k = 1 / 2 / 3: %.3f / %.3f / %.3f' % (ee[1], ee[2], ee[3]))
        b2 = 'MOST-LOST' if ee[2] >= 0.5 else 'LESS-LOST'
        print('VERDICT B1: %s | VERDICT B2: %s' % (b1, b2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('part', nargs='?'); ap.add_argument('--model'); ap.add_argument('--runs', default='s0,full300,trunc300')
    ap.add_argument('--weights', default=None); ap.add_argument('--seqs', default='0:16'); ap.add_argument('--tag', default='x')
    ap.add_argument('--analyze', nargs='*')
    A_ = ap.parse_args()
    if A_.analyze is not None:
        analyze(A_.analyze or [OUT])
    elif A_.part == 'A':
        part_a(A_.model, A_.runs.split(','), A_.weights)
    elif A_.part == 'B':
        lo, hi = (int(x) for x in A_.seqs.split(':')); part_b((lo, hi), A_.tag)
