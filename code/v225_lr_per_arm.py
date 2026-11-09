"""v225 (plan frozen in docs/prereg_v225_lr_per_arm.md before any run): learning rates chosen per arm, on data the
cost is not scored on.

Part O, Ouro-1.4B (S18's recipe; v219_ouro_cost.py imported unchanged). S18 chose its LRs on GSM8K test, the split it
scored on, and its truncated arms used full backprop's LR. Here GSM8K test is split by problem index: even-indexed
problems = SEL (LR selection), odd = SCORE (the cost). Every arm (k = 4 full, 2, 1, 0 floor) gets its own LR, the
lowest SEL loss on the selection seed; the cost is then scored on SCORE over 5 new seeds.
Part P, the two Part II models (S11's recipe; runs by v200_ifm_trunc.py through build_kaggle_v200.py, both unchanged):
k = 1, 2, 4 at LRs 3e-5, 1e-4, 3e-4 on v200's pilot seed 9001, selected on val as v200 selected its controls. For a
model with any cell whose selected LR is not 1e-4, every arm of that model is rerun on v200's scored seeds 2001101-05:
full (1e-4), floor (3e-4), each such cell at its selected LR and at 1e-4.

  python v225_lr_per_arm.py --phase selftest                         Ouro (v219's selftest)
  python v225_lr_per_arm.py --phase train --cells k:lr:seed,...      Ouro; appended to v225_ouro_runs.json
  python v225_lr_per_arm.py --select-ouro DIR [DIR ...]              the locked selection: next cells to run
  python v225_lr_per_arm.py --analyze-ouro DIR [DIR ...]             the locked Ouro analysis
  python v225_lr_per_arm.py --select-ifm DIR [DIR ...]               Part P selection (v200 run files)
  python v225_lr_per_arm.py --analyze-ifm DIR [DIR ...]              Part P analysis (v200 study runs + v225 runs)
Env: TINY=1 (tiny random Ouro on CPU), OUT_DIR.
"""
import argparse, glob, hashlib, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
TINY = os.environ.get('TINY') == '1'
OUT = os.environ.get('OUT_DIR', '/kaggle/working' if os.path.isdir('/kaggle/working') else '.')
T90 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943}      # t quantile 0.95, by degrees of freedom

# ------------------------------------------------------------------ Part O, locked at the freeze (prereg section 2)
R = 4
SEL_SEED = 2251001
GRID = (3e-5, 1e-4, 3e-4)                         # every arm
EXT = {3e-5: 1e-5, 3e-4: 1e-3}                    # one step beyond an edge, run once on the selection seed
SCORE_SEEDS = (2251011, 2251012, 2251013, 2251014, 2251015, 2251016, 2251017)   # 5 scored + 2 backups
N_SCORE = 5
L_PRED = {1: 0.2581, 2: 0.1872}                   # S18's predictions (v219: S6 curve at Ouro's lost gradient)
KP = {1: 0.224, 2: 0.100}                         # Parcae-140m's own cost at the same k
V219 = {1: 1.133, 2: 0.652}                       # S18's costs (full test, all arms at 1e-4), for description

# ------------------------------------------------------------------ Part P, locked at the freeze
IFM = ('huginn-s-fixed-pln5', 'huginn-s-fixed-r5')
IFM_GRID = (3e-5, 1e-4, 3e-4)
IFM_EXT = {3e-5: 1e-5, 3e-4: 1e-3}
IFM_BASE_LR, IFM_FLOOR_LR, IFM_R = 1e-4, 3e-4, 5
IFM_SEL_SEED = 9001
IFM_SEEDS = (2001101, 2001102, 2001103, 2001104, 2001105)
S11 = {  # S11's locked predictions (prereg v200 section 2) and v200's costs at 1e-4 (Table 9)
    'L': {'huginn-s-fixed-pln5': {1: 0.138, 2: 0.058, 4: 0.005}, 'huginn-s-fixed-r5': {1: 0.182, 2: 0.097, 4: 0.011}},
    'KP': {1: 0.224, 2: 0.100, 4: 0.024}, 'KS': {1: 0.009, 2: -0.001, 4: 0.0}}


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    open(os.path.join(OUT, 'v225_log.txt'), 'a').write(s + '\n')


def driver_sha():
    h = hashlib.sha256()
    here = os.path.dirname(os.path.abspath(__file__))
    for f in ('v225_lr_per_arm.py', 'v219_ouro_cost.py'):
        h.update(open(os.path.join(here, f), 'rb').read().replace(b'\r\n', b'\n'))
    return h.hexdigest()[:12]


def ci(x):
    x = np.asarray(x, float); m = float(x.mean())
    if len(x) < 2:
        return m, float('nan'), float('nan'), float('nan')
    se = float(x.std(ddof=1) / math.sqrt(len(x))); t = T90[len(x) - 1]
    return m, se, m - t * se, m + t * se


# ------------------------------------------------------------------------------------------------- Part O: training
def ouro_data():
    """dict(train, val = SEL chunks (even test problems), test = SCORE chunks (odd test problems)): v219's train_cell
    evaluates data['val'] and data['test'], so its records' 'val' / 'test' are SEL / SCORE here (renamed on save)."""
    import v219_ouro_cost as O
    import loop_grad_probe as P
    tok = P.load()[0] if O.TINY else __import__('transformers').AutoTokenizer.from_pretrained(P.REPO, trust_remote_code=True)
    base = O.gsm8k(tok)
    if O.TINY:                                   # O.gsm8k's TINY world: test = the third draw of the same rng
        g = np.random.default_rng(0)
        mk = lambda n: [list(g.integers(1, 1000, size=g.integers(20, 60))) for _ in range(n)]
        _, _, te = mk(60), mk(10), mk(10)
    else:
        from datasets import load_dataset
        te = [tok(ex['question'] + '\n' + ex['answer']).input_ids for ex in load_dataset('openai/gsm8k', 'main', split='test')]
    return O, base, dict(train=base['train'], val=O.pack(te[0::2]), test=O.pack(te[1::2]))


def ouro_train(cells):
    O, base, data = ouro_data()
    log('v225 driver %s | train %d examples, SEL %d chunks, SCORE %d chunks' % (
        driver_sha(), len(data['train']), len(data['val']), len(data['test'])))
    f = os.path.join(OUT, 'v225_ouro_runs.json')
    out = json.load(open(f)) if os.path.exists(f) else dict(runs=[])
    for c in cells:
        k, lr, seed = c.split(':'); k, lr, seed = int(k), float(lr), int(seed)
        if any((r['k'], r['lr'], r['seed']) == (k, lr, seed) for r in out['runs']):
            log('  skip k=%d lr=%g seed=%d (present)' % (k, lr, seed)); continue
        rec = O.train_cell(data, k, lr, seed)
        rec['sel'], rec['score'] = rec.pop('val'), rec.pop('test'); rec['v225_sha'] = driver_sha()
        log('  RUN k=%d lr=%g seed=%d: SEL %.4f / %.4f, SCORE %.4f / %.4f (final / exit) | non-finite %d | %.1f s/step' % (
            k, lr, seed, rec['sel']['final'], rec['sel']['exit'], rec['score']['final'], rec['score']['exit'],
            rec['nonfinite_steps'], rec['s_train'] / O.STEPS))
        out['runs'].append(rec)
        json.dump(out, open(f, 'w'), indent=1)


def ouro_selftest():
    O, base, data = ouro_data()
    log('v225 driver %s: SEL %d chunks, SCORE %d chunks, disjoint by problem index' % (
        driver_sha(), len(data['val']), len(data['test'])))
    O.selftest(base)


# ------------------------------------------------------------------------------------------------ Part O: selection
def ouro_runs(dirs):
    runs = []
    for d in dirs:
        for f in glob.glob(os.path.join(d, '**', 'v225_ouro_runs.json'), recursive=True):
            runs += json.load(open(f))['runs']
    return runs


def run_ok(r):
    return not r['diverged'] and all(math.isfinite(x) for x in list(r['sel'].values()) + list(r['score'].values())) \
        and (r['k'] != R or r['nonfinite_steps'] <= 10)


def select_ouro(runs):
    """Per arm, the LR with the lowest SEL final-output loss on the selection seed among valid runs. A choice at an
    edge of the LRs run asks for the one extension step beyond it (EXT), once. Returns (choice {k: lr}, cells to run,
    table lines)."""
    lines, todo, choice = [], [], {}
    for k in (R, 2, 1, 0):
        rs = {r['lr']: r for r in runs if r['k'] == k and r['seed'] == SEL_SEED and run_ok(r)}
        missing = [lr for lr in GRID if lr not in rs and not any(r['k'] == k and r['seed'] == SEL_SEED and r['lr'] == lr for r in runs)]
        if missing:
            todo += ['%d:%g:%d' % (k, lr, SEL_SEED) for lr in missing]
            lines.append('  k=%d: grid incomplete, run %s' % (k, missing)); continue
        if not rs:
            lines.append('  k=%d: no valid run on the grid' % k); continue
        best = min(rs, key=lambda lr: rs[lr]['sel']['final'])
        tab = '  '.join('%g: SEL %.4f SCORE %.4f' % (lr, rs[lr]['sel']['final'], rs[lr]['score']['final']) for lr in sorted(rs))
        ext = EXT.get(best) if best in (min(GRID), max(GRID)) else None
        ran = {r['lr'] for r in runs if r['k'] == k and r['seed'] == SEL_SEED}
        if ext is not None and ext not in ran:
            todo.append('%d:%g:%d' % (k, ext, SEL_SEED))
            lines.append('  k=%d: %s -> best %g at the grid edge: extend to %g' % (k, tab, best, ext)); continue
        choice[k] = best
        lines.append('  k=%d: %s -> SELECTED %g%s' % (k, tab, best, ' (at the edge of the extended grid)'
                                                   if best in (min(rs), max(rs)) and len(rs) > len(GRID) else ''))
    return choice, todo, lines


def score_cells(choice):
    return ['%d:%g:%d' % (k, choice[k], s) for s in SCORE_SEEDS[:N_SCORE] for k in (R, 2, 1, 0)]


def analyze_ouro(dirs):
    runs = ouro_runs(dirs)
    lines = []; P = lines.append
    P('v225 Part O (Ouro-1.4B): driver %s; %d runs read' % (driver_sha(), len(runs)))
    choice, todo, tab = select_ouro(runs)
    P('LR selection (seed %d, SEL = even GSM8K test problems, final-output loss):' % SEL_SEED); lines += tab
    if todo or len(choice) < 4:
        P('SELECTION INCOMPLETE: run %s' % ','.join(todo)); return '\n'.join(lines)
    P('selected LRs: full %g, k=2 %g, k=1 %g, floor %g' % (choice[R], choice[2], choice[1], choice[0]))
    cell = lambda k, s: [r for r in runs if r['k'] == k and r['seed'] == s and math.isclose(r['lr'], choice[k], rel_tol=1e-9)]
    valid = []
    for s in SCORE_SEEDS:
        cs = {k: cell(k, s) for k in (R, 2, 1, 0)}
        if any(len(v) != 1 for v in cs.values()):
            P('  seed %d: cells present %s -> not scored' % (s, {k: len(v) for k, v in cs.items()})); continue
        ok = all(run_ok(v[0]) for v in cs.values())
        P('  seed %d: full non-finite %d -> %s' % (s, cs[R][0]['nonfinite_steps'], 'valid' if ok else 'EXCLUDED'))
        if ok:
            valid.append({k: v[0] for k, v in cs.items()})
    if len(valid) < N_SCORE:
        P('FEWER THAN %d VALID SEEDS (%d): INCOMPLETE' % (N_SCORE, len(valid))); return '\n'.join(lines)
    valid = valid[:N_SCORE]
    res = {}
    for obj in ('final', 'exit'):
        L = lambda v, k: v[k]['score'][obj]
        res[obj] = dict(gap=ci([L(v, 0) - L(v, R) for v in valid]))
        for k in (1, 2):
            res[obj][k] = ci([(L(v, k) - L(v, R)) / (L(v, 0) - L(v, R)) for v in valid])
            res[obj]['nats%d' % k] = ci([L(v, k) - L(v, R) for v in valid])
    F = res['final']
    P('SCORE loss, final output, per seed (full / k=2 / k=1 / floor):')
    for v in valid:
        P('  seed %d: %.4f / %.4f / %.4f / %.4f' % (v[R]['seed'], *[v[k]['score']['final'] for k in (R, 2, 1, 0)]))
    P('floor gap: %.4f (s.e. %.4f; 90%% %.4f to %.4f)' % F['gap'])
    gate = F['gap'][0] >= 0.03 and F['gap'][2] > 0
    P('floor gate (>= 0.03 nats, 90%% lower bound > 0): %s' % ('PASS' if gate else 'FAIL'))
    for k in (1, 2):
        P('cost k=%d: %.4f (s.e. %.4f; 90%% %.4f to %.4f) | added nats %.4f (90%% %.4f to %.4f) | L %.3f, KP %.3f, S18 %.3f'
          % (k, *F[k], F['nats%d' % k][0], F['nats%d' % k][2], F['nats%d' % k][3], L_PRED[k], KP[k], V219[k]))
    if not gate:
        P('VERDICT V1: UNINFORMATIVE (floor gate failed) | V2: UNINFORMATIVE'); return '\n'.join(lines)
    lo = {k: min(L_PRED[k], KP[k]) for k in (1, 2)}; hi = {k: max(L_PRED[k], KP[k]) for k in (1, 2)}
    mae_l = float(np.mean([abs(F[k][0] - L_PRED[k]) for k in (1, 2)]))
    mae_k = float(np.mean([abs(F[k][0] - KP[k]) for k in (1, 2)]))
    v1 = ('BELOW-BOTH' if all(F[k][3] < lo[k] for k in (1, 2)) else 'ABOVE-BOTH' if all(F[k][2] > hi[k] for k in (1, 2))
          else 'L-CLOSER' if mae_l <= 0.5 * mae_k else 'KP-CLOSER' if mae_k <= 0.5 * mae_l else 'TIE')
    v2 = 'FLOOR-LEVEL' if F[1][3] >= 1.0 else 'BETTER-THAN-FLOOR'
    P('VERDICT V1 (S18 rule, per-arm LRs, SCORE): %s (MAE L %.4f, KP %.4f)' % (v1, mae_l, mae_k))
    P('VERDICT V2 (k=1 against the floor): %s (90%% upper bound of the k=1 cost %.3f)' % (v2, F[1][3]))
    E = res['exit']
    P('exit-weighted objective (descriptive): gap %.4f; cost k=1 %.4f (90%% %.4f to %.4f), k=2 %.4f (90%% %.4f to %.4f)'
      % (E['gap'][0], E[1][0], E[1][2], E[1][3], E[2][0], E[2][2], E[2][3]))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------------------------------------- Part P
def ifm_runs(dirs):
    runs = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, '**', 'v200_runs_*.jsonl'), recursive=True):
            for l in open(f):
                r = json.loads(l)
                runs.setdefault((r['variant'], r['k'], round(r['lr'], 12), r['seed']), []).append(dict(r, file=f))
    return runs


def ifm_ok(r):
    return not r['diverged'] and math.isfinite(r['val']) and math.isfinite(r['test']) and r['nonfinite_steps'] <= 10


def select_ifm(runs):
    """Per model and k = 1, 2, 4: the LR with the lowest val loss on seed 9001 (v225 runs only: tag v225 in the file
    name), extending once beyond an edge. Returns ({(variant, k): lr}, cells to run {variant: [...]}, lines)."""
    lines, todo, choice = [], {}, {}
    for v in IFM:
        for k in (1, 2, 4):
            rs = {lr: [r for r in rr if 'v225' in os.path.basename(r['file'])]
                  for (vv, kk, lr, s), rr in runs.items() if vv == v and kk == k and s == IFM_SEL_SEED}
            rs = {lr: rr[0] for lr, rr in rs.items() if rr}
            missing = [lr for lr in IFM_GRID if round(lr, 12) not in rs]
            if missing:
                todo.setdefault(v, []).extend('%d:%g:%d' % (k, lr, IFM_SEL_SEED) for lr in missing)
                lines.append('  %s k=%d: grid incomplete, run %s' % (v, k, missing)); continue
            ok = {lr: r for lr, r in rs.items() if ifm_ok(r)}
            if not ok:
                lines.append('  %s k=%d: no valid run' % (v, k)); continue
            best = min(ok, key=lambda lr: ok[lr]['val'])
            tab = '  '.join('%g: val %.4f test %.4f' % (lr, rs[lr]['val'], rs[lr]['test']) for lr in sorted(rs))
            ext = next((x for e, x in IFM_EXT.items() if math.isclose(best, e)), None)
            if ext is not None and round(ext, 12) not in rs:
                todo.setdefault(v, []).append('%d:%g:%d' % (k, ext, IFM_SEL_SEED))
                lines.append('  %s k=%d: %s -> best %g at the grid edge: extend to %g' % (v, k, tab, best, ext)); continue
            choice[(v, k)] = best
            lines.append('  %s k=%d: %s -> SELECTED %g' % (v, k, tab, best))
    return choice, todo, lines


def rescore_cells(choice):
    """For each model with a cell whose selected LR is not 1e-4: full, floor, and each such cell at its LR and at 1e-4,
    on the 5 scored seeds."""
    out = {}
    for v in IFM:
        moved = [k for k in (1, 2, 4) if (v, k) in choice and not math.isclose(choice[(v, k)], IFM_BASE_LR)]
        if moved:
            cells = [(IFM_R, IFM_BASE_LR), (0, IFM_FLOOR_LR)] + [(k, choice[(v, k)]) for k in moved] + \
                    [(k, IFM_BASE_LR) for k in moved]
            out[v] = ['%d:%g:%d' % (k, lr, s) for s in IFM_SEEDS for k, lr in cells]
    return out


def analyze_ifm(dirs):
    runs = ifm_runs(dirs)
    lines = []; P = lines.append
    P('v225 Part P (Part II models): %d run keys read' % len(runs))
    choice, todo, tab = select_ifm(runs)
    P('per-k LR selection (seed %d, val):' % IFM_SEL_SEED); lines += tab
    if todo:
        P('SELECTION INCOMPLETE: run %s' % todo); return '\n'.join(lines)
    resc = rescore_cells(choice)
    if not resc:
        P('VERDICT P: SAME-LR (every cell selected full backprop\'s LR %g; S11 stands as scored)' % IFM_BASE_LR)
        return '\n'.join(lines)
    get = lambda v, k, lr, s: [r for r in runs.get((v, k, round(lr, 12), s), []) if 'v225' in os.path.basename(r['file'])]
    tuned = {}
    for v in IFM:
        if v not in resc:
            continue
        moved = [k for k in (1, 2, 4) if not math.isclose(choice[(v, k)], IFM_BASE_LR)]
        ok = True
        for s in IFM_SEEDS:
            for k, lr in [(IFM_R, IFM_BASE_LR), (0, IFM_FLOOR_LR)] + [(k, choice[(v, k)]) for k in moved] + [(k, IFM_BASE_LR) for k in moved]:
                if len(get(v, k, lr, s)) != 1 or not ifm_ok(get(v, k, lr, s)[0]):
                    ok = False; P('  %s seed %d k=%d lr=%g: missing or invalid' % (v, s, k, lr))
        if not ok:
            P('  %s: INCOMPLETE' % v); continue
        L = lambda k, lr, s: get(v, k, lr, s)[0]['test']
        gap = ci([L(0, IFM_FLOOR_LR, s) - L(IFM_R, IFM_BASE_LR, s) for s in IFM_SEEDS])
        P('  %s floor gap %.4f (90%% %.4f to %.4f)' % (v, gap[0], gap[2], gap[3]))
        for k in moved:
            ph = lambda lr: [(L(k, lr, s) - L(IFM_R, IFM_BASE_LR, s)) / (L(0, IFM_FLOOR_LR, s) - L(IFM_R, IFM_BASE_LR, s))
                             for s in IFM_SEEDS]
            t, u = ph(choice[(v, k)]), ph(IFM_BASE_LR)
            d = ci(np.subtract(u, t))
            tuned[(v, k)] = ci(t)[0]
            P('  %s k=%d: cost at its LR %g %.4f, at 1e-4 %.4f; untuned minus tuned %.4f (90%% %.4f to %.4f)' % (
                v, k, choice[(v, k)], ci(t)[0], ci(u)[0], d[0], d[2], d[3]))
    if any(v in resc and not all((v, k) in tuned for k in (1, 2, 4) if not math.isclose(choice[(v, k)], IFM_BASE_LR))
           for v in IFM):
        P('VERDICT P: INCOMPLETE'); return '\n'.join(lines)
    # S11 re-scored: v200's costs (Table 9), with each moved cell's cost replaced by its tuned cost
    v200 = {'huginn-s-fixed-pln5': {1: 0.162, 2: 0.071, 4: 0.015}, 'huginn-s-fixed-r5': {1: 0.291, 2: 0.149, 4: 0.023}}
    phi = {(v, k): tuned.get((v, k), v200[v][k]) for v in IFM for k in (1, 2, 4)}
    mae = {p: float(np.mean([abs(phi[(v, k)] - (S11['L'][v][k] if p == 'L' else S11[p][k])) for v in IFM for k in (1, 2, 4)]))
           for p in ('L', 'KP', 'KS')}
    best_k = min(mae['KP'], mae['KS'])
    vt = ('L-BEATS-K' if mae['L'] <= 0.5 * best_k else 'K-AS-GOOD' if best_k <= mae['L'] else 'INCONCLUSIVE')
    P('S11 re-scored with per-k LRs: MAE L %.4f, KP %.4f, KS %.4f -> %s (as scored: K-AS-GOOD)' % (mae['L'], mae['KP'], mae['KS'], vt))
    P('VERDICT P: LR-MOVED %s; S11 re-scored %s' % (sorted('%s k=%d -> %g' % (v[-4:], k, choice[(v, k)]) for (v, k) in tuned), vt))
    return '\n'.join(lines)


# -------------------------------------------------------------------------------------------------------------- main
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase'); ap.add_argument('--cells', default='')
    for a in ('--select-ouro', '--analyze-ouro', '--select-ifm', '--analyze-ifm'):
        ap.add_argument(a, nargs='+')
    A = ap.parse_args()
    if A.select_ouro:
        choice, todo, tab = select_ouro(ouro_runs(A.select_ouro)); print('\n'.join(tab))
        print('NEXT: %s' % (','.join(todo) if todo else 'score ' + ','.join(score_cells(choice))))
    elif A.analyze_ouro:
        txt = analyze_ouro(A.analyze_ouro); print(txt); open(os.path.join(OUT, 'v225_ouro_analysis.txt'), 'w').write(txt + '\n')
    elif A.select_ifm:
        choice, todo, tab = select_ifm(ifm_runs(A.select_ifm)); print('\n'.join(tab))
        print('NEXT: %s' % (todo if todo else rescore_cells(choice) or 'nothing (every cell selected 1e-4)'))
    elif A.analyze_ifm:
        txt = analyze_ifm(A.analyze_ifm); print(txt); open(os.path.join(OUT, 'v225_ifm_analysis.txt'), 'w').write(txt + '\n')
    elif A.phase == 'selftest':
        ouro_selftest()
    elif A.phase == 'train':
        ouro_train([c for c in A.cells.split(',') if c])
