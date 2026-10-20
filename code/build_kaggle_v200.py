"""Build v200 Kaggle kernels (truncated-loop fine-tuning of Part II looped LMs; docs/v200_plan_notes.md). Build only;
does NOT push. One kernel = T4 x2, one v200_ifm_trunc.py process per GPU, cells balanced over the two GPUs by
estimated time. Every process runs the selftest (fp16 vs fp32 truncated gradients, v161 negative control, floor,
checkpointing, decomposition) before it trains; a failed selftest stops that process before any training.
Kernel id: <account>/astrocyte-v200-<variant short>-<tag>, so variants and tags run in parallel.

  python build_kaggle_v200.py pilot-lr --variant V --account A [--tag pilot-lr]
        k = 5 (full) and k = 0 (floor) at LR 1e-4, 3e-4, 9e-4, seed 9001, plus the frozen model's val / test loss
  python build_kaggle_v200.py spread --variant V --account A [--lr-floor F --lr-trunc T] [--tag spread]
        seed spread at k = 1: full and floor at seeds 9002, 9003 and k = 1 at 9001-9003. The seed-9001 controls are
        reused from the LR pilot. Without --lr-floor/--lr-trunc the kernel reads the LR pilot's output
        (kernel_sources: <account>/astrocyte-v200-<short>-pilot-lr) and takes, per arm, the LR with the lowest val
        loss; it STOPS if a choice sits at the edge of the grid (extend the grid first, as v167 pilot 2), unless
        built with --allow-edge.
  python build_kaggle_v200.py study --variant V --lr-floor F --lr-trunc T [--seeds ...] [--kernels 2]
        [--accounts A,B] [--tag study]     cells k = 0, 1, 2, 4, 5 x seeds (5 scored + 2 backups)
Push (NOT done here): kaggle kernels push -p kaggle-kernel-v200-<short>-<tag> --accelerator NvidiaTeslaT4
"""
import argparse, base64, hashlib, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ['v200_ifm_trunc.py', 'ifm_looped.py']
VARIANTS = {'huginn-s-fixed-pln5': 'c581f369846e74b40072cae77ef19fa51437dfa7',   # as v198
            'huginn-s-fixed-r5': '17cdf78e0d9fd8d19bba07ed800875e1de5675fd'}
ACCOUNTS = ('aurmerin', 'aurshalevmerin')
LR_GRID = (1e-4, 3e-4, 9e-4)
PILOT_SEED = 9001
STUDY_SEEDS = tuple(range(2001101, 2001108))          # 5 scored + 2 backups (v165 exclusion rule), provisional
# T4 seconds per 600-step run incl. load, eval (val + test) and rho_end: FLOP model calibrated on v167's T4 runs
# (fp16, batch 8 x 512; about 8.5 TFLOPS effective) - see docs/v200_plan_notes.md
EST = {0: 450, 1: 585, 2: 685, 4: 885, 5: 990}
SETUP_S = 360                                          # pip, weights, GSM8K, selftest (+ step-0 eval in the LR pilot)

ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=('pilot-lr', 'spread', 'study', 'cells'))
ap.add_argument('--cells', default=None, help="cells mode: explicit 'k:lr:seed,...' (e.g. extending the LR grid)")
ap.add_argument('--variant', required=True, choices=sorted(VARIANTS))
ap.add_argument('--account', default=None, choices=ACCOUNTS)
ap.add_argument('--accounts', default=None, help='study: comma-separated accounts, one per kernel (cycled)')
ap.add_argument('--tag', default=None)
ap.add_argument('--lr-floor', type=float, default=None); ap.add_argument('--lr-trunc', type=float, default=None)
ap.add_argument('--allow-edge', action='store_true')
ap.add_argument('--seeds', default=None); ap.add_argument('--kernels', type=int, default=2)
ap.add_argument('--dry-run', action='store_true', help='print the plan and estimates; write nothing')
A = ap.parse_args()
SHORT = A.variant.replace('huginn-s-', '').replace('fixed-', '')
TAG = A.tag or A.mode
DEFAULT_ACCOUNT = {'huginn-s-fixed-pln5': 'aurmerin', 'huginn-s-fixed-r5': 'aurshalevmerin'}[A.variant]   # as v198


def cells_str(cells):
    return ','.join('%d:%g:%d' % c for c in cells)


def balance(cells, n):
    bins = [[0.0, []] for _ in range(n)]
    for c in sorted(cells, key=lambda c: -EST[c[0]]):
        b = min(bins, key=lambda b: b[0]); b[0] += EST[c[0]]; b[1].append(c)
    return bins


KERNELS = []                                           # (slug, account, [args for GPU 0, args for GPU 1], lr_auto, sources)
if A.mode == 'pilot-lr':
    acct = A.account or DEFAULT_ACCOUNT
    jobs = [['--selftest', '--pilot-lr', '--part', '%d/2' % g, '--tag', '%s_g%d' % (TAG, g)] for g in (0, 1)]
    cells = [(k, lr, PILOT_SEED) for k in (5, 0) for lr in LR_GRID]
    est = [b[0] for b in balance(cells, 2)]
    KERNELS.append(('astrocyte-v200-%s-%s' % (SHORT, TAG), acct, jobs, None, []))
elif A.mode == 'spread':
    acct = A.account or DEFAULT_ACCOUNT
    seeds = (9001, 9002, 9003)
    auto = A.lr_floor is None or A.lr_trunc is None
    lf, lt = (None, None) if auto else (A.lr_floor, A.lr_trunc)
    cells = [(5, -1, s) for s in seeds[1:]] + [(0, -2, s) for s in seeds[1:]] + [(1, -1, s) for s in seeds]
    bins = balance(cells, 2)                           # LR placeholders: -1 = trunc / full LR, -2 = floor LR
    jobs = [['--selftest', '--train', cells_str(b[1]), '--tag', '%s_g%d' % (TAG, g)] for g, b in enumerate(bins)]
    if not auto:
        jobs = [[x.replace(':-1:', ':%g:' % lt).replace(':-2:', ':%g:' % lf) if i == 2 else x for i, x in enumerate(j)]
                for j in jobs]
    est = [b[0] for b in bins]
    src = ['%s/astrocyte-v200-%s-pilot-lr' % (acct, SHORT)] if auto else []
    KERNELS.append(('astrocyte-v200-%s-%s' % (SHORT, TAG), acct, jobs, 'auto' if auto else None, src))
elif A.mode == 'cells':                                # explicit cells, e.g. a second LR pilot below the grid's edge
    if not A.cells:
        raise SystemExit('cells mode needs --cells k:lr:seed,...')
    acct = A.account or DEFAULT_ACCOUNT
    cells = [(int(k), float(lr), int(s)) for k, lr, s in (c.split(':') for c in A.cells.split(','))]
    bins = balance(cells, 2)
    jobs = [['--selftest', '--train', cells_str(b[1]), '--tag', '%s_g%d' % (TAG, g)] for g, b in enumerate(bins)]
    est = [b[0] for b in bins]
    KERNELS.append(('astrocyte-v200-%s-%s' % (SHORT, TAG), acct, jobs, None, []))
else:
    if A.lr_floor is None or A.lr_trunc is None:
        raise SystemExit('study needs --lr-floor and --lr-trunc (fixed by the pilots / prereg)')
    seeds = [int(s) for s in A.seeds.split(',')] if A.seeds else list(STUDY_SEEDS)
    cells = [(k, A.lr_floor if k == 0 else A.lr_trunc, s) for s in seeds for k in (0, 5, 1, 2, 4)]
    bins = balance(cells, 2 * A.kernels)
    accts = (A.accounts.split(',') if A.accounts else [A.account or DEFAULT_ACCOUNT])
    est = []
    for i in range(A.kernels):
        jobs = [['--selftest', '--train', cells_str(sorted(bins[2 * i + g][1], key=lambda c: (c[2], c[0]))),
                 '--tag', '%s%d_g%d' % (TAG, i, g)] for g in (0, 1)]
        est += [bins[2 * i][0], bins[2 * i + 1][0]]
        KERNELS.append(('astrocyte-v200-%s-%s%d' % (SHORT, TAG, i), accts[i % len(accts)], jobs, None, []))

emb = {}
for f in FILES:
    b = open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n')
    emb[f] = base64.b64encode(zlib.compress(b, 9)).decode()
    print('  %s sha256 (LF) %s' % (f, hashlib.sha256(b).hexdigest()[:12]))

RUN = r'''FILES = %r
VARIANT, REPO, REV, W = %r, %r, %r, %r
JOBS = %r
LR_AUTO, ALLOW_EDGE, PILOT_SEED, SHORT = %r, %r, %d, %r
IN, WORK = '/kaggle/input', '/kaggle/working'

import base64, glob, json, os, shutil, subprocess, sys, threading, time, zlib


def run_gpu(gpu, arg_lists):
    out = os.path.join(WORK, 'gpu%%d' %% gpu)
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OUT_DIR=out, PYTHONUNBUFFERED='1',
               PYTORCH_ALLOC_CONF='expandable_segments:True', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    for args in arg_lists:
        with open(os.path.join(out, 'stdout.txt'), 'a') as log:
            rc = subprocess.call([sys.executable, '-u', 'v200_ifm_trunc.py', '--variant', VARIANT, '--weights', W] + args,
                                 stdout=log, stderr=subprocess.STDOUT, env=env)
        print('gpu%%d %%s exit %%d after %%ds' %% (gpu, args[-1], rc, time.time() - T0), flush=True)


def lr_from_pilot():
    """Per arm, the LR with the lowest val loss in the LR pilot (seed PILOT_SEED, valid runs); reuse its controls."""
    rows = []
    for f in glob.glob(os.path.join(IN, '**', 'v200_runs_*.jsonl'), recursive=True):
        rows += [json.loads(l) for l in open(f)]
    rows = [r for r in rows if r['variant'] == VARIANT and r['seed'] == PILOT_SEED and not r['diverged']
            and r['nonfinite_steps'] <= 10]
    pick, edge = {}, []
    for k in (5, 0):
        rs = sorted((r for r in rows if r['k'] == k), key=lambda r: r['lr'])
        if len(rs) < 2:
            raise SystemExit('LR pilot output missing for k=%%d' %% k)
        best = min(rs, key=lambda r: r['val'])
        pick[k] = best
        if best['lr'] in (rs[0]['lr'], rs[-1]['lr']):
            edge.append(k)
    choice = dict(lr_trunc=pick[5]['lr'], lr_floor=pick[0]['lr'], edge=edge,
                  table={str(k): [(r['lr'], r['val'], r['test']) for r in sorted(rows, key=lambda r: r['lr']) if r['k'] == k] for k in (5, 0)})
    json.dump(choice, open(os.path.join(WORK, 'v200_lr_choice.json'), 'w'), indent=1)
    print('LR choice from the pilot:', choice, flush=True)
    if edge and not ALLOW_EDGE:
        raise SystemExit('best LR at the edge of the grid for k=%%s: extend the grid before the spread pilot' %% edge)
    os.makedirs(os.path.join(WORK, 'reused'), exist_ok=True)   # the pilot's seed-PILOT_SEED controls at these LRs
    with open(os.path.join(WORK, 'reused', 'v200_runs_%%s_reused.jsonl' %% SHORT), 'w') as f:
        for k in (5, 0):
            f.write(json.dumps(pick[k]) + '\n')
    return choice['lr_floor'], choice['lr_trunc']


if __name__ == '__main__':
    T0 = time.time()
    os.chdir(WORK)
    for fn, e in FILES.items():
        open(fn, 'wb').write(zlib.decompress(base64.b64decode(e)))
    jobs = [list(j) for j in JOBS]
    if LR_AUTO:
        lf, lt = lr_from_pilot()
        jobs = [[x.replace(':-1:', ':%%g:' %% lt).replace(':-2:', ':%%g:' %% lf) for x in j] for j in jobs]
    subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'datasets', 'huggingface_hub', 'safetensors', 'tokenizers'])
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=REPO, revision=REV, local_dir=W)
    for root, _, fs in os.walk(W):
        for f in fs:
            p = os.path.join(root, f)
            if os.path.islink(p):
                t = os.path.realpath(p); os.unlink(p); shutil.copyfile(t, p)
    subprocess.call([sys.executable, '-c', "from datasets import load_dataset; [load_dataset('openai/gsm8k', 'main', split=s) for s in ('train', 'test')]"])
    print('setup done after %%ds; jobs %%s' %% (time.time() - T0, jobs), flush=True)
    import torch
    lanes = [[j] for j in jobs] if torch.cuda.device_count() >= 2 else [jobs]     # one GPU: both job lists in turn
    th = []
    for gpu, arg_lists in enumerate(lanes):
        t = threading.Thread(target=run_gpu, args=(gpu, arg_lists)); t.start(); th.append(t); time.sleep(30)
    for t in th:
        t.join()
    for gpu in range(len(lanes)):
        print('---- gpu%%d (tail)\n%%s' %% (gpu, open(os.path.join(WORK, 'gpu%%d' %% gpu, 'stdout.txt')).read()[-5000:]), flush=True)
    with open('v200_analysis.txt', 'w') as f:
        subprocess.call([sys.executable, 'v200_ifm_trunc.py', '--analyze', WORK], stdout=f, stderr=subprocess.STDOUT)
    print(open('v200_analysis.txt').read(), flush=True)
    print('total %%ds' %% (time.time() - T0), flush=True)
'''

W = '/tmp/ifm_weights'
total = 0.0
for i, (slug, acct, jobs, lr_auto, sources) in enumerate(KERNELS):
    dd = os.path.join(HERE, 'kaggle-kernel-v200-%s' % slug.replace('astrocyte-v200-', ''))
    meta = dict(id='%s/%s' % (acct, slug), title=slug, code_file='run.py', language='python', kernel_type='script',
                is_private=True, enable_gpu=True, enable_internet=True)
    if sources:
        meta['kernel_sources'] = sources
    if not A.dry_run:
        os.makedirs(dd, exist_ok=True)
        open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write(
            RUN % (emb, A.variant, 'IFM/LoopedLM-P2-' + A.variant, VARIANTS[A.variant], W, jobs, lr_auto, A.allow_edge,
                   PILOT_SEED, SHORT))
        json.dump(meta, open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
    e = est[2 * i:2 * i + 2] if A.mode == 'study' else est
    total += sum(e)
    print('%s %s -> %s/%s | GPU est %s min + setup ~%d min%s | not pushed' % (
        'would build' if A.dry_run else 'built', dd, acct, slug, [round(x / 60) for x in e], SETUP_S // 60,
        ' | LRs from %s' % sources[0] if sources else ''))
    for g, j in enumerate(jobs):
        print('    gpu%d: %s' % (g, ' '.join(j)))
print('total %.1f T4-hours of training (%d kernel(s))' % (total / 3600, len(KERNELS)))
