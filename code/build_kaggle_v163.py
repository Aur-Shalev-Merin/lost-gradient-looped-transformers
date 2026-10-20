"""Build Kaggle GPU kernels for v163 (damped-loop truncation on Parcae-140m). Prereg:
docs/prereg_v163_parcae_damped_truncation.md. Build only; this script does not push.

  python build_kaggle_v163.py pilot    PRE-FREEZE (controls + frozen model only): selftest, measurement, LR sweep of
                                       k = 0 and k = 32 at a in {1.0, 0.5}, pilot seed 9001. 2 kernels x 2 T4.
  python build_kaggle_v163.py tune     all 12 cells x LR grid x tuning seed, packed longest-first on 8 GPUs
  python build_kaggle_v163.py ext      edge-rule extension (v163_selection.json 'extend')
  python build_kaggle_v163.py eval     all cells at the selected LRs x 4 evaluation seeds
Each kernel embeds v163_parcae_damped_trunc.py and v162_parcae_trunc_train.py and clones Parcae at the pinned commit.
"""
import base64, hashlib, json, os, sys, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ['v163_parcae_damped_trunc.py', 'v162_parcae_trunc_train.py']
PARCAE_COMMIT = '69284c13746e849104f738d6d1a347b1f457df76'
A1, A2 = 'aurshalevmerin', 'aurmerin'
ALPHAS = ['1.0', '0.5']
KS = [0, 1, 2, 4, 8, 32]
GRID = ['1e-4', '3e-4', '9e-4']
TUNE = [1631001]
EVAL = [1631101, 1631102, 1631103, 1631104]
PILOT_SEED = 9001


def est(a, k):        # T4 seconds per run at R = 32, from the pilot: k 0 ~ 500 s, k 32 ~ 1,600 s (+ ~30 s end-of-run rho)
    return 530 + 34 * k


def pack(name, runs, parts):
    bins = [[0, []] for _ in range(2 * len(parts))]
    for r in sorted(runs, key=lambda r: -est(r[0], r[1])):
        b = min(bins, key=lambda b: b[0]); b[0] += est(r[0], r[1]); b[1].append(r)
    ks = []
    for i, (part, acct) in enumerate(parts):
        jobs = [['--train', ','.join('%s:%d:%s:%d' % r for r in bins[2 * i + g][1]), '--tag', '%s%s%d' % (name, part, g)]
                for g in (0, 1) if bins[2 * i + g][1]]
        if jobs:
            print('  %s-%s: est %s min' % (name, part, [round(bins[2 * i + g][0] / 60) for g in (0, 1)]))
            ks.append(('%s-%s' % (name, part), acct, jobs))
    return ks


mode = sys.argv[1] if len(sys.argv) > 1 else 'pilot'
PARTS4 = [('a', A1), ('b', A1), ('c', A2), ('d', A2)]
if mode == 'pilot':
    c = lambda a, k: ','.join('%s:%d:%s:%d' % (a, k, lr, PILOT_SEED) for lr in GRID)
    KERNELS = [('pilot-a', A1, [['--train', c('1.0', 32), '--tag', 'pilotA0'], ['--train', c('0.5', 32), '--tag', 'pilotA1']]),
               ('pilot-b', A2, [['--selftest', '--measure', '--train', c('1.0', 0), '--tag', 'pilotB0'],
                                ['--train', c('0.5', 0), '--tag', 'pilotB1']])]
elif mode == 'tune':
    KERNELS = pack('tune', [(a, k, lr, s) for a in ALPHAS for k in KS for lr in GRID for s in TUNE], PARTS4)
elif mode in ('ext', 'eval', 'evalfinal', 'evalrest'):
    # 'evalfinal' = eval runs of cells whose LR is final (not in 'extend'), run alongside 'ext'; 'evalrest' = eval runs
    # of the extended cells after re-selection. Together = 'eval' (same runs; logistics only, as in v161/v162).
    sel = json.load(open(os.path.join(HERE, 'v163_selection.json')))
    ext_cells = {'%.1f:%d' % (a, k) for a, k, lr in sel['extend']}
    allev = [(c_.split(':')[0], int(c_.split(':')[1]), '%g' % lr, s) for c_, lr in sel['selection'].items() for s in EVAL]
    if mode == 'ext':
        KERNELS = pack('ext', [('%.1f' % a, k, '%g' % lr, s) for a, k, lr in sel['extend'] for s in TUNE], [('e', A1)])
    elif mode == 'evalfinal':
        KERNELS = pack('eval', [r for r in allev if '%s:%d' % r[:2] not in ext_cells], [('b', A1), ('c', A2), ('d', A2)])
    elif mode == 'evalrest':
        assert not sel['extend'], 're-select after the extension first'
        KERNELS = pack('evalrest', [r for r in allev if r[1] == 0], [('e', A1)])
    else:
        assert not sel['extend'], 'run the extension and re-select first'
        runs = [(c_.split(':')[0], int(c_.split(':')[1]), '%g' % lr, s) for c_, lr in sel['selection'].items() for s in EVAL]
        KERNELS = pack('eval', runs, PARTS4)
else:
    raise SystemExit('unknown mode')

emb = {}
for f in FILES:
    b = open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n')
    emb[f] = base64.b64encode(zlib.compress(b, 9)).decode()
    print('  %s sha256 (LF) %s' % (f, hashlib.sha256(b).hexdigest()[:12]))

body = r'''import subprocess, sys, os, base64, zlib, time
if __name__ == '__main__':
    os.chdir('/kaggle/working')
    for fn, e in FILES.items():
        open(fn, 'wb').write(zlib.decompress(base64.b64decode(e)))
    subprocess.call(['git', 'clone', '-q', 'https://github.com/sandyresearch/parcae', '/tmp/parcae_src'])
    subprocess.call(['git', '-C', '/tmp/parcae_src', 'checkout', '-q', PARCAE_COMMIT])
    subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'datasets'])
    t0 = time.time(); procs = []
    for gpu, args in enumerate(JOBS):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PARCAE_SRC='/tmp/parcae_src', PYTHONUNBUFFERED='1')
        out = open('v163_stdout_gpu%d.txt' % gpu, 'w')
        procs.append((subprocess.Popen([sys.executable, '-u', 'v163_parcae_damped_trunc.py'] + args, stdout=out,
                                       stderr=subprocess.STDOUT, env=env), out))
        time.sleep(60)
    for p, out in procs:
        rc = p.wait(); out.close(); print('job exit', rc, 'elapsed %ds' % (time.time() - t0), flush=True)
    for gpu in range(len(JOBS)):
        print(open('v163_stdout_gpu%d.txt' % gpu).read()[-6000:], flush=True)
'''

for name, acct, jobs in KERNELS:
    d = os.path.join(HERE, 'kaggle-kernel-v163-%s' % name); os.makedirs(d, exist_ok=True)
    src = 'FILES = %r\nPARCAE_COMMIT = %r\nJOBS = %r\n' % (emb, PARCAE_COMMIT, jobs) + body
    open(os.path.join(d, 'run.py'), 'w', encoding='utf-8', newline='\n').write(src)
    json.dump(dict(id='%s/astrocyte-v163-%s' % (acct, name), title='astrocyte-v163-%s' % name, code_file='run.py',
                   language='python', kernel_type='script', is_private=True, enable_gpu=True, enable_internet=True),
              open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=1)
    print('built', d, acct, [len(j) for j in jobs])
