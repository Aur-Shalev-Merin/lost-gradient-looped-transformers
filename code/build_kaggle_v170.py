"""Build Kaggle kernels for v170 (SGD + momentum arm of truncated loop training). Build only; does not push.
  python build_kaggle_v170.py pilot    PRE-FREEZE: SGD LR sweep of the floor (k 0) and full (k R) controls, 1 seed
Parcae kernel (aurshalevmerin): frozen v164/v163/v162 drivers via v170_sgd.py, Parcae source pinned.
Retro kernel (aurmerin): frozen v167 driver via v170_sgd.py, dataset retro-smollm2-healed, transformers 4.51.0.
"""
import base64, hashlib, json, os, sys, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
PARCAE_COMMIT = '69284c13746e849104f738d6d1a347b1f457df76'
A1, A2 = 'aurshalevmerin', 'aurmerin'
PFILES = ['v170_sgd.py', 'v164_parcae_predict.py', 'v163_parcae_damped_trunc.py', 'v162_parcae_trunc_train.py']
RFILES = ['v170_sgd.py', 'v167_retrofit_trunc.py']
LRS = ['0.01', '0.03', '0.1', '0.3', '1.0']

mode = sys.argv[1] if len(sys.argv) > 1 else 'pilot'
if mode == 'pilot':
    p = lambda k, lr: '32:1.0:%d:%s:9001' % (k, lr)
    r = lambda k, lr: '1.0:%d:%s:9001' % (k, lr)
    KERNELS = [('pilot-parcae', A1, PFILES, 'parcae',
                [['--selftest', '--train', ','.join([p(32, x) for x in LRS[:3]] + [p(0, x) for x in LRS[:2]]), '--tag', 'pilot0'],
                 ['--train', ','.join([p(32, x) for x in LRS[3:]] + [p(0, x) for x in LRS[2:]]), '--tag', 'pilot1']]),
               ('pilot-retro', A2, RFILES, 'retro',
                [['--selftest', '--train', ','.join(r(8, x) for x in LRS), '--tag', 'pilot0'],
                 ['--train', ','.join(r(0, x) for x in LRS), '--tag', 'pilot1']])]
elif mode == 'eval-parcae':   # v170 Parcae arm (prereg docs/prereg_v170_sgd_vs_adam.md): R 32 a 1, k 0/1/2/4/32, 7 seeds
    est = lambda k: 500 + 36 * k
    runs = [(k, '0.3' if k == 0 else '0.1', s) for s in range(1701101, 1701108) for k in (0, 32, 1, 2, 4)]
    print('  parcae: %d runs, est %.1f T4-h' % (len(runs), sum(est(r[0]) for r in runs) / 3600))
    bins = [[0, []] for _ in range(4)]
    for r in sorted(runs, key=lambda r: -est(r[0])):
        b = min(bins, key=lambda b: b[0]); b[0] += est(r[0]); b[1].append(r)
    KERNELS = []
    for i, part in enumerate(('a', 'b')):
        jobs = [['--selftest', '--train', ','.join('32:1.0:%d:%s:%d' % r for r in bins[2 * i + g][1]), '--tag', 'eval%s%d' % (part, g)] for g in (0, 1)]
        print('  eval-parcae-%s: est %s min' % (part, [round(bins[2 * i + g][0] / 60) for g in (0, 1)]))
        KERNELS.append(('eval-parcae-%s' % part, A1, PFILES, 'parcae', jobs))
elif mode == 'pilot2-retro':  # second retrofit SGD LR pilot (controls only)
    r = lambda a, k, lr: '%s:%d:%s:9001' % (a, k, lr)
    KERNELS = [('pilot2-retro', A2, RFILES, 'retro',
                [['--train', r('1.0', 8, '0.02') + ',' + r('1.0', 8, '0.05'), '--tag', 'p2a'],
                 ['--train', r('0.3', 0, '0.3') + ',' + r('0.3', 8, '0.03') + ',' + r('0.3', 8, '0.05'), '--tag', 'p2b']])]
elif mode == 'eval-retro':   # v170 retrofit arm: a 1 / 0.3, k 0/1/2/8; LRs by the locked rule (pilot 1+2): floor 0.3, full/trunc 0.03
    est = lambda k: 340 + 72 * k
    runs = [(a, k, '0.3' if k == 0 else '0.03', s) for s in range(1701101, 1701108) for a in ('1.0', '0.3') for k in (0, 8, 1, 2)]
    print('  retro: %d runs, est %.1f T4-h' % (len(runs), sum(est(r[1]) for r in runs) / 3600))
    bins = [[0, []] for _ in range(4)]
    for r in sorted(runs, key=lambda r: -est(r[1])):
        b = min(bins, key=lambda b: b[0]); b[0] += est(r[1]); b[1].append(r)
    KERNELS = []
    for i, part in enumerate(('c', 'd')):
        jobs = [['--selftest', '--train', ','.join('%s:%d:%s:%d' % r for r in bins[2 * i + g][1]), '--tag', 'eval%s%d' % (part, g)] for g in (0, 1)]
        print('  eval-retro-%s: est %s min' % (part, [round(bins[2 * i + g][0] / 60) for g in (0, 1)]))
        KERNELS.append(('eval-retro-%s' % part, A2, RFILES, 'retro', jobs))
elif mode == 'sweep-retro':  # v170 follow-up: SGD LR sweep of the TRUNCATED retrofit cells (a 1 / 0.3, k 1 / 2) at LR < 0.03, 2 seeds
    runs = [(a, k, lr, s) for s in (1701101, 1701102) for a in ('1.0', '0.3') for k in (1, 2) for lr in ('0.003', '0.01', '0.02')]
    est = lambda k: 340 + 72 * k
    bins = [[0, []] for _ in range(4)]
    for r in sorted(runs, key=lambda r: -est(r[1])):
        b = min(bins, key=lambda b: b[0]); b[0] += est(r[1]); b[1].append(r)
    KERNELS = []
    for i, (part, acct) in enumerate((('e', A1), ('f', A1))):   # aurmerin hit its weekly GPU quota 2026-10-05
        jobs = [['--train', ','.join('%s:%d:%s:%d' % r for r in bins[2 * i + g][1]), '--tag', 'sweep%s%d' % (part, g)] for g in (0, 1)]
        print('  sweep-retro-%s: est %s min' % (part, [round(bins[2 * i + g][0] / 60) for g in (0, 1)]))
        KERNELS.append(('sweep-retro-%s' % part, acct, RFILES, 'retro', jobs))
else:
    raise SystemExit('unknown mode')

body = r'''import subprocess, sys, os, base64, zlib, time, glob
if __name__ == '__main__':
    os.chdir('/kaggle/working')
    for fn, e in FILES.items():
        open(fn, 'wb').write(zlib.decompress(base64.b64decode(e)))
    env0 = dict(os.environ, PYTHONUNBUFFERED='1')
    if WHICH == 'parcae':
        subprocess.call(['git', 'clone', '-q', 'https://github.com/sandyresearch/parcae', '/tmp/parcae_src'])
        subprocess.call(['git', '-C', '/tmp/parcae_src', 'checkout', '-q', PARCAE_COMMIT])
        subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'datasets'])
        env0['PARCAE_SRC'] = '/tmp/parcae_src'
    else:
        subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.51.0', 'datasets'])
        cfg = [q for q in glob.glob('/kaggle/input/**/config.json', recursive=True) if os.path.exists(os.path.join(os.path.dirname(q), 'raven_modeling_minimal.py'))]
        assert len(cfg) == 1, cfg
        env0['RETRO_DIR'] = os.path.dirname(cfg[0])
    t0 = time.time(); procs = []
    for gpu, args in enumerate(JOBS):
        od = '/kaggle/working/gpu%d' % gpu; os.makedirs(od, exist_ok=True)
        env = dict(env0, CUDA_VISIBLE_DEVICES=str(gpu), OUT_DIR=od)
        out = open('v170_stdout_gpu%d.txt' % gpu, 'w')
        procs.append((subprocess.Popen([sys.executable, '-u', 'v170_sgd.py', WHICH] + args, stdout=out, stderr=subprocess.STDOUT, env=env), out))
        time.sleep(60)
    for q, out in procs:
        rc = q.wait(); out.close(); print('job exit', rc, 'elapsed %ds' % (time.time() - t0), flush=True)
    for gpu in range(len(JOBS)):
        print(open('v170_stdout_gpu%d.txt' % gpu).read()[-6000:], flush=True)
    subprocess.call('rm -rf /kaggle/working/gpu*/retro_model', shell=True)
'''

for name, acct, files, which, jobs in KERNELS:
    emb = {}
    for f in files:
        b = open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n')
        emb[f] = base64.b64encode(zlib.compress(b, 9)).decode()
        print('  %s sha256 (LF) %s' % (f, hashlib.sha256(b).hexdigest()[:12]))
    dd = os.path.join(HERE, 'kaggle-kernel-v170-%s' % name); os.makedirs(dd, exist_ok=True)
    src = 'FILES = %r\nPARCAE_COMMIT = %r\nJOBS = %r\nWHICH = %r\n' % (emb, PARCAE_COMMIT, jobs, which) + body
    open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write(src)
    meta = dict(id='%s/astrocyte-v170-%s' % (acct, name), title='astrocyte-v170-%s' % name, code_file='run.py', language='python',
                kernel_type='script', is_private=True, enable_gpu=True, enable_internet=True)
    if which == 'retro':
        meta['dataset_sources'] = ['%s/retro-smollm2-healed' % acct]
    json.dump(meta, open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
    print('built', dd, acct, [len(j) for j in jobs])
