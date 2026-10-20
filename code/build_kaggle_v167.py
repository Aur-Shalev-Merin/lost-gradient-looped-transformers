"""Build Kaggle kernels for v167 (retrofit opposite-direction test). Build only; does not push.
  python build_kaggle_v167.py pilot     PRE-FREEZE: selftest + frozen err/rho + step-0 loss; control LR checks (1 kernel)
Embeds v167_retrofit_trunc.py; the healed model comes from a private Kaggle dataset '<account>/retro-smollm2-healed'
(one copy per account: private datasets don't cross accounts). transformers pinned to 4.51.0 (the retrofit's version).
"""
import base64, hashlib, json, os, sys, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ['v167_retrofit_trunc.py']
A1, A2 = 'aurshalevmerin', 'aurmerin'
DATASET = 'retro-smollm2-healed'
LR_TRUNC, LR_FLOOR = '3e-4', '9e-4'

mode = sys.argv[1] if len(sys.argv) > 1 else 'pilot'
if mode == 'pilot':
    c = lambda a, k, lr: '%s:%d:%s:9001' % (a, k, lr)
    # kernel 1 GPU 0: selftest, frozen measurement (a 1 / 0.5 / 0.3), undamped floor LR check; GPU 1: undamped full LR check
    # kernel 2: damped (a 0.3) floor and full controls at the standard LRs
    KERNELS = [('pilot', A1, [['--selftest', '--measure', '--train', ','.join(c('1.0', 0, lr) for lr in ('3e-4', '9e-4', '2.7e-3')), '--tag', 'floor'],
                              ['--train', ','.join(c('1.0', 8, lr) for lr in ('1e-4', '3e-4', '9e-4')), '--tag', 'full']]),
               ('pilotd', A2, [['--train', c('0.3', 0, '9e-4') + ',' + c('0.3', 8, '3e-4'), '--tag', 'damped']])]
elif mode == 'pilot2':   # pilot 1 put the best floor (3e-4) and full (1e-4) LRs at grid edges: extend downwards
    c = lambda a, k, lr: '%s:%d:%s:9001' % (a, k, lr)
    KERNELS = [('pilot2', A1, [['--train', c('1.0', 0, '1e-4') + ',' + c('1.0', 8, '3e-5'), '--tag', 'lr2a'],
                               ['--train', c('0.3', 0, '3e-4') + ',' + c('0.3', 0, '1e-4') + ',' + c('0.3', 8, '1e-4'), '--tag', 'lr2b']])]
elif mode == 'eval':     # v167 (prereg docs/prereg_v167_retrofit_transfer.md): 3 variants x (floor, full, k 1, k 2) x 7 seeds
    LRF, LRT = '3e-4', '1e-4'      # pilot 1 + 2: floor best 3e-4, full best 1e-4 (both variants, val and test agree)
    est = lambda k: 350 + 68 * k   # T4 seconds (pilot: floor 350, full k 8 ~890)
    runs = [(a, k, LRF if k == 0 else LRT, s) for s in range(1671101, 1671108) for a in ('1.0', '0.5', '0.3') for k in (0, 8, 1, 2)]
    print('  v167: %d runs, est %.1f T4-h' % (len(runs), sum(est(r[1]) for r in runs) / 3600))
    bins = [[0, []] for _ in range(8)]
    for r in sorted(runs, key=lambda r: -est(r[1])):
        b = min(bins, key=lambda b: b[0]); b[0] += est(r[1]); b[1].append(r)
    KERNELS = []
    for i, (part, acct) in enumerate([('a', A1), ('b', A1), ('c', A2), ('d', A2)]):
        jobs = [['--selftest', '--train', ','.join('%s:%d:%s:%d' % r for r in bins[2 * i + g][1]), '--tag', 'eval%s%d' % (part, g)] for g in (0, 1)]
        print('  eval-%s: est %s min' % (part, [round(bins[2 * i + g][0] / 60) for g in (0, 1)]))
        KERNELS.append(('eval-%s' % part, acct, jobs))
else:
    raise SystemExit('unknown mode')

emb = {}
for f in FILES:
    b = open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n')
    emb[f] = base64.b64encode(zlib.compress(b, 9)).decode()
    print('  %s sha256 (LF) %s' % (f, hashlib.sha256(b).hexdigest()[:12]))

body = r'''import subprocess, sys, os, base64, zlib, time, glob
if __name__ == '__main__':
    os.chdir('/kaggle/working')
    for fn, e in FILES.items():
        open(fn, 'wb').write(zlib.decompress(base64.b64decode(e)))
    subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.51.0', 'datasets'])
    cfg = [p for p in glob.glob('/kaggle/input/**/config.json', recursive=True) if os.path.exists(os.path.join(os.path.dirname(p), 'raven_modeling_minimal.py'))]
    assert len(cfg) == 1, cfg
    retro = os.path.dirname(cfg[0]); print('model dir', retro, os.listdir(retro), flush=True)
    t0 = time.time(); procs = []
    for gpu, args in enumerate(JOBS):
        out_dir = '/kaggle/working/gpu%d' % gpu; os.makedirs(out_dir, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), RETRO_DIR=retro, OUT_DIR=out_dir, PYTHONUNBUFFERED='1')
        out = open('v167_stdout_gpu%d.txt' % gpu, 'w')
        procs.append((subprocess.Popen([sys.executable, '-u', 'v167_retrofit_trunc.py'] + args, stdout=out,
                                       stderr=subprocess.STDOUT, env=env), out))
        time.sleep(60)
    for p, out in procs:
        rc = p.wait(); out.close(); print('job exit', rc, 'elapsed %ds' % (time.time() - t0), flush=True)
    for gpu in range(len(JOBS)):
        print(open('v167_stdout_gpu%d.txt' % gpu).read()[-6000:], flush=True)
    subprocess.call('rm -rf /kaggle/working/gpu*/retro_model', shell=True)     # don't ship model copies as output
'''

for name, acct, jobs in KERNELS:
    dd = os.path.join(HERE, 'kaggle-kernel-v167-%s' % name); os.makedirs(dd, exist_ok=True)
    src = 'FILES = %r\nJOBS = %r\n' % (emb, jobs) + body
    open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write(src)
    json.dump(dict(id='%s/astrocyte-v167-%s' % (acct, name), title='astrocyte-v167-%s' % name, code_file='run.py',
                   language='python', kernel_type='script', is_private=True, enable_gpu=True, enable_internet=True,
                   dataset_sources=['%s/%s' % (acct, DATASET)]),
              open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
    print('built', dd, acct, [len(j) for j in jobs])
