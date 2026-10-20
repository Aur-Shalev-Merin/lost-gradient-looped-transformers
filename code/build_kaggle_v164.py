"""Build Kaggle kernels for v164 (prereg docs/prereg_v164_parcae_prediction.md). Build only; does not push.
  python build_kaggle_v164.py measure    PRE-FREEZE: selftest at R 64 + frozen-model err for the test variants (1 kernel)
  python build_kaggle_v164.py eval       all cells x 5 evaluation seeds, fixed LRs, packed on 4 kernels x 2 T4
Embeds v164_parcae_predict.py + v163_parcae_damped_trunc.py + v162_parcae_trunc_train.py; clones Parcae (pinned).
"""
import base64, hashlib, json, os, sys, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ['v164_parcae_predict.py', 'v166_prefreeze.py', 'v163_parcae_damped_trunc.py', 'v162_parcae_trunc_train.py']
PARCAE_COMMIT = '69284c13746e849104f738d6d1a347b1f457df76'
A1, A2 = 'aurshalevmerin', 'aurmerin'
VARIANTS = [(32, '1.0'), (32, '0.7'), (64, '0.3')]
TEST_K = {(32, '1.0'): [3, 6], (32, '0.7'): [2, 3, 4, 6], (64, '0.3'): [2, 3, 4, 6]}
EVAL = [1641101, 1641102, 1641103, 1641104, 1641105]
LR_TRUNC, LR_FLOOR = '3e-4', '9e-4'


def est(R, k):     # T4 seconds: R 32 from the v163 pilot (530 + 34 k); R 64 forward cost ~doubles
    return 530 * R // 32 + 34 * k


def pack(name, runs, parts):
    bins = [[0, []] for _ in range(2 * len(parts))]
    for r in sorted(runs, key=lambda r: -est(r[0], r[2])):
        b = min(bins, key=lambda b: b[0]); b[0] += est(r[0], r[2]); b[1].append(r)
    ks = []
    for i, (part, acct) in enumerate(parts):
        jobs = [['--train', ','.join('%d:%s:%d:%s:%d' % r for r in bins[2 * i + g][1]), '--tag', '%s%s%d' % (name, part, g)]
                for g in (0, 1) if bins[2 * i + g][1]]
        print('  %s-%s: est %s min' % (name, part, [round(bins[2 * i + g][0] / 60) for g in (0, 1)]))
        ks.append(('%s-%s' % (name, part), acct, jobs))
    return ks


mode = sys.argv[1] if len(sys.argv) > 1 else 'measure'
if mode == 'measure':
    KERNELS = [('measure', A1, [['--selftest', '--measure', '--tag', 'measure']])]
elif mode == 'eval':
    runs = []
    for R, a in VARIANTS:
        for s in EVAL:
            runs.append((R, a, 0, LR_FLOOR, s)); runs.append((R, a, R, LR_TRUNC, s))
            runs += [(R, a, k, LR_TRUNC, s) for k in TEST_K[(R, a)]]
    print('  %d runs, est %.1f T4-h' % (len(runs), sum(est(r[0], r[2]) for r in runs) / 3600))
    # 3 kernels (A1 has one slot held by the measurement kernel); every job runs the R-64 selftest before training
    KERNELS = pack('eval', runs, [('a', A1), ('c', A2), ('d', A2)])
    KERNELS = [(n, acct, [['--selftest'] + j for j in jobs]) for n, acct, jobs in KERNELS]
elif mode == 'v165':   # v165 (prereg docs/prereg_v165_undamped_rerun.md): undamped cells, fresh seeds, 2 kernels
    seeds = list(range(1651101, 1651108))
    runs = []
    for s in seeds:
        runs += [(32, '1.0', 0, LR_FLOOR, s), (32, '1.0', 32, LR_TRUNC, s), (32, '1.0', 3, LR_TRUNC, s), (32, '1.0', 6, LR_TRUNC, s)]
    print('  v165: %d runs, est %.1f T4-h' % (len(runs), sum(est(r[0], r[2]) for r in runs) / 3600))
    KERNELS = pack('rerun', runs, [('a', A1), ('c', A2)])
    KERNELS = [(n, acct, [['--selftest'] + j for j in jobs]) for n, acct, jobs in KERNELS]
elif mode == 'v166pre':   # v166 pre-freeze: block-A measurement + R 48 a 0.4 control LR checks (seed 9001)
    c = lambda k, lr: '48:0.4:%d:%s:9001' % (k, lr)
    KERNELS = [('pre-a', A1, [['--measure', '--train', ','.join(c(0, lr) for lr in ('3e-4', '9e-4', '2.7e-3')), '--tag', 'pre0'],
                              ['--train', c(48, '1e-4') + ',' + c(48, '3e-4'), '--tag', 'full1']]),
               ('pre-c', A2, [['--train', c(48, '9e-4'), '--tag', 'full2']])]
elif mode == 'v166':      # v166 (prereg docs/prereg_v166_slam_dunk.md): block A, 4 scored cells + controls, 7 seeds
    seeds = list(range(1661101, 1661108)); TK = {(64, '0.3'): [8, 12], (48, '0.4'): [4, 8]}
    runs = []
    for s in seeds:
        for (R, a), ks in TK.items():
            runs += [(R, a, 0, LR_FLOOR, s), (R, a, R, LR_TRUNC, s)] + [(R, a, k, LR_TRUNC, s) for k in ks]
    print('  v166: %d runs, est %.1f T4-h' % (len(runs), sum(est(r[0], r[2]) for r in runs) / 3600))
    KERNELS = pack('eval', runs, [('a', A1), ('b', A1), ('c', A2), ('d', A2)])
    KERNELS = [(n, acct, [['--selftest'] + j for j in jobs]) for n, acct, jobs in KERNELS]
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
        out = open('v164_stdout_gpu%d.txt' % gpu, 'w')
        procs.append((subprocess.Popen([sys.executable, '-u', SCRIPT] + args, stdout=out,
                                       stderr=subprocess.STDOUT, env=env), out))
        time.sleep(60)
    for p, out in procs:
        rc = p.wait(); out.close(); print('job exit', rc, 'elapsed %ds' % (time.time() - t0), flush=True)
    for gpu in range(len(JOBS)):
        print(open('v164_stdout_gpu%d.txt' % gpu).read()[-6000:], flush=True)
'''

for name, acct, jobs in KERNELS:
    pre = {'v165': 'v165', 'v166pre': 'v166', 'v166': 'v166'}.get(mode, 'v164')
    dd = os.path.join(HERE, 'kaggle-kernel-%s-%s' % (pre, name)); os.makedirs(dd, exist_ok=True)
    script = 'v166_prefreeze.py' if mode == 'v166pre' else 'v164_parcae_predict.py'   # v166 eval: frozen v164 driver
    src = 'FILES = %r\nPARCAE_COMMIT = %r\nJOBS = %r\nSCRIPT = %r\n' % (emb, PARCAE_COMMIT, jobs, script) + body
    open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write(src)
    json.dump(dict(id='%s/astrocyte-%s-%s' % (acct, pre, name), title='astrocyte-%s-%s' % (pre, name), code_file='run.py',
                   language='python', kernel_type='script', is_private=True, enable_gpu=True, enable_internet=True),
              open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
    print('built', dd, acct, [len(j) for j in jobs])
