"""Build the v174 PILOT Kaggle kernels (prereg draft 3.1, §2; non-scored seed, agreement only). Build only; does not push.
  python build_kaggle_v174.py ACCOUNT
Kernel pilot-a: GPU 0 selftest + frozen gate + k=1 lam 0.25/0.5; GPU 1 k=1 lam 0.75/1.0.
Kernel pilot-b: GPU 0 k=2 lam 0.25/0.5; GPU 1 k=2 lam 0.75/1.0.
"""
import base64, hashlib, json, os, sys, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ACCT = sys.argv[1] if len(sys.argv) > 1 else 'aurmerin'
MODE = sys.argv[2] if len(sys.argv) > 2 else 'pilot'
PARCAE_COMMIT = '69284c13746e849104f738d6d1a347b1f457df76'
FILES = ['v174_mix_driver.py', 'v163_parcae_damped_trunc.py', 'v162_parcae_trunc_train.py']
KERNELS = [('pilot-a', [['--selftest', '--gate', '--pilot', '1:0.25,1:0.5', '--tag', 'a0'], ['--pilot', '1:0.75,1:1.0', '--tag', 'a1']]),
           ('pilot-b', [['--pilot', '2:0.25,2:0.5', '--tag', 'b0'], ['--pilot', '2:0.75,2:1.0', '--tag', 'b1']])]
if MODE == 'gate4':      # pre-registered fallback: frozen gate for k = 1 vs k = 4 before any k = 4 training arm
    KERNELS = [('gate4', [['--gate', '--gate-grid', '1:0.5,1:0.75,1:1.0,4:0.5,4:0.75,4:1.0,4:1.25,4:1.5', '--tag', 'k4']])]
if MODE == 'pilot4':     # fallback pilot: k = 4 arms (lam > 1 allowed) + k = 1 interpolation points; agreement only
    KERNELS = [('pilot4-a', [['--pilot', '4:0.75,4:1.0', '--tag', 'p4a0'], ['--pilot', '4:1.25,4:1.5', '--tag', 'p4a1']]),
               ('pilot4-b', [['--pilot', '4:2.0,4:2.25', '--tag', 'p4b0'], ['--pilot', '1:0.625,1:0.875', '--tag', 'p4b1']])]
if MODE == 'remeasure':  # pre-freeze: re-measure the interpolated k = 4 arms of P1-P4 and re-gate them (agreement only)
    G = '1:0.5,1:0.625,1:0.875,1:1.0,4:0.94,4:1.17,4:1.75,4:2.03'
    KERNELS = [('remeas-a', [['--gate', '--gate-grid', G, '--pilot', '4:0.94', '--tag', 'rmA0'], ['--pilot', '4:1.17', '--tag', 'rmA1']]),
               ('remeas-b', [['--pilot', '4:1.75', '--tag', 'rmB0'], ['--pilot', '4:2.03', '--tag', 'rmB1']])]
if MODE == 'remeasure2':  # pre-freeze: second matching iteration for P3 (k4 lam 1.857) and P4' (k1 lam 0.924 vs k4 2.03)
    KERNELS = [('remeas2', [['--pilot', '4:1.857', '--tag', 'rm2a'], ['--pilot', '1:0.924', '--tag', 'rm2b']])]
if MODE == 'remeasure3':  # pre-freeze: P3 third iteration, k1 side moved (k1 lam 0.82 vs k4 lam 1.75)
    KERNELS = [('remeas3', [['--pilot', '1:0.82', '--tag', 'rm3']])]
if MODE == 'remeasure3b':  # as remeasure3 on GPU 0, plus a bracketing k1 lam 0.78 on GPU 1 (both T4s used)
    KERNELS = [('remeas3', [['--pilot', '1:0.82', '--tag', 'rm3'], ['--pilot', '1:0.78', '--tag', 'rm3b']])]
if MODE == 'p5':  # pre-freeze: re-gate P3 at k1 0.82 + backup high-dose pair P5 (k1 0.875, Abar 0.932, vs k4 ~1.93-1.97)
    G = '1:0.82,1:0.875,4:1.75,4:1.93,4:1.97'
    KERNELS = [('p5', [['--gate', '--gate-grid', G, '--pilot', '4:1.93', '--tag', 'p5a'], ['--pilot', '4:1.97', '--tag', 'p5b']])]
if MODE == 'p5b':  # pre-freeze: P5 re-match, k1 side moved to match k4 lam 1.97 (Abar 0.9223); 0.90 brackets
    KERNELS = [('p5b', [['--pilot', '1:0.91', '--tag', 'p5c'], ['--pilot', '1:0.90', '--tag', 'p5d']])]
if MODE == 'gate5':  # pre-freeze: frozen gate for the final P5 (k1 0.91 vs k4 1.97); no training
    KERNELS = [('gate5', [['--gate', '--gate-grid', '1:0.91,4:1.97', '--tag', 'g5']])]
if MODE == 'scored':  # FROZEN scored study: 12 arms x 9 seeds (8 + backup 1741109) + one FULL pipeline check, 4 kernels x 2 GPUs
    PAIR_ARMS = ['1:0.5', '4:0.94', '1:0.625', '4:1.17', '1:0.82', '4:1.75', '1:0.924', '4:2.03', '1:0.91', '4:1.97']
    CELLS = ['full', 'floor'] + PAIR_ARMS
    EST = lambda c: 22 if c in ('full', 'full163') else 20 if c == 'floor' else 42      # minutes on a T4 (pilot timings)
    runs = [(c, sd) for sd in range(1741101, 1741110) for c in CELLS] + [('full163', 1741101)]
    slots = [[] for _ in range(8)]; load = [0] * 8
    for c, sd in sorted(runs, key=lambda r: -EST(r[0])):          # longest-first greedy balance
        i = load.index(min(load)); slots[i].append((c, sd)); load[i] += EST(c)
    order = {c: i for i, c in enumerate(CELLS + ['full163'])}
    for sl in slots:
        sl.sort(key=lambda r: (r[1], order[r[0]]))                  # evaluation seeds first, backup seed last
    ACCTS = ['aurmerin', 'aurmerin', 'aurshalevmerin', 'aurshalevmerin']
    KERNELS = [('scored-%d' % (j + 1), [['--runs', ','.join('%s@%d' % r for r in slots[2 * j + g]), '--tag', 's%d%d' % (j + 1, g)]
                                         for g in range(2)], ACCTS[j]) for j in range(4)]
    print('  per-GPU estimated minutes', load, 'total runs', len(runs))
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
        od = '/kaggle/working/gpu%d' % gpu; os.makedirs(od, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OUT_DIR=od, PARCAE_SRC='/tmp/parcae_src', PYTHONUNBUFFERED='1')
        out = open('v174_stdout_gpu%d.txt' % gpu, 'w')
        procs.append((subprocess.Popen([sys.executable, '-u', 'v174_mix_driver.py'] + args, stdout=out, stderr=subprocess.STDOUT, env=env), out))
        time.sleep(60)
    for p, out in procs:
        rc = p.wait(); out.close(); print('job exit', rc, 'elapsed %ds' % (time.time() - t0), flush=True)
    for gpu in range(len(JOBS)):
        print(open('v174_stdout_gpu%d.txt' % gpu).read()[-6000:], flush=True)
'''
for kern in KERNELS:
    name, jobs = kern[0], kern[1]; acct = kern[2] if len(kern) > 2 else ACCT
    dd = os.path.join(HERE, 'kaggle-kernel-v174-%s' % name); os.makedirs(dd, exist_ok=True)
    open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write(
        'FILES = %r\nPARCAE_COMMIT = %r\nJOBS = %r\n' % (emb, PARCAE_COMMIT, jobs) + body)
    json.dump(dict(id='%s/astrocyte-v174-%s' % (acct, name), title='astrocyte-v174-%s' % name, code_file='run.py',
                   language='python', kernel_type='script', is_private=True, enable_gpu=True, enable_internet=True),
              open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
    print('built', dd, acct)
