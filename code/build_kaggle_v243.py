"""Build the v243 Kaggle kernels (docs/prereg_v243_sgd_k1_cleanup.md): v241's SGD driver unchanged, k = 1 cells only,
new seeds. Build only; does NOT push. 4 kernels x 2 T4s; every GPU lane runs the v241 selftest on the real model first,
then its share of the 60 runs (12 seeds x 5 cells), balanced longest-first and ordered by seed.
  python build_kaggle_v243.py ACCT_1 ACCT_2 ACCT_3 ACCT_4
Push (NOT done here): kaggle kernels push -p kaggle-kernel-v243-<n> --accelerator NvidiaTeslaT4
"""
import base64, hashlib, json, os, py_compile, sys, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
PARCAE_COMMIT = '69284c13746e849104f738d6d1a347b1f457df76'
sys.path.insert(0, HERE)
import v241_sgd_split as V                                               # noqa: E402
from v243_sgd_k1_cleanup import SEEDS, CELLS                             # noqa: E402
FILES = list(V.FILES) + ['v243_sgd_k1_cleanup.py']
assert V.LR_G == 0.1 and V.LR_G_FLOOR == 0.1 and V.CLIP_G == 5.0          # v241's frozen setting G
EST = lambda c: 29 if c == 'full' else 9 if c == 'floor' else 38        # minutes on a T4 (S24's runs)
runs = [(c, s) for s in SEEDS for c in CELLS]
slots = [[] for _ in range(8)]; load = [0] * 8
for c, s in sorted(runs, key=lambda r: -EST(r[0])):
    i = load.index(min(load)); slots[i].append((c, s)); load[i] += EST(c)
order = {c: i for i, c in enumerate(CELLS)}
for sl in slots:
    sl.sort(key=lambda r: (r[1], order[r[0]]))                          # earlier seeds first

body = r'''import os, subprocess, sys, threading, time, zlib, base64


def lane(gpu, args, env0):
    od = '/kaggle/working/gpu%d' % gpu; os.makedirs(od, exist_ok=True)
    env = dict(env0, CUDA_VISIBLE_DEVICES=str(gpu), OUT_DIR=od)
    with open(os.path.join(od, 'stdout.txt'), 'w') as out:      # stream to the console (live log) and the file
        p = subprocess.Popen([sys.executable, '-u', 'v241_sgd_split.py'] + args, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, env=env, text=True, bufsize=1)
        for line in p.stdout:
            print('[gpu%d] %s' % (gpu, line), end='', flush=True); out.write(line); out.flush()
        rc = p.wait()
    print('gpu%d exit %d elapsed %ds' % (gpu, rc, time.time() - T0), flush=True)


if __name__ == '__main__':
    T0 = time.time(); os.chdir('/kaggle/working')
    for fn, e in FILES.items():
        open(fn, 'wb').write(zlib.decompress(base64.b64decode(e)))
    subprocess.call(['git', 'clone', '-q', 'https://github.com/sandyresearch/parcae', '/tmp/parcae_src'])
    subprocess.call(['git', '-C', '/tmp/parcae_src', 'checkout', '-q', PARCAE_COMMIT])
    subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'datasets'])
    env0 = dict(os.environ, PYTHONUNBUFFERED='1', PARCAE_SRC='/tmp/parcae_src')
    print('setup done after %ds' % (time.time() - T0), flush=True)
    th = []
    for gpu, args in enumerate(JOBS):
        t = threading.Thread(target=lane, args=(gpu, args, env0)); t.start(); th.append(t); time.sleep(60)
    for t in th:
        t.join()
    print('total %ds' % (time.time() - T0), flush=True)
'''

if __name__ == '__main__':
    accts = sys.argv[1:5]
    assert len(accts) == 4, 'give four account names (one per kernel)'
    emb = {}
    for f in FILES:
        b = open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n')
        emb[f] = base64.b64encode(zlib.compress(b, 9)).decode()
        print('  %s sha256 (LF) %s' % (f, hashlib.sha256(b).hexdigest()[:12]))
    print('  per-GPU estimated minutes', load, '; runs', len(runs))
    for j in range(4):
        name = str(j + 1)
        jobs = [['--selftest', '--runs', ','.join('%s@%d' % r for r in slots[2 * j + g]), '--tag', 'e%d%d' % (j + 1, g)]
                for g in range(2)]
        dd = os.path.join(HERE, 'kaggle-kernel-v243-%s' % name); os.makedirs(dd, exist_ok=True)
        src = 'FILES = %r\nPARCAE_COMMIT = %r\nJOBS = %r\n' % (emb, PARCAE_COMMIT, jobs) + body
        path = os.path.join(dd, 'run.py')
        open(path, 'w', encoding='utf-8', newline='\n').write(src)
        py_compile.compile(path, doraise=True)
        json.dump(dict(id='%s/astrocyte-v243-%s' % (accts[j], name), title='astrocyte-v243-%s' % name, code_file='run.py',
                       language='python', kernel_type='script', is_private=True, enable_gpu=True, enable_internet=True),
                  open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
        print('built + compiled', dd, '->', '%s/astrocyte-v243-%s' % (accts[j], name), [len(x) for x in [slots[2 * j], slots[2 * j + 1]]])
