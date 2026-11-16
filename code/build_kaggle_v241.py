"""Build the v241 Kaggle kernels (docs/prereg_v241_sgd_split.md). Build only; does NOT push.
Main study: 4 kernels x 2 T4s; every GPU lane runs the v241 selftest on the real model first (an assertion stops that
lane), then its share of the 60 runs (6 seeds x 10 cells), balanced longest-first and ordered by seed. Refuses to build
until LR_G, LR_G_FLOOR and R_MATCH are set from the pilot.
  python build_kaggle_v241.py ACCT_1 ACCT_2 ACCT_3 ACCT_4
Pilot (controls only, before the freeze): 1 kernel x 2 T4s, FULL and FLOOR at clip 5 on the pilot grids.
  python build_kaggle_v241.py --pilot ACCT [ITEMS]     (ITEMS: a second pilot round, one lane, e.g. 'floor@0.0333')
Push (NOT done here): kaggle kernels push -p kaggle-kernel-v241-<n> --accelerator NvidiaTeslaT4
"""
import base64, hashlib, json, os, py_compile, sys, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v241_sgd_split as V                            # noqa: E402
from v241_sgd_split import CELLS, SEEDS, FILES        # noqa: E402

PARCAE_COMMIT = '69284c13746e849104f738d6d1a347b1f457df76'
EST = lambda c: 29 if c.startswith('full') else 9 if c.startswith('floor') else 38   # minutes on a T4 (S8 / S22 runs)
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

PILOT_JOBS = [['--selftest', '--pilot', 'full@0.1,full@0.3,floor@0.1', '--tag', 'p0'],
              ['--selftest', '--pilot', 'full@0.03,floor@0.3,floor@1.0', '--tag', 'p1']]


def write_kernel(name, acct, jobs, emb):
    dd = os.path.join(HERE, 'kaggle-kernel-v241-%s' % name); os.makedirs(dd, exist_ok=True)
    src = 'FILES = %r\nPARCAE_COMMIT = %r\nJOBS = %r\n' % (emb, PARCAE_COMMIT, jobs) + body
    path = os.path.join(dd, 'run.py')
    open(path, 'w', encoding='utf-8', newline='\n').write(src)
    py_compile.compile(path, doraise=True)
    json.dump(dict(id='%s/astrocyte-v241-%s' % (acct, name), title='astrocyte-v241-%s' % name, code_file='run.py',
                   language='python', kernel_type='script', is_private=True, enable_gpu=True, enable_internet=True),
              open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
    print('built + compiled', dd, '->', '%s/astrocyte-v241-%s' % (acct, name))


if __name__ == '__main__':
    pilot = sys.argv[1:2] == ['--pilot']
    accts = sys.argv[2:3] if pilot else sys.argv[1:5]
    round2 = pilot and len(sys.argv) > 3                                # second pilot round (grid-end extension)
    if round2:
        PILOT_JOBS = [['--selftest', '--pilot', sys.argv[3], '--tag', 'p2']]
    assert len(accts) == (1 if pilot else 4), 'give one account (--pilot) or four (main study)'
    if not pilot:
        assert V.LR_G and V.LR_G_FLOOR and all(V.R_MATCH.values()), 'set LR_G, LR_G_FLOOR, R_MATCH from the pilot first'
    emb = {}
    for f in FILES:
        b = open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n')
        emb[f] = base64.b64encode(zlib.compress(b, 9)).decode()
        print('  %s sha256 (LF) %s' % (f, hashlib.sha256(b).hexdigest()[:12]))
    if pilot:
        write_kernel('pilot2' if round2 else 'pilot', accts[0], PILOT_JOBS, emb); sys.exit()
    print('  per-GPU estimated minutes', load, '; runs', len(runs))
    for j in range(4):
        name = str(j + 1)
        jobs = [['--selftest', '--runs', ','.join('%s@%d' % r for r in slots[2 * j + g]), '--tag', '%d%d' % (j + 1, g)]
                for g in range(2)]
        dd = os.path.join(HERE, 'kaggle-kernel-v241-%s' % name); os.makedirs(dd, exist_ok=True)
        src = 'FILES = %r\nPARCAE_COMMIT = %r\nJOBS = %r\n' % (emb, PARCAE_COMMIT, jobs) + body
        path = os.path.join(dd, 'run.py')
        open(path, 'w', encoding='utf-8', newline='\n').write(src)
        py_compile.compile(path, doraise=True)
        json.dump(dict(id='%s/astrocyte-v241-%s' % (accts[j], name), title='astrocyte-v241-%s' % name, code_file='run.py',
                       language='python', kernel_type='script', is_private=True, enable_gpu=True, enable_internet=True),
                  open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
        print('built + compiled', dd, '->', '%s/astrocyte-v241-%s' % (accts[j], name), [len(x) for x in [slots[2 * j], slots[2 * j + 1]]])
