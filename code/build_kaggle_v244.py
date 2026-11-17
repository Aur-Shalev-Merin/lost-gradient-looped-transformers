"""Build the v244 Kaggle kernel (unregistered descriptive check; see v244_group_shrink.py): v241's setting-G full-backprop
run on two new unscored seeds, one per T4, probing |g_k| / |g_R| per parameter group every 10 steps. Build only; does NOT push.
  python build_kaggle_v244.py ACCT
Push (NOT done here): kaggle kernels push -p kaggle-kernel-v244 --accelerator NvidiaTeslaT4
"""
import base64, hashlib, json, os, py_compile, sys, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
PARCAE_COMMIT = '69284c13746e849104f738d6d1a347b1f457df76'
sys.path.insert(0, HERE)
import v241_sgd_split as V                                               # noqa: E402
from v244_group_shrink import SEEDS                                      # noqa: E402
FILES = list(V.FILES) + ['v244_group_shrink.py']
assert V.LR_G == 0.1 and V.CLIP_G == 5.0

body = r'''import os, subprocess, sys, threading, time, zlib, base64


def lane(gpu, args, env0):
    od = '/kaggle/working/gpu%d' % gpu; os.makedirs(od, exist_ok=True)
    env = dict(env0, CUDA_VISIBLE_DEVICES=str(gpu), OUT_DIR=od)
    with open(os.path.join(od, 'stdout.txt'), 'w') as out:      # stream to the console (live log) and the file
        p = subprocess.Popen([sys.executable, '-u', 'v244_group_shrink.py'] + args, stdout=subprocess.PIPE,
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
    acct = sys.argv[1]
    emb = {}
    for f in FILES:
        b = open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n')
        emb[f] = base64.b64encode(zlib.compress(b, 9)).decode()
        print('  %s sha256 (LF) %s' % (f, hashlib.sha256(b).hexdigest()[:12]))
    jobs = [['--seed', str(s)] for s in SEEDS]
    dd = os.path.join(HERE, 'kaggle-kernel-v244'); os.makedirs(dd, exist_ok=True)
    src = 'FILES = %r\nPARCAE_COMMIT = %r\nJOBS = %r\n' % (emb, PARCAE_COMMIT, jobs) + body
    path = os.path.join(dd, 'run.py')
    open(path, 'w', encoding='utf-8', newline='\n').write(src)
    py_compile.compile(path, doraise=True)
    json.dump(dict(id='%s/astrocyte-v244' % acct, title='astrocyte-v244', code_file='run.py', language='python',
                   kernel_type='script', is_private=True, enable_gpu=True, enable_internet=True),
              open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
    print('built + compiled', dd, '->', '%s/astrocyte-v244' % acct, jobs)
