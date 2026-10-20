"""Build the v204 Kaggle kernel (prereg docs/prereg_v204_drift_prediction.md): the unchanged v196 driver at R = 96, 128.
Build only; does not push. (Header below is v196's.)
GPU 0: Parcae-140m then Parcae-370m at R = 32, 48, 64 (frozen v162 / v163 machinery, pinned Parcae source), then Huginn
probe sequences 20-23 at R = 48. GPU 1: Huginn probe sequences 0-7 and 16-19 at R = 48 (loop_grad_probe, transformers
4.54.1). Huginn's 16 sequences are those of v192 / v195.
"""
import base64, hashlib, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ['v196_band.py', 'v195_rho_modes.py', 'loop_grad_probe.py', 'v163_parcae_damped_trunc.py', 'v162_parcae_trunc_train.py']
PARCAE_COMMIT = '69284c13746e849104f738d6d1a347b1f457df76'
emb = {}
for f in FILES:
    b = open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n')
    emb[f] = base64.b64encode(zlib.compress(b, 9)).decode()
    print('  %s sha256 (LF) %s' % (f, hashlib.sha256(b).hexdigest()[:12]))
JOBS = [[['parcae', '--repo', 'SandyResearch/parcae-140m', '--R', '96,128', '--tag', '140m']],
        [['parcae', '--repo', 'SandyResearch/parcae-370m', '--R', '96,128', '--tag', '370m']]]

RUN = r'''FILES = %r
PARCAE_COMMIT = %r
JOBS = %r

import base64, os, subprocess, sys, threading, time, zlib

def run_gpu(gpu, cmds):
    out = '/kaggle/working/gpu%%d' %% gpu
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PARCAE_SRC='/tmp/parcae_src', PYTHONUNBUFFERED='1', OUT_DIR=out,
               LOOP_MODEL='huginn', PYTORCH_ALLOC_CONF='expandable_segments:True')
    with open(os.path.join(out, 'stdout.txt'), 'w') as log:
        for args in cmds:
            rc = subprocess.call([sys.executable, '-u', 'v196_band.py'] + args, stdout=log, stderr=subprocess.STDOUT, env=env)
            print('gpu%%d %%s exit %%d after %%ds' %% (gpu, args[-1], rc, time.time() - T0), flush=True)

if __name__ == '__main__':
    os.chdir('/kaggle/working')
    for fn, e in FILES.items():
        open(fn, 'wb').write(zlib.decompress(base64.b64decode(e)))
    subprocess.call(['git', 'clone', '-q', 'https://github.com/sandyresearch/parcae', '/tmp/parcae_src'])
    subprocess.call(['git', '-C', '/tmp/parcae_src', 'checkout', '-q', PARCAE_COMMIT])
    subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'datasets', 'transformers==4.54.1'])
    T0 = time.time(); th = []
    for gpu, cmds in enumerate(JOBS):
        t = threading.Thread(target=run_gpu, args=(gpu, cmds)); t.start(); th.append(t); time.sleep(120)
    for t in th:
        t.join()
'''

dd = os.path.join(HERE, 'kaggle-kernel-v204-drift'); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write(RUN % (emb, PARCAE_COMMIT, JOBS))
json.dump(dict(id='aurmerin/astrocyte-v204-drift', title='astrocyte-v204-drift', code_file='run.py',
               language='python', kernel_type='script', is_private=True, enable_gpu=True, enable_internet=True),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd)
