"""Build the v195 Kaggle kernel (prereg docs/prereg_v195_rho_modes.md). Build only; does not push.
GPU 0: Parcae-140m and Parcae-370m at R 32 (K 64, m 32; frozen v162 / v163 machinery, pinned Parcae source), then Huginn
probe sequences 20-23. GPU 1: Huginn-3.5B probe sequences 0-7 and 16-19 (loop_grad_probe, transformers 4.54.1).
Huginn's 16 sequences are those of v192 (WikiText 0-7, GSM8K 16-23).
"""
import base64, hashlib, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ['v195_rho_modes.py', 'loop_grad_probe.py', 'v163_parcae_damped_trunc.py', 'v162_parcae_trunc_train.py']
PARCAE_COMMIT = '69284c13746e849104f738d6d1a347b1f457df76'
emb = {}
for f in FILES:
    b = open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n')
    emb[f] = base64.b64encode(zlib.compress(b, 9)).decode()
    print('  %s sha256 (LF) %s' % (f, hashlib.sha256(b).hexdigest()[:12]))
JOBS = [[['parcae', '--repo', 'SandyResearch/parcae-140m', '--R', '32', '--K', '64', '--M', '32', '--tag', '140m'],
         ['parcae', '--repo', 'SandyResearch/parcae-370m', '--R', '32', '--K', '64', '--M', '32', '--tag', '370m'],
         ['probe', '--R', '32', '--K', '32', '--M', '32', '--seqs', '20-23', '--tag', 'huginn_part1']],
        [['probe', '--R', '32', '--K', '32', '--M', '32', '--seqs', '0-7,16-19', '--tag', 'huginn_part0']]]

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
            rc = subprocess.call([sys.executable, '-u', 'v195_rho_modes.py'] + args, stdout=log, stderr=subprocess.STDOUT, env=env)
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

dd = os.path.join(HERE, 'kaggle-kernel-v195-rho-modes'); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write(RUN % (emb, PARCAE_COMMIT, JOBS))
json.dump(dict(id='aurmerin/astrocyte-v195-rho-modes', title='astrocyte-v195-rho-modes', code_file='run.py',
               language='python', kernel_type='script', is_private=True, enable_gpu=True, enable_internet=True),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd)
