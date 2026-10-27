"""Build the v219 PILOT kernel (Ouro-1.4B truncation cost: memory/feasibility and seed spread; before any v219 prereg).
Build only; does NOT push.
  pilot (aurshalevmerin), version 2: Ouro split across the kernel's two T4s (layers 0-11 on GPU 0, 12-23 on GPU 1).
    One process at a time on both GPUs: the memory probe (each configuration in a fresh process), then short fine-tunes
    with the optimiser that fit, final-output objective k = 4 (seed 1), 0, 2, 4 (seed 2), then the exit-weighted
    objective the same.
Push (NOT done here): kaggle kernels push -p kaggle-kernel-v219-pilot --accelerator NvidiaTeslaT4
"""
import base64, hashlib, json, os, py_compile, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ['v219_ouro_cost_pilot.py', 'loop_grad_probe.py', 'v197_ouro_allparam.py']
JOBS = ['4:1:final,0:1:final,2:1:final,4:2:final', '4:1:exit,0:1:exit,2:1:exit,4:2:exit']

body = r'''import os, shutil, subprocess, sys, time, zlib, base64
if __name__ == '__main__':
    T0 = time.time(); os.chdir('/kaggle/working')
    for fn, e in FILES.items():
        open(fn, 'wb').write(zlib.decompress(base64.b64decode(e)))
    subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.54.1', 'datasets', 'bitsandbytes'])
    subprocess.call([sys.executable, '-c', "from datasets import load_dataset; load_dataset('openai/gsm8k', 'main', split='train')"])
    env0 = dict(os.environ, PYTHONUNBUFFERED='1', LOOP_MODEL='ouro', PYTORCH_ALLOC_CONF='expandable_segments:True')
    print('setup done after %ds' % (time.time() - T0), flush=True)
    od = '/kaggle/working/split'; os.makedirs(od, exist_ok=True)
    env = dict(env0, CUDA_VISIBLE_DEVICES='0,1', OUT_DIR=od)       # one process, both T4s (model split across them)
    for args in [['--phase', 'probe']] + [['--phase', 'runs', '--runs', r] for r in JOBS]:
        with open(os.path.join(od, 'stdout.txt'), 'a') as out:          # stream to the console (live log) and the file
            p = subprocess.Popen([sys.executable, '-u', 'v219_ouro_cost_pilot.py'] + args, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, env=env, text=True, bufsize=1)
            for line in p.stdout:
                print(line, end='', flush=True); out.write(line); out.flush()
            rc = p.wait()
        print('step', args[:2], 'exit', rc, 'elapsed %ds' % (time.time() - T0), flush=True)
    print(open(os.path.join(od, 'stdout.txt')).read()[-6000:], flush=True)
    print('total %ds' % (time.time() - T0), flush=True)
'''

emb = {}
for f in FILES:
    b = open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n')
    emb[f] = base64.b64encode(zlib.compress(b, 9)).decode()
    print('  %s sha256 (LF) %s' % (f, hashlib.sha256(b).hexdigest()[:12]))
dd = os.path.join(HERE, 'kaggle-kernel-v219-pilot'); os.makedirs(dd, exist_ok=True)
path = os.path.join(dd, 'run.py')
open(path, 'w', encoding='utf-8', newline='\n').write('FILES = %r\nJOBS = %r\n' % (emb, JOBS) + body)
py_compile.compile(path, doraise=True)
json.dump(dict(id='aurshalevmerin/astrocyte-v219-pilot', title='astrocyte-v219-pilot', code_file='run.py', language='python',
               kernel_type='script', is_private=True, enable_gpu=True, enable_internet=True),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built + compiled', dd)
