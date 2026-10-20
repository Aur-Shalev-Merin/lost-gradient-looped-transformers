"""Build the v197 Kaggle kernel (Ouro-1.4B all-parameter lost gradient, Table 1 protocol). Build only; does not push.
One kernel on aurmerin, T4 x2: GPU 0 measures sequences 0-7, GPU 1 sequences 8-15.
"""
import base64, hashlib, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ['v197_ouro_allparam.py', 'loop_grad_probe.py']
JOBS = [['--seqs', '0:8', '--tag', 'g0'], ['--seqs', '8:16', '--tag', 'g1']]
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
    subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.54.1', 'datasets'])
    t0 = time.time(); procs = []
    for gpu, args in enumerate(JOBS):
        od = '/kaggle/working/gpu%d' % gpu; os.makedirs(od, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OUT_DIR=od, LOOP_MODEL='ouro', PYTHONUNBUFFERED='1',
                   PYTORCH_ALLOC_CONF='expandable_segments:True')
        out = open('v197_stdout_gpu%d.txt' % gpu, 'w')
        procs.append((subprocess.Popen([sys.executable, '-u', 'v197_ouro_allparam.py'] + args, stdout=out,
                                       stderr=subprocess.STDOUT, env=env), out))
        time.sleep(150)                                  # stagger the two fp32 model loads (CPU RAM)
    for p, out in procs:
        rc = p.wait(); out.close(); print('job exit', rc, 'elapsed %ds' % (time.time() - t0), flush=True)
    for gpu in range(len(JOBS)):
        print(open('v197_stdout_gpu%d.txt' % gpu).read()[-6000:], flush=True)
    subprocess.call([sys.executable, 'v197_ouro_allparam.py', '--analyze', '/kaggle/working'])
'''
dd = os.path.join(HERE, 'kaggle-kernel-v197-ouro'); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write('FILES = %r\nJOBS = %r\n' % (emb, JOBS) + body)
json.dump(dict(id='aurmerin/astrocyte-v197-ouro', title='astrocyte-v197-ouro', code_file='run.py', language='python',
               kernel_type='script', is_private=True, enable_gpu=True, enable_internet=True),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd)
