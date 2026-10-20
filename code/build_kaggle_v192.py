"""v192: per-loop contributions on Huginn-3.5B at R = 32, 16 probe sequences split over two T4s, one kernel.
Plan: docs/v192_huginn_contrib_plan.md. Writes kaggle-kernel-v192-huginn-contrib/{run.py,kernel-metadata.json}.
"""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, 'kaggle-kernel-v192-huginn-contrib')
os.makedirs(D, exist_ok=True)
emb = {f: base64.b64encode(zlib.compress(open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n'), 9)).decode()
       for f in ('loop_grad_probe.py', 'v192_huginn_contrib.py')}
JOBS = [('0-7', 'huginn_part0'), ('16-23', 'huginn_part1')]      # WikiText 0-7 on GPU 0, GSM8K 0-7 on GPU 1

RUN = r'''FILES = %r
JOBS = %r

import base64, os, subprocess, sys, time, zlib

def launch(gpu, seqs, tag):
    out = '/kaggle/working/gpu%%d' %% gpu
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, LOOP_MODEL='huginn', LOOP_R='32', SEQS=seqs, TAG=tag, OUT_DIR=out,
               CUDA_VISIBLE_DEVICES=str(gpu), PYTORCH_ALLOC_CONF='expandable_segments:True')
    log = open(os.path.join(out, 'stdout.txt'), 'w')
    return subprocess.Popen([sys.executable, '-u', 'v192_huginn_contrib.py'], env=env, stdout=log, stderr=subprocess.STDOUT)

if __name__ == '__main__':
    os.chdir('/kaggle/working')
    for fn, e in FILES.items():
        open(fn, 'wb').write(zlib.decompress(base64.b64decode(e)))
    subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.54.1'])
    t0 = time.time(); procs = []
    for gpu, (seqs, tag) in enumerate(JOBS):
        procs.append(launch(gpu, seqs, tag))
        time.sleep(240)                                  # stagger the two model loads (CPU RAM)
    for gpu, p in enumerate(procs):
        print('gpu%%d exit=%%d after %%ds' %% (gpu, p.wait(), time.time() - t0), flush=True)
''' % (emb, JOBS)

open(os.path.join(D, 'run.py'), 'w', encoding='utf-8', newline='\n').write(RUN)
json.dump({'id': 'aurmerin/astrocyte-v192-huginn-contrib', 'title': 'astrocyte-v192-huginn-contrib', 'code_file': 'run.py',
           'language': 'python', 'kernel_type': 'script', 'is_private': True, 'enable_gpu': True,
           'enable_internet': True}, open(os.path.join(D, 'kernel-metadata.json'), 'w'), indent=1)
print('wrote', D, len(RUN) // 1024, 'KB')
