"""v187: Huginn-3.5B loop-gradient probe at R = 8 and R = 64 (fallback 48), one loop count per T4, one kernel.
Plan: docs/v187_huginn_rho_transfer_plan.md. Writes kaggle-kernel-v187-huginn-rho/{run.py,kernel-metadata.json}.
"""
import base64, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, 'kaggle-kernel-v187-huginn-rho')
os.makedirs(D, exist_ok=True)
emb = base64.b64encode(zlib.compress(open(os.path.join(HERE, 'loop_grad_probe.py'), 'rb').read(), 9)).decode()

RUN = r'''EMB = %r

import base64, os, subprocess, sys, time, zlib

def launch(R, gpu):
    out = '/kaggle/working/R%%d' %% R
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, LOOP_MODEL='huginn', LOOP_R=str(R), OUT_DIR=out, CUDA_VISIBLE_DEVICES=str(gpu),
               BUDGET_S=str(10.5 * 3600), PYTORCH_ALLOC_CONF='expandable_segments:True')
    log = open(os.path.join(out, 'stdout.txt'), 'w')
    return subprocess.Popen([sys.executable, '-u', 'loop_grad_probe.py'], env=env, stdout=log, stderr=subprocess.STDOUT)

if __name__ == '__main__':
    os.chdir('/kaggle/working')
    open('loop_grad_probe.py', 'wb').write(zlib.decompress(base64.b64decode(EMB)))
    subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.54.1'])
    t0 = time.time()
    p8 = launch(8, 0)
    time.sleep(240)                                  # stagger the two model loads (CPU RAM)
    p64 = launch(64, 1)
    rc64 = p64.wait()
    print('R64 exit=%%d after %%ds' %% (rc64, time.time() - t0), flush=True)
    if rc64 != 0:                                    # e.g. out of memory: fall back to 48 loops on the same GPU
        rc48 = launch(48, 1).wait()
        print('R48 exit=%%d after %%ds' %% (rc48, time.time() - t0), flush=True)
    print('R8 exit=%%d after %%ds' %% (p8.wait(), time.time() - t0), flush=True)
''' % emb

open(os.path.join(D, 'run.py'), 'w').write(RUN)
json.dump({'id': 'aurshalevmerin/astrocyte-v187-huginn-rho', 'title': 'astrocyte-v187-huginn-rho', 'code_file': 'run.py',
           'language': 'python', 'kernel_type': 'script', 'is_private': True, 'enable_gpu': True,
           'enable_internet': True}, open(os.path.join(D, 'kernel-metadata.json'), 'w'), indent=1)
print('wrote', D, len(RUN) // 1024, 'KB')
