"""Build Kaggle kernels for v168 (EXPLORATORY direction probe; v168_direction_probe.py). Build only; does not push.
  parcae kernel (aurshalevmerin, T4 x2): every Parcae-140m variant trained in v162-v166, split over the 2 GPUs
  retro kernel  (aurmerin, needs dataset retro-smollm2-healed): the three v167 retrofit variants
"""
import base64, hashlib, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
PARCAE_COMMIT = '69284c13746e849104f738d6d1a347b1f457df76'
A1, A2 = 'aurshalevmerin', 'aurmerin'
PARCAE_JOBS = [['--model', 'parcae', '--tag', 'parcae0', '--variants', '8:1.0:1,2,4', '16:1.0:1,2,4,8', '32:1.0:1,2,3,4,6,8', '32:0.5:1,2,4,8'],
               ['--model', 'parcae', '--tag', 'parcae1', '--variants', '32:0.7:2,3,4,6', '64:0.3:2,3,4,6,8,12', '48:0.4:4,8']]
RETRO_JOBS = [['--model', 'retro', '--tag', 'retro', '--variants', '8:1.0:1,2', '8:0.5:1,2', '8:0.3:1,2']]
KERNELS = [('parcae', A1, ['v168_direction_probe.py', 'v163_parcae_damped_trunc.py', 'v162_parcae_trunc_train.py'], PARCAE_JOBS),
           ('retro', A2, ['v168_direction_probe.py', 'v167_retrofit_trunc.py'], RETRO_JOBS)]

body = r'''import subprocess, sys, os, base64, zlib, time, glob
if __name__ == '__main__':
    os.chdir('/kaggle/working')
    for fn, e in FILES.items():
        open(fn, 'wb').write(zlib.decompress(base64.b64decode(e)))
    env0 = dict(os.environ, PYTHONUNBUFFERED='1')
    if NAME == 'parcae':
        subprocess.call(['git', 'clone', '-q', 'https://github.com/sandyresearch/parcae', '/tmp/parcae_src'])
        subprocess.call(['git', '-C', '/tmp/parcae_src', 'checkout', '-q', PARCAE_COMMIT])
        subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'datasets'])
        env0['PARCAE_SRC'] = '/tmp/parcae_src'
    else:
        subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.51.0', 'datasets'])
        cfg = [p for p in glob.glob('/kaggle/input/**/config.json', recursive=True) if os.path.exists(os.path.join(os.path.dirname(p), 'raven_modeling_minimal.py'))]
        assert len(cfg) == 1, cfg
        env0['RETRO_DIR'] = os.path.dirname(cfg[0])
    t0 = time.time(); procs = []
    for gpu, args in enumerate(JOBS):
        od = '/kaggle/working/gpu%d' % gpu; os.makedirs(od, exist_ok=True)
        env = dict(env0, CUDA_VISIBLE_DEVICES=str(gpu), OUT_DIR=od)
        out = open('v168_stdout_gpu%d.txt' % gpu, 'w')
        procs.append((subprocess.Popen([sys.executable, '-u', 'v168_direction_probe.py'] + args, stdout=out, stderr=subprocess.STDOUT, env=env), out))
        time.sleep(30)
    for p, out in procs:
        rc = p.wait(); out.close(); print('job exit', rc, 'elapsed %ds' % (time.time() - t0), flush=True)
    for gpu in range(len(JOBS)):
        print(open('v168_stdout_gpu%d.txt' % gpu).read()[-6000:], flush=True)
    subprocess.call('rm -rf /kaggle/working/gpu*/retro_model', shell=True)
'''

for name, acct, files, jobs in KERNELS:
    emb = {}
    for f in files:
        b = open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n')
        emb[f] = base64.b64encode(zlib.compress(b, 9)).decode()
        print('  %s sha256 (LF) %s' % (f, hashlib.sha256(b).hexdigest()[:12]))
    dd = os.path.join(HERE, 'kaggle-kernel-v168-%s' % name); os.makedirs(dd, exist_ok=True)
    src = 'FILES = %r\nPARCAE_COMMIT = %r\nJOBS = %r\nNAME = %r\n' % (emb, PARCAE_COMMIT, jobs, name) + body
    open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write(src)
    meta = dict(id='%s/astrocyte-v168-%s' % (acct, name), title='astrocyte-v168-%s' % name, code_file='run.py', language='python',
                kernel_type='script', is_private=True, enable_gpu=True, enable_internet=True)
    if name == 'retro':
        meta['dataset_sources'] = ['%s/retro-smollm2-healed' % acct]
    json.dump(meta, open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
    print('built', dd, acct, [len(j) for j in jobs])
