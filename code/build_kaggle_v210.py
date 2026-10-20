"""Build the v210 Kaggle kernels (review-6 follow-ups; docs/prereg_v210_followups.md). Build only; does NOT push.
  parcae (aurmerin):       gpu0 Part A s0 + full300, gpu1 Part A trunc300 (Parcae-140m; setup as v201)
  ifm    (aurshalevmerin): gpu0 Part A pln5 (all runs), gpu1 Part A r5 (all runs) (weights as v200)
  ouro   (aurmerin):       gpu0 Part B sequences 0-7, gpu1 sequences 8-15 (setup as v197)
Every generated run.py is compile-checked here.
Push (NOT done here): kaggle kernels push -p kaggle-kernel-v210-<name> --accelerator NvidiaTeslaT4
"""
import base64, hashlib, json, os, py_compile, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
PARCAE_COMMIT = '69284c13746e849104f738d6d1a347b1f457df76'
IFM_REV = {'pln5': ('IFM/LoopedLM-P2-huginn-s-fixed-pln5', 'c581f369846e74b40072cae77ef19fa51437dfa7'),
           'r5': ('IFM/LoopedLM-P2-huginn-s-fixed-r5', '17cdf78e0d9fd8d19bba07ed800875e1de5675fd')}
KERNELS = [
    ('parcae', 'aurmerin', ['v210_followups.py', 'v201_batch_lost.py', 'v162_parcae_trunc_train.py', 'v163_parcae_damped_trunc.py'],
     [['A', '--model', 'parcae', '--runs', 's0,full300'], ['A', '--model', 'parcae', '--runs', 'trunc300']]),
    ('ifm', 'aurshalevmerin', ['v210_followups.py', 'v201_batch_lost.py', 'v200_ifm_trunc.py', 'ifm_looped.py'],
     [['A', '--model', 'pln5', '--weights', '/tmp/ifm_pln5'], ['A', '--model', 'r5', '--weights', '/tmp/ifm_r5']]),
    ('ouro', 'aurmerin', ['v210_followups.py', 'v197_ouro_allparam.py', 'loop_grad_probe.py'],
     [['B', '--seqs', '0:8', '--tag', 'g0'], ['B', '--seqs', '8:16', '--tag', 'g1']]),
]

body = r'''import os, shutil, subprocess, sys, time, zlib, base64
if __name__ == '__main__':
    T0 = time.time(); os.chdir('/kaggle/working')
    for fn, e in FILES.items():
        open(fn, 'wb').write(zlib.decompress(base64.b64decode(e)))
    env0 = dict(os.environ, PYTHONUNBUFFERED='1')
    if NAME == 'parcae':
        subprocess.call(['git', 'clone', '-q', 'https://github.com/sandyresearch/parcae', '/tmp/parcae_src'])
        subprocess.call(['git', '-C', '/tmp/parcae_src', 'checkout', '-q', PARCAE_COMMIT])
        subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'datasets'])
        env0['PARCAE_SRC'] = '/tmp/parcae_src'
    elif NAME == 'ifm':
        subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'datasets', 'huggingface_hub', 'safetensors', 'tokenizers'])
        from huggingface_hub import snapshot_download
        for short, (repo, rev) in IFM_REV.items():
            w = '/tmp/ifm_%s' % short
            snapshot_download(repo_id=repo, revision=rev, local_dir=w)
            for root, _, fs in os.walk(w):
                for f in fs:
                    p = os.path.join(root, f)
                    if os.path.islink(p):
                        t = os.path.realpath(p); os.unlink(p); shutil.copyfile(t, p)
    else:
        subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.54.1', 'datasets'])
        env0.update(LOOP_MODEL='ouro', PYTORCH_ALLOC_CONF='expandable_segments:True')
    subprocess.call([sys.executable, '-c', "from datasets import load_dataset; [load_dataset('openai/gsm8k', 'main', split=s) for s in ('train', 'test')]"])
    print('setup done after %ds' % (time.time() - T0), flush=True)
    procs = []
    for gpu, args in enumerate(JOBS):
        od = '/kaggle/working/gpu%d' % gpu; os.makedirs(od, exist_ok=True)
        env = dict(env0, CUDA_VISIBLE_DEVICES=str(gpu), OUT_DIR=od)
        out = open(os.path.join(od, 'stdout.txt'), 'w')
        procs.append((subprocess.Popen([sys.executable, '-u', 'v210_followups.py'] + args, stdout=out,
                                       stderr=subprocess.STDOUT, env=env), out))
        time.sleep(150 if NAME == 'ouro' else 30)       # stagger large fp32 model loads (CPU RAM)
    for p, out in procs:
        rc = p.wait(); out.close(); print('job exit', rc, 'elapsed %ds' % (time.time() - T0), flush=True)
    for gpu in range(len(JOBS)):
        print('---- gpu%d (tail)\n%s' % (gpu, open('/kaggle/working/gpu%d/stdout.txt' % gpu).read()[-6000:]), flush=True)
    subprocess.call([sys.executable, 'v210_followups.py', '--analyze', '/kaggle/working'])
    print('total %ds' % (time.time() - T0), flush=True)
'''

for name, acct, files, jobs in KERNELS:
    emb = {}
    for f in files:
        b = open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n')
        emb[f] = base64.b64encode(zlib.compress(b, 9)).decode()
        print('  %s sha256 (LF) %s' % (f, hashlib.sha256(b).hexdigest()[:12]))
    dd = os.path.join(HERE, 'kaggle-kernel-v210-%s' % name); os.makedirs(dd, exist_ok=True)
    src = 'FILES = %r\nPARCAE_COMMIT = %r\nIFM_REV = %r\nJOBS = %r\nNAME = %r\n' % (emb, PARCAE_COMMIT, IFM_REV, jobs, name) + body
    path = os.path.join(dd, 'run.py')
    open(path, 'w', encoding='utf-8', newline='\n').write(src)
    py_compile.compile(path, doraise=True)
    json.dump(dict(id='%s/astrocyte-v210-%s' % (acct, name), title='astrocyte-v210-%s' % name, code_file='run.py',
                   language='python', kernel_type='script', is_private=True, enable_gpu=True, enable_internet=True),
              open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
    print('built + compiled', dd, '->', '%s/astrocyte-v210-%s' % (acct, name))
