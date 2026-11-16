"""Build the v237 Kaggle kernels (docs/prereg_v237_early_ft.md). Build only; does NOT push.
  ifm    : gpu0 Part A pln5, gpu1 Part A r5 (weights as v200 / v218)
  parcae : gpu0 Part A parcae (Parcae source as v218)
  ouro   : gpu0 Part B test chunks 0:8, gpu1 8:16 (transformers 4.54.1, as v210 / v219)
Every generated run.py is compile-checked here.
  python build_kaggle_v237.py ACCOUNT_IFM ACCOUNT_PARCAE ACCOUNT_OURO
Push (NOT done here): kaggle kernels push -p kaggle-kernel-v237-<name> --accelerator NvidiaTeslaT4
"""
import base64, hashlib, json, os, py_compile, sys, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
PARCAE_COMMIT = '69284c13746e849104f738d6d1a347b1f457df76'
IFM_REV = {'pln5': ('IFM/LoopedLM-P2-huginn-s-fixed-pln5', 'c581f369846e74b40072cae77ef19fa51437dfa7'),
           'r5': ('IFM/LoopedLM-P2-huginn-s-fixed-r5', '17cdf78e0d9fd8d19bba07ed800875e1de5675fd')}
A_FILES = ['v237_early_ft.py', 'v218_robustness.py', 'v201_batch_lost.py']
KERNELS = [
    ('ifm', A_FILES + ['v200_ifm_trunc.py', 'ifm_looped.py'],
     [['A', '--model', 'pln5', '--weights', '/tmp/ifm_pln5'], ['A', '--model', 'r5', '--weights', '/tmp/ifm_r5']]),
    ('parcae', A_FILES + ['v162_parcae_trunc_train.py', 'v163_parcae_damped_trunc.py'], [['A', '--model', 'parcae']]),
    ('ouro', ['v237_early_ft.py', 'v210_followups.py', 'v197_ouro_allparam.py', 'loop_grad_probe.py'],
     [['B', '--seqs', '0:8', '--tag', 'g0'], ['B', '--seqs', '8:16', '--tag', 'g1']]),
]

body = r'''import os, shutil, subprocess, sys, threading, time, zlib, base64


def lane(gpu, args, env0):
    od = '/kaggle/working/gpu%d' % gpu; os.makedirs(od, exist_ok=True)
    env = dict(env0, CUDA_VISIBLE_DEVICES=str(gpu), OUT_DIR=od)
    with open(os.path.join(od, 'stdout.txt'), 'w') as out:      # stream to the console (live log) and the file
        p = subprocess.Popen([sys.executable, '-u', 'v237_early_ft.py'] + args, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, env=env, text=True, bufsize=1)
        for line in p.stdout:
            print('[gpu%d] %s' % (gpu, line), end='', flush=True); out.write(line); out.flush()
        rc = p.wait()
    print('gpu%d %s exit %d elapsed %ds' % (gpu, args, rc, time.time() - T0), flush=True)


if __name__ == '__main__':
    T0 = time.time(); os.chdir('/kaggle/working')
    for fn, e in FILES.items():
        open(fn, 'wb').write(zlib.decompress(base64.b64decode(e)))
    env0 = dict(os.environ, PYTHONUNBUFFERED='1', PYTORCH_ALLOC_CONF='expandable_segments:True')
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
        env0['LOOP_MODEL'] = 'ouro'
    if NAME != 'ouro':
        subprocess.call([sys.executable, '-c', "from datasets import load_dataset; [load_dataset('openai/gsm8k', 'main', split=s) for s in ('train', 'test')]; load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='test')"])
    else:
        subprocess.call([sys.executable, '-c', "from datasets import load_dataset; load_dataset('openai/gsm8k', 'main')"])
    print('setup done after %ds' % (time.time() - T0), flush=True)
    th = []
    for gpu, args in enumerate(JOBS):
        t = threading.Thread(target=lane, args=(gpu, args, env0)); t.start(); th.append(t); time.sleep(30)
    for t in th:
        t.join()
    print('total %ds' % (time.time() - T0), flush=True)
'''

if __name__ == '__main__':
    accts = dict(zip(('ifm', 'parcae', 'ouro'), sys.argv[1:4]))
    for name, files, jobs in KERNELS:
        emb = {}
        for f in files:
            b = open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n')
            emb[f] = base64.b64encode(zlib.compress(b, 9)).decode()
            print('  %s sha256 (LF) %s' % (f, hashlib.sha256(b).hexdigest()[:12]))
        dd = os.path.join(HERE, 'kaggle-kernel-v237-%s' % name); os.makedirs(dd, exist_ok=True)
        src = 'FILES = %r\nPARCAE_COMMIT = %r\nIFM_REV = %r\nJOBS = %r\nNAME = %r\n' % (emb, PARCAE_COMMIT, IFM_REV, jobs, name) + body
        path = os.path.join(dd, 'run.py')
        open(path, 'w', encoding='utf-8', newline='\n').write(src)
        py_compile.compile(path, doraise=True)
        acct = accts.get(name, 'aurshalevmerin')
        json.dump(dict(id='%s/astrocyte-v237-%s' % (acct, name), title='astrocyte-v237-%s' % name, code_file='run.py',
                       language='python', kernel_type='script', is_private=True, enable_gpu=True, enable_internet=True),
                  open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
        print('built + compiled', dd, '->', '%s/astrocyte-v237-%s' % (acct, name))
