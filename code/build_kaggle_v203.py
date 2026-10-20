"""Build the v203 Kaggle kernels (held-out check of v201; docs/prereg_v203_heldout_lost.md). Build only; does NOT push.
Each kernel is T4 x2 with one v203_heldout_lost.py process per GPU (layout as v201):
  parcae (aurshalevmerin): gpu0 stage 0, gpu1 stage 300 (Parcae-140m, R = 32; setup as v171)
  retro  (aurshalevmerin, dataset retro-smollm2-healed): gpu0 stage 0, gpu1 stage 300 (SmolLM2 retrofit (A), R = 8)
  ifm    (aurmerin): gpu0 pln5, gpu1 r5, each stage 0 then stage 300 (Part II, R = 5; weights as v200)
Push (NOT done here): kaggle kernels push -p kaggle-kernel-v203-<name> --accelerator NvidiaTeslaT4
"""
import base64, hashlib, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
PARCAE_COMMIT = '69284c13746e849104f738d6d1a347b1f457df76'                       # as v171
IFM_REV = {'pln5': ('IFM/LoopedLM-P2-huginn-s-fixed-pln5', 'c581f369846e74b40072cae77ef19fa51437dfa7'),
           'r5': ('IFM/LoopedLM-P2-huginn-s-fixed-r5', '17cdf78e0d9fd8d19bba07ed800875e1de5675fd')}  # as v198 / v200
A1, A2 = 'aurshalevmerin', 'aurmerin'
# 2026-10-15, after the freeze: aurshalevmerin had reached its weekly GPU quota, so parcae and retro run on aurmerin
# (which also holds retro-smollm2-healed). Only the account changes; every run.py is byte-identical to the frozen build.
A1 = A2
KERNELS = [
    ('parcae', A1, ['v203_heldout_lost.py', 'v201_batch_lost.py', 'v162_parcae_trunc_train.py', 'v163_parcae_damped_trunc.py'],
     [['--model', 'parcae', '--stages', '0'], ['--model', 'parcae', '--stages', '300']]),
    ('parcae-rerun', A1, ['v203_heldout_lost.py', 'v201_batch_lost.py', 'v162_parcae_trunc_train.py', 'v163_parcae_damped_trunc.py'],
     [['--model', 'parcae', '--stages', '300', '--ft-seed', '2012'], ['--model', 'parcae', '--stages', '300', '--ft-seed', '2013']]),
    ('retro', A1, ['v203_heldout_lost.py', 'v201_batch_lost.py', 'v167_retrofit_trunc.py'],
     [['--model', 'retro', '--stages', '0'], ['--model', 'retro', '--stages', '300']]),
    ('ifm', A2, ['v203_heldout_lost.py', 'v201_batch_lost.py', 'v200_ifm_trunc.py', 'ifm_looped.py'],
     [['--model', 'pln5', '--stages', '0,300', '--weights', '/tmp/ifm_pln5'],
      ['--model', 'r5', '--stages', '0,300', '--weights', '/tmp/ifm_r5']]),
]

body = r'''import glob, os, shutil, subprocess, sys, time, zlib, base64
if __name__ == '__main__':
    T0 = time.time(); os.chdir('/kaggle/working')
    for fn, e in FILES.items():
        open(fn, 'wb').write(zlib.decompress(base64.b64decode(e)))
    env0 = dict(os.environ, PYTHONUNBUFFERED='1')
    if NAME.startswith('parcae'):
        subprocess.call(['git', 'clone', '-q', 'https://github.com/sandyresearch/parcae', '/tmp/parcae_src'])
        subprocess.call(['git', '-C', '/tmp/parcae_src', 'checkout', '-q', PARCAE_COMMIT])
        subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'datasets'])
        env0['PARCAE_SRC'] = '/tmp/parcae_src'
    elif NAME == 'retro':
        subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.51.0', 'datasets'])
        cfg = [p for p in glob.glob('/kaggle/input/**/config.json', recursive=True)
               if os.path.exists(os.path.join(os.path.dirname(p), 'raven_modeling_minimal.py'))]
        assert len(cfg) == 1, cfg
        env0['RETRO_DIR'] = os.path.dirname(cfg[0])
    else:
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
    subprocess.call([sys.executable, '-c', "from datasets import load_dataset; [load_dataset('openai/gsm8k', 'main', split=s) for s in ('train', 'test')]"])
    print('setup done after %ds' % (time.time() - T0), flush=True)
    procs = []
    for gpu, args in enumerate(JOBS):
        od = '/kaggle/working/gpu%d' % gpu; os.makedirs(od, exist_ok=True)
        env = dict(env0, CUDA_VISIBLE_DEVICES=str(gpu), OUT_DIR=od)
        out = open(os.path.join(od, 'stdout.txt'), 'w')
        procs.append((subprocess.Popen([sys.executable, '-u', 'v203_heldout_lost.py'] + args, stdout=out,
                                       stderr=subprocess.STDOUT, env=env), out))
        time.sleep(30)
    for p, out in procs:
        rc = p.wait(); out.close(); print('job exit', rc, 'elapsed %ds' % (time.time() - T0), flush=True)
    subprocess.call('rm -rf /kaggle/working/gpu*/retro_model', shell=True)
    for gpu in range(len(JOBS)):
        print('---- gpu%d (tail)\n%s' % (gpu, open('/kaggle/working/gpu%d/stdout.txt' % gpu).read()[-6000:]), flush=True)
    with open('v203_analysis.txt', 'w') as f:
        f.write('analysis needs v201 outputs: run v203_heldout_lost.py --analyze locally')
    print(open('v203_analysis.txt').read(), flush=True)
    print('total %ds' % (time.time() - T0), flush=True)
'''

for name, acct, files, jobs in KERNELS:
    emb = {}
    for f in files:
        b = open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n')
        emb[f] = base64.b64encode(zlib.compress(b, 9)).decode()
        print('  %s sha256 (LF) %s' % (f, hashlib.sha256(b).hexdigest()[:12]))
    dd = os.path.join(HERE, 'kaggle-kernel-v203-%s' % name); os.makedirs(dd, exist_ok=True)
    src = 'FILES = %r\nPARCAE_COMMIT = %r\nIFM_REV = %r\nJOBS = %r\nNAME = %r\n' % (emb, PARCAE_COMMIT, IFM_REV, jobs, name) + body
    open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write(src)
    meta = dict(id='%s/astrocyte-v203-%s' % (acct, name), title='astrocyte-v203-%s' % name, code_file='run.py',
                language='python', kernel_type='script', is_private=True, enable_gpu=True, enable_internet=True)
    if name == 'retro':
        meta['dataset_sources'] = ['%s/retro-smollm2-healed' % acct]
    json.dump(meta, open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
    print('built', dd, '->', meta['id'], '| gpu jobs:', [' '.join(j) for j in jobs])
