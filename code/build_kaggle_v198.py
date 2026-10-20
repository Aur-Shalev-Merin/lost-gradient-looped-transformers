"""Build the v198 Kaggle kernel (fade of a Part II looped LM, frozen; docs/v198_ifm_port_notes.md). Build only; does
not push. One kernel on aurmerin, T4 x2:
  stage 1 (both GPUs at once): GPU 0 runs the selftest on the real model; GPU 1 runs the reference check, our port vs
          their xLLM code at a pinned commit (their CUDA extension replaced by the PyTorch RMSNorm shim in
          v198_ifm_fade.py; their attention on its plain PyTorch path).
  stage 2, unless the selftest fails or the reference check FAILs (a refcheck ERROR means their code would not
          import or run here; it is reported and does not block):
          GPU 0: sequences 0-11 + direction after Adam at a = 1;  GPU 1: sequences 12-15 + Adam at a = 0.5, 0.3
          (balanced from timings: about 15-17 minutes each on a T4).
  stage 3: --analyze over /kaggle/working, also saved to v198_analysis.txt.
Weights: IFM/LoopedLM-P2-<variant> at a pinned revision, snapshot_download into a new directory under /tmp (not saved
as kernel output); both loaders verify it against artifact_manifest.json.

  python build_kaggle_v198.py [--variant huginn-s-fixed-pln5] [--b 24]
Push (NOT done here): kaggle kernels push -p kaggle-kernel-v198-ifm --accelerator NvidiaTeslaT4
"""
import argparse, base64, hashlib, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ['v198_ifm_fade.py', 'ifm_looped.py']
XLLM_COMMIT = 'df0ed1ea415b55f0a944a291f90da9d47c98b5d5'          # github.com/ifm-ai/xllm-loop, 2026-10-06
VARIANTS = {                                                       # Hugging Face revisions, read 2026-10-14
    'huginn-s-fixed-pln5': 'c581f369846e74b40072cae77ef19fa51437dfa7',          # Table 2 S, PLN-5, backprop last 10
    'huginn-s-fixed-r5': '17cdf78e0d9fd8d19bba07ed800875e1de5675fd',            # Table 2 S, fixed R = 5, full BPTT
    'huginn-s-learned-entropy0': '73ff74fa7844892ab4a3d356c80bc6b2ca9833f9',    # Table 2 S, learned prior, lambda 0
    'huginn-s-learned-entropy0p01': 'e83177781693848c631142dcf9de3b0688e328e1',  # Table 2 S, learned prior, 0.01
}
ap = argparse.ArgumentParser()
ap.add_argument('--variant', default='huginn-s-fixed-pln5', choices=sorted(VARIANTS))
ap.add_argument('--b', type=int, default=24)
ap.add_argument('--account', default='aurmerin', choices=('aurmerin', 'aurshalevmerin'))
ap.add_argument('--predictors', action='store_true', help='v200 predictor inputs: embedding frozen, R = 5, a = 1')
A = ap.parse_args()
SHORT = A.variant.replace('huginn-s-', '').replace('learned-', 'l-').replace('fixed-', '')   # pln5, r5, l-entropy0p01
SHORT += '-pred' if A.predictors else ''
SLUG = 'astrocyte-v198-ifm-' + SHORT                                  # one kernel per variant, so variants can run in parallel
emb = {}
for f in FILES:
    b = open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n')
    emb[f] = base64.b64encode(zlib.compress(b, 9)).decode()
    print('  %s sha256 (LF) %s' % (f, hashlib.sha256(b).hexdigest()[:12]))
W, X = '/tmp/ifm_weights', '/tmp/xllm_ref'
STAGE1 = [[['--weights', W, '--selftest', '--tag', 'gpu0']], [['--weights', W, '--refcheck', X]]]
STAGE2 = [[['--weights', W, '--seqs', '0:12', '--adam', '1.0', '--tag', 'g0']],
          [['--weights', W, '--seqs', '12:16', '--adam', '0.5,0.3', '--tag', 'g1']]]
EXTRA_ENV = {}
if A.predictors:                                   # the port was checked against the reference code in v198
    STAGE1 = [[['--weights', W, '--selftest', '--tag', 'gpu0']], []]
    STAGE2 = [[['--weights', W, '--seqs', '0:16', '--adam', '', '--tag', 'p0']],
              [['--weights', W, '--seqs', '0:0', '--adam', '1.0', '--tag', 'p1']]]
    EXTRA_ENV = dict(V198_FREEZE_EMBED='1', V198_RLIST='5', V198_ALPHAS='1.0')

RUN = r'''FILES = %r
REPO, REV, XLLM_COMMIT, W, X, B = %r, %r, %r, %r, %r, %d
STAGE1 = %r
STAGE2 = %r
EXTRA_ENV = %r

import base64, json, os, subprocess, sys, threading, time, zlib


def run_gpu(gpu, cmds, stage):
    out = '/kaggle/working/gpu%%d' %% gpu
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OUT_DIR=out, PYTHONUNBUFFERED='1', V198_B=str(B),
               PYTORCH_ALLOC_CONF='expandable_segments:True', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    with open(os.path.join(out, 'stdout_%%s.txt' %% stage), 'a') as log:
        for args in cmds:
            rc = subprocess.call([sys.executable, '-u', 'v198_ifm_fade.py'] + args, stdout=log, stderr=subprocess.STDOUT,
                                 env=env)
            print('%%s gpu%%d %%s exit %%d after %%ds' %% (stage, gpu, ' '.join(args[2:]), rc, time.time() - T0), flush=True)


def stage(jobs, name):
    import torch
    n = torch.cuda.device_count()
    if n < 2:                                          # one GPU: everything in order on GPU 0
        jobs = [[c for cmds in jobs for c in cmds]]
    th = []
    for gpu, cmds in enumerate(jobs):
        t = threading.Thread(target=run_gpu, args=(gpu, cmds, name)); t.start(); th.append(t); time.sleep(20)
    for t in th:
        t.join()
    for gpu in range(len(jobs)):
        p = '/kaggle/working/gpu%%d/stdout_%%s.txt' %% (gpu, name)
        if os.path.exists(p):
            print('---- %%s gpu%%d (tail)\n%%s' %% (name, gpu, open(p).read()[-5000:]), flush=True)


def status(pattern):
    import glob
    fs = glob.glob('/kaggle/working/**/' + pattern, recursive=True)
    return json.load(open(fs[0])).get('status') if fs else 'MISSING'


if __name__ == '__main__':
    T0 = time.time()
    os.environ.update(EXTRA_ENV)
    os.chdir('/kaggle/working')
    for fn, e in FILES.items():
        open(fn, 'wb').write(zlib.decompress(base64.b64decode(e)))
    subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'datasets', 'huggingface_hub', 'safetensors',
                     'tokenizers', 'einops', 'typeguard>=4.5.0'])
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=REPO, revision=REV, local_dir=W)
    import shutil
    for root, _, fs in os.walk(W):                     # their artifact loader rejects symbolic links
        for f in fs:
            p = os.path.join(root, f)
            if os.path.islink(p):
                t = os.path.realpath(p); os.unlink(p); shutil.copyfile(t, p)
    subprocess.call(['git', 'clone', '-q', 'https://github.com/ifm-ai/xllm-loop', X])
    subprocess.call(['git', '-C', X, 'checkout', '-q', XLLM_COMMIT])
    print('xllm-loop at', subprocess.run(['git', '-C', X, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip())
    subprocess.call([sys.executable, '-c', "from datasets import load_dataset; load_dataset('openai/gsm8k', 'main', split='train')"])
    print('setup done after %%ds' %% (time.time() - T0), flush=True)
    stage(STAGE1, 'stage1')
    st, rc = status('v198_selftest_gpu0.json'), status('v198_refcheck.json')
    print('selftest %%s | refcheck %%s' %% (st, rc), flush=True)
    if st == 'PASS' and rc != 'FAIL':
        stage(STAGE2, 'stage2')
    else:
        print('measurement NOT run: the selftest must PASS and the reference check must not FAIL', flush=True)
    with open('v198_analysis.txt', 'w') as f:
        subprocess.call([sys.executable, 'v198_ifm_fade.py', '--analyze', '/kaggle/working'], stdout=f, stderr=subprocess.STDOUT)
    print(open('v198_analysis.txt').read(), flush=True)
    print('total %%ds' %% (time.time() - T0), flush=True)
'''

dd = os.path.join(HERE, 'kaggle-kernel-v198-ifm-' + SHORT); os.makedirs(dd, exist_ok=True)
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write(
    RUN % (emb, 'IFM/LoopedLM-P2-' + A.variant, VARIANTS[A.variant], XLLM_COMMIT, W, X, A.b, STAGE1, STAGE2, EXTRA_ENV))
json.dump(dict(id='%s/%s' % (A.account, SLUG), title=SLUG, code_file='run.py',
               language='python', kernel_type='script', is_private=True, enable_gpu=True, enable_internet=True),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd, '| model IFM/LoopedLM-P2-%s @ %s | B = %d | not pushed' % (A.variant, VARIANTS[A.variant][:8], A.b))
