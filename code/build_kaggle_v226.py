"""Build the v226 Kaggle kernel (fine-tune seeds 2012, 2013 for S15-S17 on the Part II models;
docs/prereg_v226_ft_seeds.md). Build only; does NOT push.
  ifm (one kernel, T4 x2): gpu0 pln5 seed 2012 then 2013, gpu1 r5 seed 2012 then 2013 (weights as v200 / v218)
Scoring (locally, after download):
  python v226_ft_seeds.py --analyze results_v210 results_v211 results_v218 results_v226
Push (NOT done here): kaggle kernels push -p kaggle-kernel-v226-ifm --accelerator NvidiaTeslaT4
"""
import base64, hashlib, json, os, py_compile, sys, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
IFM_REV = {'pln5': ('IFM/LoopedLM-P2-huginn-s-fixed-pln5', 'c581f369846e74b40072cae77ef19fa51437dfa7'),
           'r5': ('IFM/LoopedLM-P2-huginn-s-fixed-r5', '17cdf78e0d9fd8d19bba07ed800875e1de5675fd')}
FILES = ['v226_ft_seeds.py', 'v218_robustness.py', 'v210_followups.py', 'v211_k0_control.py', 'v201_batch_lost.py',
         'v200_ifm_trunc.py', 'ifm_looped.py']
ACCOUNT = sys.argv[1] if len(sys.argv) > 1 else 'aurmerin'
LANES = [[['--model', 'pln5', '--seed', '2012', '--weights', '/tmp/ifm_pln5'],
          ['--model', 'pln5', '--seed', '2013', '--weights', '/tmp/ifm_pln5']],
         [['--model', 'r5', '--seed', '2012', '--weights', '/tmp/ifm_r5'],
          ['--model', 'r5', '--seed', '2013', '--weights', '/tmp/ifm_r5']]]

body = r'''import os, shutil, subprocess, sys, threading, time, zlib, base64


def lane(gpu, arg_lists):
    for args in arg_lists:
        od = '/kaggle/working/gpu%d/%s_%s' % (gpu, args[1], args[3]); os.makedirs(od, exist_ok=True)
        env = dict(os.environ, PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES=str(gpu), OUT_DIR=od)
        with open(os.path.join(od, 'stdout.txt'), 'w') as out:      # stream to the console (live log) and the file
            p = subprocess.Popen([sys.executable, '-u', 'v226_ft_seeds.py'] + args, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, env=env, text=True, bufsize=1)
            for line in p.stdout:
                print('[gpu%d %s %s] %s' % (gpu, args[1], args[3], line), end='', flush=True); out.write(line); out.flush()
            rc = p.wait()
        print('gpu%d %s exit %d elapsed %ds' % (gpu, args[:4], rc, time.time() - T0), flush=True)


if __name__ == '__main__':
    T0 = time.time(); os.chdir('/kaggle/working')
    for fn, e in FILES.items():
        open(fn, 'wb').write(zlib.decompress(base64.b64decode(e)))
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
    subprocess.call([sys.executable, '-c', "from datasets import load_dataset; [load_dataset('openai/gsm8k', 'main', split=s) for s in ('train', 'test')]; load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='test')"])
    print('setup done after %ds' % (time.time() - T0), flush=True)
    th = []
    for gpu, arg_lists in enumerate(LANES):
        t = threading.Thread(target=lane, args=(gpu, arg_lists)); t.start(); th.append(t); time.sleep(30)
    for t in th:
        t.join()
    print('total %ds' % (time.time() - T0), flush=True)
'''

if __name__ == '__main__':
    emb = {}
    for f in FILES:
        b = open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n')
        emb[f] = base64.b64encode(zlib.compress(b, 9)).decode()
        print('  %s sha256 (LF) %s' % (f, hashlib.sha256(b).hexdigest()[:12]))
    dd = os.path.join(HERE, 'kaggle-kernel-v226-ifm'); os.makedirs(dd, exist_ok=True)
    src = 'FILES = %r\nIFM_REV = %r\nLANES = %r\n' % (emb, IFM_REV, LANES) + body
    path = os.path.join(dd, 'run.py')
    open(path, 'w', encoding='utf-8', newline='\n').write(src)
    py_compile.compile(path, doraise=True)
    json.dump(dict(id='%s/astrocyte-v226-ifm' % ACCOUNT, title='astrocyte-v226-ifm', code_file='run.py', language='python',
                   kernel_type='script', is_private=True, enable_gpu=True, enable_internet=True),
              open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
    print('built + compiled', dd, '->', '%s/astrocyte-v226-ifm' % ACCOUNT)
