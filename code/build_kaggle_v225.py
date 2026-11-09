"""Build the v225 Ouro kernels (per-arm LRs on a split of GSM8K test; docs/prereg_v225_lr_per_arm.md). Build only; does
NOT push. Each kernel is one process at a time on both T4s (the model is split across them, as v219): the selftest, then
its training cells in order.
  python build_kaggle_v225.py select                      the selection grid on seed 2251001, 3 kernels (sel-a/b/c)
  python build_kaggle_v225.py cells NAME ACCOUNT CELLS    any cells, e.g. an edge extension printed by --select-ouro
  python build_kaggle_v225.py score LR_FULL LR_K2 LR_K1 LR_FLOOR [SEEDS]
                                                          one kernel per scored seed (default 2251011-2251015)
The Part II runs (Part P) use build_kaggle_v200.py cells mode, unchanged, with tags v225sel / v225score.
Push (NOT done here): kaggle kernels push -p kaggle-kernel-v225-<name> --accelerator NvidiaTeslaT4
"""
import base64, hashlib, json, os, py_compile, sys, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ['v225_lr_per_arm.py', 'v219_ouro_cost.py', 'loop_grad_probe.py']
SEL = 2251001
c = lambda k, lr, s=SEL: '%d:%g:%d' % (k, lr, s)
SELECT = {   # balanced by v219's per-run times: full 2.9 h, k = 2 1.8 h, k = 1 1.25 h, floor 0.65 h (+ 6 min eval each)
    'sel-a': ('aurshalevmerin', [c(4, 1e-4), c(4, 3e-5), c(4, 3e-4)]),
    'sel-b': ('aurmerin', [c(2, 1e-4), c(2, 3e-5), c(2, 3e-4), c(0, 1e-4)]),
    'sel-c': ('aurmerin', [c(1, 1e-4), c(1, 3e-5), c(1, 3e-4), c(0, 3e-5), c(0, 3e-4)]),
}

body = r'''import os, subprocess, sys, time, zlib, base64
if __name__ == '__main__':
    T0 = time.time(); os.chdir('/kaggle/working')
    for fn, e in FILES.items():
        open(fn, 'wb').write(zlib.decompress(base64.b64decode(e)))
    subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.54.1', 'datasets'])
    subprocess.call([sys.executable, '-c', "from datasets import load_dataset; load_dataset('openai/gsm8k', 'main')"])
    od = '/kaggle/working/out'; os.makedirs(od, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED='1', LOOP_MODEL='ouro', PYTORCH_ALLOC_CONF='expandable_segments:True',
               CUDA_VISIBLE_DEVICES='0,1', OUT_DIR=od)
    print('setup done after %ds' % (time.time() - T0), flush=True)
    for args in [['--phase', 'selftest'], ['--phase', 'train', '--cells', CELLS]]:
        with open(os.path.join(od, 'stdout.txt'), 'a') as out:          # stream to the console (live log) and the file
            p = subprocess.Popen([sys.executable, '-u', 'v225_lr_per_arm.py'] + args, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, env=env, text=True, bufsize=1)
            for line in p.stdout:
                print(line, end='', flush=True); out.write(line); out.flush()
            rc = p.wait()
        print('step', args[:2], 'exit', rc, 'elapsed %ds' % (time.time() - T0), flush=True)
        if args[1] == 'selftest' and rc != 0:
            print('SELFTEST FAILED: stopping', flush=True); break
    print('total %ds' % (time.time() - T0), flush=True)
'''


def build(name, acct, cells):
    emb = {}
    for f in FILES:
        b = open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n')
        emb[f] = base64.b64encode(zlib.compress(b, 9)).decode()
    dd = os.path.join(HERE, 'kaggle-kernel-v225-%s' % name); os.makedirs(dd, exist_ok=True)
    path = os.path.join(dd, 'run.py')
    open(path, 'w', encoding='utf-8', newline='\n').write('FILES = %r\nCELLS = %r\n' % (emb, ','.join(cells)) + body)
    py_compile.compile(path, doraise=True)
    json.dump(dict(id='%s/astrocyte-v225-%s' % (acct, name), title='astrocyte-v225-%s' % name, code_file='run.py',
                   language='python', kernel_type='script', is_private=True, enable_gpu=True, enable_internet=True),
              open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
    print('built + compiled', dd, acct, cells)


if __name__ == '__main__':
    for f in FILES:
        b = open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n')
        print('  %s sha256 (LF) %s' % (f, hashlib.sha256(b).hexdigest()[:12]))
    mode = sys.argv[1]
    if mode == 'select':
        for name, (acct, cells) in SELECT.items():
            build(name, acct, cells)
    elif mode == 'cells':
        build(sys.argv[2], sys.argv[3], sys.argv[4].split(','))
    elif mode == 'score':
        lrs = dict(zip((4, 2, 1, 0), (float(x) for x in sys.argv[2:6])))
        seeds = [int(s) for s in sys.argv[6].split(',')] if len(sys.argv) > 6 else list(range(2251011, 2251016))
        for i, s in enumerate(seeds):
            build('s%d' % (s - 2251010), ('aurshalevmerin', 'aurmerin')[i % 2], [c(k, lrs[k], s) for k in (4, 2, 1, 0)])
