"""Build the v219 kernels (Ouro-1.4B truncation cost). Build only; does NOT push.
Each kernel is one process at a time on both T4s (the model is split across them): the selftest, then its phases.
  python build_kaggle_v219.py lr        LR pilot before the freeze (seed 9001; full backprop and floor LR grids;
                                        the step-0 lost-gradient measurement)
  python build_kaggle_v219.py study     the scored study, one seed per kernel (after the freeze)
Push (NOT done here): kaggle kernels push -p kaggle-kernel-v219-<name> --accelerator NvidiaTeslaT4
"""
import base64, hashlib, json, os, py_compile, sys, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ['v219_ouro_cost.py', 'loop_grad_probe.py']
PILOT_SEED = 9001
KERNELS = {
    'lr': {
        'lr-a': ('aurshalevmerin', [['--phase', 'measure'],
                                    ['--phase', 'train', '--cells', '4:1e-05:%d,4:3e-06:%d' % (PILOT_SEED, PILOT_SEED)]]),
        'lr-b': ('aurmerin', [['--phase', 'train', '--cells', '4:3e-05:%d,0:0.0001:%d,0:0.0003:%d,0:0.001:%d,0:0.003:%d'
                               % ((PILOT_SEED,) * 5)]]),
    },
    # both grids' best test loss was at an edge (full 3e-5, floor 1e-4): extend each; val (the last 473 GSM8K train
    # problems) disagrees with test for full backprop, so the base model's val and test loss are measured too
    'lr2': {
        'lr2-a': ('aurshalevmerin', [['--phase', 'eval0'],
                                     ['--phase', 'train', '--cells', '4:0.0001:%d,0:3e-05:%d,0:1e-05:%d' % ((PILOT_SEED,) * 3)]]),
        'lr2-b': ('aurmerin', [['--phase', 'train', '--cells', '4:0.0003:%d' % PILOT_SEED]]),
    },
}
# the scored study (after the freeze): one seed per kernel, all four cells, full backprop first
STUDY_CELLS = lambda s: [['--phase', 'train', '--cells', '4:0.0001:%d,2:0.0001:%d,1:0.0001:%d,0:0.0001:%d' % ((s,) * 4)]]
KERNELS['study'] = {'s1': ('aurshalevmerin', STUDY_CELLS(2191001)), 's2': ('aurmerin', STUDY_CELLS(2191002)),
                    's3': ('aurmerin', STUDY_CELLS(2191003))}
KERNELS['study-more'] = {'s%d' % i: ('aurshalevmerin' if i % 2 else 'aurmerin', STUDY_CELLS(2191000 + i)) for i in (4, 5, 6, 7)}

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
    for args in [['--phase', 'selftest']] + PHASES:
        with open(os.path.join(od, 'stdout.txt'), 'a') as out:          # stream to the console (live log) and the file
            p = subprocess.Popen([sys.executable, '-u', 'v219_ouro_cost.py'] + args, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, env=env, text=True, bufsize=1)
            for line in p.stdout:
                print(line, end='', flush=True); out.write(line); out.flush()
            rc = p.wait()
        print('step', args[:2], 'exit', rc, 'elapsed %ds' % (time.time() - T0), flush=True)
        if args[1] == 'selftest' and rc != 0:
            print('SELFTEST FAILED: stopping', flush=True); break
    print('total %ds' % (time.time() - T0), flush=True)
'''

if __name__ == '__main__':
    which = sys.argv[1]
    emb = {}
    for f in FILES:
        b = open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n')
        emb[f] = base64.b64encode(zlib.compress(b, 9)).decode()
        print('  %s sha256 (LF) %s' % (f, hashlib.sha256(b).hexdigest()[:12]))
    for name, (acct, phases) in KERNELS[which].items():
        dd = os.path.join(HERE, 'kaggle-kernel-v219-%s' % name); os.makedirs(dd, exist_ok=True)
        path = os.path.join(dd, 'run.py')
        open(path, 'w', encoding='utf-8', newline='\n').write('FILES = %r\nPHASES = %r\n' % (emb, phases) + body)
        py_compile.compile(path, doraise=True)
        json.dump(dict(id='%s/astrocyte-v219-%s' % (acct, name), title='astrocyte-v219-%s' % name, code_file='run.py',
                       language='python', kernel_type='script', is_private=True, enable_gpu=True, enable_internet=True),
                  open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
        print('built + compiled', dd, acct, phases)
