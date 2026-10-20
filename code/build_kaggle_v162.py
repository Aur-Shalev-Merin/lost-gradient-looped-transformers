"""Build Kaggle GPU kernels for v162 (Parcae-140m truncated-loop training on GSM8K).
Prereg: docs/prereg_v162_parcae_truncation_training_rerun.md. Build only; this script does not push.

  python build_kaggle_v162.py pilot     PRE-FREEZE pilot (controls only): selftest, frozen-model measurement at R = 8, 16,
                                        step-0 losses, and an LR sweep of the two CONTROL arms (k = R full backprop,
                                        k = 0 no loop credit) on pilot seed 9001. No truncated cell (0 < k < R) is run.
  python build_kaggle_v162.py tune      wave 1 (after the freeze): every cell x LR grid {1e-4, 3e-4, 9e-4} x tuning seeds
  python build_kaggle_v162.py ext       edge-rule extension: cells listed in v162_selection.json 'extend', tuning seeds
  python build_kaggle_v162.py eval      wave 2: every cell at its selected LR (v162_selection.json) x 5 evaluation seeds
Study waves use 4 kernels (2 per account, the GPU-session cap) x 2 T4s = 8 GPUs; runs are packed longest-first onto
the GPU with the least estimated time (pilot timings: 200 s / 305 s at k = 0 for R = 8 / 16, ~26 s per backprop loop).

Each kernel runs two driver processes, one per T4 (CUDA_VISIBLE_DEVICES 0 / 1), and clones the Parcae source at the
pinned commit (the PyPI parcae-lm wheel is empty).
Fetch: kaggle kernels output ACCT/astrocyte-v162-NAME -p kaggle-output-v162/NAME --file-pattern "v162_.*"
"""
import base64, hashlib, json, os, sys, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = 'v162_parcae_trunc_train.py'
PARCAE_COMMIT = '69284c13746e849104f738d6d1a347b1f457df76'
A1, A2 = 'aurshalevmerin', 'aurmerin'
PILOT_LRS = ['1e-5', '3e-5', '1e-4', '3e-4']
PILOT_SEED = 9001


def cells(R, ks, lrs, seeds):
    return ','.join('%d:%d:%s:%d' % (R, k, lr, s) for k in ks for lr in lrs for s in seeds)


TUNE = [1621001, 1621002]
EVAL = [1621101, 1621102, 1621103, 1621104, 1621105]
KS = {8: [0, 1, 2, 4, 8], 16: [0, 1, 2, 4, 8, 16]}
GRID = ['1e-4', '3e-4', '9e-4']


def est(R, k):
    return {8: 200, 16: 305}[R] + 26 * k


def pack_waves(name, runs, parts='abcd'):
    """runs: list of (R, k, lr_str, seed) -> len(parts) kernels x 2 GPU jobs, longest-processing-time-first."""
    bins = [[0, []] for _ in range(2 * len(parts))]
    for r in sorted(runs, key=lambda r: -est(r[0], r[1])):
        b = min(bins, key=lambda b: b[0]); b[0] += est(r[0], r[1]); b[1].append(r)
    ks = []
    for i, part in enumerate(parts):
        jobs = []
        for g in (0, 1):
            t, rs = bins[2 * i + g]
            if rs:
                jobs.append(['--train', ','.join('%d:%d:%s:%d' % r for r in rs), '--tag', '%s%s%d' % (name, part, g)])
        if jobs:
            print('  %s-%s: est %s min' % (name, part, [round(bins[2 * i + g][0] / 60) for g in (0, 1)]))
            ks.append(('%s-%s' % (name, part), A1 if part in 'abe' else A2, jobs))
    return ks


mode = sys.argv[1] if len(sys.argv) > 1 else 'pilot'
ENVX = {}   # extra environment for the driver processes (exploratory modes only)
KERNELS = {'pilot': [('pilot', A1, [
    # v162 pilot: extended selftest (truncated fp16-vs-fp32 gradients + cache-on negative control) and the control-identity
    # run (full, R 8, 3e-4, seed 9001: must reproduce v161's 1.3759 within 0.005); the frozen-model measurement is
    # repeated to confirm the locked law values (fp32, unaffected by the bug).
    ['--selftest', '--check_r', '8', '--train', '8:8:3e-4:%d' % PILOT_SEED, '--tag', 'pilot'],
    ['--selftest', '--check_r', '16', '--measure', '8,16', '--tag', 'pilotmeasure'],
])]}
if mode == 'tune':
    KERNELS['tune'] = pack_waves('tune', [(R, k, lr, s) for R in (8, 16) for k in KS[R] for lr in GRID for s in TUNE])
elif mode in ('ext', 'eval', 'evalfinal', 'evalrest'):
    # 'evalfinal' = evaluation runs for cells whose LR is already final (not in 'extend'), run alongside 'ext';
    # 'evalrest' = evaluation runs for the extended cells after re-selection. Together they equal 'eval'.
    sel = json.load(open(os.path.join(HERE, 'v162_selection.json')))
    ext_cells = {'%d:%d' % (R, k) for R, k, lr in sel['extend']}
    allev = [(int(c.split(':')[0]), int(c.split(':')[1]), '%g' % lr, s) for c, lr in sel['selection'].items() for s in EVAL]
    if mode == 'ext':
        KERNELS[mode] = pack_waves('ext', [(R, k, '%g' % lr, s) for R, k, lr in sel['extend'] for s in TUNE], parts='e')
    elif mode == 'evalfinal':
        KERNELS[mode] = pack_waves('eval', [r for r in allev if '%d:%d' % r[:2] not in ext_cells], parts='bcd')
    elif mode == 'evalrest':
        assert not sel['extend'], 're-select after the extension first'
        KERNELS[mode] = pack_waves('evalrest', [r for r in allev if r[1] == 0], parts='e')
    else:
        assert not sel['extend'], 'run the edge extension and re-select first'
        KERNELS[mode] = pack_waves('eval', allev)
elif mode == 'rminus1':   # EXPLORATORY, post-freeze: k = R - 1 on eval seeds 1-3, compared with the existing full runs
    KERNELS[mode] = [('rminus1', A1, [['--train', ','.join('8:7:3e-4:%d' % s for s in EVAL[:3]) + ',' + '16:15:3e-4:%d' % EVAL[0], '--tag', 'rm1a'],
                                      ['--train', ','.join('16:15:3e-4:%d' % s for s in EVAL[1:3]), '--tag', 'rm1b']])]
elif mode == 'oneepoch':   # EXPLORATORY: 290 steps (< 1 pass over GSM8K, no repeated example), 2 tuning seeds
    cs = [(R, k) for R, ks in ((8, (0, 1, 2, 4, 8)), (16, (0, 1, 4, 8, 16))) for k in ks]
    KERNELS[mode] = pack_waves('onep', [(R, k, '9e-4' if k == 0 else '3e-4', s) for R, k in cs for s in TUNE], parts='bcd')
    ENVX = {'V162_STEPS': '290'}
elif mode == 'gradcheck':   # EXPLORATORY: training path vs measurement path, k = R vs R - 1, frozen model
    KERNELS[mode] = [('gradcheck', A1, [['v162_gradcheck.py', '16'], ['v162_gradcheck.py', '8']])]
elif mode == 'drift':   # EXPLORATORY, post-freeze: v162_drift_probe.py (imports the frozen driver unchanged)
    KERNELS[mode] = [('drift', A2, [['v162_drift_probe.py', '16:16,8:8', '--tag', 'full'],
                                    ['v162_drift_probe.py', '16:8,8:4', '--tag', 'trunc']])]
drv = open(os.path.join(HERE, DRIVER), 'rb').read().replace(b'\r\n', b'\n')
emb = base64.b64encode(zlib.compress(drv, 9)).decode()
extra = {}
for fn in {'drift': ['v162_drift_probe.py'], 'gradcheck': ['v162_gradcheck.py']}.get(mode, []):
    probe = open(os.path.join(HERE, fn), 'rb').read().replace(b'\r\n', b'\n')
    extra[fn] = base64.b64encode(zlib.compress(probe, 9)).decode()
print('  %s sha256 (LF-normalised) %s' % (DRIVER, hashlib.sha256(drv).hexdigest()[:12]))

body = r'''import subprocess, sys, os, base64, zlib, time
if __name__ == '__main__':
    os.chdir('/kaggle/working')
    open(DRIVER, 'wb').write(zlib.decompress(base64.b64decode(EMBEDDED)))
    for fn, e in EXTRA.items():
        open(fn, 'wb').write(zlib.decompress(base64.b64decode(e)))
    subprocess.call(['git', 'clone', '-q', 'https://github.com/sandyresearch/parcae', '/tmp/parcae_src'])
    subprocess.call(['git', '-C', '/tmp/parcae_src', 'checkout', '-q', PARCAE_COMMIT])
    subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'datasets'])
    t0 = time.time(); procs = []
    for gpu, args in enumerate(JOBS):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PARCAE_SRC='/tmp/parcae_src', PYTHONUNBUFFERED='1')
        env.update(ENVX)
        out = open('v162_stdout_gpu%d.txt' % gpu, 'w')
        cmd = args if args[0].endswith('.py') else [DRIVER] + args
        procs.append((subprocess.Popen([sys.executable, '-u'] + cmd, stdout=out, stderr=subprocess.STDOUT, env=env), out))
        time.sleep(60)          # stagger the HF downloads
    for p, out in procs:
        rc = p.wait(); out.close(); print('job exit', rc, 'elapsed %ds' % (time.time() - t0), flush=True)
    for gpu in range(len(JOBS)):
        print(open('v162_stdout_gpu%d.txt' % gpu).read()[-6000:], flush=True)
'''

for name, acct, jobs in KERNELS[mode]:
    d = os.path.join(HERE, 'kaggle-kernel-v162-%s' % name); os.makedirs(d, exist_ok=True)
    src = ('DRIVER = %r\nPARCAE_COMMIT = %r\nJOBS = %r\nEMBEDDED = %r\nEXTRA = %r\nENVX = %r\n'
           % (DRIVER, PARCAE_COMMIT, jobs, emb, extra, ENVX)) + body
    open(os.path.join(d, 'run.py'), 'w', encoding='utf-8', newline='\n').write(src)
    json.dump(dict(id='%s/astrocyte-v162-%s' % (acct, name), title='astrocyte-v162-%s' % name, code_file='run.py',
                   language='python', kernel_type='script', is_private=True, enable_gpu=True, enable_internet=True),
              open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=1)
    print('built', d, acct, [len(j) for j in jobs])
