"""Build Kaggle kernels for v182 (EXPLORATORY review checks; v182_review_checks.py). Build only; does not push.
  parcae kernel: GPU 0 part-of-loop vs all-parameter at R 16 (Table 1 protocol), then loop_grad_probe.py (the starred
                 rows' script) on Parcae-140m: subset / whole loop / subset in bf16; GPU 1 Table 2's Parcae cells in float64
  retro kernel (needs dataset retro-smollm2-healed): GPU 0 part-of-loop vs all-parameter at R 8; GPU 1 Table 2's cells
  python build_kaggle_v182.py ACCOUNT
"""
import base64, hashlib, json, os, sys, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ACCT = sys.argv[1] if len(sys.argv) > 1 else 'aurmerin'
PARCAE_COMMIT = '69284c13746e849104f738d6d1a347b1f457df76'
PARCAE_JOBS = [[('', ['--model', 'parcae', '--tag', 'p_subset', '--subset', '16:1,2,4,8'])]
               + [('probe_' + v, ['--model', 'parcae', '--probe', v]) for v in ('subset', 'loop', 'subset_bf16')],
               [('', ['--model', 'parcae', '--tag', 'p_table2', '--table2', '32:1.0:1,2,4', '48:0.4:4', '64:0.3:8'])]]
RETRO_JOBS = [[('', ['--model', 'retro', '--tag', 'r_subset', '--subset', '8:1,2,4'])],
              [('', ['--model', 'retro', '--tag', 'r_table2', '--table2', '8:1.0:1,2', '8:0.5:1,2', '8:0.3:1,2'])]]
KERNELS = [('parcae', ['v182_review_checks.py', 'loop_grad_probe.py', 'v163_parcae_damped_trunc.py', 'v162_parcae_trunc_train.py'], PARCAE_JOBS),
           ('retro', ['v182_review_checks.py', 'v167_retrofit_trunc.py'], RETRO_JOBS)]

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
    for gpu, seq in enumerate(JOBS):             # one shell chain per GPU: its jobs run one after another
        cmds = []
        for sub, args in seq:
            od = '/kaggle/working/gpu%d/%s' % (gpu, sub); os.makedirs(od, exist_ok=True)
            cmds.append('OUT_DIR=%s %s -u v182_review_checks.py %s' % (od, sys.executable, ' '.join(args)))
        env = dict(env0, CUDA_VISIBLE_DEVICES=str(gpu), BUDGET_S='3600')
        out = open('v182_stdout_gpu%d.txt' % gpu, 'w')
        procs.append((subprocess.Popen(' ; '.join(cmds), shell=True, stdout=out, stderr=subprocess.STDOUT, env=env), out))
        time.sleep(30)
    for p, out in procs:
        rc = p.wait(); out.close(); print('job exit', rc, 'elapsed %ds' % (time.time() - t0), flush=True)
    for gpu in range(len(JOBS)):
        print(open('v182_stdout_gpu%d.txt' % gpu).read()[-6000:], flush=True)
    subprocess.call('rm -rf /kaggle/working/gpu*/retro_model', shell=True)
'''

for name, files, jobs in KERNELS:
    emb = {}
    for f in files:
        b = open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n')
        emb[f] = base64.b64encode(zlib.compress(b, 9)).decode()
        print('  %s sha256 (LF) %s' % (f, hashlib.sha256(b).hexdigest()[:12]))
    dd = os.path.join(HERE, 'kaggle-kernel-v182-%s' % name); os.makedirs(dd, exist_ok=True)
    src = 'FILES = %r\nPARCAE_COMMIT = %r\nJOBS = %r\nNAME = %r\n' % (emb, PARCAE_COMMIT, jobs, name) + body
    open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write(src)
    meta = dict(id='%s/astrocyte-v182-%s' % (ACCT, name), title='astrocyte-v182-%s' % name, code_file='run.py', language='python',
                kernel_type='script', is_private=True, enable_gpu=True, enable_internet=True)
    if name == 'retro':
        meta['dataset_sources'] = ['%s/retro-smollm2-healed' % ACCT]
    json.dump(meta, open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
    print('built', dd, ACCT, [len(j) for j in jobs])
