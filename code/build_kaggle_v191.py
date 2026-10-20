"""Build the v191 Kaggle kernel (plan docs/v191_parcae_contrib_plan.md). Build only; does not push.
Embeds v191_parcae_contrib.py + the frozen v164/v163/v162 Parcae machinery; clones Parcae (pinned, as v164).
GPU 0: Parcae-370m R 32 then 16. GPU 1: Parcae-140m R 32 then 16. Per-loop contributions for every k.
"""
import base64, hashlib, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ['v191_parcae_contrib.py', 'v164_parcae_predict.py', 'v163_parcae_damped_trunc.py', 'v162_parcae_trunc_train.py']
PARCAE_COMMIT = '69284c13746e849104f738d6d1a347b1f457df76'
JOBS = [[['--repo', 'SandyResearch/parcae-370m', '--R', '32,16', '--tag', '370m']],
        [['--repo', 'SandyResearch/parcae-140m', '--R', '32,16', '--tag', '140m']]]

emb = {}
for f in FILES:
    b = open(os.path.join(HERE, f), 'rb').read().replace(b'\r\n', b'\n')
    emb[f] = base64.b64encode(zlib.compress(b, 9)).decode()
    print('  %s sha256 (LF) %s' % (f, hashlib.sha256(b).hexdigest()[:12]))

body = r'''import subprocess, sys, os, base64, zlib, time, threading
def run_gpu(gpu, cmds):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PARCAE_SRC='/tmp/parcae_src', PYTHONUNBUFFERED='1')
    with open('v191_stdout_gpu%d.txt' % gpu, 'w') as out:
        for args in cmds:
            rc = subprocess.call([sys.executable, '-u', 'v191_parcae_contrib.py'] + args, stdout=out, stderr=subprocess.STDOUT, env=env)
            print('gpu%d %s exit %d after %ds' % (gpu, args[-1], rc, time.time() - T0), flush=True)
if __name__ == '__main__':
    os.chdir('/kaggle/working')
    for fn, e in FILES.items():
        open(fn, 'wb').write(zlib.decompress(base64.b64decode(e)))
    subprocess.call(['git', 'clone', '-q', 'https://github.com/sandyresearch/parcae', '/tmp/parcae_src'])
    subprocess.call(['git', '-C', '/tmp/parcae_src', 'checkout', '-q', PARCAE_COMMIT])
    subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', 'datasets'])
    T0 = time.time(); th = []
    for gpu, cmds in enumerate(JOBS):
        t = threading.Thread(target=run_gpu, args=(gpu, cmds)); t.start(); th.append(t); time.sleep(60)
    for t in th:
        t.join()
    for gpu in range(len(JOBS)):
        print(open('v191_stdout_gpu%d.txt' % gpu).read()[-4000:], flush=True)
'''

dd = os.path.join(HERE, 'kaggle-kernel-v191-parcae-contrib'); os.makedirs(dd, exist_ok=True)
src = 'FILES = %r\nPARCAE_COMMIT = %r\nJOBS = %r\n' % (emb, PARCAE_COMMIT, JOBS) + body
open(os.path.join(dd, 'run.py'), 'w', encoding='utf-8', newline='\n').write(src)
json.dump(dict(id='aurshalevmerin/astrocyte-v191-parcae-contrib', title='astrocyte-v191-parcae-contrib', code_file='run.py',
               language='python', kernel_type='script', is_private=True, enable_gpu=True, enable_internet=True),
          open(os.path.join(dd, 'kernel-metadata.json'), 'w'), indent=1)
print('built', dd)
