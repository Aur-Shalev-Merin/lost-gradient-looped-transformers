"""Check the file hashes recorded in the pre-registrations against the files in this repository.

Shared by the release repositories of "How Much Gradient Does Truncated Backprop Lose in Looped Transformers?" and
"Present but Hard to Read".

Each pre-registration (preregs/*.md) records frozen files as `name` followed by the first 12 to 16 hex characters of the
file's SHA-256, computed on LF line endings (binary .npz files as stored), for example
    `v210_followups.py` 619235c6b191      `name.py` (sha256 (LF) 1a2b...)      `name.py` is now f71f7b33174c
and amendments as `name` ... old → new (the arrow may wrap onto the next line, with or without bold).

Within one pre-registration a file passes if
  - its current version matches any hash recorded for it there (an amendment's hash supersedes the frozen one), or
  - an earlier version kept as code/frozen_versions/<stem>.<hash><ext> matches one of those hashes, or
  - the hash is a kernel's "driver sha", one SHA-256 over several files concatenated in order (COMBINED below).
A recorded file that is not in the repository is a problem.

    python tools/verify_hashes.py            (from the repository root; exit status 1 if any problem)
"""
import glob, hashlib, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = r'(?:py|json|npz|md|txt)'
PAT = re.compile(r'`([^`\s]+\.%s)`[^`]{0,60}?\b([0-9a-f]{12,16})\b' % EXT)              # `name` ... hash
PAT2 = re.compile(r'`([^`\s]+\.%s)`[^`]{0,80}?→\s*\**\s*([0-9a-f]{12,16})\b' % EXT)     # `name` ... old → new
COMBINED = {   # driver sha = sha256(file1 + file2 + ...) over LF bytes, as written by the kernels' sha() helper
    'feaf9a0ba438': ['v164_parcae_predict.py', 'v163_parcae_damped_trunc.py', 'v162_parcae_trunc_train.py'],
    'd5c2647126de': ['v163_parcae_damped_trunc.py', 'v162_parcae_trunc_train.py'],
}


def data(path):
    b = open(path, 'rb').read()
    return b if path.endswith('.npz') else b.replace(b'\r\n', b'\n')


def sha(path, n):
    return hashlib.sha256(data(path)).hexdigest()[:n]


def find(name):
    base = os.path.basename(name)
    for d in ('code', 'results', 'paper', 'preregs', 'results_docs'):
        hits = [h for h in glob.glob(os.path.join(ROOT, d, '**', base), recursive=True) if 'frozen_versions' not in h]
        hits = [h for h in hits if h.replace('\\', '/').endswith(name.replace('\\', '/'))] or hits
        if hits:
            return hits[0]
    return None


def frozen(name, h):
    stem, ext = os.path.splitext(os.path.basename(name))
    p = os.path.join(ROOT, 'code', 'frozen_versions', '%s.%s%s' % (stem, h, ext))
    return p if os.path.exists(p) and sha(p, len(h)) == h else None


def combined(h):
    files = [find(f) for f in COMBINED.get(h, [])]
    if not files or None in files:
        return False
    return hashlib.sha256(b''.join(data(f) for f in files)).hexdigest()[:len(h)] == h


def main():
    bad = checked = 0
    for pre in sorted(glob.glob(os.path.join(ROOT, 'preregs', '*.md'))):
        text = open(pre, encoding='utf-8').read()
        listed = {}
        for m in list(PAT.finditer(text)) + list(PAT2.finditer(text)):
            listed.setdefault(m.group(1), set()).add(m.group(2))
        for name, hashes in sorted(listed.items()):
            path = find(name)
            if path is None:
                bad += 1
                print('MISSING   %-40s (%s)' % (name, os.path.basename(pre)))
                continue
            checked += 1
            how = None
            for h in sorted(hashes):
                if sha(path, len(h)) == h:
                    how = (h, 'current file'); break
            if how is None:
                for h in sorted(hashes):
                    if frozen(name, h):
                        how = (h, 'code/frozen_versions/' + os.path.basename(frozen(name, h))); break
                    if combined(h):
                        how = (h, 'driver sha over ' + ' + '.join(COMBINED[h])); break
            if how:
                print('ok        %-40s %s  %s  (%s)' % (name, how[0], how[1], os.path.basename(pre)))
            else:
                bad += 1
                print('MISMATCH  %-40s current %s, recorded %s  (%s)' % (
                    name, sha(path, 16), ' '.join(sorted(hashes)), os.path.basename(pre)))
    print('\n%d files checked, %d problems' % (checked, bad))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
