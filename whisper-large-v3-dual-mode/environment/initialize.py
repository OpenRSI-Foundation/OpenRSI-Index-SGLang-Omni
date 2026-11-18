"""Image-build preparation only; no serving, inference or evaluation."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path('/opt/asr-reference')
SOURCE = ROOT / 'source'
ARCHIVE_SHA = '3e39d850f47f7a65034429ff0727362c4ab9569eb5bc09a9d58a9523cf8075a2'
SURFACES = ['models/whisper_asr', 'scheduling', 'comm', 'http', 'serve',
            'model_runner', 'pipeline', 'relay', 'sampling', 'utils', 'kernels']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def inventory(root):
    result = {}
    for p in sorted(root.rglob('*')):
        key = p.relative_to(root).as_posix()
        if p.is_symlink():
            result[key] = {'type': 'symlink', 'target': os.readlink(p)}
        elif p.is_file():
            result[key] = {'type': 'file', 'sha256': digest(p)}
        elif p.is_dir():
            result[key] = {'type': 'directory'}
        else:
            raise RuntimeError('unexpected build product: ' + str(p))
    return result


def prepare():
    archive = ROOT / 'source.tar.gz'
    assert digest(archive) == ARCHIVE_SHA
    SOURCE.mkdir()
    with tarfile.open(archive) as t:
        members = t.getmembers()
        prefix = members[0].name.split('/')[0] + '/'
        for m in members:
            if not m.name.startswith(prefix):
                continue
            m.name = m.name[len(prefix):]
            if m.name:
                t.extract(m, SOURCE, filter='data')
    expected = json.loads((ROOT / 'source-manifest.json').read_text())
    for rel, sha in expected.items():
        assert digest(SOURCE / rel) == sha, rel
    subprocess.run([sys.executable, '-I', '-B', str(ROOT / 'adapt_baseline.py'), str(SOURCE)], check=True)
    model = Path('/opt/models/whisper-large-v3')
    model.mkdir(parents=True)
    for entry in json.loads((ROOT / 'model-manifest.json').read_text()):
        p = model / entry['path']
        p.parent.mkdir(parents=True, exist_ok=True)
        url = ('https://huggingface.co/openai/whisper-large-v3/resolve/'
               '06f233fe06e710322aca913c1bc4249a0d71fce1/' + entry['path'])
        with urllib.request.urlopen(url, timeout=300) as src, p.open('xb') as dst:
            shutil.copyfileobj(src, dst, 8 * 1024**2)
        assert p.stat().st_size == entry['size']
        if entry['sha256']:
            assert digest(p) == entry['sha256']
        else:
            b = p.read_bytes()
            assert hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest() == entry['git_sha1']

    eval_dir = ROOT / 'eval-data'
    eval_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    import pyarrow.parquet as pq
    for lang, spec in {'en': {'url': 'https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-arrow/resolve/27f4c1adee83b5b29b7c4b375f6b976324bda308/data/en-00000-of-00001.parquet', 'path': '/opt/asr-reference/eval-data/en.parquet', 'sha256': '5849b41b49cae996328c06d2c5791717c3bafc369bddfa1ec4f86761bb8bc0ca', 'rows': 1088}, 'zh': {'url': 'https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-arrow/resolve/27f4c1adee83b5b29b7c4b375f6b976324bda308/data/zh-00000-of-00001.parquet', 'path': '/opt/asr-reference/eval-data/zh.parquet', 'sha256': '5d73be79cf2e84fd0bebf5d47212438a88dd264a591031d092164ba84545f1de', 'rows': 2020}}.items():
        dest_pq = Path(spec['path'])
        with urllib.request.urlopen(spec['url'], timeout=300) as src, dest_pq.open('xb') as dst:
            shutil.copyfileobj(src, dst, 8 * 1024**2)
        dest_pq.chmod(0o600)
        assert digest(dest_pq) == spec['sha256'], str(dest_pq)
        assert pq.read_table(dest_pq).num_rows == spec['rows'], str(dest_pq)


def close():
    # Installation used the reference source, never the candidate tree.
    for p in list(SOURCE.rglob('__pycache__')) + list(SOURCE.glob('*.egg-info')) + list(SOURCE.glob('build')):
        shutil.rmtree(p)
    work = Path('/workspace')
    work.mkdir(exist_ok=True)
    shutil.copytree(SOURCE, work / 'sglang-omni', symlinks=True)
    shutil.copyfile(ROOT / 'candidate.yaml', work / 'candidate.yaml')
    (work / 'research').mkdir()
    # Captured by the host before any Agent runs, then injected under /tests.
    record = {'workspace': inventory(work),
              'assets': {str(p): inventory(p) for p in
                         [ROOT, Path('/opt/models/whisper-large-v3'), Path('/opt/asr-venv')]}}
    (ROOT / 'build-inventory.json').write_text(json.dumps(record, sort_keys=True))


if __name__ == '__main__':
    import sys
    {'prepare': prepare, 'close': close}[sys.argv[1]]()
