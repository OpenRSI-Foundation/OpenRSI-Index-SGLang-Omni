"""Full bilingual HTTP/SSE evaluation, auto NUMA physical-core pinning, and fixed-B0 scoring."""
import asyncio
from collections import deque
import contextlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import string
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
import wave

sys.path.insert(0, '/tests')
from contract import COUNTS, WORKLOADS, aggregate, compare, finite
from gates import CandidateInvalid, EvaluationFailure, FIXED, authority_json, check_with_changes, sha

TESTS = Path('/tests')
MODEL = '/opt/models/qwen3-asr-1.7b'
MODEL_NAME = 'Qwen/Qwen3-ASR-1.7B'
URL = 'http://127.0.0.1:18080'
PREFIX = 'ASR_AGGREGATE '


def emit(obj):
    print(json.dumps(obj, allow_nan=False, sort_keys=True), flush=True)


def _read_cpu_busy_ticks():
    ticks = {}
    try:
        with open('/proc/stat', encoding='ascii', errors='replace') as f:
            for line in f:
                if line.startswith('cpu') and line[3:4].isdigit():
                    parts = line.split()
                    cid = int(parts[0][3:])
                    nums = list(map(int, parts[1:9]))
                    ticks[cid] = sum(nums) - nums[3] - nums[4]
    except OSError:
        pass
    return ticks


def _gpu_numa_node(pci_bus_id):
    raw = pci_bus_id.strip().lower()
    parts = raw.split(':')
    candidates = [raw]
    if len(parts) == 3 and len(parts[0]) > 4:
        candidates.append(parts[0][-4:] + ':' + parts[1] + ':' + parts[2])
    for bdf in candidates:
        p = Path('/sys/bus/pci/devices') / bdf / 'numa_node'
        if p.is_file():
            try:
                val = int(p.read_text().strip())
                if val >= 0:
                    return val
            except ValueError:
                pass
    return None


def bind_hardware():
    try:
        q = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,driver_version,pci.bus_id',
                            '--format=csv,noheader,nounits'], check=False, text=True,
                           capture_output=True, timeout=30)
    except OSError:
        raise EvaluationFailure('infrastructure', 'gpu_query_unavailable',
                                'Check the NVIDIA driver tools and container GPU runtime.',
                                condition='nvidia-smi could not be launched.') from None
    except subprocess.TimeoutExpired:
        raise EvaluationFailure('infrastructure', 'gpu_query_timeout',
                                'Check driver health before repeating evaluation.',
                                condition='nvidia-smi exceeded its 30-second query timeout.') from None
    if q.returncode:
        print(q.stderr, file=sys.stderr, flush=True)
        raise EvaluationFailure('infrastructure', 'gpu_query_failed', 'nvidia-smi failed; see diagnostics.')
    fields = ['index', 'uuid', 'name', 'memory.total', 'driver_version', 'pci.bus_id']
    rows = [dict(zip(fields, [s.strip() for s in line.split(',')])) for line in q.stdout.splitlines() if line.strip()]
    expected_env = os.environ.get('RSI_HARNESS_EXPECTED_GPU_UUIDS', '').strip()
    if expected_env:
        expected = [x.strip() for x in expected_env.split(',') if x.strip()]
        by_uuid = {r['uuid']: r for r in rows}
        if len(expected) != 1 or expected[0] not in by_uuid:
            raise EvaluationFailure('infrastructure', 'gpu_visibility', 'Assigned GPU UUID is not visible via nvidia-smi.')
        gpu = by_uuid[expected[0]]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu['index']
    else:
        if len(rows) != 1:
            raise EvaluationFailure('infrastructure', 'gpu_count_mismatch',
                                    'Expose the single exclusive GPU reserved for this run.',
                                    condition=f'Expected 1 visible GPU; found {len(rows)}.')
        gpu = rows[0]

    allowed = sorted(os.sched_getaffinity(0))
    raw_cpus = os.environ.get('RSI_ASR_CPU_LANES', '').strip()
    if raw_cpus and raw_cpus != 'null':
        lane = json.loads(raw_cpus) if raw_cpus.lstrip().startswith('[') else [
            int(x) for part in raw_cpus.split(',') if part.strip()
            for x in (range(int(part.split('-')[0]), int(part.split('-')[1]) + 1) if '-' in part else [int(part)])
        ]
        if lane and isinstance(lane[0], list):
            lane = lane[0]
    else:
        t0 = _read_cpu_busy_ticks()
        time.sleep(0.2)
        t1 = _read_cpu_busy_ticks()
        busy_delta = {c: max(0, t1.get(c, 0) - t0.get(c, 0)) for c in allowed}
        cores_by_key = {}
        for cpu in allowed:
            p = Path(f'/sys/devices/system/cpu/cpu{cpu}')
            pkg = (p / 'topology/physical_package_id').read_text().strip()
            core = (p / 'topology/core_id').read_text().strip()
            nodes = sorted(int(n.name[4:]) for n in p.glob('node[0-9]*'))
            entry = cores_by_key.setdefault((pkg, core), {'cpus': [], 'nodes': nodes, 'busy': 0})
            entry['cpus'].append(cpu)
            entry['busy'] += busy_delta.get(cpu, 0)
        units = [{'cpu': min(v['cpus']), 'nodes': v['nodes'], 'busy': v['busy']} for v in cores_by_key.values()]
        if len(units) < 16:
            raise EvaluationFailure('infrastructure', 'cpu_smt_overlap',
                                    'Reserve at least 16 distinct non-SMT physical CPU cores.',
                                    condition=f'Found only {len(units)} distinct physical cores.')
        pref_node = _gpu_numa_node(gpu['pci.bus_id'])
        chosen = None
        if pref_node is not None:
            on_node = [u for u in units if pref_node in u['nodes']]
            if len(on_node) >= 16:
                on_node.sort(key=lambda u: (u['busy'], u['cpu']))
                chosen = sorted(u['cpu'] for u in on_node[:16])
        if chosen is None:
            by_node = {}
            for u in units:
                for n in u['nodes']:
                    by_node.setdefault(n, []).append(u)
            viable = [(sum(u['busy'] for u in sorted(us, key=lambda x: x['busy'])[:16]), n, us)
                      for n, us in by_node.items() if len(us) >= 16]
            if viable:
                viable.sort(key=lambda x: (x[0], x[1]))
                us_sorted = sorted(viable[0][2], key=lambda u: (u['busy'], u['cpu']))
                chosen = sorted(u['cpu'] for u in us_sorted[:16])
            else:
                units.sort(key=lambda u: (u['busy'], u['cpu']))
                chosen = sorted(u['cpu'] for u in units[:16])
        lane = chosen

    if len(lane) != 16 or len(set(lane)) != 16 or not (set(lane) <= set(allowed)):
        raise EvaluationFailure('infrastructure', 'cpu_lane_size', 'Requires 16 distinct allowed CPU IDs.')
    phys = set()
    for cpu in lane:
        p = Path(f'/sys/devices/system/cpu/cpu{cpu}')
        key = ((p / 'topology/physical_package_id').read_text().strip(), (p / 'topology/core_id').read_text().strip())
        if key in phys:
            raise EvaluationFailure('infrastructure', 'cpu_smt_overlap', 'Use 16 distinct physical cores without SMT siblings.')
        phys.add(key)
    os.sched_setaffinity(0, set(lane))
    return {'gpu': f"{gpu['uuid']}, {gpu['name']}, {gpu['memory.total']}, {gpu['driver_version']}",
            'gpu_name': gpu['name'],
            'cpu_affinity': sorted(os.sched_getaffinity(0))}


def load_baseline():
    override = os.environ.get('RSI_BASELINE_JSON', '').strip()
    if override:
        if len(override) > 1_000_000:
            raise EvaluationFailure('infrastructure', 'baseline_oversized', 'RSI_BASELINE_JSON exceeds 1 MB.')
        p = Path(override)
        raw = authority_json(p, 'baseline_override') if p.is_file() else json.loads(override)
    else:
        raw = authority_json(TESTS / 'baseline.json', 'baseline')
    means = raw.get('means') or raw.get('b0', {}).get('means')
    if not isinstance(means, dict) or set(means) != set(WORKLOADS):
        raise EvaluationFailure('infrastructure', 'baseline_schema', 'Baseline must define means for all 4 workloads.')
    for wk, m in means.items():
        if not isinstance(m, dict) or not finite(m.get('qps'), positive=True) or not finite(m.get('rtfx'), positive=True):
            raise EvaluationFailure('infrastructure', 'baseline_metrics', f'Invalid baseline metrics for {wk}.')
    return raw, means


def load_rows():
    import pyarrow.parquet as pq
    manifest = authority_json(TESTS / 'data-manifest.json', 'data_manifest')
    rows = {}
    for lang, count in COUNTS.items():
        entries = []
        for shard in manifest[lang]['shards']:
            if shard['start'] != len(entries):
                raise EvaluationFailure('infrastructure', 'dataset_order_mismatch', 'Restore native shard order from the fixed data manifest.')
            raw_path = Path(shard['path'])
            path = raw_path if raw_path.is_absolute() else TESTS / raw_path
            if sha(path) != shard['sha256']:
                raise EvaluationFailure('infrastructure', 'dataset_digest_mismatch', 'Dataset shard digest differs from manifest.')
            table = pq.read_table(path)
            if table.num_rows != shard['rows']:
                raise EvaluationFailure('infrastructure', 'dataset_shard_coverage', 'Dataset shard row count differs from manifest.')
            for row in table.to_pylist():
                audio = row['ref_audio']['bytes']
                with wave.open(io.BytesIO(audio), 'rb') as w:
                    duration = w.getnframes() / w.getframerate()
                if not math.isfinite(duration) or duration <= 0 or not isinstance(row['ref_text'], str):
                    raise EvaluationFailure('infrastructure', 'dataset_input_invalid', 'Dataset audio duration or reference text is invalid.')
                entries.append((audio, row['ref_text'], duration))
        if len(entries) != count or manifest[lang]['rows'] != count:
            raise EvaluationFailure('infrastructure', 'dataset_coverage', 'Full-language dataset row coverage is incomplete.')
        rows[lang] = entries
    return rows


def normalizer():
    from whisper.normalizers import EnglishTextNormalizer
    from zhon.hanzi import punctuation
    en = EnglishTextNormalizer()

    def normalize(text, lang):
        if lang == 'en':
            return en(text)
        for ch in punctuation + string.punctuation:
            if ch != "'":
                text = text.replace(ch, '')
        text = text.replace(' ', '').replace('\u3000', '').strip()
        return ' '.join(list(text))

    return normalize


async def request(session, row, lang, stream, protected):
    import aiohttp
    form = aiohttp.FormData()
    for k, v in [('model', MODEL_NAME), ('language', lang), ('response_format', 'json')]:
        form.add_field(k, v)
    if stream:
        form.add_field('stream', 'true')
    form.add_field('file', row[0], filename='audio.wav', content_type='audio/wav')
    headers = {'x-sglang-omni-route-stream': 'true'} if stream else None
    start = time.perf_counter()
    deltas, times, empty = [], [], 0
    final, done = None, False
    try:
        async with session.post(URL + '/v1/audio/transcriptions', data=form, headers=headers, allow_redirects=False) as response:
            if response.status != 200:
                raise EvaluationFailure('candidate_runtime', 'http_status', f'HTTP {response.status}; inspect sanitized server diagnostics.')
            if stream:
                if 'text/event-stream' not in response.headers.get('Content-Type', ''):
                    raise EvaluationFailure('candidate_invalid', 'sse_content_type', 'Return text/event-stream for streaming requests.')
                size = 0
                async for raw in response.content:
                    size += len(raw)
                    if size > 4 * 1024**2:
                        raise EvaluationFailure('candidate_invalid', 'response_size', 'Response exceeds 4 MiB safety bound.')
                    line = raw.decode('utf-8').strip()
                    if not line or line.startswith(':') or line.startswith('event:'):
                        continue
                    if not line.startswith('data:') or done:
                        raise EvaluationFailure('candidate_invalid', 'sse_order', 'Malformed event or data after DONE.')
                    payload = line[5:].lstrip()
                    if payload == '[DONE]':
                        if final is None:
                            raise EvaluationFailure('candidate_invalid', 'sse_done_order', 'Final text must precede DONE.')
                        done = True
                        continue
                    item = json.loads(payload)
                    if not isinstance(item, dict):
                        raise ValueError('event must be an object')
                    typ = item.get('type')
                    if typ == 'transcript.text.delta':
                        delta = item.get('delta')
                        if final is not None or not isinstance(delta, str):
                            raise ValueError('invalid delta ordering or type')
                        if delta:
                            deltas.append(delta)
                            times.append(time.perf_counter())
                        else:
                            empty += 1
                    elif typ == 'transcript.text.done':
                        if final is not None or not isinstance(item.get('text'), str):
                            raise ValueError('invalid final text')
                        final = item['text']
                        protected.append(final)
                    else:
                        raise ValueError('unsupported or error event')
                if not done or final is None or not deltas or ''.join(deltas) != final:
                    raise ValueError('incomplete or inconsistent stream')
            else:
                parts, size = [], 0
                async for part in response.content.iter_chunked(65536):
                    size += len(part)
                    if size > 4 * 1024**2:
                        raise ValueError('oversized response')
                    parts.append(part)
                raw = b''.join(parts)
                item = json.loads(raw)
                if not isinstance(item, dict) or not isinstance(item.get('text'), str):
                    raise ValueError('response must contain string text')
                final = item['text']
                protected.append(final)
    except asyncio.TimeoutError:
        raise EvaluationFailure('candidate_runtime', 'request_timeout', 'A request exceeded the fixed 300-second timeout; no retries are allowed.') from None
    except (UnicodeError, ValueError, aiohttp.ClientError):
        raise EvaluationFailure('candidate_invalid', 'response_protocol', 'Malformed HTTP/SSE output; preserve event order, final text, DONE and delta concatenation.') from None
    finally:
        if deltas:
            protected.append(''.join(deltas))
    end = time.perf_counter()
    return {'text': final.strip(), 'raw_text': final, 'deltas': deltas, 'latency': end - start,
            'ttft': times[0] - start if times else None,
            'intervals': [b - a for a, b in zip(times, times[1:])],
            'tail': end - times[-1] if times else None, 'empty': empty}


async def pass_requests(rows, lang, stream, protected):
    import aiohttp
    queue = deque(enumerate(rows))
    outputs = [None] * len(rows)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300),
                                     connector=aiohttp.TCPConnector(limit=32), trust_env=False) as session:
        async def worker():
            while queue:
                index, row = queue.popleft()
                outputs[index] = await request(session, row, lang, stream, protected)

        started = time.perf_counter()
        tasks = [asyncio.create_task(worker()) for _ in range(32)]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        wall = time.perf_counter() - started
    if any(x is None for x in outputs) or wall <= 0:
        raise EvaluationFailure('incomplete', 'coverage', 'Every native row must complete exactly once.')
    return outputs, wall


def metrics(rows, outputs, wall, lang, stream, normalize):
    import numpy as np
    from jiwer import process_words
    errors = words = 0
    for row, out in zip(rows, outputs, strict=True):
        ref, hyp = normalize(row[1], lang), normalize(out['text'], lang)
        if not ref:
            raise EvaluationFailure('infrastructure', 'reference_normalization_empty', 'A reference normalizes to empty text.')
        result = process_words(ref, hyp)
        errors += result.substitutions + result.deletions + result.insertions
        words += result.substitutions + result.deletions + result.hits
    lat = [x['latency'] for x in outputs]
    rtfs = [x['latency'] / row[2] for row, x in zip(rows, outputs, strict=True)]

    def p95(values):
        return float(np.percentile(values, 95, method='linear'))

    m = {'qps': len(rows) / wall, 'rtfx': sum(r[2] for r in rows) / wall,
         'latency_mean_s': statistics.mean(lat), 'latency_p95_s': p95(lat),
         'rtf_mean': statistics.mean(rtfs), 'rtf_p95': p95(rtfs),
         'corpus_wer': errors / words, 'coverage': len(outputs), 'warmup_coverage': len(rows)}
    if stream:
        ttft = [x['ttft'] for x in outputs]
        intervals = [v for x in outputs for v in x['intervals']]
        m.update(ttft_mean_s=statistics.mean(ttft), ttft_p95_s=p95(ttft),
                 inter_chunk_mean_s=statistics.mean(intervals) if intervals else None,
                 inter_chunk_p95_s=p95(intervals) if intervals else None,
                 single_delta_fraction=sum(len(x['deltas']) == 1 for x in outputs) / len(outputs),
                 empty_delta_count=sum(x['empty'] for x in outputs),
                 last_delta_to_receipt_mean_s=statistics.mean(x['tail'] for x in outputs))
    return m


def scrub(lines, protected):
    secrets = {part.strip() for s in protected for part in s.splitlines() if part.strip()}
    fields = re.compile(r'''(?i)(?:["']?\b(?:prompt|transcript|ref_text|hyp_text|audio_fingerprint|payload|delta|text|content)\b["']?\s*[:=]\s*|sampling temp\s*[:=]\s*)''')
    long_text = sorted((s for s in secrets if len(s) >= 8), key=len, reverse=True)
    pattern = re.compile('|'.join(re.escape(s) for s in long_text)) if long_text else None
    for line in lines:
        field = fields.search(line)
        if field:
            line = line[:field.end()] + '[protected request/output fields redacted]'
        elif line.strip() in secrets:
            line = '[protected text]'
        if pattern:
            line = pattern.sub('[protected text]', line)
        print('server: ' + line, flush=True)


@contextlib.contextmanager
def service(cfg, protected):
    with tempfile.TemporaryDirectory(prefix='asr-judge-', dir='/tmp') as tmp:
        env = {k: os.environ[k] for k in ('PATH', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES',
                                           'NVIDIA_VISIBLE_DEVICES', 'NVIDIA_DRIVER_CAPABILITIES') if k in os.environ}
        env.update(HOME=tmp, TMPDIR=tmp, XDG_CACHE_HOME=tmp + '/cache',
                   PYTHONPATH='/workspace/sglang-omni', PYTHONDONTWRITEBYTECODE='1',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                   TOKENIZERS_PARALLELISM='false', TORCHINDUCTOR_CACHE_DIR=tmp + '/inductor',
                   TRITON_CACHE_DIR=tmp + '/triton', TORCH_EXTENSIONS_DIR=tmp + '/extensions',
                   CUDA_CACHE_PATH=tmp + '/cuda', FLASHINFER_WORKSPACE_BASE=tmp + '/flashinfer',
                   TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE='0', TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE='0')
        command = ['/opt/asr-venv/bin/python', '-B', '-m', 'sglang_omni.cli', 'serve',
                   '--model-path', MODEL, '--model-name', MODEL_NAME, '--host', '127.0.0.1',
                   '--port', '18080', '--skip-server-warmup']
        for key, value in {**cfg, **FIXED}.items():
            command += ['--' + key, str(value).lower() if isinstance(value, bool) else str(value)]
        try:
            proc = subprocess.Popen(command, cwd=tmp, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        except OSError:
            raise EvaluationFailure('infrastructure', 'service_launch_failed',
                                    'Restore the baked Python executable and check permissions.',
                                    path=command[0], condition='Service launcher could not be executed.') from None
        captured = deque(maxlen=2000)

        def drain():
            while True:
                raw = proc.stdout.readline(16385)
                if not raw:
                    break
                if len(raw) > 16384:
                    while raw and not raw.endswith(b'\n'):
                        raw = proc.stdout.readline(16385)
                    captured.append('[oversized diagnostic line omitted]')
                else:
                    captured.append(raw.decode('utf-8', errors='replace').rstrip())

        thread = threading.Thread(target=drain, daemon=True)
        thread.start()
        try:
            import requests
            session = requests.Session()
            session.trust_env = False
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise EvaluationFailure('candidate_runtime', 'service_exit', f'Service exited with code {proc.returncode}; inspect server diagnostics.')
                try:
                    if session.get(URL + '/health', timeout=2).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.5)
            else:
                raise EvaluationFailure('incomplete', 'readiness_timeout', 'Service did not become ready within 600 seconds.')
            session.close()
            yield proc
            if proc.poll() is not None:
                raise EvaluationFailure('candidate_runtime', 'service_exit', 'Service exited before pass cleanup.')
        finally:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=15)
            thread.join(timeout=10)
            if thread.is_alive():
                raise EvaluationFailure('incomplete', 'service_cleanup_incomplete', 'Service output remained open after process-group termination.')
            scrub(captured, protected)


def _start_contention_sampler(cpuset):
    sampler_path = Path('/opt/asr-reference/source/tests/utils/ci_cpu_contention.py')
    if not sampler_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location('rsi_asr_contention', sampler_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    if Path('/.dockerenv').exists():
        mod.tree_pids = lambda procs, _root: set(procs)
    sampler = mod.ContentionSampler(set(cpuset), interval_s=30.0, root_pid=os.getpid())
    sampler.start()
    return sampler


def main(calibrate=False):
    if Path('/logs/verifier/reward.json').exists():
        raise EvaluationFailure('infrastructure', 'stale_reward', 'Judge must start with an empty reward destination.')
    hw = bind_hardware()
    cfg, changed = check_with_changes()
    if calibrate and changed:
        raise EvaluationFailure('infrastructure', 'calibration_dirty_workspace', 'Calibration requires a pristine unchanged workspace.')
    b0_record, b0_means = (None, None) if calibrate else load_baseline()
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise EvaluationFailure('infrastructure', 'cuda_preflight_failed', 'Check Torch/CUDA installation and single-GPU visibility.')
    try:
        torch.cuda.synchronize()
    except RuntimeError:
        raise EvaluationFailure('infrastructure', 'cuda_synchronize_failed', 'CUDA runtime could not synchronize before candidate launch.') from None
    try:
        rows = load_rows()
    except (OSError, ValueError, KeyError, TypeError, wave.Error, EOFError):
        raise EvaluationFailure('infrastructure', 'dataset_decode_failed', 'Check the baked SeedTTS Parquet assets and PyArrow installation.') from None

    normalize = normalizer()
    protected = [row[1] for group in rows.values() for row in group]
    sampler = _start_contention_sampler(hw['cpu_affinity'])
    repeats = {key: [] for key in WORKLOADS}
    try:
        for key in WORKLOADS:
            mode, lang = key.split('/')
            for repetition in range(3):
                with service(cfg, protected):
                    warm, _ = asyncio.run(pass_requests(rows[lang], lang, mode == 'stream', protected))
                    del warm
                    torch.cuda.synchronize()
                    outputs, wall = asyncio.run(pass_requests(rows[lang], lang, mode == 'stream', protected))
                    torch.cuda.synchronize()
                    result = metrics(rows[lang], outputs, wall, lang, mode == 'stream', normalize)
                    repeats[key].append(result)
                    del outputs
                emit({'status': 'repeat_complete', 'workload': key, 'repetition': repetition + 1, 'metrics': result})
    finally:
        contention = None
        if sampler is not None:
            sampler.stop()
            contention = {
                'windows': sampler.window_count(),
                'errors': sampler.errors,
                'peak_foreign_cores': sampler.peak_foreign_cores(),
                'mean_foreign_cores': sampler.mean_foreign_cores(),
            }
    if contention is not None and (contention['errors'] > 0 or contention['peak_foreign_cores'] > 2.0):
        raise EvaluationFailure('infrastructure', 'cpu_contention',
                                f"Foreign CPU contention peaked at {contention['peak_foreign_cores']:.2f} cores (> 2.0) on pinned cpuset.")

    means, ranges = aggregate(repeats)
    if calibrate:
        cal = {
            'schema': 1,
            'task': 'rsi/qwen3-asr-1.7b-dual-mode',
            'reference_gpu': hw['gpu_name'],
            'means': means,
            'ranges': ranges,
            'run_id': str(uuid.uuid4()),
            'created_unix': time.time(),
        }
        print('RSI_BASELINE_JSON=' + json.dumps(cal, allow_nan=False, sort_keys=True), flush=True)
        return 0

    result = {
        'run_id': str(uuid.uuid4()),
        'hardware': hw,
        'contention': contention,
        'pristine': not changed,
        'changed_paths': changed,
        'repeats': repeats,
        'means': means,
        'ranges': ranges,
        'scored_requests': 18648,
        'warmup_requests': 18648,
    }
    comp = compare(means, b0_means)
    result.update(comp)
    if not changed:
        integrity_failures = [
            f for f in result['failed']
            if 'corpus_wer' in f or 'missing_inter_chunk' in f or 'single_delta_fraction' in f
        ]
        result['pristine_oracle'] = True
        result['failed'] = integrity_failures
        result['accepted'] = not integrity_failures
        result['reward'] = 1.0 if not integrity_failures else None
    else:
        result['accepted'] = not result['failed']
        if not result['accepted']:
            result['reward'] = None
    print(PREFIX + json.dumps(result, allow_nan=False, sort_keys=True), flush=True)
    if not result['accepted']:
        return 1
    Path('/logs/verifier').mkdir(parents=True, exist_ok=True)
    with open('/logs/verifier/reward.json', 'x', encoding='utf-8') as f:
        json.dump({'reward': result['reward']}, f, allow_nan=False)
    return 0


def run_cli(calibrate=False):
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        raise SystemExit(main(calibrate=calibrate))
    except CandidateInvalid as exc:
        emit({'status': 'candidate_invalid', 'errors': exc.errors})
        sys.exit(1)
    except EvaluationFailure as exc:
        emit({'status': exc.category, 'code': exc.code, 'condition': exc.condition,
               'path': exc.path, 'field': exc.field, 'hint': exc.hint})
        sys.exit(1)
    except ImportError as exc:
        emit({'status': 'infrastructure', 'code': 'dependency_import_failed',
               'module': exc.name, 'condition': 'A required dependency could not be imported.',
               'hint': 'Restore the locked Environment dependencies.'})
        sys.exit(1)
    except SystemExit:
        raise
    except BaseException as exc:
        frames = [{'path': f.filename, 'line': f.lineno, 'function': f.name}
                  for f in traceback.extract_tb(exc.__traceback__)
                  if f.filename.startswith('/workspace/')]
        emit({'status': 'infrastructure_or_incomplete', 'exception': type(exc).__name__,
               'candidate_frames': frames,
               'hint': 'Check fixed assets, CUDA availability, and sanitized service diagnostics; no reward was written.'})
        sys.exit(1)


if __name__ == '__main__':
    run_cli(calibrate=False)
