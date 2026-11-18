"""Fixed-B0 comparison, multi-metric guards, and optional RSI_BASELINE_JSON override."""
import json
import os
from pathlib import Path

from common import HERE, PROTOCOL, evaluation_cases, finite, load, lower_metric, metrics_for, require


def require_policy_ready():
    policy = PROTOCOL['evaluation_policy']
    require(policy.get('calibration_evaluations') == 1
            and PROTOCOL.get('repetitions') == 3
            and policy.get('other_performance_gates_relative_tolerance') == 0.03
            and policy.get('wer_absolute_max') == 0.02
            and policy.get('baseline_update') == 'fixed-initial-b0',
            'evaluation_policy', HERE / 'protocol.json',
            'Protocol requires 3-repetition means, fixed-initial-b0 comparison, 3% performance guards, and 2% absolute WER cap')
    primary = policy.get('primary_objective', {})
    weights = policy.get('reward_weights', {})
    expected_names = [case['name'] for case in evaluation_cases()]
    require(primary == dict(models=expected_names, metric='throughput_qps', comparison='>', require_all=True)
            and set(weights) == set(expected_names)
            and abs(sum(weights.values()) - 1.0) <= 1e-9,
            'evaluation_policy_models', HERE / 'protocol.json',
            'Primary objective models and reward weights must match evaluation_cases()')
    return policy


def objective_ratios(candidate, reference):
    names = require_policy_ready()['primary_objective']['models']
    return {name: candidate[name]['throughput_qps'] / reference[name]['throughput_qps']
            for name in names}


def objective_reward(candidate, reference):
    weights = require_policy_ready()['reward_weights']
    score = 1.0
    for name, ratio in objective_ratios(candidate, reference).items():
        require(finite(ratio, positive=True), 'objective_ratio', 'reward', 'Mode QPS ratio must be finite and positive')
        score *= ratio ** weights[name]
    require(finite(score, positive=True), 'objective_reward', 'reward', 'Weighted reward must be finite and positive')
    return score


def validate_baseline_record(record, location='baseline.json'):
    require_policy_ready()
    require(type(record) is dict, 'baseline_type', location, 'Baseline must be a JSON object')
    means_table = record.get('means') or record.get('initial_means')
    names = {case['name'] for case in evaluation_cases()}
    require(type(means_table) is dict and set(means_table) == names,
            'baseline_models', location, 'Baseline must include all required model/mode workloads')
    for name, values in means_table.items():
        req_keys = set(metrics_for(name))
        require(type(values) is dict and req_keys <= set(values)
                and all(finite(values[k], positive=k in PROTOCOL['higher_metrics']) for k in req_keys)
                and ('c50' not in values or values['c50'] <= 100),
                'baseline_metrics', location, 'Complete finite per-workload baseline metrics are required', field=name)
    formats = record.get('audio_formats')
    require(type(formats) is dict and set(formats) == names,
            'baseline_audio_formats', location, 'All workloads need frozen native audio formats')
    normalized_formats = {}
    for name, value in formats.items():
        require(type(value) is dict and set(value) == {'sample_rates', 'formats'}
                and type(value['sample_rates']) is list and bool(value['sample_rates'])
                and all(type(v) is int and v > 0 for v in value['sample_rates'])
                and type(value['formats']) is list and bool(value['formats']),
                'baseline_audio_format_shape', location, 'Nonempty native audio formats are required', field=name)
        normalized_formats[name] = {
            'sample_rates': list(value['sample_rates']),
            'formats': [list(item) for item in value['formats']],
        }
    return {
        'schema': record.get('schema', 1),
        'task': record.get('task', ''),
        'reference_gpu': record.get('reference_gpu', 'custom'),
        'source_commit': record.get('source_commit', PROTOCOL['source']),
        'means': {name: {k: means_table[name][k] for k in metrics_for(name)} for name in sorted(names)},
        'audio_formats': normalized_formats,
    }


def load_baseline():
    override = os.environ.get('RSI_BASELINE_JSON', '').strip() or os.environ.get('RSI_TTS_BASELINE_JSON', '').strip()
    if override:
        require(len(override) <= 1_000_000, 'baseline_oversized', 'RSI_BASELINE_JSON', 'Baseline override exceeds 1 MB')
        p = Path(override)
        raw = load(p) if p.is_file() else json.loads(override)
        return validate_baseline_record(raw, location='RSI_BASELINE_JSON')
    return validate_baseline_record(load(HERE / 'baseline.json'), location=str(HERE / 'baseline.json'))


def compare(candidate, baseline):
    policy = require_policy_ready()
    comparisons, guards, strict = {}, {}, {}
    for lane in evaluation_cases():
        name = lane['name']
        values = candidate[name]
        comparisons[name], guards[name], strict[name] = {}, {}, {}
        for key in metrics_for(name):
            value, reference = values[key], baseline['means'][name][key]
            lower = lower_metric(key)
            primary = name in policy['primary_objective']['models'] and key == 'throughput_qps'
            if primary:
                margin, threshold, comparator = 0, reference, '>'
                passed = value > reference
                role = 'primary_objective'
            elif key == 'wer_corpus':
                margin = None
                threshold, comparator = policy['wer_absolute_max'], '<='
                passed = value <= threshold
                role = 'quality_guard'
            else:
                relative = policy['other_performance_gates_relative_tolerance']
                margin = reference * relative
                threshold = reference * (1 + relative if lower else 1 - relative)
                comparator = '<=' if lower else '>='
                passed = value <= threshold if lower else value >= threshold
                role = 'performance_guard'
            delta = value - reference
            strict_pass = value <= reference if lower else value >= reference
            details = dict(baseline=reference, candidate=value, delta=delta,
                           relative_delta=delta / reference if reference else None,
                           direction='lower' if lower else 'higher', role=role,
                           margin=margin, relative_margin=margin / reference if reference and margin is not None else None,
                           threshold=threshold, comparator=comparator,
                           strict_pass=strict_pass, passed=passed)
            if key == 'wer_corpus':
                details.update(delta_percentage_points=100 * delta,
                               threshold_percentage_points=100 * threshold)
            comparisons[name][key] = details
            guards[name][key], strict[name][key] = passed, strict_pass
    return comparisons, guards, strict
