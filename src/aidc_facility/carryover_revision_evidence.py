"""Read-only stage 4-4 evidence helpers; never import or fit a serving model."""
from __future__ import annotations

import ast
from collections import Counter
from datetime import datetime
import hashlib
from pathlib import Path
import re

FIELDS = (
    'seed', 'num_prompts', 'dataset_name', 'dataset_path', 'no_stream',
    'no_oversample', 'hf_subset', 'hf_split', 'hf_name', 'hf_output_len',
    'custom_output_len', 'max_concurrency', 'request_rate', 'burstiness',
    'save_result', 'save_detailed', 'append_result', 'ignore_eos', 'ramp_up_strategy',
    'tokenizer', 'tokenizer_mode', 'temperature', 'metadata',
)
ANSI = re.compile(r'\x1b\[[0-9;]*m')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def parse_namespace(line):
    """Parse logged literals without evaluating the logged function object."""
    clean = re.sub(r'<function [^>]+>', 'None', line.strip())
    tree = ast.parse(clean, mode='eval').body
    if not isinstance(tree, ast.Call) or not isinstance(tree.func, ast.Name) or tree.func.id != 'Namespace':
        raise ValueError('Expected Namespace(...)')
    all_values = {k.arg: k.value for k in tree.keywords}
    result = {k: ast.literal_eval(all_values[k]) for k in FIELDS if k in all_values}
    ids = [s[3:] for s in result.pop('metadata', []) if s.startswith('id=')]
    if len(ids) != 1:
        raise ValueError('Expected one run ID')
    result['id'] = ids[0]
    return result


def scan_bench(path):
    rows, probes, oversample, completed_summaries = [], [], [], []
    with Path(path).open(encoding='utf-8', errors='replace') as f:
        for number, line in enumerate(f, 1):
            if line.startswith('Namespace('):
                row = parse_namespace(line)
                row['source_line'] = number
                rows.append(row)
            if 'Starting initial single prompt test run' in line:
                probes.append(number)
            if 'Oversampled requests to reach' in line:
                oversample.append({'source_line': number, 'message': line.strip(),
                    'preceding_run_id': rows[-1]['id'] if rows else None,
                    'preceding_dataset': rows[-1].get('dataset_path') if rows else None,
                    'preceding_rate': rows[-1].get('request_rate') if rows else None})
            if 'Successful requests:' in line:
                completed_summaries.append({'source_line': number, 'message': line.strip()})
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate run IDs')
    return {'runs': rows, 'initial_probe_count': len(probes),
            'initial_probe_lines': probes, 'oversample_log_messages': oversample,
            'successful_request_summary_lines': completed_summaries}


def summarize_settings(rows):
    keys = [k for k in FIELDS if k != 'metadata']
    return {k: sorted({repr(r.get(k, '<missing>')) for r in rows}) for k in keys}


def match_runs(config, metadata, bench):
    if len(config['experiments']) != len(metadata) or len(metadata) != len(bench):
        raise ValueError('Run count mismatch')
    by_id = {r['id']: r for r in bench}
    for cfg, meta in zip(config['experiments'], metadata):
        row = by_id[meta['id']]
        for ck, bk in (('dataset-path', 'dataset_path'), ('request_rate', 'request_rate'),
                       ('num-prompts', 'num_prompts'), ('burstiness', 'burstiness')):
            if not cfg[ck] == meta[ck] == row[bk]:
                raise ValueError(f'Inconsistent {ck} at {meta["id"]}')
    return True


def chronology(metadata):
    out = []
    names = list(dict.fromkeys(r['dataset-path'] for r in metadata))
    for name in names:
        rows = [r for r in metadata if r['dataset-path'] == name]
        starts = [datetime.strptime(r['start_time'], '%Y-%m-%d_%H:%M:%S.%f') for r in rows]
        ends = [datetime.strptime(r['end_time'], '%Y-%m-%d_%H:%M:%S.%f') for r in rows]
        rates = [r['request_rate'] for r in rows]
        gaps = [(starts[i]-ends[i-1]).total_seconds() for i in range(1, len(rows))]
        out.append({'service': name, 'runs': len(rows), 'first_id': rows[0]['id'],
                    'last_id': rows[-1]['id'], 'strict_rate_1_to_100_order': rates == list(range(1, 101)),
                    'min_metadata_gap_s': min(gaps), 'max_metadata_gap_s': max(gaps),
                    'min_duration_s': min((b-a).total_seconds() for a,b in zip(starts,ends)),
                    'max_duration_s': max((b-a).total_seconds() for a,b in zip(starts,ends)),
                    'num_prompts_equals_rate_times_300': all(r['num-prompts'] == 300*r['request_rate'] for r in rows),
                    'boundary_is_actual_arrival': False})
    return out


def scan_server(path):
    patterns = {
        'engine_init': 'Initializing a V1 LLM engine',
        'api_version': 'vLLM API server version',
        'shutdown': 'Shutting down FastAPI HTTP server',
        'cache_reset_message': 'Resetting prefix cache',
        'cache_reset_route': 'Route: /reset_prefix_cache',
        'dev_endpoints_warning': 'Development endpoints are enabled',
        'kv_capacity': 'GPU KV cache size:',
    }
    found = {k: [] for k in patterns}
    http = Counter()
    http_with_clock, http_count = 0, 0
    number = 0
    with Path(path).open(encoding='utf-8', errors='replace') as f:
        for number, raw in enumerate(f, 1):
            line = ANSI.sub('', raw).strip()
            for key, needle in patterns.items():
                if needle in line:
                    found[key].append({'source_line': number, 'text': line})
            match = re.search(r'"(GET|POST|DELETE|PUT) ([^ ]+) HTTP/[^\"]+" (\d{3})', line)
            if match:
                http_count += 1
                http[' '.join(match.groups())] += 1
                http_with_clock += bool(re.search(r'\b\d{2}:\d{2}:\d{2}\b', line))
    return {'markers': found, 'http_access_counts': dict(http),
            'http_access_rows': http_count, 'http_rows_with_clock_pattern': http_with_clock,
            'line_count_text_universal_newlines': number,
            'absence_of_reset_markers_proves_no_reset': False,
            'http_200_is_request_completion': False}


def source_anchors(source_dir):
    """Versioned source statements checked by presence, not executed."""
    targets = {
        'datasets.py': ['def get_samples(', 'output_len=args.hf_output_len,',
            'class HuggingFaceDataset(', 'self.load_stream = not no_stream',
            'self.data = self.data.shuffle(seed=self.random_seed)',
            'def maybe_oversample_requests(', 'class InstructCoderDataset(',
            'DEFAULT_OUTPUT_LEN = 200', 'class MLPerfDataset(',
            'expected_output_len = ref_out_len if dynamic_output else output_len'],
        'serve.py': ['async def get_request(', 'target_total_delay_s = total_requests / request_rate',
            'print("Starting initial single prompt test run...")', 'random.seed(args.seed)',
            'if not args.save_detailed:', 'if args.save_result or args.append_result:'],
        'endpoint_request_func.py': ['"max_tokens": request_func_input.output_len,',
            'if request_func_input.ignore_eos:', '"vllm": async_request_openai_completions,'],
        'kv_cache_utils.py': ['class PrefixCachingMetrics:',
            'def __init__(self, max_recent_requests: int = 1000):'],
        'api_server.py': ['if envs.VLLM_SERVER_DEV_MODE:',
            '@router.post("/reset_prefix_cache")', 'async def reset_prefix_cache('],
    }
    out = []
    for name, needles in targets.items():
        text = (Path(source_dir)/name).read_text(encoding='utf-8')
        ast.parse(text)  # Syntax validation only. Do not import downloaded code.
        lines = text.splitlines()
        for needle in needles:
            hits = [i for i, line in enumerate(lines, 1) if needle in line]
            if not hits:
                raise ValueError(f'Missing source anchor: {name}: {needle}')
            out.append({'file': name, 'statement': needle, 'line_numbers': hits})
    return out


def work_scale_counterexample():
    """Analytic example, not a fit: w and mu have an arbitrary common scale."""
    a = {'work_units': 1., 'capacity_units_per_s': 40., 'running': 100}
    b = {'work_units': 2., 'capacity_units_per_s': 80., 'running': 100}
    for row in (a,b):
        row['completion_time_s_in_fixed_state'] = row['work_units']/(row['capacity_units_per_s']/row['running'])
    return {'case_a': a, 'case_b': b, 'same_completion_time': a['completion_time_s_in_fixed_state'] == b['completion_time_s_in_fixed_state'],
            'interpretation': 'Algebraic scale non-identifiability; not proof that every work-distribution shape is equivalent, and not NLR data.'}
