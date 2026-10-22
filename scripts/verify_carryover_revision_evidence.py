"""Independent raw-literal and provenance check for stage 4-4 outputs."""
from pathlib import Path
from html.parser import HTMLParser
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/carryover_revision_evidence_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.table_depth = 0
        self.headers = 0

    def handle_starttag(self, tag, attrs):
        if tag == 'a': self.links.extend(v for k,v in attrs if k == 'href')
        if tag == 'table': self.table_depth += 1
        if tag == 'h1': self.headers += 1

    def handle_endtag(self, tag):
        if tag == 'table': self.table_depth -= 1
        assert self.table_depth >= 0


def main():
    data = read(OUT/'evidence.json')
    raw = ROOT.parent/'NLR_dataset/00_raw_datasets/inference_online_rate_llama3_70b'
    bench_lines = (raw/'vllm_bench.log').read_text(encoding='utf-8').splitlines()
    namespaces = [s for s in bench_lines if s.startswith('Namespace(')]
    by_id = {re.search(r"metadata=\['id=(\d+)'\]", s).group(1):s for s in namespaces}
    assert len(by_id) == len(namespaces) == 200
    independently_checked = 0
    for r in data['benchmark']['runs']:
        s = by_id[r['id']]
        for k in ('seed','num_prompts','request_rate','burstiness'):
            val = float(re.search(r'\b'+k+r'=([\d.]+)(?:,|\))',s).group(1))
            assert val == r[k]
            independently_checked += 1
        for k, value in [('hf_output_len', 'None'),('ignore_eos','False'),
                         ('append_result','True'),('save_detailed','False'),('max_concurrency','None')]:
            assert re.search(r'\b'+k+'='+value+r'(?:,|\))',s)
            independently_checked += 1
        assert bench_lines[r['source_line']-1] == s
    over = data['benchmark']['oversample_log_messages']
    assert len(over) == 20 and [x['preceding_run_id'] for x in over] == [f'{n:06d}' for n in range(181,201)]
    for r in over:
        assert bench_lines[r['source_line']-1] == r['message']
    meta = read(raw/'log.json')
    assert [r['request_rate'] for r in meta] == list(range(1,101))*2
    assert all(r['num-prompts'] == 300*r['request_rate'] for r in meta)
    # Recheck the exact saved server evidence lines; do not treat absent events as proved absent.
    wanted = {r['source_line']:r['text'] for group in data['server']['markers'].values() for r in group}
    checked = 0
    with (raw/'vllm_server.log').open(encoding='utf-8', errors='replace') as f:
        for n,line in enumerate(f,1):
            if n in wanted:
                assert re.sub(r'\x1b\[[0-9;]*m','',line).strip() == wanted[n]
                checked += 1
    assert checked == len(wanted)
    manifest = read(OUT/'source_manifest.json')
    assert all(digest(x['path']) == x['sha256'] for x in manifest)
    protected = read(OUT/'preservation_hashes.json')
    assert all(digest(p) == h for p,h in protected.items())
    p = Links()
    p.feed((OUT/'report_ko.html').read_text(encoding='utf-8'))
    assert p.table_depth == 0 and p.headers == 1
    local = [h for h in p.links if not h.startswith(('https:','http:','#'))]
    assert all((OUT/h).exists() for h in local)
    decision, plan = read(OUT/'decision.json'), read(OUT/'experiment_plan.json')
    assert not any(x['implemented'] for x in decision['candidate_priority'])
    assert plan['acceptance']['numeric_tolerances'] is None
    result = {'namespace_numeric_and_option_fields_independently_checked': independently_checked,
              'namespace_source_lines_checked': 200,'oversampling_lines_checked':20,
              'server_evidence_lines_rechecked':checked,
              'source_hashes_checked':len(manifest),'protected_files_checked':len(protected),
              'local_report_links_checked':len(local),
              'report_structure_checked':True,'model_not_refitted':True,
              'verification_script_sha256':digest(__file__)}
    (OUT/'independent_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
