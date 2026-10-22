"""Stage 4-4: source-grounded minimal-revision decision, not a model fit."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from aidc_facility.carryover_revision_evidence import (
    sha256, scan_bench, scan_server, summarize_settings, match_runs, chronology,
    source_anchors, work_scale_counterexample,
)

URLS = {
    'datasets.py': 'vllm/benchmarks/datasets.py',
    'serve.py': 'vllm/benchmarks/serve.py',
    'endpoint_request_func.py': 'vllm/benchmarks/lib/endpoint_request_func.py',
    'kv_cache_utils.py': 'vllm/v1/core/kv_cache_utils.py',
    'api_server.py': 'vllm/entrypoints/openai/api_server.py',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def table(headers, rows):
    return '<div class="table"><table><thead><tr>'+''.join('<th>'+escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def decisions():
    return {
        'stage': '4-4',
        'decision': 'Do not calibrate a new work law or cache-dependent capacity from these observations alone.',
        'status_ko': '최소 보완 방향은 조건부로 정리했으나 수치 보정·최종 채택은 보류',
        'candidate_priority': [
            {'candidate': 'S1-W: heterogeneous request work', 'implemented': False,
             'basis': 'Input lengths may differ; MLPerf output limits depend on reference lengths in version-matched upstream source.',
             'missing': 'Exact benchmark request list/tokenizer revision, actual outputs, lifetime/remaining work, and work-to-service mapping.',
             'gate': 'Recover or measure per-request work covariates and processing events; normalize work scale before estimation.'},
            {'candidate': 'S1-C: additional cache/workload state in capacity', 'implemented': False,
             'basis': 'Prefix caching enabled; history/condition order not independently controlled; stage 4-3 throughput drift.',
             'missing': 'Matched cold/warm controls, per-request cache queries/hits and separately measured admissions/completions.',
             'gate': 'Only consider after separating work-composition and admission effects; no regression on rolling cache ratio alone.'},
        ],
        'minimal_next_action': 'Request original benchmark artifacts or design a controlled one-serving-unit experiment; no further blind fitting to the same state traces.',
        'no_change': ['S1 equations/parameters', 'M3', 'R3', 'facility sizing', 'power-input definition'],
        'not_proved': ['cache caused the intermediate-rate error', 'all stock v0.11.0 code exactly matches the deployed binary',
                       'same seed establishes the exact historical request sequence', '5-minute power accuracy',
                       'safe operating capacity or SLO', 'actual per-request completions reconstructed'],
    }


def experiment_plan():
    return {
        'status': 'proposal only; not executed; researcher-designed protocol, not copied from a paper',
        'first_choice': 'Ask data owners for already-collected artifacts before commissioning a new GPU experiment.',
        'artifact_request': [
            'Exact vLLM commit/local diff and package lock; model/tokenizer/chat-template revisions.',
            'Dataset revisions/shards and ordered sampled request list (hashed IDs acceptable), prompt tokens and max_tokens.',
            'Actual admission, scheduler start, completion/cancel/timeout records with matched request IDs and clocks.',
            'Serving launch config including max_num_seqs, token budget, cache settings and restart/reset history.',
            'Benchmark runner/time limit/termination policy and whether any request queue was naturally drained.',
        ],
        'minimal_experiment': {
            'unit': 'One serving unit matching the NLR TP4 configuration, or explicitly label a new hardware/configuration domain.',
            'workload': 'Two services separately; fixed, versioned request list and output limits; hold token composition comparable across phases.',
            'sequence_proposal_rps': [20, 50, 20, 0],
            'targeted_intermediate_plateaus': {'InstructCoder':[40], 'mlperf-inference-llama2-data':[46,48]},
            'plateau_note': 'Include independent matched-composition plateaus at the observed failure rates; the 20-50 sequence alone does not retest those failures.',
            'rationale': 'Low input, congestion challenge, low-input recovery, no-new-arrival drain. 50 is a challenge condition, not certified capacity.',
            'nonzero_phase_duration_proposal_s': 300,
            'drain_rule': 'Stop arrivals but retain the server until all admitted work exits; if a safety timeout intervenes, record residual stock and cancellations as censored, not completed.',
            'cache_control': 'Paired cold/warm starts with the same request list; document initial probe/warmup. Use controlled restarts or verified in-process reset in an isolated test environment; do not expose development endpoints publicly.',
            'repeats_proposal': 3,
            'repeat_note': 'Three is a starting pilot design, not a literature-backed sufficiency or confidence guarantee.',
            'order_control': 'Vary workload ordering across repeats using recorded permutations; counterbalance cold/warm run order.',
            'separation': 'Fit on designated runs; evaluate on unseen request orders and full up/down/drain runs. Do not tune on held-out outcomes.',
        },
        'measurements': [
            {'name': 'request_id / dataset_row_hash / sequence_index', 'need': 'Connect workload, arrival and actual exit; preserve sequence without publishing prompt text.'},
            {'name': 'scheduled_send / actual_send / server_admit / scheduler_start / first_token / last_token / exit_reason', 'need': 'Separate dispatch lag, queueing, service and cancellation; client TTFT is not scheduler start.'},
            {'name': 'prompt_tokens / requested_max_tokens / actual_output_tokens / per-request prefix queries and hits', 'need': 'Separate configured token limit, actual work and cache history.'},
            {'name': 'Running / Waiting / finished / aborted / preempted / scheduled prefill and decode tokens', 'need': 'Event counters preferred; proposed 1-s snapshots as a supplement, not a replacement for events.'},
            {'name': 'GPU and non-overlapping CPU/node power / clock offset / sampling interval', 'need': 'Preserve native power samples and energy integrate to 300-s bins; do not double-count CPU domains.'},
        ],
        'validation': [
            'Exact request-count reconciliation with separately recorded cancellations and pending requests.',
            'Running/Waiting RMSE, MAE, bias by phase and run; no weighted composite J.',
            'Waiting-onset timing and false/missed onset; backlog area (request*s), peak and end-of-phase stock.',
            'Drain time and remaining-work trajectory; mark unobserved/censored drain explicitly.',
            'Only after a separate power mapping: 300-s energy and mean-power bias/MAE; compare carryover versus no-carryover on identical arrivals.',
            'Repeat 300-s aggregation with a 150-s offset; disclose partial edge bins, do not hide carryover by alignment to input steps.',
        ],
        'acceptance': {
            'numeric_tolerances': None,
            'reason': 'Need intended DOE/PCC energy-error budget and repeated-measurement variability; do not create a post-hoc pass threshold.',
            'gate': 'Predeclare tolerances before opening the new evaluation set. A lower fit error alone does not establish validity.',
        },
    }


def report(out, data):
    bench, srv = data['benchmark'], data['server']
    counts = {k: len(v) for k,v in srv['markers'].items()}
    settings = summarize_settings(bench['runs'])
    assert len(bench['runs']) == 200
    for k, value in {'seed':['0'], 'hf_output_len':['None'], 'ignore_eos':['False'],
                     'append_result':['True'], 'save_detailed':['False'],
                     'max_concurrency':['None']}.items():
        assert settings[k] == value, (k, settings[k])
    assert counts['engine_init'] == counts['shutdown'] == 1
    assert all(r['strict_rate_1_to_100_order'] for r in data['run_chronology'])
    matrix = [
        ['요청률·실행 순서', '원자료 확인', '200개 설정·실행 metadata·benchmark Namespace 일치. 서비스별 1→100 순서.', '실제 서버 admission 시각과는 다름.'],
        ['seed·출력 길이 옵션', '원자료 확인', 'seed=0, hf_output_len=None, ignore_eos=False. 두 서비스 모두 동일 옵션.', '동일 옵션이 동일한 실제 출력 길이를 뜻하지 않음.'],
        ['요청 token 상한 규칙', '버전 대응 원본 코드로 조건부 해석', 'InstructCoder: 기본 200 token 상한. MLPerf: reference answer token 길이를 상한으로 전달.*2,*3', '배포 코드 무수정·동일 tokenizer 가정. 실제 생성 길이 미확인. custom_output_len=256은 HF 경로에 적용하지 않음.'],
        ['요청 목록·순서', '부분적으로만 식별', 'HF streaming 및 seed 기반 shuffle, 순차 표본 선택 규칙 확인.*2', '실험 당시 dataset/tokenizer snapshot·선택 ID 미확보. 정확한 목록·시간별 token 분포 복원 불가.'],
        ['cache·실행 이력', '설정 확인, 인과 미식별', f'prefix caching 켜짐. engine 초기화 {counts["engine_init"]}회, 종료 {counts["shutdown"]}회 기록.', 'cache reset 미기록이 reset 부재 증명은 아님. prompt 반복·보유 block·개별 hit 여부 미확인.'],
        ['Running·Waiting·token/s', '기존 집계 관측', '기존 약 10초 상태 기록 및 직전 구간 처리량 사용 가능.', '요청별 service·remaining work·prefill/decode 분리 식별 불가.'],
        ['요청별 완료·실제 출력', '기존 부재 판단 유지, 저장 옵션 해석 정정', 'save_detailed=False로 상세 배열은 제거. append_result=True이므로 stock 코드상 요약 저장 경로는 켜져 있음.*3', 'save_result=False만으로 결과 부재를 설명할 수 없음. HTTP 200도 streaming 완료가 아님. 실제 종료·파일 누락 원인 미확인.'],
        ['동시 처리 한도', '런타임 설정 미확인', '약 1,024 Running 관측은 유지. client max_concurrency=None 확인.', 'client 제한·server max_num_seqs·관측 최대값을 동일시하지 않음.'],
    ]
    body = '''<h1>④-4 S1 최소 보완 근거 및 검증 계획</h1>
<p>2026-10-06 · 요청 처리상태 모델 검토 · 기존 S1/M3/R3 변경 없음</p>
<p class="lead"><strong>결론:</strong> 요청 설정과 cache 이력의 추가 단서는 확보했다. 그러나 요청별 실제 작업량과 처리속도 변화를 분리 식별할 수 없어,
새로운 작업량 분포나 cache 보정식을 수치적으로 채택하지 않는다. 다음 작업은 원 데이터 생산자의 기록 확보 또는 통제된 상승·하강·drain 실험이다.</p>
<h2>1. 이번에 확인한 범위</h2><p>기존에 없다고 확인한 요청별 완료 로그를 반복 탐색하지 않았다.
online-rate config·실행 metadata·benchmark 옵션을 교차 확인하고, 서버의 cache/수명주기 기록을 제한된 패턴으로 점검했다.
공개 vLLM v0.11.0 소스 5개는 분석용으로 내려받아 SHA-256과 위치를 저장했으며 실행하지 않았다.</p>'''
    body += table(['정보', '근거 수준', '확인한 내용', '남은 한계'], matrix)
    over = bench['oversample_log_messages']
    if over:
        body += '<p><strong>추가 관측:</strong> oversampling 메시지 '+str(len(over))+'건. 연결된 run ID·요청률은 evidence.json에 보존했다. '+\
            '이는 요청 목록 생성 중 일부 재표집을 실제로 수행했다는 기록이며, 정확한 요청 목록·실제 도착·완료를 복원한 것은 아니다.</p>'
        assert {r['preceding_dataset'] for r in over} == {'mgoin/mlperf-inference-llama2-data'}
        assert [r['preceding_rate'] for r in over] == list(range(81,101))
        body += '<p>해당 기록은 Llama2-data의 81–100 request/s 실행에 있다. 따라서 기존 고율 benchmark는 혼잡뿐 아니라 '+\
            '재표집을 포함하는 workload 조건의 결과로 설명해야 한다. 재표집이 R3의 성능 또는 S1 오차를 유발했다고 입증한 것은 아니다.</p>'
    body += '<p>초기 단일 요청 시험 시작 기록은 '+str(bench['initial_probe_count'])+'건이다. '+\
        'Successful requests 요약 출력 기록은 '+str(len(bench['successful_request_summary_lines']))+'건이다. '+\
        '후자의 부재는 모든 요청 실패 또는 특정 강제 종료 원인을 증명하지 않는다.</p>'
    body += '''<h2>2. 새롭게 확인한 핵심 단서</h2>
<h3>2.1 같은 hf_output_len=None도 서비스별 의미가 다르다</h3>
<p>원본 HF loader는 hf_output_len을 서비스별 sampler에 전달한다. InstructCoder의 기본 요청 출력 상한은 200이고,
MLPerf는 reference answer를 token화한 길이를 사용한다. 이 값은 HTTP payload의 max_tokens로 전달된다.*2,*3
ignore_eos=False이므로 상한까지 항상 생성했다고 가정할 수 없다. 코드 서비스도 입력 길이와 실제 출력 길이가 같다는 근거는 없다.</p>
<p>따라서 요청당 작업량=1은 물리적으로 동일한 요청이라는 관측이 아니라 S1의 정규화·단순화 가정이다.
token 길이 차이는 이질적 작업량 후보의 근거는 되지만, token을 곧바로 계산시간에 선형 환산할 근거는 아니다.</p>
<h3>2.2 요청률과 실험 순서·cache 이력이 분리되지 않았다</h3>
<p>두 서비스 각각 1→100 request/s 순서이고 모든 실행 seed는 0이다. 원본은 같은 seed로 데이터셋을 shuffle한 뒤 앞에서 요청을 선택하며,
실행별 요청 수는 300×요청률이다.*2 동일 dataset snapshot·패키지 동작·tokenizer와 filtering을 가정하면 실행 간 요청이 겹칠 수 있다.
다만 정확한 역사적 목록, oversampling 영향, 실제 cache 보유·eviction을 알 수 없어 반복 비율이나 cache 효과를 실측값으로 제시하지 않는다.</p>
<p>engine 초기화 한 번과 켜진 prefix caching은 순서·이력 영향의 가능성을 제시한다. 그것이 ④-3 오차의 단독 원인이라는 뜻은 아니다.
기존 홀수/짝수 rate 비교도 이 이력을 독립적으로 통제한 실험은 아니므로 탐색 결과로 유지한다. 기존 M3/R3의 정확성이 자동으로 부정되는 것도 아니다.</p>
<h3>2.3 cache 비율을 바로 처리율 보정 변수로 넣지 않는다</h3>
<p>prefix cache 지표는 최근 요청을 대상으로 하는 token-query hit 비율이며 약 10초 token/s와 집계창이 다르다.*4
cache·요청 조성·유입률을 분리하지 않은 채 μ(R, cache)를 맞추면 시간 추세를 원인처럼 학습할 수 있다.
서버 reset 관련 기록 부재를 cache 미초기화의 확정 증거로 사용하지 않는다.</p>
<h2>3. 현재 자료로 분리하기 어려운 것</h2>
<p>S1은 활성 요청에 총 처리능력을 균등 분배한다: <span class="eq">ψ(R)=μ(R)/R</span>.*1
이 틀에서 모든 작업량과 μ를 같은 배수로 바꾸면 완료 궤적은 동일하다. 아래는 NLR fitting이 아닌 단위 정규화의 대수적 예시다.</p>'''
    body += table(['예시', '요청당 작업량', '총 작업 처리율', 'Running', '고정 상태에서 완료까지'],
                  [['A',1,'40 work-unit/s',100,'2.5 s'],['B',2,'80 work-unit/s',100,'2.5 s']])
    body += '''<p>따라서 평균 작업량의 단위를 먼저 고정해야 한다. 그 이후에도 현재 집계 상태만으로 작업량 분포,
시간별 처리율, 실제 유입률, 취소를 모두 동시에 식별할 수는 없다. 이 예시는 모든 작업량 분포가 동등하다는 증명이 아니다.</p>
<h2>4. 최소 수정안 결정</h2>'''
    body += table(['후보', '수정 내용', '이번 결정'], [
        ['S1-W', '동일 작업량 대신 요청별 작업량 차이 반영', '첫 보완 후보로 유지하되 실제 작업 공변량·완료 기록 없이는 분포 선택/보정하지 않음.'],
        ['S1-C', 'Running 외에 workload/cache 상태에 따른 처리율', '작업량과 유입 영향을 통제한 뒤에도 차이가 남을 때만 검토. 현재 보류.'],
        ['현 S1', '기존 식·파라미터 유지', '④-3의 제한적 탐색 범위만 유지. 시간가변·5분 전력 검증 완료로 표시하지 않음.'],
    ])
    body += '''<p>두 수정안을 동시에 도입하거나, 시간 경과에 따라 μ를 임의로 낮추거나, 가장 잘 맞았던 유입률 배율을 보정값으로 채택하지 않는다.
M3의 입력을 Running·완료율로 치환하지 않고 R3도 변경하지 않는다.</p>
<h2>5. 가장 작은 추가 자료 요청</h2>
<ol><li>실제 benchmark 요청 목록/순서와 dataset·tokenizer·vLLM revision, 로컬 수정 여부.</li>
<li>요청별 입력 token·출력 상한·실제 출력 token과 admission/start/first-token/completion/abort 기록.</li>
<li>서버 max_num_seqs·token budget·cache 설정, 재시작/초기화 이력, warmup/초기 시험 요청과 실행 종료 정책.</li></ol>
<p>파일을 확보하기 전 최신 Hugging Face 자료와 다른 tokenizer로 token을 재계산하여 원 실험값이라고 표시하지 않는다.
finite workload나 Azure token 분포를 online-rate의 실제 작업량 분포로 대체하지 않는다. 원 저자에게 메시지를 보내지는 않았다.</p>
<h2>6. 추가 측정이 필요할 때의 최소 실험안</h2>
<p>아래 수치·순서는 우리 연구의 파일럿 제안이며 문헌에서 정해 준 값이나 이미 실행한 실험이 아니다.</p>'''
    body += table(['항목', '제안'], [
        ['구성', 'NLR과 같은 한 serving 단위(TP4)를 우선 사용. 다른 장비라면 새로운 검증 영역으로 구분.'],
        ['입력', '서비스별 20→50→20→0 request/s. 비영(非零) 단계는 우선 각 300초. 50은 혼잡 유도 후보이며 처리용량 확정값 아님.'],
        ['기존 실패 조건 재검증', '별도 동일 작업구성 plateau: InstructCoder 40, Llama2-data 46·48 request/s. 위 20→50 전이만으로 중간 영역 검증을 대신하지 않음.'],
        ['종료', '마지막 0 구간에서 새 요청만 중단하고 미완료 작업이 소진될 때까지 관측. 안전 timeout 발생 시 남은 작업·취소를 별도 기록.'],
        ['작업 구성', 'token 구성과 출력 상한이 비교 가능하도록 versioned 요청 목록 사용. 반복별 순서 permutation 기록.'],
        ['cache', '동일 요청 목록으로 cold/warm 초기조건 쌍 비교. 순서 교차 및 초기 시험 요청 기록. 실험 전 조건 확인.'],
        ['반복', '초기 파일럿 3회 제안. 충분한 통계 검증 횟수라는 의미는 아님.'],
        ['계측', '요청 사건 시각·ID·token 정보 우선. 상태 1초 기록은 보조안. 전력은 원해상도 보존 후 300초 에너지 집계.'],
    ])
    body += '''<p>v0.11.0의 reset_prefix_cache HTTP 경로는 개발 모드 조건 아래 있고 응답만으로 성공을 보증하지 않는다.*5
공개 서비스에 개발 endpoint를 열지 않는다. 격리된 실험에서 제어된 재시작 또는 성공 여부를 확인할 수 있는 reset을 사용한다.</p>
<h2>7. 검증 지표와 채택 조건</h2>
<ul><li>유입·완료·취소·남은 요청의 수지 일치. HTTP 200이나 run 종료를 완료로 집계하지 않음.</li>
<li>Running·Waiting RMSE/MAE/bias를 phase별·run별로 평가. 임의 가중 합산 J를 추가하지 않음.</li>
<li>대기 발생 시점, 놓친/잘못 예측한 대기, 누적 backlog 면적(request·s), peak·종료 잔량과 drain 시간.</li>
<li>처리상태→전력 관계를 별도로 검증한 후에만 같은 요청자료에서 이월 전후 5분 평균전력·에너지 오차 비교.
5분 집계 경계를 150초 이동한 결과도 비교하고 불완전 경계 구간 처리방법을 기록.</li></ul>
<p>새 평가자료를 열기 전에 허용오차를 정해야 한다. 지금 임의의 합격 숫자를 만들지 않으며,
PCC/DOE 운영에서 허용 가능한 에너지 오차와 반복 실험 변동을 바탕으로 결정한다.
기존 상태 궤적의 fitting 오차 감소만으로 실제 시간가변 전력을 검증했다고 보지 않는다.</p>
<h2>8. 산출물·검증 기록</h2>
<p><a href="evidence.json">원자료 근거</a> · <a href="decision.json">최소 수정 결정</a> ·
<a href="experiment_plan.json">추가 측정 및 검증 계획</a> · <a href="source_manifest.json">출처와 해시</a> ·
<a href="verification.json">검증 결과</a> · <a href="../carryover_scope_diagnostic_v1/report_ko.html">④-3</a></p>
<h2>Reference 및 차용 범위</h2>
<p>*1 Gupta &amp; Zhang, <em>Approximations and Optimal Control for State-Dependent Limited Processor Sharing Queues</em>, Stochastic Systems, 2022,
<a href="https://doi.org/10.1287/stsy.2021.0087">DOI</a>. 처리공유 구조의 근거. NLR 경험적 μ와 작업량 단위 설정은 우리 재구성이다.</p>
<p>*2 <a href="https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/benchmarks/datasets.py">vLLM v0.11.0 datasets.py</a>:
get_samples, HuggingFaceDataset, InstructCoderDataset, MLPerfDataset. stock 소스 해석이며 배포 파일 일치 여부 미확인.</p>
<p>*3 <a href="https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/benchmarks/lib/endpoint_request_func.py">endpoint_request_func.py</a> 및
<a href="https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/benchmarks/serve.py">serve.py</a>: max_tokens 전달, 초기 시험 요청, 도착 생성과 저장 옵션.</p>
<p>*4 <a href="https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/core/kv_cache_utils.py">PrefixCachingMetrics</a>: 최근 요청 cache 지표 정의.</p>
<p>*5 <a href="https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/entrypoints/openai/api_server.py">api_server.py</a>: 개발 모드 cache reset 경로.</p>
<p>*6 NLR [3] online-rate config/log 및 benchmark/server 원자료. 이번 점검은 dataset 자체의 새로운 실험이 아니다.
[43]의 FIFO·활성 구간 구조는 이전 F1/F2 참고이며 S1의 직접 근거는 *1이다. 이번 추가 실험 설계는 우리 제안이다.</p>'''
    html = '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>④-4 최소 보완 근거</title><style>body{font:16px/1.7 "Malgun Gothic",Arial,sans-serif;color:#17212b;max-width:1120px;margin:40px auto;padding:0 24px}h1{font-size:28px}h2{font-size:22px;margin-top:36px;border-bottom:1px solid #ccd2d8}h3{font-size:18px;margin-top:28px}.lead{border-left:4px solid #315c83;padding:10px 18px;background:#f3f6f9}.table{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px;margin:18px 0}th,td{text-align:left;vertical-align:top;border-bottom:1px solid #d7dfe4;padding:10px;overflow-wrap:anywhere}th{background:#edf2f6}a{color:#205f91}.eq{font-family:Georgia,serif}@media print{body{margin:0;max-width:none}h2,h3{break-after:avoid}tr{break-inside:avoid}}</style><body>'+body+'</body></html>'
    (out/'report_ko.html').write_text(html, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/carryover_revision_evidence_v1')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if args.report_only:
        manifest = read(out/'source_manifest.json')
        for item in manifest:
            if Path(item['path']).resolve() != Path(__file__).resolve():
                assert sha256(item['path']) == item['sha256'], item['path']
        for p, digest in read(out/'preservation_hashes.json').items():
            assert sha256(p) == digest, p
        report(out, read(out/'evidence.json'))
        save(out/'decision.json', decisions())
        save(out/'experiment_plan.json', experiment_plan())
        for item in manifest:
            if Path(item['path']).resolve() == Path(__file__).resolve():
                item['sha256'] = sha256(__file__)
                item['bytes'] = Path(__file__).stat().st_size
        save(out/'source_manifest.json', manifest)
        print('Report rendered from stored evidence; model not executed.')
        return
    if (out/'evidence.json').exists():
        raise FileExistsError('Existing evidence is preserved; choose a new output.')
    out.mkdir(parents=True, exist_ok=True)
    raw = ROOT.parent/'NLR_dataset/00_raw_datasets/inference_online_rate_llama3_70b'
    sources = out/'sources'
    for name in URLS:
        if not (sources/name).is_file():
            raise FileNotFoundError(f'Acquire version-pinned public source first: {name}')
    previous = read(ROOT/'outputs/carryover_scope_diagnostic_v1/preservation_hashes.json')
    protected = {Path(k): v for k,v in previous.items()}
    for p in (ROOT/'outputs/carryover_scope_diagnostic_v1').glob('*'):
        if p.is_file(): protected[p] = sha256(p)
    for p in (ROOT/'src/aidc_facility').glob('*.py'):
        if p.name != 'carryover_revision_evidence.py': protected[p] = sha256(p)
    for p, old in protected.items():
        if sha256(p) != old: raise ValueError(f'Previous state changed before audit: {p.name}')
    source_paths = [raw/n for n in ('config.json','log.json','vllm_bench.log','vllm_server.log')]
    source_paths += [sources/n for n in URLS]
    source_paths += [Path(__file__),ROOT/'src/aidc_facility/carryover_revision_evidence.py']
    source_paths += [ROOT/'outputs/carryover_parameter_audit_v1/parameter_contract.json',
                     ROOT/'outputs/carryover_scope_diagnostic_v1/applicability_decision.json']
    manifest = []
    for p in source_paths:
        item = {'path': str(p), 'sha256': sha256(p), 'bytes': p.stat().st_size}
        if p.parent == sources:
            item['url'] = 'https://raw.githubusercontent.com/vllm-project/vllm/v0.11.0/'+URLS[p.name]
            item['version'] = 'v0.11.0 upstream tag; exact deployment commit not verified'
        manifest.append(item)
    known_server = read(ROOT/'outputs/nlr_processing_diagnostic_v1/run_summary.json')['raw_scan']['sha256']
    assert next(x['sha256'] for x in manifest if Path(x['path']).name == 'vllm_server.log') == known_server
    config, metadata = read(raw/'config.json'), read(raw/'log.json')
    bench = scan_bench(raw/'vllm_bench.log')
    match_runs(config, metadata, bench['runs'])
    server = scan_server(raw/'vllm_server.log')
    inventory = [{'relative_path': p.name, 'bytes': p.stat().st_size} for p in sorted(raw.iterdir()) if p.is_file()]
    data = {'created_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'NLR online-rate metadata, targeted server cache/lifecycle markers, versioned upstream loader; no deployment-source equality claim',
            'benchmark': bench, 'settings': summarize_settings(bench['runs']),
            'run_chronology': chronology(metadata), 'server': server, 'raw_directory_inventory': inventory,
            'source_anchors': source_anchors(sources), 'work_scale_example': work_scale_counterexample(),
            'historical_request_sequence_reconstructed': False,
            'new_capacity_or_work_distribution_estimated': False,
            'request_completion_absence': 'Previously established; no repeated broad search. Save flags rechecked as source evidence.',
            'source_snapshot_missing': 'Exact historical HF dataset/cache snapshot, selected request IDs, tokenizer/chat-template revision not supplied in the inspected online-rate package.'}
    save(out/'evidence.json', data)
    save(out/'decision.json', decisions())
    save(out/'experiment_plan.json', experiment_plan())
    save(out/'source_manifest.json', manifest)
    report(out, data)
    unchanged = all(sha256(p) == old for p,old in protected.items())
    sources_unchanged = all(sha256(x['path']) == x['sha256'] for x in manifest)
    assert unchanged and sources_unchanged
    checks = {'metadata_config_namespace_matched_runs': len(metadata),
              'namespace_ids_unique': len({r['id'] for r in bench['runs']}) == len(metadata),
              'source_anchors_checked': len(data['source_anchors']),
              'downloaded_source_files_syntax_checked_not_executed': len(URLS),
              'raw_server_matches_prior_sha256': True,
              'protected_files_unchanged': unchanged, 'protected_file_count': len(protected),
              'source_hashes_unchanged': sources_unchanged,
              'work_scale_algebra_check': data['work_scale_example']['same_completion_time'],
              'model_fit_or_simulation_executed': False, 'new_power_profile_generated': False,
              'external_message_sent': False}
    save(out/'preservation_hashes.json', {str(k):v for k,v in protected.items()})
    save(out/'verification.json', checks)
    print(json.dumps({'output':str(out),'verification':checks,
                      'server_marker_counts':{k:len(v) for k,v in server['markers'].items()}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
