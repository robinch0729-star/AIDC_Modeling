"""Phase 4-3. Diagnose frozen S1; do not refit it or generate facility power."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from aidc_facility.carryover_replay import make_segments
from aidc_facility.carryover_scope_diagnostic import (window_summary,selected_errors,
    waiting_onset,interval_diagnostics,replay_diagnostic,exposure_rates,size_for_mean_rate,block_rates)
from aidc_facility.nlr_processing_diagnostic import SERVICES,sha256,parse_metric_line
from run_carryover_replay import dump,SHORT
import run_nlr_processing_diagnostic as graphics

PRIOR=ROOT/'outputs/carryover_replay_v1'
ARRIVALS=ROOT/'outputs/facility_arrival_scaling_v1'
RAW=ROOT.parent/'NLR_dataset/00_raw_datasets/inference_online_rate_llama3_70b/vllm_server.log'
CORE=((SERVICES[0],40.),(SERVICES[1],46.),(SERVICES[1],48.))


def read_json(path): return json.loads(path.read_text(encoding='utf-8'))


def raw_check(wanted):
    """Recheck selected extracted snapshots against exact source lines and hash all bytes."""
    expected=next(x['sha256'] for x in read_json(ROOT/'outputs/nlr_processing_diagnostic_v1/source_manifest.json') if Path(x['path'])==RAW)
    h=hashlib.sha256(); checked=0
    wanted=wanted.set_index('source_line')
    assert wanted.index.is_unique
    ids=set(wanted.index)
    with RAW.open('rb') as stream:
        for number,line in enumerate(stream,1):
            h.update(line)
            if number not in ids: continue
            parsed=parse_metric_line(line.decode('utf-8','replace'),2026,number)
            original=wanted.loc[number]
            assert parsed is not None and pd.Timestamp(parsed['timestamp'])==original.timestamp
            for key in ('running_requests','waiting_requests','generation_tokens_per_s','prompt_tokens_per_s','gpu_kv_cache_pct','prefix_cache_hit_pct'):
                assert np.isclose(parsed[key],original[key],equal_nan=True)
            checked+=1
    assert checked==len(ids) and h.hexdigest()==expected
    return {'selected_raw_snapshots_reparsed':checked,'raw_log_sha256':h.hexdigest(),'raw_log_matches_prior_hash':True}


def facility_exposure(protocol):
    summary=read_json(ARRIVALS/'run_summary.json')
    assert summary['config']['instance_count'] is None
    assert summary['primary_multiplier']==1000
    files={
        'scaled_replay':'facility_arrivals_replay_K1000_seed20261005.npz',
        'conditional_60s':'facility_arrivals_conditional_60s_K1000_seed20261005.npz',
        'conditional_300s':'facility_arrivals_conditional_300s_K1000_seed20261005.npz'}
    demand=pd.read_csv(ARRIVALS/'facility_demand_5min.csv')
    result=[]; sizing=[]; summaries=[]; verified=0
    reference={}
    for method,filename in files.items():
        with np.load(ARRIVALS/filename,allow_pickle=False) as data:
            for service in ('code','conversation'):
                counts=data[service]; rates300=block_rates(counts,300)
                listed=demand[(demand.service==service)&(demand.multiplier==1000)].sort_values('bin_start_utc')
                np.testing.assert_array_equal(rates300*300,listed.facility_offered_requests.to_numpy())
                verified+=1
                if service in reference: np.testing.assert_array_equal(rates300,reference[service])
                else: reference[service]=rates300.copy()
                mean=float(counts.mean())
                if method=='scaled_replay':
                    summaries.append({'service':service,'seconds':len(counts),'five_minute_bins':len(rates300),
                        'mean_facility_rps':mean,'p95_facility_5min_rps':float(np.quantile(rates300,.95)),
                        'max_facility_5min_rps':float(rates300.max()),
                        'instances_to_keep_5min_mean_at_most_30':int(np.ceil(rates300.max()/30)),
                        'this_is_not_an_SLO_capacity_size':True})
                for target in protocol['facility_mapping']['mean_rate_targets_rps_per_instance']:
                    n=size_for_mean_rate(mean,target)
                    if method=='scaled_replay': sizing.append({'service':service,'illustrative_mean_target_rps':target,
                        'illustrative_instances':n,'achieved_mean_rps_per_instance':mean/n})
                    for seconds in (1,300):
                        rates=block_rates(counts,seconds)/n
                        result.append({'service':service,'method':method,'bin_seconds':seconds,
                            'illustrative_mean_target_rps':target,'illustrative_instances':n,
                            'actual_mean_rps_per_instance':float(rates.mean()),
                            'p95_rps_per_instance':float(np.quantile(rates,.95)),
                            'max_rps_per_instance':float(rates.max()),**exposure_rates(rates)})
    return pd.DataFrame(result),sizing,summaries,verified


def figures(out,core,params,windows,exposure):
    graphics.COLORS=['#111827','#7659a5','#3375b9','#df8230']
    panels=[]
    for service,rate in CORE:
        c=core[(service,rate)]; obs=c['obs']; base=c['base']; env=np.stack([x['running'].to_numpy() for x in c['initial_traces']])
        panels.append({'title':f'{SHORT[service]} {rate:.0f} req/s: Running',
            'series':[('NLR snapshots',obs.relative_run_s,obs.running_requests),
                ('Frozen S1',obs.relative_run_s,base.running),
                ('Initial-work grid min',obs.relative_run_s,env.min(axis=0)),
                ('Initial-work grid max',obs.relative_run_s,env.max(axis=0))],
            'xlabel':'Time from run start (s)','unit':'Requests','zero':True})
    graphics.plot(out/'focus_running_sensitivity.png','Initial-work assumptions do not uniquely explain state errors',
        'Curve parameters fixed. Bounds are pointwise extrema across assumptions, not confidence intervals.',panels,
        'Observed first state supplied near run +60 s. Initial work quantile powers: 0.25, 0.5, 1, 2, 4; arrival phases: 0.1, 0.5, 0.9.')
    panels=[]
    for service,rate in CORE[1:]:
        c=core[(service,rate)]; obs=c['obs']; env=np.stack([x['waiting'].to_numpy() for x in c['initial_traces']])
        panels.append({'title':f'{SHORT[service]} {rate:.0f} req/s: Waiting',
            'series':[('NLR snapshots',obs.relative_run_s,obs.waiting_requests),('Frozen S1',obs.relative_run_s,c['base'].waiting),
                ('Initial-work grid min',obs.relative_run_s,env.min(axis=0)),('Initial-work grid max',obs.relative_run_s,env.max(axis=0))],
            'xlabel':'Time from run start (s)','unit':'Requests','zero':True})
    graphics.plot(out/'focus_waiting_sensitivity.png','Initial-work changes still miss observed Waiting onset',
        'First positive logger snapshot is not an exact queue-onset timestamp.',panels,
        'Sensitivity cases are not measured initial remaining-work distributions. No parameter was selected or adopted.')
    panels=[]
    for service,rate in CORE:
        c=core[(service,rate)]; rows=pd.DataFrame(c['intervals'])
        panels.append({'title':f'{SHORT[service]} {rate:.0f} req/s: conditional departures',
            'series':[('Balance-based proxy',rows.relative_end_s,rows.conditional_departure_rps),
                      ('Frozen curve at observed R',rows.relative_end_s,rows.curve_endpoint_average)],
            'xlabel':'Logger interval end from run start (s)','unit':'Conditional requests / s','zero':False})
    graphics.plot(out/'conditional_rate_drift.png','A static rate curve leaves time-dependent discrepancies',
        'Proxy = configured rate - change in unfinished requests / interval. Not measured completion rate.',panels,
        'The curve is a work-service law; boundary remaining work, actual admissions and other departures are not observed.')
    panels=[]
    for service,rate in CORE:
        rows=pd.DataFrame(core[(service,rate)]['intervals'])
        panels.append({'title':f'{SHORT[service]} {rate:.0f} req/s: generation token throughput',
            'series':[('Logged generation throughput',rows.relative_end_s,rows.generation_tokens_per_s)],
            'xlabel':'Logger interval end from run start (s)','unit':'Tokens / s','zero':False})
    graphics.plot(out/'generation_throughput_drift.png','Token throughput changes within fixed-request-rate runs',
        'Logger interval-average generation throughput. This is not request completion throughput.',panels,
        'Variation may reflect work composition, serving dynamics or other factors; this plot alone does not identify the cause.')
    panels=[]
    for service in ('code','conversation'):
        x=exposure[(exposure.service==service)&(exposure.method=='scaled_replay')&(exposure.bin_seconds==300)]
        panels.append({'title':f'{service.capitalize()}: five-minute rate exposure',
            'series':[('2-30 req/s',x.illustrative_mean_target_rps,100*x['2_to_30']),
                      ('>30 to <60 req/s',x.illustrative_mean_target_rps,100*x.over_30_below_60),
                      ('60-100 req/s',x.illustrative_mean_target_rps,100*x['60_to_100']),
                      ('Outside 2-100 req/s',x.illustrative_mean_target_rps,100*(x.below_2+x.over_100))],
            'xlabel':'Illustrative mean offered rate per fixed instance (req/s)','unit':'Share of equal-duration bins (%)','zero':True})
    graphics.plot(out/'facility_rate_exposure.png','Facility relevance depends on the as-yet-unfixed instance allocation',
        'Same K=1000 demand. Fixed instance count = ceil(mean service demand / illustrative mean rate target).',panels,
        'Equal fractional allocation. No 17 MW sizing, SLO certification, queue simulation or power conversion.')


def write_report(out):
    """Report frozen diagnostics; do not turn the best sensitivity case into a fit."""
    load=lambda name:pd.DataFrame(read_json(out/name))
    w=load('window_diagnostics.json'); init=load('initial_work_sensitivity.json')
    admission=load('arrival_rate_sensitivity.json'); baseline=load('focus_baseline.json')
    e=load('facility_exposure.json'); v=read_json(out/'verification.json')
    old_scores=pd.DataFrame(read_json(PRIOR/'per_run_scores.json'))
    old_scores=old_scores[old_scores.model.eq('S1_state_dependent_lps')]
    bands=old_scores.groupby(['service','band'],sort=False).agg(runs=('run_id','size'),
        Running_RMSE_mean=('running_rmse','mean'),Running_RMSE_max=('running_rmse','max'),
        Waiting_RMSE_mean=('waiting_rmse','mean'),Waiting_RMSE_max=('waiting_rmse','max')).reset_index()
    sensitivities=init.groupby(['service','rate'],sort=False).agg(
        Running_RMSE_min=('running_rmse','min'),Running_RMSE_max=('running_rmse','max'),
        late_Running_RMSE_min=('late_running_rmse','min'),late_Running_RMSE_max=('late_running_rmse','max'),
        predicted_Waiting_max=('predicted_waiting_max','max')).reset_index()
    early=w[w.window.eq('fit_span')].set_index(['service','rate'])
    late=w[w.window.eq('later_span')].set_index(['service','rate'])
    drift=[]
    for service,rate in CORE+((SERVICES[1],90.),):
        a,b=early.loc[(service,rate)],late.loc[(service,rate)]
        drift.append({'service':service,'rate':rate,'early_departure_proxy':a.conditional_departure_rps,
            'late_departure_proxy':b.conditional_departure_rps,
            'early_generation_tps':a.generation_tokens_per_s_aligned_mean,'late_generation_tps':b.generation_tokens_per_s_aligned_mean,
            'generation_change_pct':100*(b.generation_tokens_per_s_aligned_mean/a.generation_tokens_per_s_aligned_mean-1),
            'early_prefix_cache_pct':a.prefix_cache_hit_pct_mean,'late_prefix_cache_pct':b.prefix_cache_hit_pct_mean,
            'late_frozen_curve_at_observed_R':b.frozen_curve_at_observed_running_mean})
    drift=pd.DataFrame(drift); dump(out/'key_temporal_evidence.json',drift.to_dict('records'))
    scope=[
        {'input_scope':'Observed even constant rates 2-30 req/s per NLR benchmark instance',
         'decision':'Limited exploratory use for Running/Waiting magnitude under matched initial conditions',
         'not_authorized_claim':'Safe capacity, latency/SLO, variable-arrival or energy accuracy'},
        {'input_scope':'Intermediate regime >30 and <60 req/s; specific failures at code40 and conversation46/48',
         'decision':'Withhold quantitative Waiting onset and backlog use in final facility model',
         'not_authorized_claim':'This whole continuous interval has been exhaustively tested or has a single capacity threshold'},
        {'input_scope':'Observed even high constant rates 60-100 req/s',
         'decision':'Exploratory congestion trend only, conditional on initial state and offered-rate assumption',
         'not_authorized_claim':'Validated long-horizon queue accumulation or acceptable service operation'},
        {'input_scope':'Outside observed rates, time-varying requests, workload/token distribution changes, 5-min energy',
         'decision':'Unvalidated; no production/final-study adoption from these diagnostics alone',
         'not_authorized_claim':'Five-minute averaging removes unresolved processing-state error'}]
    dump(out/'applicability_decision.json',{'status':'provisional research-use scope, not numerical acceptance certification',
        'model_parameters_changed':False,'scope':scope,'next_priority':'Identify available request/token/cache evidence before adding a new service law; power linking remains pending'})
    display_e=e[(e.method=='scaled_replay')&(e.bin_seconds==300)&e.illustrative_mean_target_rps.isin([20,30,40])].copy()
    for field in ('below_2','2_to_30','over_30_below_60','60_to_100','over_100'): display_e[field]*=100
    def table(frame):
        frame=frame.copy()
        if 'service' in frame: frame['service']=frame.service.replace(SHORT)
        return frame.to_html(index=False,border=0,float_format=lambda x:f'{x:,.2f}')
    footer_refs='''<h2>Reference 및 차용 범위</h2>
<p>*1 NLR [3]의 online-rate 원자료. 이번 핵심 비교의 snapshot 197개를 server 원문 줄과 다시 대조하고 파일 해시를 확인했다.
Running·Waiting은 scheduler 상태 기록이며 약 10초 간격이다. <a href="https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/metrics/loggers.py">vLLM v0.11.0 logger 원문</a>.</p>
<p>*2 V. Gupta and J. Zhang, <a href="https://doi.org/10.1287/stsy.2021.0087">Approximations and Optimal Control for State-Dependent Limited Processor Sharing Queues</a>, Stochastic Systems 12(2),205–225,2022.
S1의 처리공유·동시 처리 한도 구조만 차용했다. NLR 경험적 μ̂(R), 모든 신규 요청의 작업량=1, 초기 잔여량 가정은 우리 변경사항이다.
이 원문이 NLR 모델의 성능이나 적용 범위를 보증하는 것은 아니다.</p>
<p>*3 [43] Wilkins et al., <a href="https://arxiv.org/html/2603.18383v1#S3.SS3">From Servers to Sites</a>, arXiv:2603.18383v1,2026,§3.3.
이전 F1/F2는 FIFO·활성 구간 집계 구조를 단순화한 비교 모델이었다. 이번 S1 진단은 [43] 전체 모델 재현이 아니다.</p>
<p>*4 <a href="https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/core/kv_cache_utils.py">vLLM PrefixCachingMetrics</a>.
Prefix cache 지표는 기본 최대 최근 1,000개 요청을 대상으로 한 token query hit 비율이다.
generation token 처리량의 10초 logger 구간과 같은 집계 창이 아니다. 두 지표의 동시 변화만으로 인과관계를 주장하지 않았다.</p>
<p>*5 ②에서 구성한 Azure 요청자료의 공통 관측기간(2024-05-12 00:05–2024-05-16 23:55 UTC), K=1000 예시.
<a href="https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md">Azure LLM inference trace 2024</a>.
요청률 확대·instance 수 산식·균등 분배는 우리 시나리오 가정이며 Azure가 제시한 실제 시설 구성이 아니다.</p>'''
    html=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>④-3 이월 모델 오차 진단 및 적용 범위</title><style>
body{{font-family:'Malgun Gothic',Arial,sans-serif;max-width:1280px;margin:32px auto;padding:0 22px;color:#223447;line-height:1.8}}
h1{{font-size:28px}}h2{{font-size:22px;margin-top:34px}}h3{{font-size:18px}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}}
td,th{{padding:8px;border-bottom:1px solid #dbe3eb;text-align:right}}td:first-child,th:first-child{{text-align:left}}th{{background:#eff3f7}}
img{{max-width:100%;height:auto}}.scroll{{overflow-x:auto}}.note{{background:#fff4db;padding:15px 20px}}.eq{{background:#f2f5f8;padding:15px;font-family:Cambria,serif;font-size:20px}}a{{color:#21659a}}</style>
<h1>④-3 중간 요청률 오차 진단 및 S1의 잠정 적용 범위</h1>
<p><strong>결론:</strong> 시험한 초기 잔여량 가정만으로 중간 요청률 오차를 해소하지 못했다.
실행 중 token 처리량·cache 지표 변화와 고정 처리율 곡선의 불일치가 함께 관찰된다.
다만 실제 admission·완료·요청별 작업량이 없으므로 인과 원인은 하나로 식별하지 못했다.
현재 S1은 <strong>제한된 조건의 탐색용 모델</strong>로 유지하고, 최종 시설 부하생성기로의 채택은 보류한다.</p>
<p class="note">S1·M3·R3를 수정하거나 재보정하지 않았다. 이미 검토한 짝수 rate 100개 run에 대한 후속 진단이며 새로운 독립 검증이 아니다.
민감도에서 가장 잘 맞는 설정을 선택해 기존 성능을 갱신하지 않았다.</p>
<h2>1. 수행한 진단</h2>
<ul><li>핵심 11개 run, 원문 snapshot {v['selected_raw_snapshots_reparsed']}개 재대조.</li>
<li>초기 잔여 작업량 5종 × arrival 위상 3종 × 11개 run = {v['initial_work_cases']}개 민감도.</li>
<li>설정률 대비 실제 유입률 가정 0.95/0.98/1/1.02/1.05배, 총 {v['arrival_rate_cases']}개 민감도.</li>
<li>각 run의 초기(60–120초)·중기(120–180초)·후기(180–240초), 그리고 60–150/150–240초 비교.
실제 계산은 각 창에 포함된 첫–마지막 snapshot 사이만 사용했다.</li>
<li>기존 시설 요청자료의 1초·5분 구간별 조건부 요청률 분포 비교. instance 수는 확정하지 않았다.</li></ul>
<h2>2. 초기 조건만의 문제인가?</h2>
<p>모든 신규 요청의 작업량은 기존처럼 1로 유지하고, 초기 Running의 잔여 작업 분위수 u만 u<sup>0.25</sup>, √u, u, u², u⁴로 바꿨다.
아래 최소·최대는 가정별 결과 범위이며 신뢰구간이 아니다. 지표별 최솟값이 서로 다른 설정에서 나올 수도 있다.</p>
<div class="scroll">{table(sensitivities)}</div>
<ul><li>InstructCoder 40 req/s는 초기 상태를 바꿔도 전체 Running RMSE 최솟값이 약 109개, 후기 최솟값이 약 135개다.</li>
<li>Llama2-data 46·48 req/s는 15개 초기 가정 모두 Waiting=0으로 예측한다. 실제 최대 Waiting은 5·166개다.</li>
<li>모델 내부에서는 핵심 세 run의 최초 Running 요청들이 첫 비교 시점부터 10–20초 안에 모두 완료된다.
이것은 모델의 계산 결과이지 NLR 실제 완료시간 관측이 아니다. 초기 차이가 후속 재고에 남을 수 있으므로 초기 영향이 완전히 없다는 뜻도 아니다.</li></ul>
<img src="focus_running_sensitivity.png" alt="초기 잔여량 민감도 Running">
<img src="focus_waiting_sensitivity.png" alt="초기 잔여량 민감도 Waiting">
<h2>3. 실행 중 처리 양상이 달라지는가?</h2>
<p>앞부분은 run+60–150초, 뒷부분은 +150–240초다. token 처리량은 logger 구간 끝의 값에 해당 구간 길이를 가중하고,
창의 첫 snapshot이 나타내는 이전 구간은 제외했다. Running은 순간 상태, token/s는 직전 logger 구간 평균이다.*1</p>
<div class="scroll">{table(drift)}</div>
<ul><li>InstructCoder 40 req/s는 후기 조건부 이탈률이 약 39.39 req/s인데, 관측 Running 위치에서 고정 곡선은 약 38.55다.
요청 수 수지상 작은 처리율 차이가 누적 Running 오차로 확대될 수 있다.</li>
<li>Llama2-data 46·48 req/s는 생성 token 처리량이 후기에 약 14–15% 감소하고, prefix cache 표시값도 약 48%에서 35%로 내려간다.
Running이 늘어도 조건부 이탈률은 오히려 감소하는 구간이 나타난다.</li>
<li>이 현상은 46·48 두 조건에만 국한되지 않는다. 90 req/s에서도 후반 token 처리량 및 조건부 이탈률 감소가 관찰된다.
따라서 고부하 전체 평균 오차가 작아 보였다는 이유로 장기간 누적량까지 정확하다고 해석할 수 없다.</li>
<li><strong>확인한 것은 지표의 동시 변화다.</strong> cache 적중 감소가 단독 원인인지, prompt/token 구성·요청 순서·잔여 작업량·scheduler 동작이 원인인지는 구분되지 않는다.
cache 지표는 최근 요청에 대한 이동 집계라 10초 token 처리량과 집계 창도 다르다.*4</li></ul>
<p class="eq">B=Running+Waiting<br>μ̃ = r<sub>configured</sub> − slope(B) &nbsp; (조건부 요청 이탈률)<br>
ψ(R)=μ̂(R)/R &nbsp; (S1의 요청별 정규화 작업 진행률)*2</p>
<p>첫 진단식은 실제 admission≈설정률이고 취소·기타 이탈이 없다는 가정 아래 상태 수지에 OLS 추세를 적용한 우리 계산이다.
μ̃는 직접 측정한 완료율이나 확정 capacity가 아니다. μ̂는 작업 처리율 곡선이므로 짧은 구간에서는 남은 작업량의 경계효과도 있다.
두 값의 차이만으로 μ̂의 오차를 독립 식별했다고 주장하지 않는다.</p>
<img src="conditional_rate_drift.png" alt="조건부 이탈률과 고정 곡선 비교">
<img src="generation_throughput_drift.png" alt="generation token 처리량 변화">
<h2>4. 실제 유입률을 안다는 가정의 영향</h2>
<p>±2%·±5%는 측정된 불확실성 범위가 아니라 연구자가 정한 가정 민감도다.
예를 들어 InstructCoder 40 req/s에서 유입률만 0.98배로 가정하면 Running RMSE가 약 156개에서 27개로 줄지만,
<strong>실제 유입률이 39.2였다는 증거가 아니다.</strong> 처리율 곡선의 편향을 입력 변경으로 상쇄했을 수도 있다.</p>
<div class="scroll">{table(admission[admission.apply(lambda r:(r.service,r.rate) in CORE,axis=1)][['service','rate','assumed_arrival_factor','running_rmse','waiting_rmse','late_running_rmse','predicted_waiting_max']])}</div>
<p>따라서 이 설정을 모델 보정값으로 채택하지 않는다. 유입과 처리 양쪽 정보를 분리 측정해야 원인을 구분할 수 있다.</p>
<h2>5. 실제 시설 시나리오에서 중요한 구간인가?</h2>
<p>현재 ② 자료는 서비스별 시설 전체 요청량만 구성했고 instance 수는 null이다.
따라서 실제 계획의 구간 노출비율을 단일 숫자로 확정할 수 없다. 다음은 배정 수에 따른 예시다.*5</p>
<p class="eq">N<sub>d</sub>=ceil(mean(Λ<sub>d</sub>)/r<sub>target</sub>), &nbsp;
r<sub>d,k</sub>=Λ<sub>d,k</sub>/N<sub>d</sub></p>
<p>r<sub>target</sub>은 원하는 <strong>평균</strong> 요청률이며 최대값 제한이나 처리용량이 아니다.
서비스별 고정 N과 균등 분배를 가정했다. 두 서비스에 같은 평균 목표를 적용한 예시이므로 서비스별 처리능력을 반영한 최적 배정이 아니다.
K=1000은 수요 확대 배수이지 node 수가 아니며, N은 17 MW 목표로 산정한 서버 수가 아니다.</p>
<div class="scroll">{table(display_e[['service','illustrative_mean_target_rps','illustrative_instances','below_2','2_to_30','over_30_below_60','60_to_100','over_100']])}</div>
<p>구간 열의 단위는 시간 비율(%). 평균 20 req/s 예시에서도 코드 서비스의 27.4%, 대화 서비스의 2.6%가 &gt;30~&lt;60 req/s에 들어간다.
평균 30 req/s 예시에서는 각각 15.4%, 53.7%다. 코드 서비스는 평균 30일 때 60–100 구간도 약 19.5%이므로
중간 구간 비율만 보고 더 안전하다고 해석하면 안 된다.</p>
<img src="facility_rate_exposure.png" alt="시설 요청률 구간 노출">
<p>5분 총량은 세 arrival 구성법에서 일치하지만 1초 노출은 다르다. 평균 20 req/s 예시의 코드 서비스는
원형 확대(replay)에서 1초 구간의 약 1.73%가 60 req/s 이상이고, 60초 조건부 재배치에서는 약 0.11%,
300초 조건부 재배치에서는 0%다. 이는 현재 가정의 민감도이지 실제 혼잡 빈도 관측이 아니다.
5분 평균만으로 짧은 과부하나 이월이 없다고 판정할 수 없다.</p>
<h2>6. 현재 S1의 적용 범위 결정</h2>
<table><tr><th>조건</th><th>이번 단계의 판단</th><th>사용하면 안 되는 주장</th></tr>
<tr><td>관측된 저율 고정 입력<br>짝수 2–30 req/s</td><td>동일 NLR 구성·초기 상태 조건에서 Running·Waiting 규모의 탐색용 근사로 유지</td><td>안전 운전용량, latency/SLO, 시간가변 입력의 정확도 보증</td></tr>
<tr><td>중간 영역 &gt;30~&lt;60 req/s</td><td>최종 시설 모델의 대기 발생·이월량 정량 예측에 사용 보류</td><td>모든 연속 rate를 검증했거나 하나의 임계 처리용량을 확정했다는 주장</td></tr>
<tr><td>관측된 고율 고정 입력<br>짝수 60–100 req/s</td><td>혼잡·누적의 대략적 경향을 보는 스트레스 비교에 한정</td><td>장기 누적 대기량의 정확성, 실제 운영 적합성</td></tr>
<tr><td>범위 밖, 시간가변 요청,<br>다른 token 구성, 5분 전력</td><td>미검증. 현재 결과만으로 채택하지 않음</td><td>평균화만으로 처리 상태 오차가 해소된다는 주장</td></tr></table>
<p>이 구간들은 ④-2에서 사용한 설명용 분류다. 새 허용오차나 문헌의 SLO 기준으로 합격 판정을 만든 것이 아니다.</p>
<div class="scroll">{table(bands)}</div>
<h2>7. 무엇이 확인됐고 무엇이 남았나?</h2>
<table><tr><th>쟁점</th><th>판단</th></tr>
<tr><td>CSV 추출·초기화 구현이 잘못됐는가?</td><td>선택 원문 197개, 고정 모델의 이전 결과 일치, 수지 보존을 확인. 이 검사 범위에서 불일치 없음.</td></tr>
<tr><td>초기 잔여량만 고치면 해결되는가?</td><td>시험한 넓은 가정 범위에서는 아님. 특히 대화 46·48의 Waiting 발생은 계속 놓침.</td></tr>
<tr><td>항상 같은 μ(R)·요청당 작업량=1로 충분한가?</td><td>현재 보정·가정 조합은 관측 궤적 일부를 설명하지 못함. 곡선과 작업량 중 어느 쪽 문제인지는 분리 식별 불가.</td></tr>
<tr><td>cache/작업 조성 변화가 원인인가?</td><td>관련 지표가 함께 변한다는 증거는 있음. 단독 인과 원인은 미확인.</td></tr>
<tr><td>시설에서 무시할 수 있는가?</td><td>instance 배정에 따라 오차 영역 노출이 상당할 수 있어 현재로서는 무시할 근거 부족.</td></tr></table>
<h2>8. 다음 작업</h2>
<ol><li><strong>필요한 정보의 가용성 확인:</strong> request별 admission/completion, 입력·출력 token, cache 초기화·재사용 조건 및 요청 순서를 기존 자료에서 더 식별할 수 있는지 점검한다.</li>
<li><strong>추가 정보가 있으면 최소 변경 검토:</strong> 작업량 분포 또는 조건별 처리율 중 근거가 있는 것만 보완한다.
시간이 지났다는 이유로 처리율을 임의로 떨어뜨리거나, 가장 잘 맞는 arrival 배율을 채택하지 않는다.</li>
<li><strong>자료가 없으면 한계와 대안을 확정:</strong> 통제된 요청률 상승·하강 및 배출(drain) 실험 설계를 제안하고,
그 전에는 S1을 검증된 5분 전력 생성기로 연결하지 않는다. 기존 M3+R3의 별도 활용 가능성과 이 미검증 상태 모델을 구분한다.</li></ol>
<h2>검증과 산출물</h2><p>핵심 원문 {v['selected_raw_snapshots_reparsed']}개, {v['diagnostic_windows']}개 창,
초기 민감도 {v['initial_work_cases']}개, 유입 민감도 {v['arrival_rate_cases']}개를 계산했다.
기존 파일 {v['protected_file_count']}개의 해시가 유지됐다.</p>
<p><a href="verification.json">검증 기록</a> · <a href="protocol.json">진단 규칙</a> ·
<a href="applicability_decision.json">적용 범위</a> · <a href="source_manifest.json">입력 출처·해시</a> ·
<a href="facility_exposure.json">전체 구간 노출 결과</a> · <a href="../carryover_replay_v1/report_ko.html">④-2 비교 보고서</a></p>
{footer_refs}</html>'''
    (out/'report_ko.html').write_text(html,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,default=ROOT/'outputs/carryover_scope_diagnostic_v1')
    parser.add_argument('--report-only',action='store_true',help='Render the report from unchanged stored diagnostics')
    args=parser.parse_args(); out=args.output.resolve()
    if args.report_only:
        assert read_json(out/'protocol.json')['stage']=='4-3: diagnostic attribution and provisional applicability, not model refitting'
        manifest=read_json(out/'source_manifest.json')
        for item in manifest:
            if Path(item['path'])!=Path(__file__): assert sha256(Path(item['path']))==item['sha256']
        assert all(sha256(Path(p))==h for p,h in read_json(out/'preservation_hashes.json').items())
        stored_core={}; intervals=pd.DataFrame(read_json(out/'interval_diagnostics.json'))
        for row in read_json(out/'focus_plot_data.json'):
            service,rate=row['service'],row['rate']
            stored_core[(service,rate)]={'obs':pd.DataFrame({'relative_run_s':row['relative_run_s'],
                'running_requests':row['observed_running'],'waiting_requests':row['observed_waiting']}),
                'base':pd.DataFrame(row['base']),'initial_traces':[pd.DataFrame(x) for x in row['initial_work_traces']],
                'intervals':intervals[(intervals.service==service)&(intervals.rate==rate)].to_dict('records')}
        figures(out,stored_core,None,None,pd.DataFrame(read_json(out/'facility_exposure.json')))
        write_report(out)
        for item in manifest:
            if Path(item['path'])==Path(__file__): item['sha256']=sha256(Path(__file__))
        dump(out/'source_manifest.json',manifest)
        print(out/'report_ko.html'); return
    if out.exists(): raise FileExistsError(f'Refusing overwrite: {out}')
    protocol_path=ROOT/'configs/carryover_scope_diagnostic_v1.json'; protocol=read_json(protocol_path)
    protected=list((ROOT/'src/nlr_data').glob('*.py'))+[ROOT/'src/aidc_facility/carryover_replay.py']
    for folder in (PRIOR,ARRIVALS,ROOT/'outputs/nlr_processing_diagnostic_v1',ROOT/'outputs/carryover_parameter_audit_v1'):
        protected.extend(p for p in folder.rglob('*') if p.is_file())
    before={str(p):sha256(p) for p in protected}
    raw=pd.read_csv(ROOT/'outputs/nlr_processing_diagnostic_v1/raw_server_metrics.csv')
    runs=pd.read_csv(ROOT/'outputs/nlr_transition_audit_v1/inference_online_rate_llama3_70b_runs.csv',dtype={'id':str})
    segments=make_segments(runs,raw); params=read_json(PRIOR/'fitted_parameters.json')
    compare=[s for s in segments.values() if s['split']=='comparison']
    assert len(compare)==100
    old={(x['run_id'],x['model']):x for x in read_json(PRIOR/'observed_time_replays.json')}
    windows=[]; intervals=[]; phases=[]; onsets=[]; core={}; init=[]; arrival=[]; baseline=[]
    focus=[s for s in compare if s['rate'] in protocol['focus_rates'][s['service']]]
    print('Rechecking selected snapshots against raw NLR server log',flush=True)
    raw_validation=raw_check(pd.concat([s['observations'] for s in focus]).drop_duplicates('source_line'))
    for s in compare:
        obs,service,rate=s['observations'],s['service'],s['rate']; p=params[service]
        meta={'run_id':s['run_id'],'service':service,'rate':rate}
        pred=pd.DataFrame(old[(s['run_id'],'S1_state_dependent_lps')]['prediction'])
        for name,(left,right) in protocol['windows_relative_to_run_s'].items():
            windows.append({**meta,'window':name,**window_summary(obs,rate,p,left,right)})
            if name in ('early','middle','late'):
                phases.append({**meta,'window':name,**selected_errors(obs,pred,left,right)})
        onsets.append({**meta,**waiting_onset(obs,pred)})
    for s in focus:
        obs,service,rate=s['observations'],s['service'],s['rate']; p=params[service]
        meta={'run_id':s['run_id'],'service':service,'rate':rate}
        base=replay_diagnostic(obs,rate,p)
        prior=pd.DataFrame(old[(s['run_id'],'S1_state_dependent_lps')]['prediction'])
        np.testing.assert_array_equal(base[['running','waiting','arrivals','completions']],prior[['running','waiting','arrivals','completions']])
        cleared=base[base.initial_running_jobs_left_in_model.eq(0)]
        baseline.append({**meta,**selected_errors(obs,base),**{f'late_{k}':v for k,v in selected_errors(obs,base,180,240).items()},
            **waiting_onset(obs,base),'initial_running_cleared_by_model_sample_s':float(cleared.time_s.iloc[0]) if len(cleared) else None})
        rows=interval_diagnostics(obs,rate,p)
        intervals.extend([{**meta,**r} for r in rows])
        traces=[]
        for power in protocol['initial_work_powers']:
            for phase in protocol['arrival_phases']:
                pred=replay_diagnostic(obs,rate,p,power=power,phase=phase)
                assert abs(pred.conservation_error).max()==0
                init.append({**meta,'initial_work_power':power,'arrival_phase':phase,**selected_errors(obs,pred),
                    **{f'late_{k}':v for k,v in selected_errors(obs,pred,180,240).items()},**waiting_onset(obs,pred)})
                traces.append(pred)
        for factor in protocol['arrival_rate_factors']:
            pred=replay_diagnostic(obs,rate,p,arrival_factor=factor)
            assert abs(pred.conservation_error).max()==0
            arrival.append({**meta,'assumed_arrival_factor':factor,**selected_errors(obs,pred),
                **{f'late_{k}':v for k,v in selected_errors(obs,pred,180,240).items()},**waiting_onset(obs,pred)})
        core[(service,rate)]={'obs':obs,'base':base,'intervals':rows,'initial_traces':traces}
        print(f'Diagnosed {SHORT[service]} {rate:g} req/s',flush=True)
    exposure,sizing,demand_summary,exposure_checks=facility_exposure(protocol)
    out.mkdir(parents=True)
    dump(out/'protocol.json',protocol)
    outputs={'window_diagnostics.json':windows,'interval_diagnostics.json':intervals,'phase_errors.json':phases,
        'waiting_onsets.json':onsets,'focus_baseline.json':baseline,'initial_work_sensitivity.json':init,
        'arrival_rate_sensitivity.json':arrival,'facility_exposure.json':exposure.to_dict('records'),
        'illustrative_instance_allocation.json':sizing,'facility_demand_summary.json':demand_summary}
    for name,value in outputs.items(): dump(out/name,value)
    plotdata=[]
    for service,rate in CORE:
        c=core[(service,rate)]
        plotdata.append({'service':service,'rate':rate,'relative_run_s':c['obs'].relative_run_s.tolist(),
            'observed_running':c['obs'].running_requests.tolist(),'observed_waiting':c['obs'].waiting_requests.tolist(),
            'base':c['base'].to_dict('records'),
            'initial_work_traces':[p.to_dict('records') for p in c['initial_traces']]})
    dump(out/'focus_plot_data.json',plotdata)
    figures(out,core,params,pd.DataFrame(windows),exposure)
    unchanged=all(sha256(Path(path))==h for path,h in before.items()); assert unchanged
    verification={'created_utc':datetime.now(timezone.utc).isoformat(),**raw_validation,
        'comparison_runs':len(compare),'focus_runs':len(focus),'diagnostic_windows':len(windows),
        'initial_work_cases':len(init),'arrival_rate_cases':len(arrival),'facility_bin_reconciliations':exposure_checks,
        'frozen_model_reproduces_prior_focus_states_exactly':True,'request_conservation_checked':True,
        'protected_files_unchanged':unchanged,'protected_file_count':len(before),
        'S1_recalibrated':False,'M3_R3_modified':False,'facility_instance_allocation_finalized':False,
        'independent_validation_claim':False,'causal_identification_claim':False}
    dump(out/'verification.json',verification)
    dump(out/'preservation_hashes.json',before)
    sources=[protocol_path,ROOT/'src/aidc_facility/carryover_scope_diagnostic.py',Path(__file__),
        ROOT/'src/aidc_facility/carryover_replay.py',PRIOR/'fitted_parameters.json',PRIOR/'observed_time_replays.json',
        ROOT/'outputs/nlr_processing_diagnostic_v1/raw_server_metrics.csv',
        ROOT/'outputs/nlr_transition_audit_v1/inference_online_rate_llama3_70b_runs.csv',
        ARRIVALS/'facility_demand_5min.csv',ARRIVALS/'run_summary.json']+list(ARRIVALS.glob('facility_arrivals*.npz'))
    dump(out/'source_manifest.json',[{'path':str(p),'sha256':sha256(p)} for p in sources]+[{'path':str(RAW),'sha256':raw_validation['raw_log_sha256']}])
    write_report(out)
    print(json.dumps(verification,indent=2))
    print(pd.DataFrame(baseline)[['service','rate','running_rmse','waiting_rmse','late_running_rmse','initial_running_cleared_by_model_sample_s']].to_string(index=False))


if __name__=='__main__': main()
