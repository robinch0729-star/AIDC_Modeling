"""Generate an explicitly provisional five-minute IT baseline and limited sensitivity."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import html
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from aidc_facility.it_operating_baseline import (DATASETS,m3,block_mean,choose_design,
    locked_coefficients,sum_residual_seconds,carryover_bridge,paired_metrics)
from aidc_facility.facility_arrivals import scale_arrivals,sha256


def plot(path,title,subtitle,panels,footer):
    """Compact scientific raster plot; show an honest 0..1 axis for zero stock."""
    from PIL import Image,ImageDraw,ImageFont
    width,height=1650,175+375*len(panels)
    image=Image.new('RGB',(width,height),'white'); draw=ImageDraw.Draw(image)
    def font(size,bold=False):
        return ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf',size)
    draw.text((120,23),title,fill='#173348',font=font(30,True))
    draw.text((120,69),subtitle,fill='#4c606d',font=font(20))
    colors=['#2465a5','#c65c28','#299078']
    for index,p in enumerate(panels):
        left,right=130,1550; top,bottom=183+index*375,420+index*375
        draw.text((left,top-70),p['title'],fill='#18374e',font=font(23,True))
        values=np.concatenate([np.asarray(y,float) for _,_,y in p['series']])
        lo,hi=float(values.min()),float(values.max()); pad=max((hi-lo)*.08,.01)
        ymin,ymax=(0 if p.get('zero') else lo-pad),hi+pad
        if lo==hi==0: ymin,ymax=0.,1.
        xmin=min(float(np.min(x)) for _,x,_ in p['series']); xmax=max(float(np.max(x)) for _,x,_ in p['series'])
        for y in np.linspace(ymin,ymax,6):
            py=bottom-(y-ymin)/(ymax-ymin)*(bottom-top)
            draw.line((left,py,right,py),fill='#dfe6eb')
            label=f'{y:,.0f}' if ymax>200 else f'{y:.1f}'
            draw.text((left-15,py),label,anchor='rm',fill='#415562',font=font(18))
        for x in np.linspace(xmin,xmax,7):
            px=left+(x-xmin)/(xmax-xmin or 1)*(right-left)
            draw.line((px,top,px,bottom),fill='#ecf0f3')
            draw.text((px,bottom+12),f'{x:.0f}',anchor='mt',fill='#415562',font=font(18))
        for j,(label,x,y) in enumerate(p['series']):
            color=colors[j%3]; lx=left+j*1420/max(3,len(p['series']))
            draw.line((lx,top-29,lx+25,top-29),fill=color,width=3)
            draw.text((lx+34,top-40),label,fill='#304653',font=font(18))
            points=[(left+(xv-xmin)/(xmax-xmin or 1)*(right-left),bottom-(yv-ymin)/(ymax-ymin)*(bottom-top)) for xv,yv in zip(x,y)]
            if len(points)>1: draw.line(points,fill=color,width=2)
        if lo==hi==0:
            draw.text((left+35,top+25),'No waiting in this candidate-model window',fill='#415562',font=font(19))
        draw.line((left,top,left,bottom,right,bottom),fill='#5b6d79',width=2)
        draw.text((left,top-2),p['unit'],anchor='lb',fill='#415562',font=font(17))
        draw.text(((left+right)/2,bottom+42),p['xlabel'],anchor='mt',fill='#415562',font=font(18))
    draw.text((120,height-36),footer,fill='#4c606d',font=font(17))
    image.save(path)

ARR=ROOT/'outputs/facility_arrival_scaling_v1'
FROZEN=ROOT/'outputs/r3_high_rate_benchmark'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def dump(path,obj): Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def table(rows):
    if not rows: return ''
    keys=list(rows[0])
    def val(v):
        if isinstance(v,float): return f'{v:,.6g}'
        return str(v)
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(val(row[k]))+'</td>' for k in keys)+'</tr>' for row in rows)+'</tbody></table></div>'


def make_report(out,summary):
    s=summary; design=s['design']; settings=s['settings']; base=s['baseline']; sens=s['sensitivity']
    regular=next(x for x in sens if x['case']=='azure_drop_window' and x['comparison']=='full_vs_original' and x['offset_s']==0)
    stressed=next(x for x in sens if x['case']=='synthetic_overload_drop' and x['comparison']=='full_vs_original' and x['offset_s']==0)
    rows=[{'서비스':service,'요청 비율 (%)':design['source_mix'][service]*100,
           'instance 수':design['instance_counts'][service],
           'instance 비율 (%)':design['instance_mix'][service]*100,
           '평균 요청률 / instance':s['rates'][service]['mean_rps'],
           '최대 1초 요청률 / instance':s['rates'][service]['max_rps'],
           '1 rps 미만 시간 (%)':100*s['rates'][service]['below_1_fraction']} for service in DATASETS]
    sensitivity_rows=[{'조건':r['case'],'비교':r['comparison'],'집계 경계 offset(s)':r['offset_s'],
        '5분 MAE (MW)':r['mae_mw'],'5분 최대 차이 (MW)':r['max_abs_difference_mw'],
        '순 에너지 차이 (MWh)':r['difference_energy_mwh']} for r in sens if r['block_s']==300]
    baseline_rows=[{'seed':r['seed'],'평균 MW':r['mean_mw'],'5분 최소 MW':r['min_5min_mw'],
        '5분 최대 MW':r['max_5min_mw'],'기간 에너지 MWh':r['energy_mwh'],
        '5분 residual 표준편차 kW':r['residual_5min_std_kw']} for r in base]
    body=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>5분 IT 부하 기본안과 제한적 이월 민감도</title>
<style>body{{font-family:'Malgun Gothic',Arial,sans-serif;max-width:1180px;margin:35px auto;padding:0 24px;color:#213547;line-height:1.7}}h1{{font-size:28px}}h2{{font-size:22px;border-bottom:1px solid #bdcbd5;padding-bottom:7px;margin-top:32px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border-bottom:1px solid #dbe2e8;padding:9px;text-align:left}}th{{background:#edf2f6}}.scroll{{overflow-x:auto}}.note{{background:#fff7e9;padding:15px;border-left:4px solid #c78324}}.eq{{font-family:Cambria,'Malgun Gothic',serif;background:#f4f7fa;padding:13px;overflow-wrap:anywhere}}img{{width:100%;height:auto}}a{{color:#216199}}li{{margin:5px 0}}</style>
<h1>5분 IT 부하 기본안과 제한적 이월 민감도</h1>
<p>목적: DFL–DSO/AIDC 운영 연구의 평가환경을 고정하기 위한 최소 작업. GPU 처리모델의 정밀 재보정은 수행하지 않음.</p>
<div class="note"><strong>상태: 재현 가능한 연구용 기본안 생성 완료.</strong> 실제 facility 전력의 검증 완료 또는 이월 영향의 일반적 상한 확보를 의미하지 않는다. 냉각·UPS·기타 부하는 아직 포함하지 않았다.</div>
<p><strong>핵심 결과:</strong> 기본안의 선택된 하락 구간에서는 이월 연결 전후 5분 MAE가 약 {regular['mae_mw']*1000:.1f} kW, 최대 차이가 약 {regular['max_abs_difference_mw']*1000:.1f} kW였다. 반면 가상 과부하 후 급감 조건에서는 최대 약 {stressed['max_abs_difference_mw']:.2f} MW 차이가 발생했다. 따라서 이 기본안은 저부하 운영의 잠정 환경으로 사용하고, 과부하 이월을 일반적으로 무시하지 않는다. 이 수치는 해당 후보모델/전력 연결 가정의 민감도이며 실측 오차가 아니다.</p>
<h2>1. 데이터와 운전조건</h2>
<ul><li>Azure 공통 관측기간: {s['start_utc']}부터 {s['end_exclusive_utc']} 직전까지. 총 {s['duration_seconds']/3600:.3f}시간, 5분 {s['five_minute_bins']:,}구간.*1</li>
<li>원자료의 1분 서비스별 요청 총량에 공통 정수 배수 {design['request_multiplier']:,}를 적용. 각 1분 안의 도착을 균등 확률로 재배치한 조건부 시나리오 사용.*2 원시 1초 잡음을 그대로 증폭하지 않음.</li>
<li>서비스별 고정 instance 수는 1분 최대 평균 요청률 약 {design['peak_parent_rate_rps']:g} rps/instance를 기준으로 산정. 처리용량·SLA 또는 최적 instance 배치가 아닌 보수적 시나리오 설계값.</li>
<li>IT 목표 평균 {design['target_it_mean_mw']:.6f} MW = 20 MW / 1.15. PUE는 규모 설정에만 사용하며 실제 PCC 모델을 계산한 것이 아님. 전체 기간을 사용한 사후 시나리오 설계이므로 예측기의 미래 입력으로 사용하면 안 됨.</li>
<li>20 MW는 여기서 평균 규모 목표이지 최대 수전한도가 아니다. IT 5분 최대가 20 MW를 넘는 구간도 있으며, 향후 PCC 용량/DOE 제약은 별도로 정해야 한다.</li>
<li>instance는 NLR 보정 단위를 복제한 등가 serving 단위. GPU 한 개 또는 독립적으로 실측한 물리 서버 수라는 뜻이 아님. NLR TP4 구성과 계측 경계의 불확실성은 유지.</li></ul>
{table(rows)}
<p>요청 비율과 instance 비율이 다른 이유는 각 서비스의 최대/평균 요청 패턴이 다르기 때문이다. 전력 비율도 별도로 계산했다. Azure token 분포를 NLR 처리/전력 특성과 동일하다고 가정하지 않았다.</p>
<h2>2. 전력 생성과 5분 집계</h2>
<div class="eq">N<sub>d</sub> = ceil(K × max Λ<sub>d,60s</sub> / 30), &nbsp; r<sub>d,t</sub> = A<sub>d,t</sub> / N<sub>d</sub><br>
P<sub>M3</sub>(r,d) = P<sub>0,d</sub> + ΔP<sub>d</sub> / [1 + exp(−k<sub>d</sub>(ln(1+r) − x<sub>0,d</sub>))]<br>
P<sub>IT</sub>(t) = Σ<sub>d</sub> [N<sub>d</sub>P<sub>M3</sub>(r<sub>d,t</sub>,d) + E<sub>d</sub>(t)] / 1000 &nbsp; [MW]<br>
E<sub>d,k+1</sub> = φ<sub>d,k</sub>E<sub>d,k</sub> + Σ<sub>i=1…N_d</sub>η<sub>d,i,k</sub>, &nbsp; Δt = 0.1 s<br>
P̄<sub>5min,j</sub> = (1/3000)Σ<sub>k∈j</sub>P<sub>IT,k</sub>, &nbsp; Energy<sub>j</sub> = P̄<sub>5min,j</sub> × (300/3600) [MWh].*3</div>
<ul><li>M3와 R3의 보정 파라미터는 그대로 유지. 요청률은 1초 구간 강도로 해석하고 0.1초 residual 재귀식은 원래 간격으로 계산. 시간가변 조건 적용은 준정상 가정이며 전력 전이의 실측 검증이 아님.</li>
<li>서비스 내 동일 계수와 독립 residual을 가정하면 전체 residual 합을 직접 계산할 수 있다. 합성 혁신항은 다항분포 성분 수와 조건부 정규분포 합을 사용하므로, 이 가정하에서 수천 instance의 개별 R3 합과 분포가 동일하다. 중앙극한 근사나 임의 변동폭 증폭을 사용하지 않음.*4</li>
<li>상태는 구간 경계에서 유지. R3는 초기값 0에서 첫 조건으로 120초 burn-in 후 사용하며 정확한 정상분포 초기화를 보증하는 것은 아님. 구간별 평균 제거·전력 clipping·사후 출력 scale-up을 하지 않음. 무작위 seed 3개는 동일 수요의 residual realization이며 독립 실측 검증 자료가 아님. 0.1초는 기존 R3 계수의 계산 간격이지 DSO 통신에 요구하는 간격이 아님.</li>
<li>1 rps 미만은 요청량을 잘라내지 않고 M3 수학적 외삽을 명시적으로 허용. R3는 1 rps 계수 고정. 이는 idle 측정모델이 아니며, 아래에 1 rps 경계값 고정과의 차이를 따로 보고.</li></ul>
{table(baseline_rows)}
<p>사전 설계 예상 평균: {design['design_expected_mean_mw']:.6f} MW. 정수 규모 선택으로 목표와 {design['design_mean_error_mw']:+.6f} MW 차이. 기본안의 1 rps 미만 M3 연결 가정에 따른 최대 1초 facility 평균전력 차이: {s['low_support_sensitivity']['max_difference_kw']:.6f} kW, 전 기간 에너지 차이: {s['low_support_sensitivity']['energy_difference_kwh']:.6f} kWh. 이는 idle 모델의 검증이 아니다.</p>
<img src="long_5min_it_profile.png" alt="약 5일의 IT 전력과 서비스별 요청률">
<p><a href="it_profiles_5min.csv">5분 프로파일 CSV</a> · <a href="baseline_1s.npz">1초 평균전력·요청률 NPZ</a> · <a href="model_contract.json">모델 사용조건 JSON</a></p>
<h2>3. 이월 효과: 추가 보정 없이 대표 구간만 비교</h2>
<p>이미 존재하는 처리상태·이월 후보모델의 계수와 사건 기반 처리공유 규칙을 재사용했다.*5 모델 선택이나 μ 재추정은 하지 않았다. 실제 Azure 기반 기본안에서 가장 큰 하락 구간 주변 1시간과, 별도 20→70→10 rps 과부하 시험을 비교했다. 과부하 시험은 관측된 시설 운전이 아니다.</p>
<div class="eq">Original(t) = ΣN<sub>d</sub>P<sub>M3</sub>(r<sub>d</sub>(t),d)/1000<br>
Memoryless bridge(t) = ΣN<sub>d</sub>P<sub>M3</sub>(min(r<sub>d</sub>(t), μ̂<sub>d,max</sub>),d)/1000<br>
Stateful bridge(t) = ΣN<sub>d</sub>P<sub>M3</sub>(r<sub>d</sub><sup>eq</sup>(t),d)/1000, &nbsp; r<sub>d</sub><sup>eq</sup>(t)=μ̂<sub>d</sub>(Running<sub>d</sub>(t)) / w̄, &nbsp; w̄=1.*6</div>
<p><strong>*6은 본 연구가 이번 민감도 계산에 도입한 가정</strong>이다. r<sup>eq</sup>는 정상상태 균형을 이용한 등가 입력 요청률이지, 측정된 완료율이 아니다. 기존 M3의 입력을 바꾸거나 이 연결을 문헌의 전력 공식이라고 주장하지 않는다. Original과 Stateful의 차이에는 연결 규칙의 정적 차이도 포함되므로, Memoryless 대비 Stateful을 따로 비교해 순수한 모델 내 이월 효과를 분리했다. 이 연결을 최종 모델로 채택하지 않았다.</p>
<ul><li>각 서비스 대표 instance를 입력 강도에 맞춘 결정적 도착열로 모의. 전체 시설의 개별 요청 라우팅을 재현한 것이 아니며, 처리 상태의 서비스 내 다양성은 반영하지 않음.</li>
<li>시작 전 첫 요청률을 600초 동안 유지하는 warm-up을 가정. 실제 과거 이력은 미확인. 미완료 작업은 끝까지 기록하고 취소·timeout은 없다고 가정.</li>
<li>처리는 사건 기반이며 전력 적분은 0.1초 중간점 표본. 1초 표본과 비교해 수치 민감도 확인. 구조 차이에 집중하기 위해 이 비교에서는 R3를 제외했다. R3까지 재보정한 비교가 아님.</li>
<li>0초와 150초로 이동한 5분 집계 경계를 모두 확인. 부분 구간은 삭제한 시간을 명시.</li></ul>
{table(sensitivity_rows)}
<img src="carryover_azure_window.png" alt="Azure 기반 대표 구간의 제한적 이월 민감도">
<img src="carryover_stress_window.png" alt="가상 과부하 후 요청 감소 시 이월 민감도">
<p><a href="carryover_profiles_1s.csv">대표 구간 상태·전력 CSV</a> · <a href="sensitivity_metrics.csv">민감도 수치 CSV</a></p>
<p><a href="integration_sensitivity.json">0.1초와 1초 전력 적분 비교</a>도 별도로 보존했다. 표본 간격에 따른 수치 차이와 상태–전력 연결 가정의 오차는 서로 다르며, 둘을 실측 오차로 해석하지 않는다.</p>
<h2>4. 해석과 사용 범위</h2>
<ul><li>채택 범위: 이 구성의 5분 IT 기본안을 DFL–DOE 구현을 위한 <strong>잠정 평가환경 버전</strong>으로 고정. 실제 facility 검증, workload 전체 영역 검증, 초단위 ramp 검증으로 확대 해석하지 않음.</li>
<li>이월 민감도는 선정한 두 구간과 가정에만 해당한다. 중간 요청률에서 기존 상태모델 오차가 남아 있으므로, 작은 차이가 나와도 이월 영향이 항상 작다는 결론이나 일반적 오차 상한을 만들지 않음.</li>
<li>1분 내부 요청 재배치와 5분 내부 재배치의 전력 차이도 별도 확인: M3 기준 5분 MAE {s['timing_sensitivity']['mae_mw']:.6f} MW, 최대 {s['timing_sensitivity']['max_abs_difference_mw']:.6f} MW. 같은 5분 요청 총량이라도 M3 비선형성 때문에 평균전력이 반드시 같지는 않음.</li>
<li>시설 power ground truth, 장기간 cache·열·DVFS 변화, 공간적 residual 상관, token별 처리량, 실제 SLA는 검증하지 않음. PUE 1.15와 peak 30은 연구 설계값이다.</li>
<li>현재 단계에서 NLR 연구진 문의·추가 GPU 실험·처리모델 정밀 보정은 필수 경로에서 제외. 다음은 냉각·손실·기타 부하 경계 설정과 DSO–AIDC 의사결정 정식화. DFL 비교 시 동일 생성기 조건만으로 일반화 성능을 주장하지 않고 운전/모델 가정 변화도 별도로 평가해야 함.</li></ul>
<h2>5. 재현성과 확인 결과</h2>
<p>원본·기존 모델/진단 파일 {s['protected_files']}개 해시 보존. 입력 요청 총량·5분 집계 에너지·상태 요청수 보존·확률 성분 합 등을 확인. 통계적 합격 임계값을 새로 만들어 모델을 합격 판정하지 않았다.</p>
<p><a href="verification.json">수치 확인</a> · <a href="source_manifest.json">출처와 해시</a> · <a href="summary.json">전체 결과 JSON</a></p>
<h2>6. 수식·방법론 출처와 변경사항</h2>
<p>*1 <a href="https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md">Azure LLM inference trace 2024</a>; J. Stojkovic et al., <em>DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency</em>, IEEE HPCA, 2025, <a href="https://doi.org/10.1109/HPCA61900.2025.00102">DOI</a>. 도착 시각/서비스별 패턴 활용. NLR prompt workload로의 연결은 우리 가정.</p>
<p>*2 <a href="https://web.mit.edu/urban_or_book/www/book/chapter2/2.12.3.html">MIT Urban Operations Research §2.12.3</a>, 조건부 Poisson 도착 시각의 균등분포 성질. 구간 총량 고정 후 다항분포로 1초 counts 생성. Azure가 실제 Poisson이라는 검정 결과가 아님. 60초 선택과 K 배율은 우리 설계.</p>
<p>*3 NLR Paper list [3]의 요청 조건·전력자료와 기존 <a href="../r3_high_rate_benchmark/protocol.json">고정 M3/R3 protocol</a>. M3 로지스틱은 우리 경험식이며 [3]의 원문 수식을 그대로 옮긴 것이 아님. R3의 OU+jump 뼈대는 [2] <em>Dynamic Load Model for Data Centers with Pattern-Consistent Calibration</em> Eq. (2)를 참고했지만, signed Gaussian-mixture 이산 혁신항은 우리 근사. 평균+residual/AR 구조의 기존 [37] 연계도 위 protocol에 보존. 5분 평균/에너지는 시간 적분 정의.</p>
<p>*4 독립 확률변수 합과 조건부 정규분포 합의 대수적 성질. 동일 φ일 때 Σe 재귀식을 직접 합산한 계산 최적화이며, 추가 물리 상관 모델이나 새로운 문헌 보정식이 아님. 도착량 변화에 따른 공통 변동은 이미 M3 입력에 존재.</p>
<p>*5 V. Gupta and J. Zhang, <em>Approximations and Optimal Control for State-Dependent Limited Processor Sharing Queues</em>, Stochastic Systems 12(2), 205–225, 2022, §2.1, Eqs. (2)–(3), <a href="https://doi.org/10.1287/stsy.2021.0087">DOI</a>. 처리공유/동시 처리 한도 구조만 차용. NLR μ̂(R), 정규화 작업량, 초기조건은 기존 연구 구현의 가정. <a href="../carryover_replay_v1/report_ko.html">기존 비교</a>와 <a href="../carryover_scope_diagnostic_v1/report_ko.html">적용 범위 진단</a> 참조.</p>
<p>*6 정상상태 μ̂(R)/w̄와 입력 요청률의 균형을 전력에 연결한 <strong>이번 민감도 전용 가정</strong>. 직접 차용한 전력 전이식 또는 검증된 상태–전력 매핑이 아니다. 별도의 무기억 control로 정적 연결 차이를 분리.</p>
</html>'''
    (out/'report_ko.html').write_text(body,encoding='utf-8')


def refresh_figures(out):
    """Render only from saved results; do not rerun or mutate numerical data."""
    s=read(out/'summary.json'); f=pd.read_csv(out/'it_profiles_5min.csv'); hours=np.arange(len(f))/12
    plot(out/'long_5min_it_profile.png','Five-minute IT operating baseline',
        'Azure-shaped demand + frozen M3/R3; fixed instances; no post-generation scaling',[
        {'title':'IT power: three residual realizations (nearly overlapping)',
         'series':[(str(seed),hours,f[f'it_power_seed_{seed}_mw']) for seed in s['settings']['residual_seeds']],
         'unit':'MW','xlabel':'Hours from 2024-05-12 00:05 UTC'},
        {'title':'Offered requests per instance: common service demand',
         'series':[(d,hours,f[d+'_rate_per_instance_rps']) for d in DATASETS],
         'unit':'request/s/instance','xlabel':'Hours from start','zero':True}],
        'Cooling and other loads excluded. Small residual spread is conditional on independent instances.')
    c=pd.read_csv(out/'carryover_profiles_1s.csv')
    for case,filename in [('azure_drop_window','carryover_azure_window.png'),('synthetic_overload_drop','carryover_stress_window.png')]:
        x=c[c.case.eq(case)]; minutes=x.time_s/60
        plot(out/filename,'Carryover sensitivity: '+case.replace('_',' '),
            'Frozen processing candidate + explicitly hypothetical state/power bridge; not field validation',[
            {'title':'Offered rate per representative instance','series':[(d,minutes,x[d+'_offered_rate_rps']) for d in DATASETS],
             'unit':'request/s','xlabel':'Elapsed minutes','zero':True},
            {'title':'Power mappings (structural sensitivity; R3 excluded)',
             'series':[(label,minutes,x[key]) for label,key in [('Original M3','facility_original_mw'),('No-memory bridge','facility_memoryless_bridge_mw'),('Stateful bridge','facility_stateful_bridge_mw')]],
             'unit':'MW','xlabel':'Elapsed minutes'},
            {'title':'Waiting stock in candidate processing model','series':[(d,minutes,x[d+'_waiting_mean']) for d in DATASETS],
             'unit':'requests/instance','xlabel':'Elapsed minutes','zero':True}],
            'All cases keep processing state across 5-min boundaries. No timeout/drop policy or SLA guarantee.')


def run(out,config):
    start_time=time.perf_counter()
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True)
    settings=read(config)
    sources=[config,ARR/'source_arrivals_1s.npz',ARR/'source_arrivals_metadata.json',
        FROZEN/'train_validation_m3_parameters.csv',FROZEN/'train_validation_residual_parameters.csv',
        FROZEN/'protocol.json',ROOT/'outputs/carryover_replay_v1/fitted_parameters.json',
        ROOT/'outputs/carryover_scope_diagnostic_v1/applicability_decision.json',
        ROOT/'src/aidc_facility/carryover_replay.py',ROOT/'src/aidc_facility/facility_arrivals.py',
        ROOT/'src/aidc_facility/it_operating_baseline.py',Path(__file__)]
    protected={str(p):sha256(p) for p in sources[:-2]}
    for name in ('carryover_replay_v1','carryover_scope_diagnostic_v1','carryover_revision_evidence_v1','r3_high_rate_benchmark'):
        for p in (ROOT/'outputs'/name).rglob('*'):
            if p.is_file(): protected[str(p)]=sha256(p)
    dump(out/'preservation_hashes.json',protected)
    dump(out/'source_manifest.json',[{'path':str(p),'sha256':sha256(p),'bytes':p.stat().st_size} for p in sources])
    metadata=read(ARR/'source_arrivals_metadata.json')
    source={s:np.load(ARR/'source_arrivals_1s.npz')[s] for s in DATASETS}
    rows=pd.read_csv(FROZEN/'train_validation_m3_parameters.csv').set_index('prompt_dataset')
    pars={s:rows.loc[d].to_dict() for s,d in DATASETS.items()}
    library=pd.read_csv(FROZEN/'train_validation_residual_parameters.csv')
    design=choose_design(source,pars,settings['design_pcc_mean_mw']/settings['design_pue'],
        settings['design_peak_parent_rate_per_instance'],settings['design_parent_seconds'],settings['request_multiplier_search_max'])
    dump(out/'design.json',design)
    print('DESIGN',json.dumps(design),flush=True)
    k=design['request_multiplier']; nodes=design['instance_counts']; length=len(source['code'])
    counts={s:scale_arrivals(x,k,settings['arrival_method'],settings['arrival_seed']+j) for j,(s,x) in enumerate(source.items())}
    rates={s:x/nodes[s] for s,x in counts.items()}
    checks=[]
    for s in DATASETS:
        np.testing.assert_array_equal(counts[s].reshape(-1,60).sum(1),source[s].reshape(-1,60).sum(1)*k)
        checks.append({'check':s+' exact one-minute request totals','passed':True})
    steady_by_service={s:nodes[s]*m3(r,pars[s])/1000 for s,r in rates.items()}
    steady=sum(steady_by_service.values())
    low_alternative=sum(nodes[s]*m3(np.maximum(rates[s],1),pars[s])/1000 for s in DATASETS)
    alternate={s:scale_arrivals(source[s],k,'conditional_300s',settings['arrival_seed']+j) for j,s in enumerate(DATASETS)}
    alternate_power=sum(nodes[s]*m3(alternate[s]/nodes[s],pars[s])/1000 for s in DATASETS)
    for s in DATASETS:
        np.testing.assert_array_equal(alternate[s].reshape(-1,300).sum(1),counts[s].reshape(-1,300).sum(1))
    coefficients={s:locked_coefficients(library,DATASETS[s],rates[s]) for s in DATASETS}
    residuals=np.zeros((len(settings['residual_seeds']),length))
    residual_audit=[]
    for j,seed in enumerate(settings['residual_seeds']):
        for i,s in enumerate(DATASETS):
            r,audit=sum_residual_seconds(coefficients[s],nodes[s],np.random.SeedSequence([seed,i]),settings['residual_burn_in_s'])
            residuals[j]+=r/1000
            residual_audit.append({'service':s,'seed':seed,**audit})
        print('RESIDUAL seed',seed,'elapsed_s',round(time.perf_counter()-start_time,1),flush=True)
    power=steady[None,:]+residuals
    # Strong positivity check uses a conservative native residual bound, not only 1-s averages.
    for j in range(len(power)):
        native_bound=sum(a['summed_native_residual_max_abs_kw'] for a in residual_audit if a['seed']==settings['residual_seeds'][j])/1000
        assert float(steady.min())>native_bound
    np.savez_compressed(out/'baseline_1s.npz',offered_code_requests=counts['code'],offered_conversation_requests=counts['conversation'],
        code_rate_per_instance_rps=rates['code'],conversation_rate_per_instance_rps=rates['conversation'],
        steady_mw=steady,residual_mw=residuals,it_power_mw=power)
    times=pd.date_range(metadata['start_utc'],periods=length//300,freq='300s')
    frame=pd.DataFrame({'interval_start_utc':times.astype(str),'interval_s':300,
        'code_offered_requests':counts['code'].reshape(-1,300).sum(1),
        'conversation_offered_requests':counts['conversation'].reshape(-1,300).sum(1),
        'code_rate_per_instance_rps':block_mean(rates['code']),
        'conversation_rate_per_instance_rps':block_mean(rates['conversation']),
        'code_steady_mw':block_mean(steady_by_service['code']),
        'conversation_steady_mw':block_mean(steady_by_service['conversation']),
        'm3_steady_mw':block_mean(steady)})
    baseline=[]
    for j,seed in enumerate(settings['residual_seeds']):
        b=block_mean(power[j]); e=b/12
        frame[f'it_power_seed_{seed}_mw']=b; frame[f'it_energy_seed_{seed}_mwh']=e
        assert np.isclose(e.sum(),power[j].sum()/3600,atol=1e-8,rtol=0)
        baseline.append({'seed':seed,'mean_mw':float(b.mean()),'min_5min_mw':float(b.min()),
            'max_5min_mw':float(b.max()),'energy_mwh':float(e.sum()),
            'residual_5min_std_kw':float(block_mean(residuals[j]).std()*1000),
            'residual_mean_kw':float(residuals[j].mean()*1000),
            'adjacent_5min_change_p95_mw':float(np.quantile(np.abs(np.diff(b)),.95))})
    frame.to_csv(out/'it_profiles_5min.csv',index=False,encoding='utf-8-sig')
    # Limited, pre-defined diagnostics only. No model refitting or case selection by outcome.
    five=block_mean(steady)
    candidates=np.arange(6,len(five)-6)
    edge=int(candidates[np.argmax((five[:-1]-five[1:])[candidates-1])])*300
    window_s=settings['carryover_comparison']['azure_window_seconds']
    a=max(0,min(length-window_s,edge-window_s//2))
    scenarios={'azure_drop_window':{s:r[a:a+window_s] for s,r in rates.items()},
        'synthetic_overload_drop':{s:np.concatenate([np.full(duration,rate) for rate,duration in settings['carryover_comparison']['stress_plateaus']]) for s in DATASETS}}
    fitted=read(ROOT/'outputs/carryover_replay_v1/fitted_parameters.json')
    sensitivity=[]; carryframes=[]; states={}; numerical=[]; plotdata={}
    for case,case_rates in scenarios.items():
        columns={}; totals={key:np.zeros(len(next(iter(case_rates.values())))) for key in ('original_kw','memoryless_bridge_kw','stateful_bridge_kw')}
        states[case]={}
        for s in DATASETS:
            fine,audit=carryover_bridge(case_rates[s],fitted[DATASETS[s]],pars[s],
                settings['carryover_comparison']['warmup_seconds'],settings['carryover_comparison']['internal_observation_seconds'])
            coarse,_=carryover_bridge(case_rates[s],fitted[DATASETS[s]],pars[s],
                settings['carryover_comparison']['warmup_seconds'],1.)
            for key,values in fine.items(): columns[s+'_'+key]=values
            for key in totals: totals[key]+=nodes[s]*fine[key]/1000
            numerical.append({'case':case,'service':s,**paired_metrics(nodes[s]*fine['stateful_bridge_kw']/1000,nodes[s]*coarse['stateful_bridge_kw']/1000)})
            states[case][s]=audit
            assert audit['conservation_max_error_requests']==0
            print('CARRYOVER',case,s,'elapsed_s',round(time.perf_counter()-start_time,1),flush=True)
        for name,reference,other in [('full_vs_original','original_kw','stateful_bridge_kw'),
            ('memory_only','memoryless_bridge_kw','stateful_bridge_kw'),('static_bridge_offset','original_kw','memoryless_bridge_kw')]:
            for offset in (0,150):
                sensitivity.append({'case':case,'comparison':name,**paired_metrics(totals[reference],totals[other],offset=offset)})
        part=pd.DataFrame(columns)
        part.insert(0,'time_s',np.arange(len(part)))
        part.insert(0,'case',case)
        for key,value in totals.items(): part['facility_'+key.replace('_kw','_mw')]=value
        carryframes.append(part); plotdata[case]=(case_rates,totals,columns)
    pd.concat(carryframes,ignore_index=True).to_csv(out/'carryover_profiles_1s.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(sensitivity).to_csv(out/'sensitivity_metrics.csv',index=False,encoding='utf-8-sig')
    dump(out/'carryover_state_checks.json',states)
    dump(out/'integration_sensitivity.json',numerical)
    dump(out/'residual_aggregation_checks.json',residual_audit)
    hours=np.arange(len(frame))/12
    plot(out/'long_5min_it_profile.png','Five-minute IT operating baseline',
        'Azure-shaped demand + frozen M3/R3; fixed instances; no post-generation scaling',[
        {'title':'IT power: three residual realizations','series':[(str(seed),hours,frame[f'it_power_seed_{seed}_mw']) for seed in settings['residual_seeds']],
         'unit':'MW','xlabel':'Hours from 2024-05-12 00:05 UTC'},
        {'title':'Offered requests per instance: common service demand','series':[(s,hours,block_mean(rates[s])) for s in DATASETS],
         'unit':'request/s/instance','xlabel':'Hours from start','zero':True}],
        'Cooling and other loads excluded. Small residual spread is conditional on independent instances.')
    for case,filename in [('azure_drop_window','carryover_azure_window.png'),('synthetic_overload_drop','carryover_stress_window.png')]:
        rr,tt,cc=plotdata[case]; minutes=np.arange(len(next(iter(rr.values()))))/60
        panels=[{'title':'Offered rate per representative instance','series':[(s,minutes,rr[s]) for s in DATASETS],
            'unit':'request/s','xlabel':'Elapsed minutes','zero':True},
            {'title':'Power mappings (structural sensitivity; R3 excluded)','series':[(name,minutes,tt[key]) for name,key in [('Original M3','original_kw'),('No-memory bridge','memoryless_bridge_kw'),('Stateful bridge','stateful_bridge_kw')]],
            'unit':'MW','xlabel':'Elapsed minutes'},
            {'title':'Waiting stock in candidate processing model','series':[(s,minutes,cc[s+'_waiting_mean']) for s in DATASETS],
            'unit':'requests/instance','xlabel':'Elapsed minutes','zero':True}]
        plot(out/filename,'Carryover sensitivity: '+case.replace('_',' '),
            'Frozen processing candidate + explicitly hypothetical state/power bridge; not field validation',panels,
            'All cases keep processing state across 5-min boundaries. No timeout/drop policy or SLA guarantee.')
    ratesummary={s:{'mean_rps':float(r.mean()),'max_rps':float(r.max()),'min_rps':float(r.min()),
        'above_30_fraction':float(np.mean(r>30)),'below_1_fraction':float(np.mean(r<1)),
        'above_first_waiting_observed_fraction':float(np.mean(r>({'code':42,'conversation':44}[s])))} for s,r in rates.items()}
    for path,old in protected.items(): assert sha256(Path(path))==old,path
    checks.extend([{'check':'all frozen files unchanged','passed':True},
        {'check':'1s to 5min energy equality','passed':True},
        {'check':'conditional60 and conditional300 exact matched five-minute totals','passed':True},
        {'check':'native power positivity conservative bound','passed':True},
        {'check':'carryover event-level request conservation','passed':True}])
    verification={'checks':checks,'all_passed':all(x['passed'] for x in checks),'protected_files':len(protected)}
    dump(out/'verification.json',verification)
    summary={'created_utc':datetime.now(timezone.utc).isoformat(),'start_utc':metadata['start_utc'],
        'end_exclusive_utc':metadata['end_exclusive_utc'],'duration_seconds':length,'five_minute_bins':len(frame),
        'settings':settings,'design':design,'rates':ratesummary,'baseline':baseline,'sensitivity':sensitivity,
        'integration_sensitivity':numerical,'azure_window_source_offset_s':a,
        'azure_window_start_utc':str(pd.Timestamp(metadata['start_utc'])+pd.Timedelta(seconds=a)),
        'timing_sensitivity':paired_metrics(steady,alternate_power),
        'service_steady_energy_share':{s:float(v.sum()/steady.sum()) for s,v in steady_by_service.items()},
        'low_support_sensitivity':{'max_difference_kw':float(np.max(low_alternative-steady)*1000),
            'energy_difference_kwh':float(np.sum(low_alternative-steady)*1000/3600)},
        'protected_files':len(protected),'runtime_seconds':time.perf_counter()-start_time,
        'physical_facility_validated':False,'carryover_mapping_selected_as_final':False,
        'decision':'Freeze as a provisional five-minute IT operating baseline; retain carryover as sensitivity only; proceed to PCC and coordination design.'}
    dump(out/'summary.json',summary)
    contract={'version':settings['version'],'status':'provisional research evaluation environment',
        'scope':'IT only; fixed equal-intensity instances; Azure time pattern mapped to NLR workload classes',
        'design':design,'input_files_and_units':{'baseline_1s.npz':'integer facility arrivals per second; offered rps/instance; MW one-second means',
        'it_profiles_5min.csv':'left-closed UTC intervals; MW mean and MWh energy per 300 seconds'},
        'time_axis':'No compression of calendar time. Native R3 0.1s -> exact 1s mean -> 300s mean.',
        'frozen_m3':pars,'r3_refit':False,'power_rescaling':False,'state_power_link_validated':False,
        'limitations':['No physical facility validation','Independent conditional residuals','Fixed routing/counts, not autoscaling',
            'Below-one-rps explicit extension, not measured idle','Long-horizon stationarity is assumed',
            'Azure prompt/token workload is not NLR workload equivalence','Carryover tested on two windows only',
            'All-data offline sizing is not causal forecasting or held-out DFL evaluation'],
        'next_step':'Cooling/loss/auxiliary boundary and DSO-AIDC optimization; do not claim final facility truth.'}
    dump(out/'model_contract.json',contract)
    make_report(out,summary)
    print('COMPLETE',json.dumps({'output':str(out),'runtime_s':summary['runtime_seconds'],'baseline':baseline,'sensitivity':sensitivity}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/it_5min_operating_baseline_v1')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/it_5min_operating_baseline_v1.json')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    if args.report_only:
        make_report(args.output,read(args.output/'summary.json'))
        refresh_figures(args.output)
        dump(args.output/'report_refresh.json',{'renderer_path':str(Path(__file__)),
            'renderer_sha256':sha256(Path(__file__)),'numerical_outputs_changed':False,
            'original_generation_hash_retained_in_source_manifest':True})
    else: run(args.output,args.config)
