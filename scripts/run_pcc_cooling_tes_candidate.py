"""Run and document the proposed PCC candidate without modifying the IT baseline."""
from __future__ import annotations
import copy
from datetime import datetime,timezone
import hashlib
import html
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from aidc_facility.pcc_cooling_tes import simulate,summarize,aggregate,tank_capacity,maximum_discharge_tank_temperature
from run_it_operating_baseline import plot,table

OUT=ROOT/'outputs/pcc_cooling_tes_candidate_v1'
CONFIG=ROOT/'configs/pcc_cooling_tes_candidate_v1.json'


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,o): Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


REFERENCES=[
 {'id':'*1 / PL13','title':'Integrated management of workloads and energy system for data centers',
  'publication':'Guo et al., Energy 327, 136400 (2025)','doi':'10.1016/j.energy.2025.136400',
  'original':'Eq31 electricity; Eq41 cooling balance; Eq42-43 chiller; Eq53-57 cold storage',
  'adaptation':'Remove CHP/PV/absorption/free cooling; replace workload power by frozen IT input. Separate MW_th and MWh_th, explicit dt. Replace inventory-only storage by mixed-water thermal state.',
  'verified':'local PDF pages 8-9, full equation page rendered and checked'},
 {'id':'*2 / PL4','title':'Dynamic coupling real-time energy consumption modeling for data centers',
  'publication':'An and Ma, Energy Reports 8, 1184-1192 (2022), CEES special issue','doi':'10.1016/j.egyr.2022.06.085',
  'original':'Eq8 pump power; Eq10 cooling power sum; Eq12-16 heat paths',
  'adaptation':'Keep heat-flow logic and equipment split; no separate server-fan power (included provisionally in IT input). Fan/pump curves and tower ratios are illustrative reduced-order assumptions, not fitted paper coefficients.',
  'verified':'local PDF pages 5-6'},
 {'id':'*3 / PL16','title':'Short-term data center load forecasting method based on data-physics hybrid driven framework',
  'publication':'Zhao et al., Energy Reports 15, 108953 (2026)','doi':'10.1016/j.egyr.2025.108953',
  'original':'Continuous thermal balance Eq8-9, single air-temperature state',
  'adaptation':'Use thermal balance only, no neural predictor; effective room capacity is an assumed aggregate, not measured air-only capacity. Exact constant-input integration with explicit dt; add TES through actual delivered cooling.',
  'verified':'local PDF page 5; hourly original data does not validate this 5-minute use'},
 {'id':'*4 / PL14','title':'Dynamic Model and Converter-Based Emulator of a Data Center Power Distribution System',
  'publication':'Sun et al., IEEE Transactions on Power Electronics 37(7), 2022','doi':'',
  'original':'Section II electrical supply topology (UPS/PDU/PSU, cooling branches)',
  'adaptation':'Use topology/boundary only. Constant-efficiency UPS and transformer are our reduced assumptions, not copied dynamic converter equations.',
  'verified':'local PDF page 2'},
 {'id':'*5 / EnergyPlus','title':'Engineering Reference 9.5 - Water Thermal Tanks',
  'publication':'Technical model documentation, not a journal paper','doi':'',
  'url':'https://bigladdersoftware.com/epx/docs/9-5/engineering-reference/water-thermal-tanks-includes-water-heaters.html',
  'original':'Mixed tank C*dT/dt=sum heat flows; UA ambient transfer and epsilon*m_dot*cp*temperature_difference',
  'adaptation':'One fully mixed node; heat-port dispatch limited by minimum approach over the interval. Tank energy is derived from temperature, not an independent duplicated SOC state.',
  'verified':'primary model documentation, equations read on 2026-10-06'},
 {'id':'*6 / Research assumptions','title':'Candidate controller, ratings, boundary assumptions and validation protocol',
  'publication':'This research candidate; not literature-proven site parameters','doi':'',
  'original':'No source claim for assumed site hardware or dispatch',
  'adaptation':'Elapsed-clock TES schedule, 4-hour terminal recovery, temperature limits, controller gain, sizing and sensitivity ranges are review hypotheses.',
  'verified':'All assumptions exposed in generation_config.json and parameter_register.json'}
]


def parameters(c):
    rows=[]
    for group in ('electrical','room','plant','tes','dispatch'):
        for k,v in c[group].items():
            origin='연구용 가정; 실측/제조사 검증 전'
            if group=='plant' and k=='cop': origin='PL13 Table 1의 COP 5를 비교 기준으로만 사용'
            if k in ('water_density_kg_m3','water_cp_j_kg_k'): origin='물의 물성 근사; 온도 의존성 생략'
            rows.append({'분류':group,'파라미터':k,'값':v,'출처·상태':origin})
    return rows


def figures(frames,summary,c):
    tank_limit=maximum_discharge_tank_temperature(c['tes'])
    a,b=[aggregate(frames[n],c) for n in ('no_tes','with_tes')]
    x=a.time_s.to_numpy()/3600
    plot(OUT/'pcc_overview.png','PCC candidate | frozen IT input + cooling + TES',
         'Illustrative air / chilled-water plant | 5-minute means | no PUE rescaling',[
        {'title':'IT and PCC active power','unit':'MW','xlabel':'Elapsed time (h)',
         'series':[('IT input',x,a.it_mw),('PCC: no TES',x,a.pcc_mw),('PCC: TES schedule',x,b.pcc_mw)]},
        {'title':'TES changes cooling electricity, not IT workload','unit':'MW','xlabel':'Elapsed time (h)',
         'series':[('PCC(TES) - PCC(no TES)',x,b.pcc_mw-a.pcc_mw)]}],
         'Candidate only. Scheduled shifting is not optimized peak shaving or a validated AIDC site.')
    sel=(x>=24)&(x<48); xx=x[sel]-24
    plot(OUT/'tes_daily_operation.png','TES feasibility | second simulated day',
         'Mixed water tank with temperature, flow, capacity and heat-leak constraints',[
        {'title':'PCC active power','unit':'MW','xlabel':'Hours within day 2',
         'series':[('No TES',xx,a.pcc_mw[sel]),('TES schedule',xx,b.pcc_mw[sel])]},
        {'title':'Requested / feasible TES heat flow (+ discharge; - charge)','unit':'MW_th','xlabel':'Hours within day 2',
         'series':[('Requested',xx,b.tes_command_mw_th[sel]),('Feasible',xx,b.tes_net_mw_th[sel])]},
        {'title':'Tank temperature (supply ceiling applies only while discharging)','unit':'deg C','xlabel':'Hours within day 2',
         'series':[('Tank',xx,b.tank_end_c[sel]),('Tank limit for <=10 C supply',xx,np.full(sel.sum(),tank_limit)),
                   ('Charge target',xx,np.full(sel.sum(),7.5))]}],
         'Schedule: charge 02-06, discharge 16-18 in elapsed-day clock. Not a tariff/DOE control.')
    f=aggregate(frames['overcommand'],c); xx=f.time_s.to_numpy()/3600
    plot(OUT/'tes_constraint_stress.png','Constraint test | over-requesting TES discharge',
         'Synthetic constant IT demand | 8 MW_th requested for 4 h | plant supplies the remainder',[
        {'title':'Requested discharge is curtailed; it is not silently delivered','unit':'MW_th','xlabel':'Elapsed time (h)',
         'series':[('Requested',xx,f.tes_command_mw_th),('Feasible',xx,f.tes_net_mw_th)]},
        {'title':'Tank limit includes the finite heat-exchanger temperature approach','unit':'deg C','xlabel':'Elapsed time (h)',
         'series':[('Tank temperature',xx,f.tank_end_c),('Tank limit for <=10 C supply',xx,np.full(len(f),tank_limit))]},
        {'title':'Room temperature remains protected by chiller fallback','unit':'deg C','xlabel':'Elapsed time (h)',
         'series':[('Room',xx,f.room_end_c),('Lower constraint',xx,np.full(len(f),22.)),
                   ('Upper constraint',xx,np.full(len(f),27.))]}],
         'A warm tank can retain energy relative to 12 C but cannot supply the declared <=10 C service.')


def report(s,c):
    normal=s['cases']['no_tes']; tes=s['cases']['with_tes']; delta=s['comparison']; cap=tank_capacity(c['tes'])
    tank_limit=maximum_discharge_tank_temperature(c['tes'])
    rows=[{'시나리오':n,'평균 PCC (MW)':s['cases'][n]['pcc_mean_mw'],
           '5분 최대 (MW)':s['cases'][n]['pcc_5min_peak_mw'],
           '기간 PUE':s['cases'][n]['pue_energy_ratio'],
           '총 전력량 (MWh)':s['cases'][n]['pcc_energy_mwh'],
           '명시한 제약 통과':s['cases'][n]['feasible_within_declared_checks']} for n in ('no_tes','with_tes')]
    ref_rows=[{'각주':v['id'],'문헌':v['title'],'확인 범위':v['original'],'변경·제외':v['adaptation']} for v in REFERENCES]
    text=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PCC·냉각·TES 검토 후보 v1</title>
<style>body{{font-family:'Malgun Gothic',Arial,sans-serif;max-width:1220px;margin:30px auto;padding:0 24px;line-height:1.75;color:#213547}}h1{{font-size:28px}}h2{{font-size:22px;margin-top:36px;border-bottom:1px solid #ccd6de;padding-bottom:8px}}.note{{padding:18px;background:#fff6e5;border-left:4px solid #b47920}}.eq{{background:#f1f5f8;padding:15px;font-family:Cambria,'Malgun Gothic',serif;overflow-wrap:anywhere}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid #d5dfe6;text-align:left;vertical-align:top}}th{{background:#eef3f7}}.scroll{{overflow:auto}}img{{max-width:100%;height:auto}}a{{color:#2168a3}}li{{margin:6px 0}}.flow{{display:flex;flex-wrap:wrap;gap:10px;align-items:center}}.box{{border:1px solid #a8bdcc;padding:14px;border-radius:7px;background:#f6f9fb}}code{{overflow-wrap:anywhere}}</style>
<h1>PCC·냉각·TES 부하모델: 검토 후보 v1</h1>
<div class="note"><b>사용자 검토 전 후보. 확정·실측 검증 완료 아님.</b> 문헌 기반 열·전력 수지를 구현하고, 가정한 설비/온도 범위 안에서 TES 명령이 가능한지 점검했다. DFL·DSO 최적화, 실제 AIDC 냉각 설계, grid-code 적합성 검증은 수행하지 않았다.</div>
<h2>1. 구성 선택과 기존 결과 보존</h2>
<div class="flow"><div class="box">고정 IT 입력<br>기존 M3+R3</div><span>→</span><div class="box">등가 공랭 공간<br>열용량·온도</div><span>→</span><div class="box">냉수 루프<br>냉동기 ↔ 축냉조</div><span>→</span><div class="box">펌프·팬·열 방출<br>UPS·변압기 손실</div><span>→</span><div class="box">PCC 유효전력</div></div>
<ul><li>입력은 {s['start_utc']}부터 {s['end_exclusive_utc']} 직전까지 기존 IT seed 20261006. 평균 {normal['it_mean_mw']:.6f} MW. M3/R3, 요청량, instance 수는 수정하지 않았다.</li>
<li>원래 1초 평균 IT 전력을 10초 평균으로 넣고 상태를 계산, 보고 출력은 5분 평균이다. 시간 간격 비교를 별도로 수행했다. 5분 평균을 온도 제약 검사 대신 사용하지 않는다.</li>
<li>첫 후보는 <b>공랭 공간 + 냉수식 냉동기 + 간접 열교환 완전혼합 축냉조</b>. 실제 액체냉각 GPU 센터라고 주장하지 않는다. 액체냉각이면 열 분기/CDU 모델이 필요하다.</li>
<li>외기는 30°C, 탱크 주변은 25°C로 고정한 <b>가상 비교 조건</b>. Azure 날짜의 실측 기상자료가 아니다.</li>
<li>IT 경계는 잠정적으로 UPS 하류 서버 AC 입력으로 정의. 서버 내부 팬/PSU는 다시 더하지 않고, 외부 CRAH 팬·UPS·변압기만 추가. 이 경계는 NLR 계측 범위 확인에 따라 변경될 수 있다.</li></ul>
<h2>2. 문헌에서 차용한 부분과 변경한 부분</h2>{table(ref_rows)}
<p>PL13의 저장량/출력 기호를 그대로 복사하지 않고 MW_th와 MWh_th를 구분했다. PL16에서는 연속시간 열수지 Eq8-9만 사용하고 예측 네트워크·원문의 이산 예측식을 복제하지 않았다. 온도 모델의 원래 시간 단위 검증을 우리의 5분 검증으로 간주하지 않는다.</p>
<h2>3. 핵심 수식</h2>
<div class="eq">P_PCC = P_IT + P_UPS,loss + P_aux + P_fan + P_CHWpump + P_TESpump + P_chiller + P_CWpump + P_tower + P_transformer,loss. (*1, *2, *4)<br>
P_UPS,loss = P_IT(1/η_UPS − 1), P_transformer,loss = P_downstream(1/η_tr − 1). 효율 정의에 따른 연구용 단순화.(*6)<br>
C_room dT_room/dt = P_IT + f_UPS P_UPS,loss + P_aux + P_fan + UA_room(T_ambient − T_room) − Q_coil. (*3, 경계 수정)<br>
Q_chiller + Q_TES,dis − Q_TES,ch = Q_coil + P_CHWpump + P_TESpump; P_chiller = Q_chiller/COP. (*1, *2)<br>
C_tank dT_tank/dt = Q_TES,dis − Q_TES,ch + UA_tank(T_ambient,tank − T_tank). (*5)<br>
C_tank = ρVcp / 3.6×10⁹ [MWh_th/K]; E_TES = C_tank(T_ref − T_tank).<br>
E_(k+1) = E_k + (Q_ch − Q_dis − Q_leak)Δt_h. (*1, *5; explicit units)<br>
Q_dis ≤ ε m_dot,max cp(T_return − T_tank); Q_ch ≤ ε m_dot,max cp(T_tank − T_source), with W→MW conversion. (*5)<br>
T_supply,worst = T_return − ε(T_return − max(T_tank,start,T_tank,end)) ≤ 10°C. 탱크 자체 온도만 검사하지 않음. (*5, *6)<br>
P_TESpump = m_dot gH / η_pump /10⁶ [MW]. (*2, hydraulic-power identity)</div>
<ul><li>탱크 열포트는 탱크 경계에서 정의하므로 별도 충·방전 효율을 중복 적용하지 않는다. 열교환 효율은 필요한 유량·온도차와 가능한 열출력을 제한한다.</li>
<li>팬과 냉수펌프는 유량비의 세제곱 형태를 이용한 축약 곡선, 냉각탑 팬/응축수펌프는 열 방출량 비례식이다. 계수는 이번 후보의 가정이며 문헌의 검증 계수를 그대로 가져온 것이 아니다.(*2, *6)</li>
<li>냉동기는 용량 내에서 이상적으로 연속 조절되는 총합 설비다. 최소 부분부하·단계별 기동/정지·응답지연은 미포함. 현재 COP는 5로 고정하고 4/6 민감도를 제시한다.</li>
<li>실내 온도와 탱크 온도는 잘라 맞추지 않는다. 입력 일정 구간의 열수지를 해석적으로 적분하며, 제약 위반은 그대로 보고한다. 기본안에서 실내온도가 24°C로 거의 일정한 것은 충분한 냉동기 용량과 이상적 온도 추종 제어를 가정했기 때문이며, 실제 센서 온도/응답을 재현했다는 의미가 아니다.</li>
<li>TES 열교환에 필요한 유량과 펌프 전력은 구간 내 가장 불리한 온도차로 계산한 보수적 값이다. 배관의 수리학·실제 밸브 제어는 식별하지 않았다.</li></ul>
<h2>4. 검토용 설비 값과 TES 운영</h2>
<ul><li>냉동기 26 MW_th, COP 5. 팬·펌프·UPS 효율은 검토용 수치이며 제조사 성능곡선이 아니다.</li>
<li>탱크 3,000 m³, C={cap:.4f} MWh_th/K. 6–12°C 기준 총 냉열량은 {cap*6:.2f} MWh_th이나, 7–{tank_limit:.4f}°C 운전창의 활용 가능 변화량은 {cap*(tank_limit-7):.2f} MWh_th이다. 둘은 다르다.</li>
<li>초기 및 복원 목표 7.5°C, 방전 공급수 상한 10°C, 유량 상한 500 kg/s, 충·방전 정격 각각 5 MW_th, 저장 예비량 2 MWh_th. ε=0.95이므로 실제 탱크 온도는 방전 시 {tank_limit:.4f}°C 이하이어야 한다. 방전 중단 후 외부 열 유입으로 이 온도를 초과하는 것은 허용되지만, 그 상태에서는 방전할 수 없다.</li>
<li>경과시간 기준 매일 02–06시 충전, 16–18시 3 MW_th 방전. 마지막 4시간은 초기 저장량 복원을 위한 충전 구간이다. 실제 요금·DSO 지령·최적화 결과가 아니다.</li>
<li>온도·유량·저장량·냉동기 여유 용량에 의해 요청한 TES 출력을 줄인다. 줄어든 TES 방전은 충분한 냉동기 용량이 있으면 냉동기가 보충한다.</li></ul>
<details><summary>전체 파라미터와 출처 펼치기</summary>{table(parameters(c))}</details>
<h2>5. 약 5일 PCC 생성 결과</h2>{table(rows)}
<p>최대 순간 비교가 아니라 동일 시각 5분 평균 차이: TES 방전 중 최대 PCC 감소 {delta['max_reduction_mw']:.4f} MW, 충전 시 최대 증가 {delta['max_increase_mw']:.4f} MW. 전 기간 에너지 변화 {delta['energy_change_mwh']:+.4f} MWh. 고정 COP와 열손실·펌프를 포함하므로 TES를 쓴다고 총 전력량이 감소하는 구조가 아니다.</p>
<p>전체 기간의 5분 최대 PCC는 두 경우 모두 {normal['pcc_5min_peak_mw']:.3f} MW이다. 고정 시간표가 최대부하 시각을 최적으로 겨냥하지 않았으므로 피크 저감 성과를 주장하지 않는다. TES 지령 전환에서 보이는 계단은 이상적 설비 조절을 가정한 운영 모델의 결과이며, 실제 냉동기 고속 ramp 파형이 아니다.</p>
<p>초기 대비 최종 저장 냉열 변화 {tes['tes_energy_change_mwh_th']:+.8f} MWh_th, 최종 탱크 온도 {tes['tank_final_c']:.6f}°C. 저장량 차이를 숨긴 채 절감량으로 해석하지 않는다.</p>
<img src="pcc_overview.png" alt="IT와 TES 사용 여부에 따른 PCC 부하"><img src="tes_daily_operation.png" alt="일간 TES 전력 온도 및 열출력">
<div class="note"><b>PUE 1.15를 달성하도록 맞추지 않았다.</b> 계산 PUE는 무TES {normal['pue_energy_ratio']:.3f}, TES {tes['pue_energy_ratio']:.3f}. 따라서 IT 약17.39 MW에 이 후보 설비를 연결하면 평균 PCC가20 MW보다 높다. 이는 IT 모델 오류의 증거가 아니라 초기 규모 가정과 이번 냉각 설비 가정이 다르다는 뜻이다. 후보 승인 후 냉각 구성을 바꾸거나, 전체 시설 규모를 재설계할 수 있으나 이번에는 IT를 축소하지 않았다.</div>
<h2>6. 물리 제약과 수치 점검</h2>
<p>기본 조건의 전기·냉수·실내·탱크 수지 오차와 제약 위반을 <a href="verification.json">verification.json</a>에 저장했다. 이는 수식/코드 일관성 점검이지 실측 타당성 검증이 아니다.</p>
<img src="tes_constraint_stress.png" alt="과도한 TES 방전 요청을 제한하는 시험">
<p>별도 냉동기 용량 부족 시험(10 MW_th)에서는 온도 상승/위반이 실제로 발생하여 검출되었다. TES 명령 제한과 서비스 냉각 부족을 서로 다른 지표로 구분한다.</p>
{table(s['numerical_sensitivity'])}
<h2>7. 파라미터 민감도</h2>{table(s['parameter_sensitivity'])}
<p>민감도는 실측 신뢰구간이 아니다. COP와 탱크 용량을 한 번에 하나씩 변경했다. 탱크 비교에서 UA·펌프 정격 등은 고정했으므로 실물 탱크를 비례 확대한 설계 비교도 아니다.</p>
<h2>8. 판단해야 할 사항과 남은 한계</h2>
<ol><li>공랭·냉수·축냉조 후보를 평가환경으로 사용할지, 직접 액체냉각/CDU 분기부터 추가할지.</li><li>3,000 m³ TES와 26 MW_th 냉동기를 시나리오 기본값으로 둘지, 특정 설비자료로 교체할지.</li><li>PUE 1.15/평균20 MW를 계속 목표로 둘지. 현재는 계산 결과와 목표의 차이를 정직하게 유지.</li></ol>
<p>제한: 탱크 성층화, 공급수·환수 배관 상태, 냉동기 부분부하 성능, 공조 코일 상세 성능, 서버 hot spot, 액체냉각, 실제 기상, 유효·무효전력 제어, ESS, DSO/DFL은 아직 미포함. 탱크 HX는 최대 유량으로 필요한 열교환이 가능한지 점검하지만, coil 측은 정격 및 고정 환수온도 가정이며 CFD/냉각 설계 인증은 아니다.</p>
<h2>9. 파일 및 재현</h2><p><a href="pcc_profiles_5min.csv">5분 PCC 프로파일</a> · <a href="internal_profiles.npz">내부10초 전력·상태</a> · <a href="summary.json">요약 수치</a> · <a href="reference_mapping.json">문헌·수식 대조표</a> · <a href="parameter_register.json">파라미터 출처</a> · <a href="generation_config.json">입력 설정</a> · <a href="source_manifest.json">원본 해시</a></p>
<p>실행: <code>python scripts/run_pcc_cooling_tes_candidate.py</code>. 시험: <code>python -m unittest discover -s tests -p test_pcc_cooling_tes.py</code>.</p>
<h2>Reference 각주</h2><ul>'''
    for ref in REFERENCES:
        link=ref.get('url') or ('https://doi.org/'+ref['doi'] if ref['doi'] else '')
        text+=f"<li>{html.escape(ref['id'])}: {html.escape(ref['title'])}. {html.escape(ref['publication'])}."
        if link: text+=f' <a href="{html.escape(link)}">원문/문헌 정보</a>'
        text+='</li>'
    text+='</ul></html>'
    (OUT/'report_ko.html').write_text(text,encoding='utf-8')


def main():
    start=time.perf_counter(); c=read(CONFIG); OUT.mkdir(parents=True,exist_ok=True)
    protected=[p for p in (ROOT/'outputs/it_5min_operating_baseline_v1').iterdir() if p.is_file()]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in protected}
    raw=np.load(ROOT/c['it_input'])['it_power_mw'][c['it_seed_index']]
    step=c['internal_step_s']; it=raw.reshape(-1,step).mean(axis=1)
    frames={}; results={}
    for name,enabled in [('no_tes',False),('with_tes',True)]:
        frames[name]=simulate(it,c,enabled); results[name]=summarize(frames[name],c,enabled)
        print(name,results[name]['pcc_mean_mw'],results[name]['pue_energy_ratio'],flush=True)
    # Deliberately excessive request, separate from the observed-demand scenario.
    n=6*3600//step; commands=np.zeros(n); commands[3600//step:5*3600//step]=8.
    frames['overcommand']=simulate(np.full(n,it.mean()),c,True,commands)
    results['overcommand']=summarize(frames['overcommand'],c,True)
    low=copy.deepcopy(c); low['plant']['chiller_capacity_mw_th']=10.
    low_frame=simulate(np.full(3600//step,20.),low)
    results['insufficient_chiller']=summarize(low_frame,low,False)
    agg={name:aggregate(f,c) for name,f in frames.items()}
    pd.concat([a.assign(scenario=n) for n,a in agg.items()],ignore_index=True).to_csv(OUT/'pcc_profiles_5min.csv',index=False)
    np.savez_compressed(OUT/'internal_profiles.npz',**{name:f.to_numpy(float) for name,f in frames.items()})
    dump(OUT/'internal_columns.json',list(frames['no_tes'].columns))
    numerical=[]
    for dt in (5,60):
        alt=copy.deepcopy(c); alt['internal_step_s']=dt
        # Full period includes terminal recovery exactly as main run.
        f=simulate(raw.reshape(-1,dt).mean(axis=1),alt,True)
        a=aggregate(f,alt); diff=a.pcc_mw.to_numpy()-agg['with_tes'].pcc_mw.to_numpy()
        numerical.append({'내부 간격(s)':dt,'10초 대비 5분 MAE(kW)':float(np.mean(abs(diff))*1000),
            '10초 대비 최대 차이(kW)':float(max(abs(diff))*1000),
            '최대 실내온도(°C)':float(f.room_end_c.max()),'최종 탱크온도(°C)':float(f.tank_end_c.iloc[-1])})
        print('step sensitivity',dt,flush=True)
    sensitivity=[]
    for name,group,key,value in [('COP4','plant','cop',4.),('COP6','plant','cop',6.),
                                ('Tank1500m3','tes','volume_m3',1500.),('Tank6000m3','tes','volume_m3',6000.)]:
        alt=copy.deepcopy(c); alt[group][key]=value
        f=simulate(it,alt,True); m=summarize(f,alt,True)
        sensitivity.append({'조건':name,'평균 PCC(MW)':m['pcc_mean_mw'],'PUE':m['pue_energy_ratio'],
            '방전량(MWh_th)':m['tes_discharge_mwh_th'],'명령 제한(MWh_th)':m['tes_command_curtailed_mwh_th'],
            '온도 등 제약 통과':m['feasible_within_declared_checks'],
            '최종 저장량 차이(MWh_th)':m['tes_energy_change_mwh_th']})
        print('parameter sensitivity',name,flush=True)
    delta=agg['with_tes'].pcc_mw-agg['no_tes'].pcc_mw
    meta=read(ROOT/'outputs/it_5min_operating_baseline_v1/summary.json')
    summary={'status':c['status'],'created_utc':datetime.now(timezone.utc).isoformat(),
        'start_utc':meta['start_utc'],'end_exclusive_utc':meta['end_exclusive_utc'],'cases':results,
        'comparison':{'max_reduction_mw':float(-delta.min()),'max_increase_mw':float(delta.max()),
                      'energy_change_mwh':results['with_tes']['pcc_energy_mwh']-results['no_tes']['pcc_energy_mwh']},
        'numerical_sensitivity':numerical,'parameter_sensitivity':sensitivity,
        'physical_facility_validated':False,'selected_as_final':False,'dfl_or_dso_optimized':False,
        'runtime_seconds':time.perf_counter()-start}
    unchanged=all(digest(ROOT/p)==h for p,h in hashes.items())
    verification={'existing_it_files_unchanged':unchanged,'protected_file_count':len(hashes),
        'IT_energy_mwh_input':float(raw.sum()/3600),'IT_energy_mwh_internal':float(frames['no_tes'].it_mw.sum()*step/3600),
        'IT_energy_mwh_5min':float(agg['no_tes'].it_mw.sum()*300/3600),
        'balances':{n:r['max_abs_balances'] for n,r in results.items()},
        'constraints':{n:r['constraints'] for n,r in results.items()},
        'negative_control_detected':results['insufficient_chiller']['constraints']['room_temperature_violations']>0}
    assert unchanged
    assert results['no_tes']['feasible_within_declared_checks'] and results['with_tes']['feasible_within_declared_checks']
    assert abs(results['with_tes']['tes_energy_change_mwh_th'])<1e-5
    assert verification['negative_control_detected']
    dump(OUT/'source_manifest.json',hashes); dump(OUT/'generation_config.json',c)
    dump(OUT/'reference_mapping.json',REFERENCES); dump(OUT/'parameter_register.json',parameters(c))
    dump(OUT/'summary.json',summary); dump(OUT/'verification.json',verification)
    figures(frames,summary,c); report(summary,c)
    print(json.dumps({'output':str(OUT),'comparison':summary['comparison'],'verified':verification},ensure_ascii=False),flush=True)


if __name__=='__main__': main()
