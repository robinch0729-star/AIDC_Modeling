"""Reference-informed ESS addition and reproducible storage-portfolio comparison."""
from copy import deepcopy
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import scipy

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from aidc_facility.ess_tes_scheduler import solve_joint_schedule
from aidc_facility.tes_flexibility import marginal_pcc_coefficients, pcc_from_thermal
from run_tes_pcc_scheduling import read, dump, sha, cap_metrics, make_cap, table

OUT = ROOT/'outputs/ess_tes_pcc_scheduling_v1'


def solve(rows, cap, tes, legacy, cfg, strategy, ess=None):
    return solve_joint_schedule(
        rows, cap, tes, legacy, ess or cfg['ess'], dt_h=cfg['step_s']/3600,
        initial_tes_soc=cfg['initial_soc'], terminal_tes_soc=cfg['terminal_soc'],
        initial_ess_soc=cfg['initial_soc'], terminal_ess_soc=cfg['terminal_soc'],
        cop=cfg['cop'], use_tes=strategy['use_tes'], use_ess=strategy['use_ess'],
        objective=cfg['objective'], time_limit_s=cfg['time_limit_s'])


def figures(cfg, summaries, solutions, p0, hours):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False})
    strategies = cfg['strategies']
    cases = cfg['scenarios']
    grid = np.zeros((4,len(cases)))
    lookup = {(r['case_id'],r['strategy_id']):r for r in summaries}
    labels = ['No storage','TES only','ESS only','ESS + TES']
    for j, case in enumerate(cases):
        ref = lookup[(case['id'],'joint')]
        grid[0,j] = float(ref['no_storage']['cap_excess_duration_h']==0)
        for i, strategy in enumerate(strategies,1):
            item = lookup[(case['id'],strategy['id'])]
            grid[i,j] = 1 if item['feasible'] else (0 if item['status']=='infeasible' else .5)
    fig, ax = plt.subplots(figsize=(13,4),layout='constrained')
    ax.imshow(grid, cmap=ListedColormap(['#f6d8cf','#f6e9b8','#c9e8df']),vmin=0,vmax=1,aspect='auto')
    ax.set(yticks=np.arange(4),yticklabels=labels,
           xticks=np.arange(len(cases)),xticklabels=['27.0 MW','26.0 MW','25.8 MW','25.5 MW','24.8 MW','Time-varying','5 h stress'])
    for i in range(4):
        for j in range(len(cases)):
            ax.text(j,i,'Meets cap' if grid[i,j]==1 else ('Not solved' if grid[i,j]==.5 else ('Exceeds cap' if i==0 else 'Infeasible')),
                    ha='center',va='center',fontsize=10,color='#173c4d')
    ax.set_title('Storage portfolios under identical hypothetical 5-minute PCC limits',pad=14)
    fig.supxlabel('ESS: 1 MW / 2 MWh; TES: 3 MWth / 10 MWhth; enabled storage ends at its initial SOC',fontsize=10)
    fig.savefig(OUT/'portfolio_feasibility.png',dpi=180)
    plt.close(fig)
    for key, file, title in (
        ('constant_25_5','joint_dispatch_25_5.png','Joint ESS-TES scheduling: 25.5 MW PCC limit'),
        ('constant_25_8','joint_dispatch_25_8.png','Storage dispatch comparison: 25.8 MW PCC limit')):
        result = solutions[(key,'joint')]
        if not result['feasible']:
            continue
        s = result['schedule']
        edges = np.r_[hours,hours[-1]+cfg['step_s']/3600]
        fig, axes = plt.subplots(4,1,figsize=(12,10),sharex=True,layout='constrained')
        axes[0].plot(hours,p0,color='#9dabb3',lw=1,label='No storage')
        for strategy,color in [('ess_only','#b67a28'),('joint','#007e83')]:
            other = solutions[(key,strategy)]
            if other['feasible']:
                axes[0].plot(hours,other['schedule']['pcc_mw'],lw=1,label='ESS only' if strategy=='ess_only' else 'ESS + TES',color=color)
        axes[0].step(hours,s['pcc_upper_mw'],where='post',ls='--',color='#ba4c4c',label='5-minute limit')
        axes[0].set(ylabel='PCC power (MW)',title=title)
        axes[0].legend(frameon=False,ncols=4,fontsize=9)
        axes[1].stairs(s['ess_charge_mw'],edges,color='#256ba1',label='ESS charge (+)')
        axes[1].stairs(-np.asarray(s['ess_discharge_mw']),edges,color='#d37c28',label='ESS discharge (-)')
        axes[1].set(ylabel='ESS AC power (MW)',ylim=(-1.1,1.1))
        axes[1].legend(frameon=False,ncols=2)
        axes[2].stairs(s['tes_charge_mw_th'],edges,color='#256ba1',label='TES charge (+)')
        axes[2].stairs(-np.asarray(s['tes_discharge_mw_th']),edges,color='#d37c28',label='TES discharge (-)')
        axes[2].set(ylabel='TES heat rate (MWth)',ylim=(-3.3,3.3))
        axes[2].legend(frameon=False,ncols=2)
        axes[3].plot(edges,np.asarray(s['ess_energy_mwh'])/cfg['ess']['energy_capacity_mwh']*100,color='#326db2',label='ESS SOC')
        axes[3].plot(edges,np.asarray(s['tes_energy_mwh_th'])/10*100,color='#007e83',label='TES SOC')
        axes[3].axhline(50,color='#818b91',ls=':',label='Initial / final SOC')
        axes[3].set(ylabel='SOC (%)',xlabel='Elapsed simulation time (h)',ylim=(0,105))
        axes[3].legend(frameon=False,ncols=3)
        for ax in axes:
            ax.grid(alpha=.2)
        fig.suptitle('Perfect foresight; hard average-power cap; charging and terminal recovery included',fontsize=11)
        fig.savefig(OUT/file,dpi=180)
        plt.close(fig)


def report(cfg, candidate, summary, verification):
    statuses = {'optimal':'준수 가능 (최적해)','feasible_time_limit':'준수 가능 (최적성 미확인)',
                'infeasible':'제약 내 운전 불가능'}
    overview = []
    for case in cfg['scenarios']:
        r = {'PCC 상한 조건':case['label_ko']}
        group = [s for s in summary['results'] if s['case_id']==case['id']]
        r['저장장치 미사용'] = '상한 준수' if group[0]['no_storage']['cap_excess_duration_h']==0 else '상한 초과'
        for strategy in cfg['strategies']:
            item = next(s for s in group if s['strategy_id']==strategy['id'])
            r[strategy['label_ko']] = statuses.get(item['status'],item['status'])
        overview.append(r)
    detail = []
    fine = []
    for r in summary['results']:
        if not r['feasible']:
            continue
        m = r['operating_metrics']
        detail.append({'조건':r['case_label_ko'],'운영':r['strategy_label_ko'],
                       '최대 PCC (MW)':m['pcc_peak_mw'],'5분 초과시간 (h)':m['cap_excess_duration_h'],
                       '미사용 대비 추가 전력량 (MWh)':m['pcc_energy_mwh']-r['no_storage']['pcc_energy_mwh'],
                       'ESS 방전량 (MWh AC)':r['ess_discharge_mwh'],
                       'TES 방전량 (MWhth)':r['tes_discharge_mwh_th']})
        fine.append({'조건':r['case_label_ko'],'운영':r['strategy_label_ko'],
                     '10초 최대 초과 (MW)':r['held_10s']['cap_excess_max_mw'],
                     '10초 초과시간 (h)':r['held_10s']['cap_excess_duration_h']})
    sens = [{'사양':r['label'],'상한':r['case_label_ko'],'결과':statuses.get(r['status'],r['status']),
             '최대 PCC (MW)':r.get('pcc_peak_mw')} for r in summary['sensitivity']]
    matched = [{'상한':r['case_label_ko'],'ESS+TES 대기':statuses.get(r['idle_status'],r['idle_status']),
                '공동 운영':statuses.get(r['joint_status'],r['joint_status']),
                '공동운영 추가 전력량 − 대기조건 (MWh)':r['joint_minus_idle_mwh']}
               for r in summary['matched_hardware_comparison']]
    refs = ''.join(f'<li><a href="{html.escape(r["url"])}">{html.escape(r["title"])}</a>: {html.escape(r["used"])}</li>' for r in cfg['references'])
    provenance = [{'항목':r['parameter'],'값':r['value'],'근거 구분':r['classification'],'설명':r['basis']} for r in cfg['parameter_provenance']]
    text = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ESS–TES 공동 운영 검토</title><style>
body{{font-family:"Malgun Gothic",Arial,sans-serif;max-width:1200px;margin:32px auto;padding:0 24px;color:#153d4e;line-height:1.8}}h1{{font-size:28px}}h2{{margin-top:34px;font-size:22px;border-bottom:1px solid #cbdcde}}.note{{background:#fff2df;padding:18px;border-left:4px solid #c18c41}}.eq{{padding:18px;background:#edf6f8;font-family:Cambria,serif}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;vertical-align:top;padding:10px;border-bottom:1px solid #d3e0e6}}th{{background:#edf6f8}}.scroll{{overflow:auto}}img{{max-width:100%;height:auto}}a{{color:#087486}}</style></head><body>
<h1>ESS 모델 추가 및 ESS–TES 공동 운영: PCC 상한 준수 검토</h1>
<p class="note"><b>검토 범위:</b> 동일한 약 {summary['duration_h']:.3f}시간의 부하에서 TES 단독·ESS 단독·공동 운영을 비교했다. 상한은 연구용 가상 입력이며 5분 평균 유효전력에 적용된다. DSO 계통해석, 실제 접속기준 준수, 예측오차, 실시간 제어 또는 경제성 검증은 아니다.</p>
<h2>1. 이번에 추가한 내용</h2><ul><li>운영용 ESS의 에너지 수지·SOC·출력 한계·효율·동시 충방전 방지 제약 추가.</li><li>TES와 ESS의 운전량을 동시에 결정하고, 충전 시에도 동일 PCC 상한을 적용.</li><li>활성 저장장치는 시작과 종료 SOC를 모두 50%로 강제. 사용한 에너지는 실제 충전으로 복원.</li><li>ESS를 추가했으므로 단순 공동 운영 결과는 장치 추가 효과를 포함한다. 동일 설비에서 TES를 대기시키는 비교도 별도로 수행.</li></ul>
<h2>2. ESS 사양과 문헌 근거</h2><p><b>기본 후보: 1 MW / 2 MWh, 왕복효율 85%, SOC 10–90%.</b> ATB는 1–8시간 저장시간과 85% 대표 효율을 제공한다. 1 MW AC/2 MWh 자체는 현재 연구용으로 선정한 값이며 실제 AIDC의 최적 사양·제품 사양이 아니다. ATB 표의 DC 정격을 그대로 AC 출력으로 바꾼 것이 아니다.</p>
<p>ηch=ηdis=√0.85≈0.921954로 대칭 분할했다. 명목 E/P는 2시간이지만 SOC 범위와 방전효율을 고려하면 최대 SOC에서 1 MW로 방전 가능한 시간은 약 1.475시간이다. 시작 SOC50%에서는 사전충전 없이 약0.738시간이다. 열량 단위 MWhth와 전기 에너지 MWh를 혼동하지 않는다.</p>{table(provenance)}
<h2>3. 계산 경계 및 수식</h2><div class="eq">P_PCC[k] = P0[k] + αch Q_TES,ch[k] − αdis Q_TES,dis[k] + P_ESS,ch[k] − P_ESS,dis[k]<br>E_ESS[k+1] = E_ESS[k] + (ηch P_ESS,ch[k] − P_ESS,dis[k]/ηdis) Δt<br>0.2 ≤ E_ESS[k] ≤ 1.8 MWh; E_ESS[0] = E_ESS[T] = 1 MWh<br>0 ≤ P_ESS,ch[k] ≤ 1·z_ESS[k]; 0 ≤ P_ESS,dis[k] ≤ 1·(1−z_ESS[k]) MW<br>E_TES[k+1] = E_TES[k] + (0.9 Q_TES,ch[k] − Q_TES,dis[k]/0.9 − 0.004) Δt<br>1 ≤ E_TES[k] ≤ 10 MWhth; E_TES[0] = E_TES[T] = 5 MWhth<br>0 ≤ P_PCC[k] ≤ P_limit[k]; min Σ P_PCC[k] Δt; Δt=1/12 h</div>
<ul><li>ESS는 기존 UPS와 분리된 운영용 장치다. 비상전원 에너지를 임의 활용하지 않는다.</li><li>ESS 출력은 PCC에 환산한 AC 포트로 정의한다. η에 저장 경로의 유효 손실을 반영하므로 UPS/PCS/변압기 효율을 다시 곱하지 않는다. 실제 내부 접속 회로를 상세 설계했다는 의미는 아니다.</li><li>ESS 열손실은 기존 서버실 열경계 외부로 가정한다. 별도 배터리 HVAC·대기손실·열화·비상예비 제약은 미포함이며 물리적으로 0이라고 주장하지 않는다.</li><li>ESS·TES 각각 동시 충방전 금지. 서로 다른 장치의 충방전 조합은 허용하며 역할을 사전에 빠른/느린 제어로 나누지 않는다.</li><li>목적함수는 수전 전력량이며 요금·열화비용·설비투자비를 포함한 비용 최적화가 아니다.</li><li>전체 미래 부하를 안다는 오프라인 조건. 상수 효율에서 여러 충전시각이 같은 목적값을 가질 수 있으므로 그림의 충전시각을 유일하거나 경제적인 시각으로 해석하지 않는다.</li></ul>
<h2>4. 같은 부하·상한에서 저장장치 구성 비교</h2>{table(overview)}<img src="portfolio_feasibility.png" alt="Storage portfolios feasibility matrix">
<p>시간가변 조건은 경과48–72 h에26 MW, 그 외27 MW를 적용한다. 5시간 스트레스 조건은 경과36–41 h에 <b>P_limit[k]=P0[k]−약0.428 MW</b>로 두어 연속 감축을 요구한 합성 진단이다. 기준 부하를 따라 움직이는 인위적 신호이며 실제 DSO 지시나 독립적인 혼잡 신호로 해석하지 않는다.</p>
<p><b>현재 기준 사양의 핵심:</b> 25.8 MW는 ESS 단독으로 대응할 수 있으나, 25.5 MW에서는 ESS·TES 각각 단독으로는 불가능하고 공동 운전은 가능했다. ESS 단독의 피크 시각 출력 하한은 약25.553 MW(기준26.553−ESS1)이기 때문이다. 공동 운전에서도 24.8 MW는 불가능하며, 최대 두 장치 출력을 고려한 해당 시각 하한은 약24.911 MW(26.553−1−0.642)이다. 이는 전체 가능한 최저 PCC 상한을 탐색한 결과가 아니라, 지정한 상한 시나리오의 비교다.</p>
<p>운전 불가능은 원래 수요를 강제로 줄이거나 상태를 덮어써 해결하지 않았다. 충분한 출력이 있어도 누적 에너지 또는 재충전 여유가 부족하면 불가능할 수 있다. 시간 제한/검증 실패는 운전 불가능과 별도로 표시한다.</p>
<h2>5. 대표 운전계획과 세부 수치</h2><img src="joint_dispatch_25_5.png" alt="ESS TES joint dispatch under 25.5 MW limit"><img src="joint_dispatch_25_8.png" alt="ESS and TES dispatch under 25.8 MW limit">{table(detail)}
<p>TES 단독/공동에는 TES 대기 열손실이 있으나, ESS 단독은 TES 미설치로 취급한다. 따라서 전력량 차이는 제어 방식뿐 아니라 설비 구성 차이도 포함한다. 공동 운전이 모든 상한에서 전력량을 더 줄인다고 전제하지 않았다. ESS와 TES의 채택 효율에 따라 한 장치가 우선 사용될 수 있으며, 이는 본 목적함수·가정에 한정된다.</p>
<h2>6. 동일 설비를 둔 비교 및 용량 민감도</h2><p>ESS+TES 설비는 그대로 두되 TES의 방전 출력을 0으로 제한한 조건과 공동 운전을 비교했다. 대기 열손실 보충 충전과 종료 저장량 복원은 두 조건 모두 포함한다.</p>{table(matched)}
<p>아래 용량 민감도는 후보 규모의 영향을 보는 진단이며 용량 최적화나 투자 대안 비교가 아니다.</p>{table(sens)}
<h2>7. 5분 준수와 10초 구간 내부 변동의 차이</h2>{table(fine)}<p class="note">5분 평균 상한을 지켜도 같은 명령을 10초 기존 입력에 대입하면 구간 내부에서 상한을 넘을 수 있다. ESS 추가만으로 빠른 제어까지 구현된 것은 아니다. 실제 상한이 순간값이면 별도 안전여유·하위 제어를 검토해야 한다. 본 10초 검사는 고정된 입력에 명령을 대입한 진단이며 신규 열동역학 검증이 아니다.</p>
<h2>8. 검증 기록과 연구적 의미</h2><ul><li>가용 해는 독립 상태 재계산·에너지 수지·PCC·장치 출력·동시충방전·냉동기 용량·종료 SOC를 확인: {verification['all_feasible_verified']}.</li><li>이전 TES 단독 결과와 같은 조건의 해결 상태 및 목적값 비교: {verification['tes_only_regression_passed']}. 최적 운전계획은 비유일할 수 있으므로 저장량 궤적의 일치까지 요구하지 않음.</li><li>관련 단위시험 통과: {verification['unit_tests']['passed']}. 기존 보호 파일 {verification['protected_file_count']}개 변경 없음: {verification['protected_files_unchanged']}.</li><li>확보한 것은 PCC 상한을 입력으로 받아 두 저장장치의 운전을 함께 계산하는 AIDC 측 기준 운영모델이다. 아직 DSO–AIDC 정보교환이나 실제 배전망 제약을 구현한 것은 아니다.</li></ul>
<h2>9. 출처·재현 파일</h2><ul>{refs}</ul><p>TES 사양과 기존 냉각·손실 근거는 <a href="../pcc_tes_operating_candidate_v1/report_ko.html">이전 문헌 기반 TES 후보</a>를 유지한다.</p><p><a href="summary.json">요약 결과</a> · <a href="model_contract.json">설정·출처·경계</a> · <a href="verification.json">검증 기록</a> · <a href="schedules_5min.npz">운전계획 배열</a></p></body></html>'''
    for name in ('joint_dispatch_25_5.png','joint_dispatch_25_8.png'):
        if not (OUT/name).exists():
            import re
            text = re.sub(r'<img src="'+re.escape(name)+r'"[^>]*>','<p>해당 조건은 가용 해가 없어 운전계획 그림을 제공하지 않는다.</p>',text)
    (OUT/'report_ko.html').write_text(text,encoding='utf-8')


def main():
    cfg = read(ROOT/'configs/ess_tes_pcc_scheduling_v1.json')
    candidate = read(ROOT/cfg['candidate_config'])
    legacy = read(ROOT/candidate['legacy_config'])
    tes = candidate['tes']
    protected = set()
    for name in ('tes_pcc_scheduling_v1','tes_flexibility_assessment_v1','pcc_tes_operating_candidate_v1',
                 'pcc_cooling_tes_candidate_v1','it_pcc_boundary_audit_v1','pcc_capacity_cop_review_v1','pcc_temperature_conditions_v1'):
        protected.update(p for p in (ROOT/'outputs'/name).rglob('*') if p.is_file())
    for p in ('configs/tes_pcc_scheduling_v1.json',cfg['candidate_config'],candidate['legacy_config'],
              'src/aidc_facility/tes_scheduler.py','src/aidc_facility/tes_operating.py','src/aidc_facility/tes_flexibility.py'):
        protected.add(ROOT/p)
    before = {str(p.relative_to(ROOT)):sha(p) for p in sorted(protected)}
    cols = read(ROOT/candidate['baseline_columns'])
    original = np.load(ROOT/candidate['baseline_profiles'])['no_tes']
    block = cfg['step_s']//legacy['internal_step_s']
    if cfg['step_s'] % legacy['internal_step_s'] or len(original)%block:
        raise ValueError('Complete aggregation bins required')
    base = original.reshape(-1,block,len(cols)).mean(1)
    rows = [dict(zip(cols,map(float,row))) for row in base]
    dt = cfg['step_s']/3600
    hours = np.arange(len(rows))*dt
    p0 = np.array([pcc_from_thermal(r,0.,0.,legacy,cfg['cop'],tes['pump_mw_per_mw_th'])['pcc_mw'] for r in rows])
    coeff = marginal_pcc_coefficients(legacy,cfg['cop'],tes['pump_mw_per_mw_th'])
    OUT.mkdir(parents=True,exist_ok=True)
    solution_records = {}

    def save_solution(name, result):
        if name in solution_records:
            raise ValueError('Duplicate solution name in this run: '+name)
        solution_records[name] = result
        dump(OUT/name,result)

    arrays = {'time_h':hours,'no_storage_pcc_mw':p0}
    solutions, summaries = {}, []
    for case in cfg['scenarios']:
        cap = make_cap(case,p0,hours,coeff)
        arrays[case['id']+'_cap_mw'] = cap
        for strategy in cfg['strategies']:
            key = (case['id'],strategy['id'])
            print('Solving '+ '/'.join(key),flush=True)
            result = solve(rows,cap,tes,legacy,cfg,strategy)
            solutions[key] = result
            save_solution('_'.join(key)+'_solution.json',result)
            item = {'case_id':case['id'],'case_label_ko':case['label_ko'],
                    'strategy_id':strategy['id'],'strategy_label_ko':strategy['label_ko'],
                    'status':result['status'],'feasible':result['feasible'],'optimal':result['optimal'],
                    'solver':result['solver'],'no_storage':cap_metrics(p0,cap,dt)}
            if result['feasible']:
                s = result['schedule']
                item['operating_metrics'] = cap_metrics(s['pcc_mw'],cap,dt)
                item['kernel_metrics'] = result['metrics']
                item['ess_discharge_mwh'] = float(np.sum(s['ess_discharge_mw'])*dt)
                item['tes_discharge_mwh_th'] = float(np.sum(s['tes_discharge_mw_th'])*dt)
                delta = coeff['increase_mwe_per_mwth']*np.asarray(s['tes_charge_mw_th'])-coeff['reduction_mwe_per_mwth']*np.asarray(s['tes_discharge_mw_th'])+np.asarray(s['ess_charge_mw'])-np.asarray(s['ess_discharge_mw'])
                p10 = original[:,cols.index('pcc_mw')]+np.repeat(delta,block)
                q10 = original[:,cols.index('chiller_mw_th')]+np.repeat((1+tes['pump_mw_per_mw_th'])*np.asarray(s['tes_charge_mw_th'])-(1-tes['pump_mw_per_mw_th'])*np.asarray(s['tes_discharge_mw_th']),block)
                item['held_10s'] = cap_metrics(p10,np.repeat(cap,block),legacy['internal_step_s']/3600)
                item['held_10s']['minimum_chiller_headroom_mw_th'] = float(legacy['plant']['chiller_capacity_mw_th']-q10.max())
                item['held_10s']['minimum_chiller_output_mw_th'] = float(q10.min())
                item['held_10s']['reaggregation_error_mw'] = float(np.max(np.abs(p10.reshape(-1,block).mean(1)-s['pcc_mw'])))
                for name,value in s.items():
                    arrays['_'.join(key)+'_'+name] = np.asarray(value)
            summaries.append(item)
            print(json.dumps({'case':case['id'],'strategy':strategy['id'],'status':result['status'],
                              'energy_mwh':item.get('operating_metrics',{}).get('pcc_energy_mwh')},ensure_ascii=False),flush=True)
    sensitivities = []
    sensitivity_case = next(c for c in cfg['scenarios'] if c['id']=='constant_25_5')
    cap = make_cap(sensitivity_case,p0,hours,coeff)
    joint_strategy = next(s for s in cfg['strategies'] if s['id']=='joint')
    for sen in cfg['sensitivity']:
        ess = dict(cfg['ess'],energy_capacity_mwh=sen['energy_mwh'],charge_limit_mw=sen['power_mw'],discharge_limit_mw=sen['power_mw'])
        result = solutions[('constant_25_5','joint')] if sen['id']=='base' else solve(rows,cap,tes,legacy,cfg,joint_strategy,ess)
        save_solution('sensitivity_'+sen['id']+'_solution.json',result)
        sensitivities.append({'label':f"{sen['power_mw']} MW / {sen['energy_mwh']} MWh",'case_label_ko':sensitivity_case['label_ko'],
                              'status':result['status'],'feasible':result['feasible'],
                              'pcc_peak_mw':result['metrics']['pcc_peak_mw'] if result['feasible'] else None})
        print('Sensitivity '+sen['id']+': '+result['status'],flush=True)
    matched = []
    for cid in ('constant_26','constant_25_8','constant_25_5'):
        case = next(c for c in cfg['scenarios'] if c['id']==cid)
        cap = make_cap(case,p0,hours,coeff)
        idle_tes = dict(tes,discharge_limit_mw_th=0.)
        idle_result = solve(rows,cap,idle_tes,legacy,cfg,joint_strategy)
        save_solution(cid+'_ess_with_tes_idle_solution.json',idle_result)
        joint = solutions[(cid,'joint')]
        delta = joint['metrics']['pcc_energy_mwh']-idle_result['metrics']['pcc_energy_mwh'] if joint['feasible'] and idle_result['feasible'] else None
        matched.append({'case_label_ko':case['label_ko'],'idle_status':idle_result['status'],'joint_status':joint['status'],'joint_minus_idle_mwh':delta})
        print('Matched hardware '+cid+': '+idle_result['status'],flush=True)
    old = read(ROOT/'outputs/tes_pcc_scheduling_v1/summary.json')
    regression = []
    for case in cfg['scenarios']:
        prior = next((r for r in old['scenarios'] if r['id']==case['id']),None)
        if prior is None:
            continue
        current = solutions[(case['id'],'tes_only')]
        good = current['status']==prior['status']
        difference = None
        if current['feasible'] and prior['feasible']:
            difference = abs(current['metrics']['pcc_energy_mwh']-prior['optimized']['pcc_energy_mwh'])
            good = good and difference<1e-6
        regression.append({'case':case['id'],'passed':good,'objective_difference_mwh':difference})
    tests = subprocess.run([sys.executable,'-X','utf8','-B','-m','unittest','discover','-s','tests','-p','test_*tes*.py','-v'],
                           cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    after = {str(p.relative_to(ROOT)):sha(p) for p in sorted(protected)}
    fine_ok = all(r['held_10s']['minimum_chiller_headroom_mw_th']>=-1e-7 and r['held_10s']['minimum_chiller_output_mw_th']>=-1e-7 and r['held_10s']['reaggregation_error_mw']<1e-8 for r in summaries if r['feasible'])
    all_outputs = list(solution_records.values())
    expected_count = len(cfg['scenarios'])*len(cfg['strategies'])+len(cfg['sensitivity'])+3
    if len(solution_records) != expected_count:
        raise AssertionError('Missing solution records in this run')
    verified = all(all(r['checks'].values()) for r in all_outputs if r['feasible'])
    statuses_resolved = all(r['status'] in ('optimal','infeasible','feasible_time_limit') for r in all_outputs)
    verification = {'protected_file_count':len(before),'protected_files_unchanged':before==after,
                    'solution_file_count':len(solution_records),'solution_files_this_run':list(solution_records),
                    'protected_sha256':before,'all_feasible_verified':verified,'all_solver_statuses_resolved':statuses_resolved,
                    'tes_only_regression_passed':all(r['passed'] for r in regression),'tes_only_regression':regression,
                    'held_10s_chiller_and_reaggregation_passed':fine_ok,
                    'unit_tests':{'passed':tests.returncode==0,'output':tests.stdout+tests.stderr}}
    summary = {'created_utc':datetime.now(timezone.utc).isoformat(),'duration_h':len(rows)*dt,'intervals':len(rows),
               'no_storage':{'mean_pcc_mw':float(p0.mean()),'peak_pcc_mw':float(p0.max()),'energy_mwh':float(p0.sum()*dt)},
               'coefficients':coeff,'ess_config':cfg['ess'],'results':summaries,'sensitivity':sensitivities,
               'matched_hardware_comparison':matched,
               'scope':'Offline perfect-foresight storage portfolio comparison; hypothetical 5-minute mean caps; no DSO network or uncertainty implementation.'}
    dump(OUT/'summary.json',summary)
    dump(OUT/'verification.json',verification)
    dump(OUT/'model_contract.json',{'experiment':cfg,'inherited_tes_candidate':candidate,'scipy_version':scipy.__version__,
                                 'common_conditions':'Same load, horizon and caps; enabled devices start/end at50%; device absence excludes its standby loss.',
                                 'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/aidc_facility/ess_tes_scheduler.py',ROOT/'configs/ess_tes_pcc_scheduling_v1.json',Path(__file__)]}})
    np.savez_compressed(OUT/'schedules_5min.npz',**arrays)
    figures(cfg,summaries,solutions,p0,hours)
    report(cfg,candidate,summary,verification)
    print(json.dumps({'report':str(OUT/'report_ko.html'),'verified':verified,'resolved':statuses_resolved,'tests':tests.returncode==0,'old_files_unchanged':before==after},ensure_ascii=False),flush=True)
    if not (verified and statuses_resolved and fine_ok and before==after and verification['tes_only_regression_passed'] and tests.returncode==0):
        raise AssertionError('One or more validation checks failed; inspect verification.json')


if __name__=='__main__':
    main()
