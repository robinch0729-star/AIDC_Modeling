"""Plots and Korean report for the capacity-only 2-to-4 MWh experiment."""
import html
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def table(rows):
    keys = list(rows[0])
    def fmt(value):
        if value is None:
            return '—'
        return f'{value:.4f}' if isinstance(value,float) else html.escape(str(value))
    return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+fmt(r.get(k))+'</td>' for k in keys)+'</tr>' for r in rows)+'</table></div>'


def figures(cfg, candidate, solutions, p0, hours, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'axes.spines.top':False,'axes.spines.right':False})
    dt = cfg['step_s']/3600
    edges = np.r_[hours,hours[-1]+dt]
    case = solutions[('constant_25_5','joint')]
    if not case['feasible']:
        return []
    s = case['schedule']
    old = read(ROOT/'outputs/ess_tes_pcc_scheduling_v1/constant_25_5_joint_solution.json')['schedule']
    fig, ax = plt.subplots(4,1,figsize=(12,10),sharex=True,layout='constrained')
    ax[0].plot(hours,p0,color='#a6afb4',lw=1,label='No storage')
    ax[0].plot(hours,s['pcc_mw'],color='#087f82',lw=1.2,label='ESS 1 MW / 4 MWh + TES')
    ax[0].step(hours,s['pcc_upper_mw'],where='post',ls='--',color='#bd514d',label='5-minute PCC limit')
    ax[0].set(ylabel='PCC power (MW)',title='Joint ESS-TES operation with a 25.5 MW PCC limit')
    ax[0].legend(frameon=False,ncols=3,fontsize=9)
    ax[1].stairs(s['ess_charge_mw'],edges,color='#2b69a0',label='ESS charge (+)')
    ax[1].stairs(-np.asarray(s['ess_discharge_mw']),edges,color='#c37830',label='ESS discharge (-)')
    ax[1].set(ylabel='ESS AC power (MW)',ylim=(-1.1,1.1))
    ax[1].legend(frameon=False,ncols=2)
    ax[2].stairs(s['tes_charge_mw_th'],edges,color='#2b69a0',label='TES charge (+)')
    ax[2].stairs(-np.asarray(s['tes_discharge_mw_th']),edges,color='#c37830',label='TES discharge (-)')
    ax[2].set(ylabel='TES heat rate (MWth)',ylim=(-3.3,3.3))
    ax[2].legend(frameon=False,ncols=2)
    ax[3].plot(edges,np.asarray(s['ess_energy_mwh'])/cfg['ess']['energy_capacity_mwh']*100,color='#2b69a0',label='ESS SOC')
    ax[3].plot(edges,np.asarray(s['tes_energy_mwh_th'])/candidate['tes']['energy_capacity_mwh_th']*100,color='#087f82',label='TES SOC')
    ax[3].axhline(50,color='#899297',ls=':',label='Initial / final SOC')
    ax[3].set(ylabel='SOC (%)',xlabel='Elapsed simulation time (h)',ylim=(0,105))
    ax[3].legend(frameon=False,ncols=3)
    for a in ax:
        a.grid(alpha=.2)
    fig.suptitle('Fixed ESS output: 1 MW; capacity: 4 MWh; perfect-foresight 5-minute scheduling',fontsize=12)
    fig.savefig(out/'joint_dispatch_4mwh.png',dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(3,1,figsize=(12,8),sharex=True,layout='constrained')
    for data,capacity,color in [(old,2.,'#b87630'),(s,4.,'#087f82')]:
        label=f'ESS {capacity:g} MWh (same 1 MW output)'
        ax[0].plot(hours,data['ess_discharge_mw'],color=color,lw=1,label=label)
        ax[1].plot(hours,data['tes_discharge_mw_th'],color=color,lw=1,label=label)
        ax[2].plot(edges,np.asarray(data['ess_energy_mwh'])/capacity*100,color=color,lw=1.2,label=label)
    ax[0].set(ylabel='ESS discharge (MW)',title='Capacity-only comparison under the same 25.5 MW PCC limit',ylim=(-.03,1.07))
    ax[1].set(ylabel='TES discharge (MWth)',ylim=(-.1,3.2))
    ax[2].set(ylabel='ESS SOC (%)',xlabel='Elapsed simulation time (h)',ylim=(0,100))
    for a in ax:
        a.grid(alpha=.2)
        a.legend(frameon=False,ncols=2,fontsize=10)
    fig.suptitle('Same workload, limits, efficiencies and SOC percentages; different initial stored MWh',fontsize=11)
    fig.savefig(out/'capacity_2_vs_4mwh.png',dpi=180)
    plt.close(fig)
    return ['joint_dispatch_4mwh.png','capacity_2_vs_4mwh.png']


def create_report(cfg,candidate,summary,verification,solutions,p0,hours,out):
    out = Path(out)
    generated_figures = figures(cfg,candidate,solutions,p0,hours,out)
    prior = read(ROOT/'outputs/ess_tes_pcc_scheduling_v1/summary.json')
    old_lookup = {(r['case_id'],r['strategy_id']):r for r in prior['results']}
    new_lookup = {(r['case_id'],r['strategy_id']):r for r in summary['results']}
    states = {'optimal':'준수 가능','infeasible':'운전 불가능','feasible_time_limit':'준수 가능·최적성 미확인'}
    matrix,compare,fine = [],[],[]
    for scenario in cfg['scenarios']:
        row = {'상한 조건':scenario['label_ko']}
        for strategy in cfg['strategies']:
            r = new_lookup[(scenario['id'],strategy['id'])]
            row[strategy['label_ko']] = states.get(r['status'],r['status'])
        matrix.append(row)
    for r in summary['results']:
        old = old_lookup[(r['case_id'],r['strategy_id'])]
        compare.append({'조건':r['case_label_ko'],'운영':r['strategy_label_ko'],
                        '2 MWh 판정':states.get(old['status'],old['status']),
                        '4 MWh 판정':states.get(r['status'],r['status']),
                        '2 MWh ESS 방전 (MWh)':old.get('ess_discharge_mwh'),
                        '4 MWh ESS 방전 (MWh)':r.get('ess_discharge_mwh'),
                        '2 MWh TES 방전 (MWhth)':old.get('tes_discharge_mwh_th'),
                        '4 MWh TES 방전 (MWhth)':r.get('tes_discharge_mwh_th')})
        if r['feasible']:
            d=r['held_10s']
            fine.append({'조건':r['case_label_ko'],'운영':r['strategy_label_ko'],
                         '5분 상한 초과시간 (h)':r['operating_metrics']['cap_excess_duration_h'],
                         '10초 대입 최대 초과 (MW)':d['cap_excess_max_mw'],
                         '10초 대입 초과시간 (h)':d['cap_excess_duration_h']})
    picked = []
    for label,lookup in [('2 MWh',old_lookup),('4 MWh',new_lookup)]:
        r=lookup[('constant_25_5','joint')]
        if r['feasible']:
            m=r['kernel_metrics']
            picked.append({'ESS 용량':label,'최대 PCC (MW)':m['pcc_peak_mw'],
                           'ESS 방전량 (MWh)':r['ess_discharge_mwh'],'TES 방전량 (MWhth)':r['tes_discharge_mwh_th'],
                           '미사용 대비 전력량 증가 (MWh)':m['pcc_energy_increment_mwh'],
                           'ESS 시작/종료 E (MWh)':f"{m['ess_initial_energy_mwh']:.1f} / {m['ess_terminal_energy_mwh']:.1f}",
                           'TES 시작/종료 E (MWhth)':f"{m['tes_initial_energy_mwh_th']:.1f} / {m['tes_terminal_energy_mwh_th']:.1f}"})
    ess = cfg['ess']
    cap = ess['energy_capacity_mwh']
    usable = cap*(ess['max_soc']-ess['min_soc'])
    duration = usable*ess['discharge_efficiency']/ess['discharge_limit_mw']
    initial_duration = cap*(cfg['initial_soc']-ess['min_soc'])*ess['discharge_efficiency']/ess['discharge_limit_mw']
    stressold=old_lookup[('energy_stress','ess_only')]
    stressnew=new_lookup[('energy_stress','ess_only')]
    provenance = [{'항목':r['parameter'],'값':r['value'],'구분':r['classification'],'설명':r['basis']} for r in cfg['parameter_provenance']]
    refs=''.join(f'<li><a href="{html.escape(r["url"])}">{html.escape(r["title"])}</a>: {html.escape(r["used"])}</li>' for r in cfg['references'])
    imgs=''.join(f'<img src="{name}" alt="ESS four MWh dispatch and capacity comparison">' for name in generated_figures)
    tests=verification.get('unit_tests',{})
    text=f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ESS 1 MW / 4 MWh 공동 운영</title><style>
body{{font-family:"Malgun Gothic",Arial,sans-serif;max-width:1240px;margin:32px auto;padding:0 24px;color:#183f50;line-height:1.8}}h1{{font-size:28px}}h2{{font-size:22px;margin-top:32px;border-bottom:1px solid #d1e0e5}}.note{{padding:18px;background:#fff2db;border-left:4px solid #c18a35}}.eq{{padding:16px;background:#eef6f8;font-family:Cambria,serif}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:10px;border-bottom:1px solid #d1dfe5;text-align:left;vertical-align:top}}th{{background:#eef6f8}}img{{width:100%;height:auto}}a{{color:#08777d}}</style></head><body>
<h1>ESS 1 MW / 4 MWh 적용 및 ESS–TES 공동 운영 재검토</h1>
<p class="note"><b>이번 변경:</b> ESS 최대출력은 1 MW로 유지하고 명목 저장용량만 2→4 MWh로 늘렸다. 명목 출력/저장용량 비율은 0.5→0.25 h⁻¹이며, AC 포트 기준 비율을 엄밀한 셀 C-rate와 동일시하지 않는다. 기존 2 MWh 결과는 그대로 보존했다.</p>
<h2>1. 변경 조건과 유지 조건</h2><ul><li>ESS 1 MW / 4 MWh, SOC10–90%, 왕복효율85%(ηch=ηdis≈0.921954), 초기·종료SOC50%.</li><li>에너지 범위0.4–3.6 MWh, 초기·종료에너지2 MWh. 기존2 MWh 사례의0.2–1.8 MWh, 초기·종료1 MWh와 구분.</li><li>TES10 MWhth/3 MWth, COP5 및 기존 IT·냉각 입력·PCC 손실 경계는 유지.</li><li>동일한 약{summary['duration_h']:.3f}시간의 수요·5분간격·PCC 상한·수전 전력량 최소화 목적함수 적용.</li><li>초기SOC 비율은 같지만 초기 저장에너지의 절대량은 달라진다. 각 사례는 자기 초기량까지 실제 충전으로 복원하며, 같은 초기 MWh를 둔 비교나 같은 투자비 비교는 아니다.</li></ul>
<h2>2. 4시간형의 의미와 문헌 근거</h2><p>ATB2024의 상업용 ESS 모델은 1–8시간의 저장시간을 다루고4시간형 기준 사례를 사용한다. 이번1 MW AC/4 MWh는 이 범위를 참고해 사용자가 선택한 연구용 후보이며, 특정제품·최소용량·최적투자 사양이 아니다.</p>
<p>명목 E/P=4시간이지만 실제 사용 가능한 저장량은 <b>{usable:.1f} MWh</b>이다. 최대SOC90%에서10%까지1 MW AC로 방전하면 본 모델의 효율 기준 약<b>{duration:.3f}시간</b>, 초기SOC50%에서는 사전충전 없이 약<b>{initial_duration:.3f}시간</b>이다. 따라서4시간 연속 정격출력을 보장한다고 표현하지 않는다.</p>{table(provenance)}
<h2>3. 동일 PCC 상한에서의 결과</h2>{table(matrix)}
<p>가능은 충전·방전을 모두 포함한 전 구간 상한 준수와 종료 저장량 복원을 만족하는 운전계획이 존재한다는 뜻이다. 불가능한 경우 수요를 삭제하거나 SOC를 덮어쓰지 않았다.</p>
<h2>4. 저장용량 증가로 달라진 점</h2><p>25.5 MW 공동운영에서 두 저장장치의 역할과 전력량을 비교한다. ESS 방전량은 전체 기간 누적량이므로 정격 저장용량보다 클 수 있다.</p>{table(picked)}{imgs}
<p>최대출력이 여전히1 MW이므로 ESS단독의 피크 시각 하한은 약25.553 MW(26.553−1)다. 따라서4 MWh로 늘려도25.5 MW 상한의 <b>출력 부족</b>은 해결되지 않는다. 저장용량 증가는 지속시간·에너지 여유를 늘리며, 출력 한계를 높이지 않는다.</p>
<p>별도5시간 연속감축 진단에서 ESS단독 판정은 2 MWh: <b>{states.get(stressold['status'],stressold['status'])}</b>, 4 MWh: <b>{states.get(stressnew['status'],stressnew['status'])}</b>이다. 경과36–41 h에 P_limit[k]=P0[k]−약0.428 MW로 두어 총약2.14 MWh AC 감축을 요구했다. 이는 저장시간 차이를 보기 위한 합성신호이며 실제DSO 허용범위가 아니다.</p>
<p>공동운영에 두 장치가 존재한다고 항상 동시에 방전하는 것은 아니다. 본 효율·에너지 목적함수에서 ESS가 우선 사용될 수 있으며, ESS의 저장량 증가에 따라 TES 역할이 줄어들 수 있다. 배터리 열화·투자비·요금이 없으므로 경제적 우월성을 의미하지 않는다.</p>
<h2>5. 전체 2 MWh–4 MWh 비교</h2>{table(compare)}
<h2>6. 계산식과 경계</h2><div class="eq">E_ESS[k+1]=E_ESS[k]+(ηch Pch[k]−Pdis[k]/ηdis)Δt<br>0.4≤E_ESS[k]≤3.6 MWh; E_ESS[0]=E_ESS[T]=2 MWh<br>0≤Pch[k]≤1·z[k]; 0≤Pdis[k]≤1·(1−z[k]) MW<br>P_PCC[k]=P0[k]+αch Q_TES,ch[k]−αdis Q_TES,dis[k]+Pch[k]−Pdis[k]<br>0≤P_PCC[k]≤P_limit[k]; min Σ P_PCC[k]Δt; Δt=1/12 h</div>
<ul><li>ESS는 UPS와 분리된 운영용 장치. PCC환산AC 포트에 유효 효율을 적용하며 UPS/PCS/변압기 효율을 추가 중복 적용하지 않음.</li><li>TES단독/공동에는 기존TES 대기 열손실 포함, ESS단독은TES 미설치. 용량 변경 전후 비교는 각 동일 설비 구성끼리 실시.</li><li>단위수지·SOC·동시충방전·출력·냉동기 용량·PCC 상한을 독립 재계산. TES 온도/유량의 상세 검증은 추가하지 않음.</li><li>ESS의 별도열관리 전력·자기방전·열화·비상예비 제약은 여전히 미포함. 물리적으로 손실이 없다는 뜻이 아님.</li></ul>
<h2>7. 시간해상도 및 한계</h2>{table(fine)}<p class="note">이번 준수는 <b>5분 평균 PCC 전력</b>에 대한 것이다. 같은 명령을10초 입력에 대입했을 때의 구간 내부 초과를 별도 표시했다. 순간 상한·ramp·계통접속기준·빠른 제어의 준수 결과로 해석하지 않는다. 미래 부하 전체를 알고 푸는 오프라인 기준이며 DSO조류계산, 예측오차, 반복운영, DFL 또는HIL 단계는 아니다.</p>
<h2>8. 검증·보존·재현</h2><ul><li>단위시험 통과: {tests.get('passed','검증기록 참조')}.</li><li>검증 세부항목은 <a href="verification.json">verification.json</a>에 저장. 가용해의PCC·SOC·에너지 수지 및 이전파일 해시를 확인.</li><li>기존2 MWh 보고서와 결과는 <a href="../ess_tes_pcc_scheduling_v1/report_ko.html">이전 검토보고서</a>에 보존.</li><li>상한을 만족하는 최저용량 탐색이나 최적투자 설계를 수행한 것은 아니며, 이후 운영 기준후보를4 MWh로 갱신한 단계다.</li></ul>
<h2>9. 출처와 파일</h2><ul>{refs}</ul><p><a href="summary.json">결과 요약</a> · <a href="model_contract.json">설정·근거</a> · <a href="schedules_5min.npz">운전계획</a></p></body></html>'''
    (out/'report_ko.html').write_text(text,encoding='utf-8')
    return generated_figures
