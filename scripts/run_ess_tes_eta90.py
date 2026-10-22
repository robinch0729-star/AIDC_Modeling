"""Efficiency-only comparison against the preserved 1 MW / 4 MWh v2 candidate."""
from datetime import datetime, timezone
import html
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from aidc_facility.tes_flexibility import marginal_pcc_coefficients, pcc_from_thermal
from run_ess_tes_pcc_scheduling import solve
from run_ess_tes_4mwh import make_cap
from run_tes_pcc_scheduling import read, dump, sha, cap_metrics, table

CONFIG = ROOT/'configs/ess_tes_pcc_scheduling_v3_eta90.json'
OUT = ROOT/'outputs/ess_tes_pcc_scheduling_v3_eta90'


def report(cfg, summary, verification):
    matrix = []
    for case in cfg['scenarios']:
        row = {'상한 조건': case['label_ko']}
        for st in cfg['strategies']:
            item = next(r for r in summary['results'] if r['case_id']==case['id'] and r['strategy_id']==st['id'])
            row[st['label_ko']] = {'optimal':'준수 가능', 'infeasible':'불가능'}.get(item['status'], item['status'])
        matrix.append(row)
    representative = next(r for r in summary['efficiency_comparison'] if r['case_id']=='constant_25_5' and r['strategy_id']=='joint')
    comparison = []
    for field, label in [('pcc_energy_mwh','전체 PCC 전력량 (MWh)'), ('ess_discharge_energy_mwh','ESS 누적 방전량 (MWh AC)'), ('tes_discharge_energy_mwh_th','TES 누적 방전량 (MWhth)'), ('pcc_peak_mw','최대 PCC (MW)')]:
        comparison.append({'지표':label, '기존 각92.195%':representative['old_'+field], '변경 각90%':representative['new_'+field]})
    changed = [r for r in summary['efficiency_comparison'] if r['old_status']!=r['new_status']]
    note = '해당 7개 상한·3개 운영구성의 가능/불가능 판정은 기존과 동일하다.' if not changed else '일부 가능/불가능 판정이 달라졌다. summary.json 비교표를 확인한다.'
    refs = ''.join(f'<li><a href="{html.escape(r["url"])}">{html.escape(r["title"])}</a>: {html.escape(r["used"])}</li>' for r in cfg['references'])
    document = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>ESS 충방전 효율 90% 적용</title><style>body{{font-family:Malgun Gothic,Arial,sans-serif;max-width:1120px;margin:30px auto;padding:0 24px;line-height:1.8;color:#183c4b}}h2{{border-bottom:1px solid #cadde3;margin-top:30px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #d7e4e9}}th{{background:#edf6f8}}.note{{background:#fff4e5;padding:16px;border-left:4px solid #ba8340}}.eq{{background:#edf6f8;padding:16px}}.scroll{{overflow-x:auto}}a{{color:#087887}}</style>
<h1>ESS 효율 설정 변경: 충전 90%·방전 90%</h1>
<p class="note">사용자 선택에 따른 연구용 효율 가정이다. <b>각 90%이므로 왕복효율은 81%</b>이며, 기존 ATB 기반 85%와 다르다. 보편적인 제품 효율이나 실측값으로 주장하지 않는다. TES는 변경하지 않았다.</p>
<h2>1. 변경 및 유지 조건</h2><ul><li>ESS: 1 MW / 4 MWh, 충전·방전 효율 각각0.9, SOC10–90%, 초기·종료2MWh(50%).</li><li>기존 각√0.85≈0.921954에서 두 효율만 변경. 같은 부하·7개 상한·119.833시간·5분 간격으로 재최적화했다.</li><li>TES: 기존10MWhth/3MWth, 각0.9, 대기손실0.004MWth, 펌프계수0.01, 초기·종료5MWhth를 유지했다. TES효율은 PL30의 특정 문헌 사례값이지 보편값이 아니다.</li><li>기존 후보 설정에 들어 있는 고정 충방전 시간표는 이번 최적화에서 사용하지 않는다. 각 저장장치 운전량을 새로 결정한다.</li></ul>
<h2>2. 에너지 수지와 해석</h2><div class="eq">E[k+1] = E[k] + (0.9P_ch[k] − P_dis[k]/0.9)Δt<br>0.4 ≤ E[k] ≤ 3.6 MWh; E[0]=E[T]=2 MWh; 0≤P_ch,P_dis≤1MW<br>왕복효율=0.9×0.9=0.81; Δt=1/12h</div><p>허용SOC90→10%에서 가용 내부에너지는3.2MWh이며, AC인도량은2.88MWh다. 1MW정격방전은 최대2.88시간이며, 시작SOC50%에서 사전충전이 없으면1.44시간이다. 열화·대기전력은 추가하지 않았다. ESS효율은 PCC환산 AC경계의 유효경로효율이므로 PCS/UPS/변압기 손실을 중복 적용하지 않는다.</p>
<h2>3. PCC 상한 준수 결과</h2>{table(matrix)}<p>{note}</p><p>5시간 연속감축은 경과36–41시간에 P_limit=P0−약0.428MW를 적용한 합성진단이며 실제DSO지시가 아니다.</p>
<h2>4. 대표25.5MW 상한 공동운영 비교</h2>{table(comparison)}<p>낮아진 ESS효율은 재충전에 필요한 전력량과 ESS–TES분담에 영향을 준다. 전체 부하패턴·냉각설비·PCC상한·저장용량은 같다. 비용·열화·투자 최적화 결과가 아니며, 일정한 효율 아래 충전시각은 비유일할 수 있다.</p>
<h2>5. 검증과 한계</h2><ul><li>관련 단위시험 {verification['unit_tests']['count']}개 통과: {verification['unit_tests']['passed']}.</li><li>가용해 독립 에너지수지·상태·PCC상한·동시충방전·냉동기용량·종료저장량 검사: {verification['all_feasible_verified']}.</li><li>이전 TES단독 결과 유지: {verification['tes_only_unchanged']}; 이전보호파일 {verification['protected_file_count']}개 변경없음: {verification['protected_files_unchanged']}.</li><li>미래부하를 아는 오프라인 에너지 최소화이며, DSO계통해석·DFL·예측오차·실시간제어·HIL은 미포함이다.</li><li>5분 평균상한 준수는 순간상한 준수가 아니다. 같은명령을10초입력에 대입한 구간내 진단을 결과JSON에 저장했으며, 새로운 열동역학 검증은 아니다.</li></ul>
<h2>6. 출처 및 재현파일</h2><ul>{refs}</ul><p><a href="../pcc_tes_operating_candidate_v1/report_ko.html">변경하지 않은 TES문헌근거</a> · <a href="../ess_tes_pcc_scheduling_v2_4mwh/report_ko.html">이전85%왕복효율 결과</a></p><p><a href="summary.json">요약·이전결과비교</a> · <a href="model_contract.json">계산경계·설정·출처</a> · <a href="verification.json">검증기록</a> · <a href="schedules_5min.npz">새운전계획</a></p></html>'''
    (OUT/'report_ko.html').write_text(document, encoding='utf-8')


def main():
    cfg = read(CONFIG)
    prev_cfg = read(ROOT/'configs/ess_tes_pcc_scheduling_v2_4mwh.json')
    previous = read(ROOT/cfg['comparison_baseline'])
    candidate = read(ROOT/cfg['candidate_config'])
    legacy = read(ROOT/candidate['legacy_config'])
    tes = candidate['tes']
    changes = [k for k in cfg['ess'] if cfg['ess'][k]!=prev_cfg['ess'][k]]
    common = ('candidate_config','step_s','initial_soc','terminal_soc','cop','objective','time_limit_s','strategies','scenarios')
    assert set(changes)=={'charge_efficiency','discharge_efficiency'} and all(cfg[k]==prev_cfg[k] for k in common)
    assert cfg['ess']['charge_efficiency']==cfg['ess']['discharge_efficiency']==0.9
    protected = set()
    for folder in ('ess_tes_pcc_scheduling_v1','ess_tes_pcc_scheduling_v2_4mwh','tes_pcc_scheduling_v1','tes_flexibility_assessment_v1','pcc_tes_operating_candidate_v1','pcc_cooling_tes_candidate_v1','it_pcc_boundary_audit_v1','pcc_capacity_cop_review_v1','pcc_temperature_conditions_v1'):
        protected.update(p for p in (ROOT/'outputs'/folder).rglob('*') if p.is_file())
    for folder, pattern in [('configs','*.json'),('src/aidc_facility','*.py'),('scripts','*tes*.py'),('tests','*tes*.py')]:
        protected.update(p for p in (ROOT/folder).glob(pattern) if p!=CONFIG and p!=Path(__file__))
    before = {str(p.relative_to(ROOT)):sha(p) for p in sorted(protected)}
    cols = read(ROOT/candidate['baseline_columns'])
    raw = np.load(ROOT/candidate['baseline_profiles'])['no_tes']
    block = cfg['step_s']//legacy['internal_step_s']
    assert cfg['step_s']%legacy['internal_step_s']==0 and len(raw)%block==0
    base = raw.reshape(-1,block,len(cols)).mean(1)
    rows = [dict(zip(cols,map(float,r))) for r in base]
    dt = cfg['step_s']/3600
    hours = np.arange(len(rows))*dt
    p0 = np.array([pcc_from_thermal(r,0.,0.,legacy,cfg['cop'],tes['pump_mw_per_mw_th'])['pcc_mw'] for r in rows])
    coeff = marginal_pcc_coefficients(legacy,cfg['cop'],tes['pump_mw_per_mw_th'])
    OUT.mkdir(parents=True,exist_ok=True)
    arrays = {'time_h':hours,'no_storage_pcc_mw':p0}
    records, results, solutions, cap_errors = {}, [], {}, {}
    with np.load(ROOT/'outputs/ess_tes_pcc_scheduling_v2_4mwh/schedules_5min.npz') as old_arrays:
        for case in cfg['scenarios']:
            cap = make_cap(case,p0,hours,coeff)
            cap_errors[case['id']] = float(np.max(np.abs(cap-old_arrays[case['id']+'_cap_mw'])))
            assert cap_errors[case['id']]<1e-12
            arrays[case['id']+'_cap_mw'] = cap
            for st in cfg['strategies']:
                key = case['id']+'_'+st['id']
                print('Solving '+key,flush=True)
                result = solve(rows,cap,tes,legacy,cfg,st)
                records[key+'_solution.json'] = result
                solutions[(case['id'],st['id'])] = result
                dump(OUT/(key+'_solution.json'),result)
                item = {'case_id':case['id'],'case_label_ko':case['label_ko'],'strategy_id':st['id'],'strategy_label_ko':st['label_ko'],'status':result['status'],'feasible':result['feasible'],'optimal':result['optimal'],'solver':result['solver'],'no_storage':cap_metrics(p0,cap,dt)}
                if result['feasible']:
                    s = result['schedule']
                    item.update(operating_metrics=cap_metrics(s['pcc_mw'],cap,dt),kernel_metrics=result['metrics'],ess_discharge_mwh=float(np.sum(s['ess_discharge_mw'])*dt),tes_discharge_mwh_th=float(np.sum(s['tes_discharge_mw_th'])*dt))
                    delta = coeff['increase_mwe_per_mwth']*np.asarray(s['tes_charge_mw_th'])-coeff['reduction_mwe_per_mwth']*np.asarray(s['tes_discharge_mw_th'])+np.asarray(s['ess_charge_mw'])-np.asarray(s['ess_discharge_mw'])
                    p10 = raw[:,cols.index('pcc_mw')]+np.repeat(delta,block)
                    q10 = raw[:,cols.index('chiller_mw_th')]+np.repeat((1+tes['pump_mw_per_mw_th'])*np.asarray(s['tes_charge_mw_th'])-(1-tes['pump_mw_per_mw_th'])*np.asarray(s['tes_discharge_mw_th']),block)
                    item['held_10s'] = dict(cap_metrics(p10,np.repeat(cap,block),legacy['internal_step_s']/3600),minimum_chiller_headroom_mw_th=float(legacy['plant']['chiller_capacity_mw_th']-q10.max()),minimum_chiller_output_mw_th=float(q10.min()),reaggregation_error_mw=float(np.max(np.abs(p10.reshape(-1,block).mean(1)-s['pcc_mw']))))
                    arrays.update({key+'_'+k:np.asarray(v) for k,v in s.items()})
                results.append(item)
                print(result['status'],flush=True)
    matched = []
    joint_st = next(st for st in cfg['strategies'] if st['id']=='joint')
    for cid in ('constant_26','constant_25_8','constant_25_5'):
        case = next(c for c in cfg['scenarios'] if c['id']==cid)
        idle = solve(rows,make_cap(case,p0,hours,coeff),dict(tes,discharge_limit_mw_th=0.),legacy,cfg,joint_st)
        name = cid+'_ess_with_tes_idle_solution.json'
        records[name] = idle
        dump(OUT/name,idle)
        joint = solutions[(cid,'joint')]
        matched.append({'case_id':cid,'idle_status':idle['status'],'joint_status':joint['status'],'joint_minus_idle_mwh':joint['metrics']['pcc_energy_mwh']-idle['metrics']['pcc_energy_mwh'] if joint['feasible'] and idle['feasible'] else None})
    comparison = []
    for item in results:
        old = next(r for r in previous['results'] if r['case_id']==item['case_id'] and r['strategy_id']==item['strategy_id'])
        entry = {k:item[k] for k in ('case_id','case_label_ko','strategy_id','strategy_label_ko')}
        entry.update(old_status=old['status'],new_status=item['status'])
        for field in ('pcc_energy_mwh','ess_charge_energy_mwh','ess_discharge_energy_mwh','tes_charge_energy_mwh_th','tes_discharge_energy_mwh_th','ess_minimum_soc','ess_maximum_soc','pcc_peak_mw'):
            a,b = old.get('kernel_metrics',{}).get(field),item.get('kernel_metrics',{}).get(field)
            entry.update({'old_'+field:a,'new_'+field:b,'delta_'+field:b-a if a is not None and b is not None else None})
        comparison.append(entry)
    tests = subprocess.run([sys.executable,'-X','utf8','-B','-m','unittest','discover','-s','tests','-p','test_*tes*.py','-v'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    test_text = tests.stdout+tests.stderr
    count = re.search(r'Ran (\d+) tests?',test_text)
    after = {str(p.relative_to(ROOT)):sha(p) for p in sorted(protected)}
    verified = all(all(r['checks'].values()) for r in records.values() if r['feasible'])
    # Verify enabled-device initial AND terminal states directly from saved trajectories.
    endpoints = all((not any(r['schedule']['ess_energy_mwh']) or max(abs(r['schedule']['ess_energy_mwh'][i]-2) for i in (0,-1))<1e-7) and (not any(r['schedule']['tes_energy_mwh_th']) or max(abs(r['schedule']['tes_energy_mwh_th'][i]-5) for i in (0,-1))<1e-7) for r in records.values() if r['feasible'])
    tes_same = all(r['old_status']==r['new_status'] and (r['delta_pcc_energy_mwh'] is None or abs(r['delta_pcc_energy_mwh'])<1e-6) for r in comparison if r['strategy_id']=='tes_only')
    fine = all(r['held_10s']['minimum_chiller_headroom_mw_th']>=-1e-7 and r['held_10s']['minimum_chiller_output_mw_th']>=-1e-7 and r['held_10s']['reaggregation_error_mw']<1e-8 for r in results if r['feasible'])
    verification = {'protected_file_count':len(before),'protected_files_unchanged':before==after,'protected_sha256':before,'solution_file_count':len(records),'solution_files_this_run':list(records),'all_feasible_verified':verified,'all_solver_statuses_resolved':all(r['status'] in ('optimal','infeasible','feasible_time_limit') for r in records.values()),'tes_only_unchanged':tes_same,'only_ess_efficiencies_changed':True,'changed_ess_parameters':changes,'ess_2mwh_tes_5mwhth_endpoints':endpoints,'pcc_caps_identical_to_v2':all(v<1e-12 for v in cap_errors.values()),'pcc_cap_max_differences_mw':cap_errors,'held_10s_chiller_and_reaggregation_passed':fine,'no_storage_energy_difference_from_v2_mwh':abs(previous['no_storage']['energy_mwh']-float(p0.sum()*dt)),'unit_tests':{'passed':tests.returncode==0,'count':int(count.group(1)) if count else None,'output':test_text}}
    summary = {'created_utc':datetime.now(timezone.utc).isoformat(),'duration_h':len(rows)*dt,'intervals':len(rows),'no_storage':{'mean_pcc_mw':float(p0.mean()),'peak_pcc_mw':float(p0.max()),'energy_mwh':float(p0.sum()*dt)},'coefficients':coeff,'ess_config':cfg['ess'],'results':results,'matched_hardware_comparison':matched,'efficiency_comparison':comparison,'comparison_baseline':cfg['comparison_baseline'],'scope':'Only ESS two one-way efficiencies changed from sqrt(.85) to .9. RTE81%, same1MW4MWh; TES unchanged; perfect-foresight five-minute average caps.'}
    dump(OUT/'summary.json',summary)
    dump(OUT/'verification.json',verification)
    dump(OUT/'model_contract.json',{'experiment':cfg,'inherited_tes_candidate':candidate,'inherited_fixed_dispatch_is_used':False,'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [CONFIG,Path(__file__),ROOT/cfg['candidate_config'],ROOT/candidate['legacy_config'],ROOT/'src/aidc_facility/ess_tes_scheduler.py',ROOT/'scripts/run_ess_tes_4mwh.py']}})
    np.savez_compressed(OUT/'schedules_5min.npz',**arrays)
    assert len(records)==24 and verified and endpoints and tes_same and fine and before==after and tests.returncode==0 and verification['all_solver_statuses_resolved'] and verification['no_storage_energy_difference_from_v2_mwh']<1e-8
    report(cfg,summary,verification)
    print('Verified; report: '+str(OUT/'report_ko.html'),flush=True)


if __name__=='__main__':
    main()
