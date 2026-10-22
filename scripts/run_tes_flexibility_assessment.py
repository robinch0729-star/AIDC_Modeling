"""Assess the existing TES candidate without altering IT or previous outputs."""
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from aidc_facility.tes_flexibility import (duration_limits, evaluate_cycle,
                                         marginal_pcc_coefficients, pcc_from_thermal)
from aidc_facility.tes_operating import energy_step

OUT = ROOT / 'outputs/tes_flexibility_assessment_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def compact(result):
    return {k: v for k, v in result.items() if k != 'rows'}


def chart(envelope, episode, c):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(10, 5.6), layout='constrained')
    colors = ['#b46934', '#167d8d', '#638b49', '#154f83']
    for soc, color in zip((.25, .5, .75, 1.), colors):
        rows = [r for r in envelope if r['initial_soc'] == soc]
        rates = np.linspace(1., 3., 200)
        coeff = rows[0]['pcc_reduction_mw']/rows[0]['thermal_discharge_mw_th']
        continuous = [duration_limits(soc*c['tes']['energy_capacity_mwh_th'], float(q),
                                       c['step_s']/3600, c['tes'])['continuous_duration_h']*60
                      for q in rates]
        ax.plot(rates*coeff, continuous, color=color, linestyle='--', alpha=.7)
        ax.plot([r['pcc_reduction_mw'] for r in rows], [r['full_duration_min'] for r in rows],
                marker='o', linestyle='none', color=color, label=f'Initial SOC {soc:.0%}')
    ax.set(xlabel='PCC demand reduction (MW)', ylabel='Constant-output duration (min)',
           title='TES power-duration capability under the adopted model', ylim=(0, 520))
    ax.grid(alpha=.2)
    ax.legend(ncols=2, frameon=False)
    fig.suptitle('10 MWh(th) / 3 MW(th), COP 5; points: full 5-min bins, dashed: continuous energy limit', fontsize=10)
    fig.savefig(OUT/'tes_power_duration.png', dpi=180)
    plt.close(fig)

    rr = episode['rows']
    h = c['step_s']/3600
    edges = np.arange(len(rr)+1)*h
    delta = [r['delta_pcc_mw'] for r in rr]
    capacity = c['tes']['energy_capacity_mwh_th']
    soc = [rr[0]['energy_start_mwh_th']/capacity*100] + [r['energy_end_mwh_th']/capacity*100 for r in rr]
    idle_soc = [rr[0]['baseline_energy_start_mwh_th']/capacity*100] + [r['baseline_energy_end_mwh_th']/capacity*100 for r in rr]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout='constrained', sharex=True)
    axes[0].stairs(delta, edges, color='#155b88', linewidth=2)
    axes[0].axhline(0, color='#7e8c92', linewidth=.7)
    axes[0].set(ylabel='PCC difference (MW)', title='Dispatch minus installed-idle baseline (same initial and final SOC)')
    axes[1].plot(edges, soc, label='Discharge and recovery', color='#155b88', linewidth=2)
    axes[1].plot(edges, idle_soc, label='Idle and terminal restoration', color='#b46934', linestyle='--')
    axes[1].axhline(10, color='#64834d', linestyle=':', label='Minimum SOC')
    axes[1].set(xlabel='Elapsed time from activation (h)', ylabel='Stored cooling energy SOC (%)', ylim=(0, 105))
    axes[1].legend(frameon=False, ncols=1, loc='upper center', bbox_to_anchor=(.5, -.18), fontsize=10)
    for ax in axes:
        ax.axvline(episode['full_discharge_duration_h'], color='#9da8ad', linestyle='--')
        ax.grid(alpha=.2)
    fig.suptitle('Full-SOC example: 3 MW(th) discharge followed by immediate recovery', fontsize=14)
    fig.savefig(OUT/'tes_recovery_cycle.png', dpi=180)
    plt.close(fig)


def table(rows):
    if not rows:
        return ''
    keys = list(rows[0])
    def fmt(v):
        return f'{v:.3f}' if isinstance(v, float) else html.escape(str(v))
    return '<div class="scroll"><table><tr>'+''.join(f'<th>{html.escape(k)}</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join(f'<td>{fmt(r[k])}</td>' for k in keys)+'</tr>' for r in rows)+'</table></div>'


def report(s, c, v):
    rated = [{'초기 SOC (%)': int(r['initial_soc']*100), 'PCC 감소 (MW)': r['mean_pcc_reduction_mw'],
              '연속시간 한계 (분)': r['limits']['continuous_duration_h']*60,
              '5분 정격 유지 (분)': r['full_discharge_duration_h']*60,
              '초기 SOC 복원 (분)': r['recovery_duration_h']*60,
              '복원 중 최대 PCC 증가 (MW)': r['maximum_recovery_pcc_increase_mw'],
              '복원 포함 순 전력량 증가 (MWh)': r['net_electricity_increase_mwh']}
             for r in s['representative_cases'] if r['discharge_rate_mw_th'] == 3.]
    scans = s['start_time_scan']
    refs = ''.join(f'<li><a href="{r["url"]}">{html.escape(r["title"])}</a>: {html.escape(r["used"])}</li>' for r in c['references'])
    text = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TES 전력 조정량·지속시간·복원 평가</title>
<style>body{{font-family:"Malgun Gothic",Arial,sans-serif;max-width:1180px;margin:30px auto;padding:0 22px;color:#193c50;line-height:1.75}}h1{{font-size:28px}}h2{{font-size:22px;margin-top:30px;border-bottom:1px solid #d3dfe5}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid #d8e3e8;text-align:left}}th{{background:#edf5f7}}.scroll{{overflow:auto}}.note{{padding:16px;background:#fff4dc;border-left:4px solid #bc8736}}.eq{{padding:15px;background:#eef6f6;font-family:Cambria,serif}}img{{max-width:100%;height:auto}}a{{color:#006d77}}</style></head><body>
<h1>TES 가용 유연성: 전력 감소량·지속시간·충전 복원</h1>
<p class="note">기존 문헌 기반 TES 후보의 조건부 운영 능력을 계산했다. 실제 설비 시험, 공급수온 적합성, 최적 스케줄, DSO 운전 허용범위(DOE)의 확정 결과는 아니다. 기존 IT·PCC 결과와 사양은 수정하지 않았다.</p>
<h2>1. 평가 목적과 기준</h2><ul><li>기존 시간표의 특정 방전 결과를 넘어, 초기 SOC와 방전출력에 따라 얼마의 PCC 전력을 얼마나 오래 줄일 수 있는지 평가.</li><li>TES 10 MWhth, 충·방전 각각 최대 3 MWth, 효율 각각 90%, SOC 10–100%, 용량 기준 대기손실 0.04%/h, TES 펌프계수 0.01 MWe/MWth.</li><li>냉동기 26 MWth, COP 5, 기존 IT·팬·냉수펌프·열부하 입력 고정. 정격 TES 열출력을 적절한 냉수 운전조건에서 공급할 수 있다고 가정.</li><li>SOC 25·50·75·100%, 열 방전 1·2·3 MWth를 평가. 여기서 SOC는 각 호출 시점의 가정 상태이며, 실제 연속 시간표에서 모든 시각에 해당 SOC가 확보된다는 뜻은 아님.</li></ul>
<h2>2. 전력 감소량과 지속시간</h2><div class="eq">L = σ Emax = 0.004 MWth<br>tcontinuous = (E0 − Emin)/(Qdis/ηdis + L)<br>tfull = floor(tcontinuous/Δt) Δt, Δt = 5/60 h<br>Qch ≤ min(3, (26 − Qbase)/(1 + kp)); Qdis ≤ min(3, Qbase/(1 − kp))</div>
<p>연속시간 한계와 온전한 5분 구간의 같은 출력 유지시간을 구분했다. 마지막 남은 분량을 저출력으로 늘려 최대출력 지속시간으로 계산하지 않는다. 저장량 여유가 있어도 냉각수요나 냉동기 여유가 부족하면 해당 요청을 그대로 수행할 수 없다.</p>
<img src="tes_power_duration.png" alt="PCC power reduction and full five-minute duration by initial SOC">
<p>고정 COP와 선형 보조설비 조건에서 열 방전 1·2·3 MWth는 PCC 전력 약 {s['marginal_coefficients']['reduction_mwe_per_mwth']:.3f}·{2*s['marginal_coefficients']['reduction_mwe_per_mwth']:.3f}·{3*s['marginal_coefficients']['reduction_mwe_per_mwth']:.3f} MW 감소에 대응한다. 냉동기 전력만의 Q/COP에 TES 펌프, 응축수 펌프, 냉각탑, 변압기 손실 변화까지 포함한 값이다. 일반적인 TES 고정 변환계수로 해석하지 않는다.</p>
<h2>3. 최대 열출력 3 MWth의 결과와 복원</h2>{table(rated)}
<p>표는 시뮬레이션 시작시각 0 h에서 각각 독립적으로 호출한 결과다. 복원은 방전 직후 시작하며 냉동기 여유 내에서 최대 3 MWth로 충전한다. 완료시각은 5분 격자로 올림하고 마지막 구간의 출력을 낮춰 초기 저장량에 정확히 도달한다. SOC 100%에서의 호출은 사전 충전된 상태를 조건으로 한다. 방전 전 충전시간을 없다고 가정한 상시 서비스 보장은 아니다.</p>
<img src="tes_recovery_cycle.png" alt="Full-SOC discharge followed by paid recovery and matched baseline">
<h2>4. 복원 전력을 포함한 공정한 비교</h2><ul><li>동일한 설비·입력자료·평가구간·초기 SOC를 사용. 요청 시나리오는 방전 후 즉시 초기 SOC로 복원.</li><li>비교 기준은 같은 TES를 대기시킨 후 마지막 구간에 초기 SOC를 실제 충전으로 복원. 필요하면 최소 SOC 유지충전도 포함. 상태값을 직접 덮어쓰지 않음.</li><li>방전시간 전력 감소, 복원시간 추가 전력, 복원까지 포함한 순 전력량 차이를 분리. 방전시간 감소량을 에너지 절감이나 전 기간 피크 감소로 해석하지 않음.</li><li>이전 약 5일 시간표 결과의 순 전력량 증가와 본 표의 단일 호출 결과는 평가기간이 다르므로 직접 개선율을 계산하지 않음.</li></ul>
<h2>5. 시간대별 여유용량 및 제한상황 점검</h2><ul><li>기존 {s['baseline_duration_h']:.3f}시간 자료에서 1시간 간격 호출시각과 12개 SOC·출력 조합을 점검: 전체 {scans['attempted']}회, 방전·복원 완료 {scans['completed']}회, 자료 끝 도달 {scans['trace_exhausted']}회.</li><li>완료 사례에서 동일 출력 지속 및 상태·용량 검사 통과: {scans['all_complete_checks_passed']}. 자료 끝에 도달하면 미완료로 기록하며 자료를 순환 연결하지 않음.</li><li>전체 입력에서 최대 3 MWth 충전을 위한 최소 냉동기 여유(5분): {s['headroom']['minimum_margin_at_full_charge_5min_mw_th']:.3f} MWth. 기존 10초 입력에 같은 명령을 대입한 최소 여유: {s['headroom']['minimum_margin_at_full_charge_10s_mw_th']:.3f} MWth.</li><li>현재 부하 조건에서는 저장량이 주 지속시간 제한요인. 이 사실은 다른 냉각수요·설비규모에도 자동 적용되지 않음.</li><li>별도 가상 스트레스 조건에서 냉동기 여유 부족·냉각수요 부족·저장량 부족으로 출력이나 지속시간이 제한됨을 확인: {v['stress_checks_passed']}.</li></ul>
<h2>6. 연구 활용 범위와 후속 과제</h2><ul><li>향후 AIDC가 DSO에 제공할 정보의 후보: 현재 SOC, PCC 전력 감소 가능량, 유지 가능한 구간 수, 복원시간과 추가 전력 요구.</li><li>본 곡선은 초기상태별 단일 호출 능력이다. 연속·중첩 호출, 향후 수요 불확실성, 네트워크 제약 및 복원시간 수전 한도까지 포함한 다시간 최적화 가능영역은 후속 작업.</li><li>냉동기 용량에 맞는 충전이라도 배전계통이나 DOE의 수전 상한을 만족한다는 의미는 아님. 따라서 복원을 포함하여 PCC 허용범위 내에서 스케줄링해야 함.</li><li>ESS 동작·DSO 협조운영·DFL은 이번에 구현하지 않았으며, 탱크 치수·냉수온도 모델도 추가하지 않음.</li></ul>
<h2>7. 검증 및 재현</h2><ul><li>기존 파일 해시 보존: {v['protected_files_unchanged']} ({v['protected_file_count']}개).</li><li>기존 TES 계산과 새 유연성 단위시험 통과: {v['unit_tests']['passed']}.</li><li>기존 TES 미사용 PCC 재현 최대 오차: {v['baseline_max_error_mw']:.3g} MW.</li><li>수치 검사는 물리 설비 검증이나 빠른 열제어 검증과 구분.</li></ul><p><a href="summary.json">요약 수치</a> · <a href="scenario_results.json">호출별 결과</a> · <a href="model_contract.json">사양과 평가범위</a> · <a href="verification.json">검증 기록</a></p>
<h2>8. 기존 후보에서 계승한 출처</h2><p>이번 단계에서 사양을 새로 선정하지 않고 기존 문헌 기반 후보의 출처와 가정을 계승했다.</p><ul>{refs}</ul></body></html>'''
    (OUT/'report_ko.html').write_text(text, encoding='utf-8')


def main():
    cpath = ROOT/'configs/pcc_tes_operating_candidate_v1.json'
    c = read(cpath)
    legacy = read(ROOT/c['legacy_config'])
    protected = {cpath, ROOT/c['legacy_config'], ROOT/'src/aidc_facility/tes_operating.py',
                 ROOT/'src/aidc_facility/pcc_cooling_tes.py', ROOT/'tests/test_tes_operating.py'}
    for name in ('pcc_tes_operating_candidate_v1', 'pcc_cooling_tes_candidate_v1',
                 'it_pcc_boundary_audit_v1', 'pcc_capacity_cop_review_v1', 'pcc_temperature_conditions_v1'):
        protected.update(p for p in (ROOT/'outputs'/name).rglob('*') if p.is_file())
    before = {str(p.relative_to(ROOT)): sha(p) for p in sorted(protected)}
    cols = read(ROOT/c['baseline_columns'])
    with np.load(ROOT/c['baseline_profiles']) as raw:
        original = raw['no_tes'].copy()
    block = c['step_s']//legacy['internal_step_s']
    if c['step_s'] % legacy['internal_step_s'] or len(original) % block:
        raise ValueError('Need complete aggregate bins')
    base = original.reshape(-1, block, len(cols)).mean(axis=1)
    rows = [dict(zip(cols, map(float, r))) for r in base]
    dt = c['step_s']/3600
    tes = c['tes']
    coeff = marginal_pcc_coefficients(legacy, 5., tes['pump_mw_per_mw_th'])
    OUT.mkdir(parents=True, exist_ok=True)
    envelope, representative, scan, profiles = [], [], [], {}
    selected = None
    for soc in (.25, .5, .75, 1.):
        for q in (1., 2., 3.):
            limits = duration_limits(soc*tes['energy_capacity_mwh_th'], q, dt, tes)
            envelope.append({'initial_soc': soc, 'thermal_discharge_mw_th': q,
                             'pcc_reduction_mw': coeff['reduction_mwe_per_mwth']*q,
                             'continuous_duration_min': limits['continuous_duration_h']*60,
                             'full_duration_min': limits['full_interval_duration_h']*60})
            for start in range(0, len(rows), int(round(1/dt))):
                case = evaluate_cycle(rows, start, soc, q, tes, legacy, dt)
                scan.append(compact(case))
                if start == 0:
                    representative.append(compact(case))
                    profiles[f'soc{int(soc*100)}_q{int(q)}'] = case['rows']
                    if soc == 1. and q == 3.:
                        selected = case
    successful = [s for s in scan if s['completed']]
    errors = [s for s in scan if s['status'] not in ('complete', 'trace_exhausted')]
    all_checks = all(all(s['checks'].values()) for s in successful) and not errors
    no_error = max(abs(pcc_from_thermal(r, 0., 0., legacy, 5., tes['pump_mw_per_mw_th'])['pcc_mw']-r['pcc_mw']) for r in rows)
    capacity = legacy['plant']['chiller_capacity_mw_th']
    qbase5 = base[:, cols.index('chiller_mw_th')]
    qbase10 = original[:, cols.index('chiller_mw_th')]
    k = tes['pump_mw_per_mw_th']
    headroom = {'minimum_margin_at_full_charge_5min_mw_th': float(capacity-(qbase5+3*(1+k)).max()),
                'minimum_margin_at_full_charge_10s_mw_th': float(capacity-(qbase10+3*(1+k)).max()),
                'minimum_chiller_output_at_full_discharge_10s_mw_th': float((qbase10-3*(1-k)).min()),
                'note': 'Counterfactual rated-command plant capacity check; storage readiness is separate. Not fast thermal validation.'}
    limited_charge = energy_step(5., -3., dt, 25.5, capacity, tes)
    limited_discharge = energy_step(5., 3., dt, 1., capacity, tes)
    limited_energy = energy_step(1.01, 3., dt, 18., capacity, tes)
    stress = {'charge_headroom': limited_charge, 'low_cooling_demand': limited_discharge, 'low_energy': limited_energy}
    stress_ok = (abs(limited_charge['charge_mw_th']-.5/(1+k)) < 1e-9 and
                 abs(limited_discharge['discharge_mw_th']-1/(1-k)) < 1e-9 and
                 0 < limited_energy['discharge_mw_th'] < 3 and
                 all(not x['state_violation'] for x in stress.values()))
    tests = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-m', 'unittest', 'discover',
                            '-s', 'tests', '-p', 'test_*tes*.py', '-v'], cwd=ROOT,
                           capture_output=True, text=True, encoding='utf-8')
    after = {str(p.relative_to(ROOT)): sha(p) for p in sorted(protected)}
    v = {'protected_file_count': len(before), 'protected_files_unchanged': before == after,
         'protected_sha256': before, 'baseline_max_error_mw': no_error,
         'all_complete_case_checks_passed': all_checks, 'stress_checks_passed': stress_ok,
         'unit_tests': {'passed': tests.returncode == 0, 'output': tests.stdout+tests.stderr}}
    s = {'created_utc': datetime.now(timezone.utc).isoformat(),
         'status': 'conditional_energy_operation_capability_not_validated_grid_service',
         'baseline_duration_h': len(base)*dt, 'marginal_coefficients': coeff, 'headroom': headroom,
         'envelope': envelope, 'representative_cases': representative,
         'start_time_scan': {'attempted': len(scan), 'completed': len(successful),
                             'trace_exhausted': sum(r['status'] == 'trace_exhausted' for r in scan),
                             'other_errors': len(errors), 'all_complete_checks_passed': all_checks},
         'stress_cases': stress}
    contract = {'candidate_source': str(cpath.relative_to(ROOT)), 'candidate': c,
                'evaluation': {'initial_soc': [.25, .5, .75, 1.], 'discharge_mw_th': [1, 2, 3],
                    'start_interval_h': 1, 'interval_h': dt, 'trace_wrapping': False,
                    'recovery': 'Immediately after full-bin discharge, restore initial energy via constrained charging.',
                    'baseline': 'Same installed TES, same initial energy and horizon; idle then actual terminal restoration.',
                    'duration': 'Whole 5min intervals at requested rate; fractional tail excluded.',
                    'exclusions': ['DSO/network feasibility', 'new water-temperature validation', 'ESS', 'optimized dispatch', 'forecast uncertainty'],
                    'assumption': 'Each activation starts at specified SOC independently; prior charging is not guaranteed.'}}
    dump(OUT/'model_contract.json', contract)
    dump(OUT/'summary.json', s)
    dump(OUT/'scenario_results.json', scan)
    dump(OUT/'verification.json', v)
    dump(OUT/'representative_profiles.json', profiles)
    chart(envelope, selected, c)
    report(s, c, v)
    print(json.dumps({'report': str(OUT/'report_ko.html'), 'scan': s['start_time_scan'],
                      'rated_cases': [r for r in representative if r['discharge_rate_mw_th'] == 3.],
                      'headroom': headroom, 'tests_passed': tests.returncode == 0,
                      'protected_unchanged': before == after}, ensure_ascii=False, indent=2))
    if not (before == after and no_error < 1e-9 and all_checks and stress_ok and tests.returncode == 0):
        raise AssertionError('Verification failed; inspect report and verification.json')


if __name__ == '__main__':
    main()
