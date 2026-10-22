"""Perfect-information TES scheduling with transparent, hypothetical PCC caps."""
from datetime import datetime, timezone
import hashlib
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
from aidc_facility.tes_flexibility import marginal_pcc_coefficients, pcc_from_thermal
from aidc_facility.tes_scheduler import solve_schedule

CONFIG = ROOT/'configs/tes_pcc_scheduling_v1.json'
OUT = ROOT/'outputs/tes_pcc_scheduling_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def cap_metrics(power, cap, dt):
    power, cap = np.asarray(power), np.asarray(cap)
    excess = np.maximum(power-cap, 0.)
    return {'pcc_peak_mw': float(power.max()), 'pcc_energy_mwh': float(power.sum()*dt),
            'cap_excess_max_mw': float(excess.max()),
            'cap_excess_duration_h': float(np.count_nonzero(excess > 1e-7)*dt),
            'cap_excess_integral_mwh': float(excess.sum()*dt)}


def make_cap(s, p0, hour, coeff):
    if s['type'] == 'constant':
        return np.full(len(p0), s['cap_mw'])
    cap = np.full(len(p0), s['outside_cap_mw'])
    mask = (hour >= s['start_h']) & (hour < s['end_h'])
    if s['type'] == 'time_window':
        cap[mask] = s['inside_cap_mw']
    elif s['type'] == 'energy_stress':
        cap[mask] = p0[mask]-s['thermal_discharge_required_mw_th']*coeff['reduction_mwe_per_mwth']
    else:
        raise ValueError('Unknown cap scenario')
    return cap


def plots(results, baselines, cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    h = cfg['step_s']/3600
    for key, fname, title in (
        ('constant_26_2', 'pcc_scheduling_comparison.png', 'TES scheduling with a 26.2 MW PCC import limit'),
        ('time_varying', 'time_varying_pcc_limit.png', 'TES scheduling with a time-varying illustrative PCC limit')):
        result = results[key]
        if not result['feasible']:
            continue
        s = result['schedule']
        n = len(s['pcc_mw'])
        edges = np.arange(n+1)*h
        x = edges[:-1]
        fig, ax = plt.subplots(3, 1, figsize=(12, 9), layout='constrained', sharex=True)
        ax[0].plot(x, baselines['no_tes'], color='#a2a9ad', lw=1, label='No TES')
        ax[0].plot(x, baselines['preset'], color='#c88a51', lw=.8, alpha=.65, label='Prior preset schedule')
        ax[0].plot(x, s['pcc_mw'], color='#007d8a', lw=1.2, label='Optimized TES schedule')
        ax[0].step(x, s['pcc_upper_mw'], where='post', color='#b44740', ls='--', lw=1.5, label='Illustrative 5-min limit')
        ax[0].set(ylabel='PCC power (MW)', title=title)
        ax[0].legend(frameon=False, ncols=2, fontsize=10)
        ax[1].stairs(s['charge_mw_th'], edges, color='#15639a', label='Charge (+)')
        ax[1].stairs(-np.asarray(s['discharge_mw_th']), edges, color='#c07130', label='Discharge (-)')
        ax[1].set(ylabel='TES thermal power (MWth)', ylim=(-3.3, 3.3))
        ax[1].legend(frameon=False, ncols=2, loc='upper right')
        ax[2].plot(edges, np.asarray(s['energy_mwh_th'])*10, color='#007d8a', lw=1.5)
        ax[2].axhline(10, color='#ad5f44', ls=':', label='Minimum SOC')
        ax[2].axhline(50, color='#838b90', ls='--', label='Initial / final SOC')
        ax[2].set(ylabel='SOC (%)', xlabel='Elapsed simulation time (h)', ylim=(0, 105))
        ax[2].legend(frameon=False, ncols=2, loc='upper right')
        for a in ax:
            a.grid(alpha=.2)
        fig.suptitle('Perfect-foresight offline test; charging and terminal recovery obey the same 5-minute cap', fontsize=12)
        fig.savefig(OUT/fname, dpi=180)
        plt.close(fig)


def table(rows):
    keys = list(rows[0])
    def fmt(x):
        if x is None:
            return '—'
        return f'{x:.4f}' if isinstance(x, float) else html.escape(str(x))
    return '<div class="scroll"><table><tr>'+''.join(f'<th>{html.escape(k)}</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join(f'<td>{fmt(r[k])}</td>' for k in keys)+'</tr>' for r in rows)+'</table></div>'


def report(summary, candidate, verification):
    rows = []
    for s in summary['scenarios']:
        optimal = s['optimized']
        rows.append({'조건': s['label_ko'], '결과': s['status'],
                     '기존 시간표 초과시간 (h)': s['preset']['cap_excess_duration_h'],
                     '최적화 PCC 최대 (MW)': optimal['pcc_peak_mw'] if optimal else None,
                     '최적화 초과시간 (h)': optimal['cap_excess_duration_h'] if optimal else None,
                     '대기 기준 대비 전력량 증가 (MWh)': s['increment_vs_idle_mwh'],
                     '초기/종료 E (MWhth)': '5 / 5' if s['feasible'] else '운전계획 없음'})
    diagnostic = [{'조건': s['label_ko'], '출력만 고려한 최대 불가피 초과 (MW)': s['instantaneous_lower_bound_excess_mw'],
                   '에너지 스트레스 필요량 (MWhth)': s.get('required_window_inventory_mwh_th'),
                   '최대 가용 저장량 (MWhth)': 9.0} for s in summary['scenarios'] if not s['feasible']]
    fine = [{'조건': s['label_ko'], '5분 최적화 초과 (MW)': s['optimized']['cap_excess_max_mw'],
             '10초 입력 대입 최대 초과 (MW)': s['held_10s_diagnostic']['cap_excess_max_mw'],
             '10초 입력 대입 초과시간 (h)': s['held_10s_diagnostic']['cap_excess_duration_h'],
             '10초 냉동기 최소 여유 (MWth)': s['held_10s_diagnostic']['minimum_chiller_headroom_mw_th']}
            for s in summary['scenarios'] if s['feasible']]
    m = next(s for s in summary['scenarios'] if s['id'] == 'constant_26_2')
    refs = ''.join(f'<li><a href="{r["url"]}">{html.escape(r["title"])}</a>: 기존 TES 후보에서 채택한 근거·가정 유지</li>' for r in candidate['references'])
    text = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PCC 한도 내 TES 스케줄링</title><style>
body{{font-family:"Malgun Gothic",Arial,sans-serif;max-width:1200px;margin:30px auto;padding:0 24px;color:#173c4d;line-height:1.75}}h1{{font-size:28px}}h2{{font-size:22px;margin-top:32px;border-bottom:1px solid #d4e2e6}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid #d5e1e5;text-align:left}}th{{background:#edf5f7}}.scroll{{overflow:auto}}.note{{padding:16px;background:#fff3dd;border-left:4px solid #bd8532}}.eq{{padding:16px;background:#eef5f7;font-family:Cambria,serif;font-size:17px}}img{{width:100%;height:auto}}a{{color:#006c7a}}</style></head><body>
<h1>PCC 수전 한도를 고려한 TES 운영 스케줄링</h1>
<p class="note"><b>이번 진척:</b> 주어진 PCC 상한 안에서 TES 충·방전과 종료 저장량 복원을 함께 계획하는 최적화 문제를 구현했다. 수전 상한은 연구용 가상 입력이며 DSO 조류계산으로 산정한 DOE가 아니다. 미래 부하 전체를 아는 오프라인 기준실험으로, 예측오차·롤링 제어·DFL 검증은 포함하지 않는다.</p>
<h2>1. 기존 단계와 달라진 점</h2><ul><li>기존: 특정 시간표의 동작 또는 초기 SOC별 단일 호출 능력 평가.</li><li>이번: 약 {summary['duration_h']:.3f}시간의 5분 구간 전체를 함께 계획하여, 부하가 상한을 초과하는 구간의 방전과 저장량 복원 충전 모두에 같은 PCC 상한 적용.</li><li>IT·기본 냉각수요는 고정. TES 10 MWhth/3 MWth, COP 5, 냉동기 26 MWth 및 기존 손실 조건을 유지.</li><li>초기·종료 SOC 모두 50%(5 MWhth). 종료 저장량은 수식 제약과 실제 충전으로 복원하며 상태값을 덮어쓰지 않음.</li></ul>
<h2>2. 운영 문제와 계산식</h2><div class="eq">min Σ P_PCC[k] Δt<br>P_PCC[k] = P0[k] + αch Qch[k] − αdis Qdis[k]<br>0 ≤ P_PCC[k] ≤ P_limit[k]<br>E[k+1] = E[k] + (0.9 Qch[k] − Qdis[k]/0.9 − 0.004) Δt<br>1 ≤ E[k] ≤ 10; E[0] = E[T] = 5 (MWhth)<br>0 ≤ Qch[k] ≤ 3 z[k]; 0 ≤ Qdis[k] ≤ 3(1−z[k]); z[k] ∈ {{0,1}}<br>0 ≤ Qbase[k] + 1.01 Qch[k] − 0.99 Qdis[k] ≤ 26 (MWth)</div>
<p>Δt=1/12 h, αch={summary['coefficients']['increase_mwe_per_mwth']:.8f}, αdis={summary['coefficients']['reduction_mwe_per_mwth']:.8f} MWe/MWth. 냉동기 COP, TES 펌프, 응축수 펌프, 냉각탑 및 변압기 손실 변화를 포함한다. 충·방전 동시 수행은 이진변수로 금지한다.</p>
<ul><li>목적함수는 총 수전 전력량 최소화. 요금이 없으므로 운영비 최소화라고 표현하지 않음.</li><li>상한은 강제 제약이며, 숨겨진 완화변수로 초과를 허용하지 않음. 해가 없으면 운전 불가능으로 명시.</li><li>상한 이하의 여러 충전시각이 동일한 에너지 목적값을 가질 수 있다. 그림의 충전시각은 유일한 최적해나 요금 최적 시각이 아님.</li></ul>
<h2>3. 상한 시나리오와 결과</h2>{table(rows)}
<p>상한 초과시간은 허용오차 1e-7 MW를 넘는 5분 구간의 합계다. 초과량 적분(MWh)은 상한 준수 지표이며 미공급 에너지로 해석하지 않는다. 미사용·기존 시간표는 비교용이며 상한 준수를 보장하는 기준은 아니다.</p>
<p>고정 26.2 MW의 경우 TES 미사용 최대 {m['no_tes']['pcc_peak_mw']:.4f} MW를 {m['optimized']['pcc_peak_mw']:.4f} MW로 제한했다. 기존 시간표 초과시간 {m['preset']['cap_excess_duration_h']:.3f} h에서 새 계획은 {m['optimized']['cap_excess_duration_h']:.3f} h로 감소했다. 동일 TES 비시프트 기준 대비 전체 전력량은 {m['increment_vs_idle_mwh']:.4f} MWh 증가했다.</p>
<img src="pcc_scheduling_comparison.png" alt="PCC limits, optimized TES dispatch and storage state">
<h2>4. 시간에 따라 달라지는 상한</h2><p>경과시간 48–72 h는 26.0 MW, 그 외는 27.0 MW로 둔 가상 신호도 시험했다. 실제 배전계통 혼잡이나 독일 현지 시간대에 근거한 신호가 아니다.</p><img src="time_varying_pcc_limit.png" alt="Scheduling under changing illustrative PCC limits">
<h2>5. 운전이 불가능한 경우의 구분</h2>{table(diagnostic)}
<ul><li>25.8 MW 상한: 최대 열 방전 3 MWth로도 최대 부하 시각에 약 25.911 MW의 PCC 수전이 필요. 저장량이 충분해도 출력 부족으로 불가능.</li><li>5시간 연속 에너지 스트레스: 기준 PCC보다 2 MWth 방전에 해당하는 전력을 연속 감축하도록 설정. 해당 구간은 충전할 수 없으며 필요한 저장에너지가 최대 가용 9 MWhth를 초과. 전력 감소량은 정격 이내여도 지속시간 때문에 불가능.</li><li>불가능한 경우 IT 수요를 임의 축소하거나 ESS를 자동 추가하지 않았다. 향후 상한 재협의·ESS 결합·사양 조정 등 별도 선택이 필요.</li></ul>
<h2>6. 시간 해상도의 의미</h2>{table(fine)}
<p class="note">본 상한은 <b>5분 평균전력 상한</b>이다. 동일 명령을 기존 10초 입력에 대입하면 5분 내부 변동으로 순간적인 상한 초과가 발생할 수 있다. 따라서 이번 결과는 순간 전력·ramp·grid-code 준수 검증을 의미하지 않는다. 실제 계약이 순간 상한이면 여유폭 또는 하위 제어가 필요하다. 10초 대입에서도 냉동기 용량을 별도로 확인했지만 신규 열제어 검증은 아니다.</p>
<h2>7. 검증 및 한계</h2><ul><li>최적화 출력은 기존 energy_step과 PCC 전력 계산으로 다시 실행해 SOC·충방전·냉동기·PCC 상한·종료 상태를 점검.</li><li>모든 가용 해 검증 통과: {verification['all_feasible_verified']}. 두 불가능 사례 판정 확인: {verification['expected_infeasibility_confirmed']}.</li><li>단위시험: {verification['unit_tests']['passed']}. 기존 파일 {verification['protected_file_count']}개 해시 보존: {verification['protected_files_unchanged']}.</li><li>수전한도 입력의 통신·DSO 계산, 부하 예측오차, ESS, DFL, 실제 수온/유량/빠른 제어 검증은 미포함.</li></ul>
<h2>8. 다음 연구 연결</h2><ul><li>이번 모델의 입력 P_limit[k]를 향후 DSO가 산정한 허용범위로 교체할 수 있는 최소 TES 운전 최적화 기준 확보.</li><li>다음에는 평가할 계통과 DSO 제약·허용범위 산정식을 정하고, 예측과 실현 수요를 분리한 반복 운영으로 확장.</li><li>DFL은 이후 같은 운영문제에서 예측오차 중심 학습과 비교. 본 결과만으로 DSO–AIDC 협조운영이나 DFL 효과가 검증됐다고 주장하지 않음.</li></ul>
<h2>9. 재현 파일 및 출처</h2><p><a href="summary.json">요약 결과</a> · <a href="model_contract.json">가정·제약</a> · <a href="verification.json">검증 기록</a> · <a href="schedules_5min.npz">5분 운전계획 배열</a></p><ul>{refs}<li><a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html">SciPy milp 공식 문서</a>: 혼합정수 선형계획 및 solver status 해석. 실행 버전 {html.escape(scipy.__version__)}.</li></ul></body></html>'''
    (OUT/'report_ko.html').write_text(text, encoding='utf-8')


def main():
    cfg = read(CONFIG)
    cpath = ROOT/cfg['candidate_config']
    candidate = read(cpath)
    legacy = read(ROOT/candidate['legacy_config'])
    tes = candidate['tes']
    protected = {cpath, ROOT/candidate['legacy_config'], ROOT/'src/aidc_facility/tes_operating.py',
                 ROOT/'src/aidc_facility/tes_flexibility.py', ROOT/'src/aidc_facility/pcc_cooling_tes.py'}
    for name in ('pcc_tes_operating_candidate_v1', 'tes_flexibility_assessment_v1', 'pcc_cooling_tes_candidate_v1',
                 'it_pcc_boundary_audit_v1', 'pcc_capacity_cop_review_v1', 'pcc_temperature_conditions_v1'):
        protected.update(p for p in (ROOT/'outputs'/name).rglob('*') if p.is_file())
    before = {str(p.relative_to(ROOT)): sha(p) for p in sorted(protected)}
    cols = read(ROOT/candidate['baseline_columns'])
    original = np.load(ROOT/candidate['baseline_profiles'])['no_tes']
    block = cfg['step_s']//legacy['internal_step_s']
    if cfg['step_s'] % legacy['internal_step_s'] or len(original) % block:
        raise ValueError('Complete aggregation bins required')
    base = original.reshape(-1, block, len(cols)).mean(1)
    rows = [dict(zip(cols, map(float, r))) for r in base]
    dt = cfg['step_s']/3600
    hours = np.arange(len(rows))*dt
    p0 = np.array([pcc_from_thermal(r, 0., 0., legacy, cfg['cop'], tes['pump_mw_per_mw_th'])['pcc_mw'] for r in rows])
    old = ROOT/'outputs/pcc_tes_operating_candidate_v1'
    oldcols = read(old/'profile_columns.json')
    data = np.load(old/'profiles_5min.npz')
    baselines = {'no_tes': p0, 'idle': data['candidate_cop5_idle'][:, oldcols.index('pcc_mw')],
                 'preset': data['candidate_cop5_scheduled'][:, oldcols.index('pcc_mw')]}
    for name in ('candidate_cop5_idle', 'candidate_cop5_scheduled'):
        if len(data[name]) != len(rows):
            raise ValueError('Baseline horizon mismatch')
        assert abs(data[name][0,oldcols.index('tes_energy_start_mwh_th')]-5.) < 1e-8
        assert abs(data[name][-1,oldcols.index('tes_energy_end_mwh_th')]-5.) < 1e-8
    baseline_error = float(np.max(np.abs(p0-data['candidate_cop5_no_tes'][:,oldcols.index('pcc_mw')])))
    coeff = marginal_pcc_coefficients(legacy, cfg['cop'], tes['pump_mw_per_mw_th'])
    OUT.mkdir(parents=True, exist_ok=True)
    results, summaries, arrays = {}, [], {'time_h': hours}
    arrays.update({f'baseline_{key}_pcc_mw': value for key, value in baselines.items()})
    for sc in cfg['scenarios']:
        cap = make_cap(sc, p0, hours, coeff)
        print('Solving '+sc['id'], flush=True)
        result = solve_schedule(rows, cap, tes, legacy, dt_h=dt, initial_soc=cfg['initial_soc'],
                                terminal_soc=cfg['terminal_soc'], cop=cfg['cop'],
                                objective=cfg['objective'], time_limit_s=cfg['time_limit_s'])
        results[sc['id']] = result
        dump(OUT/(sc['id']+'_solution.json'), result)
        arrays[sc['id']+'_cap_mw'] = cap
        summary = {'id': sc['id'], 'label_ko': sc['label_ko'], 'status': result['status'],
                   'feasible': result['feasible'], 'optimal': result['optimal'],
                   'no_tes': cap_metrics(p0,cap,dt), 'idle': cap_metrics(baselines['idle'],cap,dt),
                   'preset': cap_metrics(baselines['preset'],cap,dt), 'optimized': None,
                   'increment_vs_idle_mwh': None, 'increment_vs_preset_mwh': None,
                   'instantaneous_lower_bound_excess_mw': float(np.maximum(p0-coeff['reduction_mwe_per_mwth']*tes['discharge_limit_mw_th']-cap,0.).max()),
                   'solver': result['solver']}
        if sc['type'] == 'energy_stress':
            span = sc['end_h']-sc['start_h']
            summary['required_window_inventory_mwh_th'] = span*(sc['thermal_discharge_required_mw_th']/tes['discharge_efficiency']+tes['capacity_loss_fraction_per_hour']*tes['energy_capacity_mwh_th'])
        if result['feasible']:
            s = result['schedule']
            summary['optimized'] = cap_metrics(s['pcc_mw'],cap,dt)
            summary['optimized'].update(result['metrics'])
            summary['increment_vs_idle_mwh'] = float((np.asarray(s['pcc_mw'])-baselines['idle']).sum()*dt)
            summary['increment_vs_preset_mwh'] = float((np.asarray(s['pcc_mw'])-baselines['preset']).sum()*dt)
            qc, qd = np.asarray(s['charge_mw_th']), np.asarray(s['discharge_mw_th'])
            p10 = original[:,cols.index('pcc_mw')]+np.repeat(coeff['increase_mwe_per_mwth']*qc-coeff['reduction_mwe_per_mwth']*qd,block)
            q10 = original[:,cols.index('chiller_mw_th')]+np.repeat((1+tes['pump_mw_per_mw_th'])*qc-(1-tes['pump_mw_per_mw_th'])*qd,block)
            diag = cap_metrics(p10,np.repeat(cap,block),legacy['internal_step_s']/3600)
            diag['minimum_chiller_headroom_mw_th'] = float(legacy['plant']['chiller_capacity_mw_th']-q10.max())
            diag['minimum_chiller_output_mw_th'] = float(q10.min())
            diag['mean_reaggregation_max_error_mw'] = float(np.max(np.abs(p10.reshape(-1,block).mean(1)-np.asarray(s['pcc_mw']))))
            summary['held_10s_diagnostic'] = diag
            for key, value in s.items():
                arrays[sc['id']+'_'+key] = np.asarray(value)
        summaries.append(summary)
        print(json.dumps({'case':sc['id'],'status':result['status'],'peak_mw':result['metrics']['pcc_peak_mw'] if result['metrics'] else None},ensure_ascii=False),flush=True)
    tests = subprocess.run([sys.executable,'-X','utf8','-B','-m','unittest','discover','-s','tests','-p','test_*tes*.py','-v'],
                           cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    after = {str(p.relative_to(ROOT)): sha(p) for p in sorted(protected)}
    expected = all(results[k]['status']=='infeasible' for k in ('constant_25_8','energy_stress'))
    all_feasible = all(r['feasible'] and r['optimal'] and all(r['checks'].values())
                       for key,r in results.items() if key not in ('constant_25_8','energy_stress'))
    fine_ok = all(s['held_10s_diagnostic']['minimum_chiller_headroom_mw_th']>=-1e-7 and
                  s['held_10s_diagnostic']['minimum_chiller_output_mw_th']>=-1e-7 and
                  s['held_10s_diagnostic']['mean_reaggregation_max_error_mw']<1e-8
                  for s in summaries if s['feasible'])
    verification = {'protected_file_count':len(before),'protected_files_unchanged':before==after,
                    'protected_sha256':before,'all_feasible_verified':all_feasible,
                    'expected_infeasibility_confirmed':expected,'baseline_reproduction_max_error_mw':baseline_error,
                    'held_10s_chiller_and_reaggregation_passed':fine_ok,
                    'unit_tests':{'passed':tests.returncode==0,'output':tests.stdout+tests.stderr}}
    summary = {'created_utc':datetime.now(timezone.utc).isoformat(),'duration_h':len(rows)*dt,
               'intervals':len(rows),'coefficients':coeff,'scenarios':summaries,
               'scope':'Perfect-foresight offline TES scheduling, hypothetical five-minute average import caps, no DSO/network/ESS/DFL implementation.'}
    contract = {'experiment':cfg,'inherited_candidate':candidate,
                'baseline_comparison':'Same TES, same IT, same horizon, E0=ET=5. Prior preset and installed idle both restore energy through actual operation.',
                'solver':{'library':'scipy.optimize.milp','version':scipy.__version__,
                          'documentation':'https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html'},
                'formulation':'Hard PCC cap; binary charge/discharge exclusion; exact linear energy and plant balances; no slack.'}
    dump(OUT/'summary.json',summary)
    dump(OUT/'verification.json',verification)
    dump(OUT/'model_contract.json',contract)
    np.savez_compressed(OUT/'schedules_5min.npz',**arrays)
    plots(results,baselines,cfg)
    report(summary,candidate,verification)
    print(json.dumps({'report':str(OUT/'report_ko.html'), 'all_feasible_verified':all_feasible,
                      'expected_infeasible':expected,'unit_tests_passed':tests.returncode==0,
                      'legacy_unchanged':before==after},ensure_ascii=False,indent=2))
    if not (before==after and all_feasible and expected and fine_ok and baseline_error<1e-8 and tests.returncode==0):
        raise AssertionError('Verification failed; inspect outputs')


if __name__ == '__main__':
    main()
