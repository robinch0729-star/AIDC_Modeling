"""Literature-based TES energy operating candidate; preserves all legacy outputs.

Reuses frozen no-TES cooling service and 5-minute averages; no hardware design,
room-temperature replay, cost optimization or DSO instruction is inferred.
"""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aidc_facility.tes_operating import energy_step, target_charge_request
from run_it_operating_baseline import plot

CONFIG = ROOT / "configs/pcc_tes_operating_candidate_v1.json"
OUT = ROOT / "outputs/pcc_tes_operating_candidate_v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def protected_files(c):
    paths = {ROOT / c['legacy_config'], ROOT / 'src/aidc_facility/pcc_cooling_tes.py',
             ROOT / 'tests/test_pcc_cooling_tes.py'}
    for name in ('it_5min_operating_baseline_v1', 'pcc_cooling_tes_candidate_v1',
                 'pcc_capacity_cop_review_v1', 'pcc_temperature_conditions_v1', 'it_pcc_boundary_audit_v1'):
        paths.update(p for p in (ROOT / 'outputs' / name).rglob('*') if p.is_file())
    return sorted(paths)


def simulate_operating(base, legacy, c, mode, cop=5.):
    t, d = c['tes'], c['dispatch']
    h = c['step_s'] / 3600
    total_h = len(base) * h
    energy = t['initial_soc'] * t['energy_capacity_mwh_th']
    low = t['min_soc'] * t['energy_capacity_mwh_th']
    target = d['terminal_target_soc'] * t['energy_capacity_mwh_th']
    rows = []
    e, p = legacy['electrical'], legacy['plant']
    for k, row in enumerate(base.to_dict('records')):
        hour, phase = k * h, (k * c['step_s'] % 86400) / 3600
        recovery = hour >= total_h - d['terminal_recovery_hours'] - 1e-9
        charge_window = d['charge_hours_elapsed_mod24'][0] <= phase < d['charge_hours_elapsed_mod24'][1]
        dis_window = d['discharge_hours_elapsed_mod24'][0] <= phase < d['discharge_hours_elapsed_mod24'][1]
        ordinary_dis = mode == 'scheduled' and dis_window and not recovery
        req = 0.
        if mode != 'no_tes':
            if recovery:
                # No free terminal-state assignment: restoration uses real heat ports.
                idle = energy - t['capacity_loss_fraction_per_hour'] * t['energy_capacity_mwh_th'] * h
                if idle > target:
                    req = min((idle - target) * t['discharge_efficiency'] / h,
                              d['discharge_command_mw_th'])
                else:
                    req = max(target_charge_request(energy, target, h, t), -d['charge_command_mw_th'])
            elif mode == 'scheduled' and charge_window:
                req = max(target_charge_request(energy, d['ordinary_charge_target_soc'] * t['energy_capacity_mwh_th'], h, t),
                          -d['charge_command_mw_th'])
            elif ordinary_dis:
                req = d['discharge_command_mw_th']
            schedule_req = req
            # Maintain the adopted minimum SOC with paid cooling, not state clipping.
            idle_end = energy - t['capacity_loss_fraction_per_hour'] * t['energy_capacity_mwh_th'] * h
            reserve_override = idle_end < low - 1e-10 and req >= 0
            if reserve_override:
                req = target_charge_request(energy, low, h, t)
            st = energy_step(energy, req, h, row['chiller_mw_th'], p['chiller_capacity_mw_th'], t)
            if st['state_violation'] or st['physical_energy_violation']:
                raise AssertionError(f'Unresolved storage infeasibility at {k}: {st}')
            energy = st['energy_end_mwh_th']
            qch, qdis, pump = st['charge_mw_th'], st['discharge_mw_th'], st['pump_mw']
        else:
            schedule_req, reserve_override = 0., False
            qch, qdis, pump = 0., 0., 0.
            st = dict(energy_start_mwh_th=np.nan, energy_end_mwh_th=np.nan,
                      standby_loss_mwh_th=0., charge_conversion_loss_mwh_th=0.,
                      discharge_conversion_loss_mwh_th=0., energy_balance_error_mwh_th=0.)
        thermal = row['chiller_mw_th'] + pump + qch - qdis
        pch = thermal / cop
        condenser = p['condenser_pump_mw_per_mw_heat'] * (thermal + pch)
        reject = thermal + pch + condenser
        tower = p['tower_fan_mw_per_mw_heat'] * reject
        cooling = row['fan_mw'] + row['water_pump_mw'] + pump + pch + condenser + tower
        downstream = row['it_mw'] / e['ups_efficiency'] + e['auxiliary_mw'] + cooling
        pcc = downstream / e['transformer_efficiency']
        vals = {
            'time_h': hour, 'it_mw': row['it_mw'], 'base_chiller_mw_th': row['chiller_mw_th'],
            'pcc_mw': pcc, 'cooling_mw': cooling, 'chiller_mw_th': thermal,
            'chiller_mw': pch, 'tes_charge_mw_th': qch, 'tes_discharge_mw_th': qdis,
            'tes_pump_mw': pump, 'tes_energy_start_mwh_th': st['energy_start_mwh_th'],
            'tes_energy_end_mwh_th': st['energy_end_mwh_th'],
            'tes_soc_end': st['energy_end_mwh_th'] / t['energy_capacity_mwh_th'],
            'standby_loss_mwh_th': st['standby_loss_mwh_th'],
            'charge_conversion_loss_mwh_th': st['charge_conversion_loss_mwh_th'],
            'discharge_conversion_loss_mwh_th': st['discharge_conversion_loss_mwh_th'],
            'scheduled_net_request_mw_th': schedule_req,
            'ordinary_discharge_period': float(ordinary_dis), 'terminal_recovery_period': float(recovery),
            'reserve_maintenance_override': float(reserve_override),
            'ordinary_discharge_shortfall_mw_th': max(d['discharge_command_mw_th'] - qdis, 0.) if ordinary_dis else 0.,
            'energy_balance_mwh_th': st['energy_balance_error_mwh_th'],
            'cooling_balance_mw_th': thermal + qdis - qch - pump - row['coil_mw_th'] - row['water_pump_mw'],
            'electrical_balance_mw': pcc - downstream - downstream * (1 / e['transformer_efficiency'] - 1),
            'chiller_headroom_mw_th': p['chiller_capacity_mw_th'] - thermal,
        }
        rows.append(vals)
    return pd.DataFrame(rows)


def metrics(f, c, installed):
    h = c['step_s'] / 3600
    t = c['tes']
    final_error = float(f.tes_energy_end_mwh_th.iloc[-1] - t['initial_soc'] * t['energy_capacity_mwh_th']) if installed else None
    result = {
        'duration_h': len(f) * h, 'it_mean_mw': float(f.it_mw.mean()),
        'pcc_mean_mw': float(f.pcc_mw.mean()), 'pcc_peak_mw': float(f.pcc_mw.max()),
        'pcc_energy_mwh': float(f.pcc_mw.sum() * h),
        'chiller_peak_mw_th': float(f.chiller_mw_th.max()),
        'chiller_min_headroom_mw_th': float(f.chiller_headroom_mw_th.min()),
        'charge_mwh_th': float(f.tes_charge_mw_th.sum() * h),
        'discharge_mwh_th': float(f.tes_discharge_mw_th.sum() * h),
        'standby_loss_mwh_th': float(f.standby_loss_mwh_th.sum()),
        'conversion_loss_mwh_th': float((f.charge_conversion_loss_mwh_th + f.discharge_conversion_loss_mwh_th).sum()),
        'ordinary_discharge_shortfall_mwh_th': float(f.ordinary_discharge_shortfall_mw_th.sum() * h),
        'initial_energy_mwh_th': float(f.tes_energy_start_mwh_th.iloc[0]) if installed else None,
        'final_energy_mwh_th': float(f.tes_energy_end_mwh_th.iloc[-1]) if installed else None,
        'final_energy_error_mwh_th': final_error,
        'soc_min': float(min(t['initial_soc'], f.tes_soc_end.min())) if installed else None,
        'soc_max': float(max(t['initial_soc'], f.tes_soc_end.max())) if installed else None,
        'max_balance_errors': {key: float(f[key].abs().max()) for key in ('energy_balance_mwh_th', 'cooling_balance_mw_th', 'electrical_balance_mw')},
    }
    result['checks'] = {
        'chiller_capacity': bool((f.chiller_mw_th >= -1e-9).all() and (f.chiller_headroom_mw_th >= -1e-9).all()),
        'no_simultaneous_charge_discharge': bool(not ((f.tes_charge_mw_th > 1e-9) & (f.tes_discharge_mw_th > 1e-9)).any()),
        'charge_rating': bool((f.tes_charge_mw_th <= t['charge_limit_mw_th'] + 1e-9).all()),
        'discharge_rating': bool((f.tes_discharge_mw_th <= t['discharge_limit_mw_th'] + 1e-9).all()),
        'energy_conservation': result['max_balance_errors']['energy_balance_mwh_th'] < 1e-9,
        'cooling_and_electrical_balance': max(result['max_balance_errors'].values()) < 1e-9,
        'soc_bounds': bool(result['soc_min'] >= t['min_soc'] - 1e-9 and result['soc_max'] <= t['max_soc'] + 1e-9) if installed else None,
        'equal_initial_final_energy': abs(final_error) < 1e-8 if installed else None,
    }
    return result


def comparison(f, ref, c):
    delta = f.pcc_mw.to_numpy() - ref.pcc_mw.to_numpy()
    mask = (f.ordinary_discharge_period > 0) & (f.tes_discharge_mw_th > 1e-9)
    return {
        'pcc_energy_difference_mwh': float(delta.sum() * c['step_s'] / 3600),
        'global_peak_difference_mw': float(f.pcc_mw.max() - ref.pcc_mw.max()),
        'maximum_same_time_reduction_mw': float(max(-delta.min(), 0.)),
        'maximum_same_time_increase_mw': float(max(delta.max(), 0.)),
        'mean_reduction_during_ordinary_discharge_mw': float(-delta[mask].mean()) if mask.any() else None,
        'maximum_reduction_during_ordinary_discharge_mw': float(max(-delta[mask].min(), 0.)) if mask.any() else None,
    }


def figure(frames):
    f, idle, no = (frames[k] for k in ('candidate_cop5_scheduled', 'candidate_cop5_idle', 'candidate_cop5_no_tes'))
    x = f.time_h.to_numpy()
    plot(OUT / 'operating_candidate.png', 'Literature-based TES candidate | 10 MWh_th / 3 MW_th',
         '5-minute energy operation | same IT input | no equipment geometry or temperature validation', [
        {'title': 'PCC active power', 'unit': 'MW', 'xlabel': 'Elapsed simulation time (h)',
         'series': [('No TES', x, no.pcc_mw), ('Scheduled TES', x, f.pcc_mw)]},
        {'title': 'Scheduled minus installed-idle baseline (equal initial / final energy)', 'unit': 'MW',
         'xlabel': 'Elapsed simulation time (h)', 'series': [('PCC difference', x, f.pcc_mw-idle.pcc_mw)]},
        {'title': 'Stored cooling energy', 'unit': 'SOC (%)', 'xlabel': 'Elapsed simulation time (h)', 'zero': True,
         'series': [('Scheduled TES', x, f.tes_soc_end*100), ('Installed idle + recovery', x, idle.tes_soc_end*100),
                    ('Minimum SOC', x, np.full(len(x),10.))]}],
         'Final 4 h: energy recovery through actual charging. Illustrative schedule, not optimized peak shaving.')


def table(rows):
    keys = list(rows[0])
    def fmt(v):
        return 'N/A' if v is None else f'{v:.5f}' if isinstance(v, float) else html.escape(str(v))
    return '<div class="scroll"><table><tr>' + ''.join(f'<th>{html.escape(k)}</th>' for k in keys) + '</tr>' + ''.join('<tr>' + ''.join(f'<td>{fmt(row[k])}</td>' for k in keys) + '</tr>' for row in rows) + '</table></div>'


def report(s, c, v):
    main = s['cases']['candidate_cop5_scheduled']
    effect = s['comparisons']['candidate_cop5_scheduled_vs_idle']
    specs = [{'항목': p['parameter'], '적용': p['value'], '구분': p['classification'], '근거·해석': p['basis']} for p in c['parameter_provenance']]
    cases = [{'경우': k, '평균 PCC (MW)': m['pcc_mean_mw'], '방전 (MWhth)': m['discharge_mwh_th'], '미수행 방전 (MWhth)': m['ordinary_discharge_shortfall_mwh_th'], '최종 E (MWhth)': m['final_energy_mwh_th'], 'SOC 최소': m['soc_min'], '냉동기 최대 (MWth)': m['chiller_peak_mw_th']} for k, m in s['cases'].items()]
    compare = [{'비교': k, '전체 전력량 차이 (MWh)': m['pcc_energy_difference_mwh'], '전체 피크 차이 (MW)': m['global_peak_difference_mw'], '정규 방전시간 평균 감소 (MW)': m['mean_reduction_during_ordinary_discharge_mw']} for k,m in s['comparisons'].items()]
    refs = ''.join(f'<li><a href="{r["url"]}">{html.escape(r["title"])}</a> ({html.escape(r["version"])}) — {html.escape(r["used"])}</li>' for r in c['references'])
    text = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>문헌 기반 TES 운영 모델</title><style>body{{font-family:"Malgun Gothic",Arial,sans-serif;max-width:1250px;margin:32px auto;padding:0 24px;color:#203d4d;line-height:1.75}}h1{{font-size:28px}}h2{{font-size:22px;margin-top:32px;border-bottom:1px solid #cddfe4}}table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{padding:9px;border-bottom:1px solid #d8e3e7;text-align:left;vertical-align:top}}th{{background:#eaf4f4}}.scroll{{overflow:auto}}.note{{padding:16px;background:#fff5df;border-left:4px solid #cb9634}}.eq{{background:#eff6f7;padding:16px;font-family:Cambria,serif;font-size:18px}}img{{width:100%;height:auto}}a{{color:#006d77}}code{{overflow-wrap:anywhere}}</style></head><body>
<h1>TES 모델링 범위 정리 및 문헌 기반 운영 후보</h1>
<p class="note"><b>결론:</b> 탱크·열교환기 하드웨어 설계 대신, 저장에너지·충방전출력·효율·손실·냉동기 여유를 갖는 운영 모델을 별도로 구현했다. 기존 IT 입력과 혼합수온 TES 모델은 보존했다. 아래는 문헌과 가정을 결합한 연구용 후보이며 실제 설비 성능 검증이나 최적 용량 선정 결과가 아니다.</p>
<h2>1. 무엇을 바꾸었는가</h2><ul><li>기존: 3,000 m³ 완전혼합 탱크의 온도·유량·열교환 온도차로 TES 동작을 계산. 장치 상세와 연구 목적 사이의 범위 조정 필요.</li><li>이번 후보: 냉수 TES를 에너지 저장상태 E와 열출력 Q로 표현. 지름·높이·배관·코일 UA·냉매 모델은 추가하지 않음.</li><li>paper list [30]의 TES 상태식·출력 제약·사양을 주 근거로 사용하고, REopt의 SOC 및 대기손실 기본값을 보완 적용. 단일 문헌의 전체 모델을 그대로 재현한 것은 아님.</li><li>[30]은 확인한 arXiv v1 preprint. 정식 저널 논문으로 표기하지 않음. [13]은 저장용량 단위와 시간 기준이 불명확해 TES 수치의 주 근거로 사용하지 않음.</li></ul>
<h2>2. 후보 사양과 출처 구분</h2>{table(specs)}
<p>10 MWhth/3 MWth는 기존 3 MWth 냉각 이동 시험을 유지하면서 [30]의 E/Q 비율(3.333 h)을 보존한 10배 확대다. 우리 IT 규모와 비례해 산정된 값이나 최적 설계가 아니다. 실제 최대 정격 방전 지속시간은 최소 SOC·효율·대기손실을 고려하면 완충 상태에서 약 {s['rated_discharge_duration_from_full_h']:.3f}시간이며, E/Q 자체와 다르다. 검증은 탱크 부피를 확정하지 않아도 수행할 수 있다.</p>
<h2>3. 사용한 식 및 계산 경계</h2><div class="eq">E[k+1] = E[k] + (ηch Qch[k] − Qdis[k]/ηdis − σ Emax) Δt<br>SOC = E/Emax; &nbsp; 0.1 ≤ SOC ≤ 1; &nbsp; Efinal = Einitial<br>Qchiller = Qbase + PTES,pump + Qch − Qdis<br>PTES,pump = kp (Qch + Qdis); &nbsp; Pchiller = Qchiller/COP<br>PPCC = (PIT,proxy/ηUPS + Pcooling + Paux)/ηtransformer</div>
<ul><li>E는 MWhth, Q는 MWth, 전기 P는 MWe, Δt는 시간(h). 5분이면 1/12 h. 효율 0.9는 시간당 손실률이 아니므로 매 스텝마다 E에 0.9를 곱하지 않는다.</li><li>σ=0.0004/h는 정격 저장용량 기준. 후보의 대기손실은 4 kWth, 5분에 0.333 kWhth. 매뉴얼 Table31의 저장량 표현과 공식 코드의 용량 기준을 구분하고, 버전 고정 코드의 수식을 따랐다.</li><li>충방전 효율은 열의 입출력 손실, 대기손실은 외부 열유입, 펌프는 별도 전기소비로 구분. 과거 탱크 UA 손실이나 REopt 내부효율을 추가로 곱하지 않음.</li><li>충전은 저장량·3 MWth 정격·냉동기 잔여 용량으로 제한. 방전은 저장량·3 MWth 정격·동시 냉각수요로 제한. 충방전 동시 수행 금지.</li><li>IT·기본 팬/펌프·코일 열수요는 기존 TES 미사용 결과를 고정 입력으로 사용. TES가 대체해도 같은 기본 냉각 서비스가 공급되도록 수지를 구성했다. 이는 신규 온도 시뮬레이션이나 공급수온 적합성 검증은 아니다.</li></ul>
<h2>4. 약 5일 실행 결과</h2><p>총 {main['duration_h']:.3f}시간, 5분 간격. IT 평균 {main['it_mean_mw']:.4f} MW를 유지했으며 PCC 평균을 20 MW로 재조정하지 않았다. 정규 시간표는 경과시간 기준 매일 2–6시간 충전(충전목표 도달 시 감소), 16–18시간 3 MWth 방전이다. 종료 전 4시간에는 초기 저장량을 실제 충방전으로 복원한다.</p>
<img src="operating_candidate.png" alt="PCC power and TES SOC candidate comparison">
<p>주 후보는 정규 방전 요구를 모두 수행했는지 <b>미수행 방전 {main['ordinary_discharge_shortfall_mwh_th']:.6f} MWhth</b>로 확인한다. 최종 E={main['final_energy_mwh_th']:.6f} MWhth, 냉동기 최대={main['chiller_peak_mw_th']:.4f} MWth. 같은 TES를 설치한 비시프트·복원 기준 대비 정규 방전시간 PCC 감소는 평균 <b>{effect['mean_reduction_during_ordinary_discharge_mw']:.4f} MW</b>다.</p>
{table(cases)}
<h2>5. 비교조건과 민감도 해석</h2>{table(compare)}
<ul><li>No TES는 설비 도입 비교용이며 저장상태 비교는 N/A. Installed idle은 같은 설비·손실·초기/종료 E를 가진 운영전략 비교 기준이다. 대기 및 복원 전력도 포함한다.</li><li>candidate는 10 MWhth/3 MWth. reference_size는 원문 1 MWhth/0.3 MWth, 나머지 SOC·손실 조건은 보완 모델과 동일. capacity5는 출력 3 MWth를 유지하고 용량만 5 MWhth로 줄인 스트레스 검사다.</li><li>COP4/6, 손실 0 또는 기준 5배, 펌프 계수 0/0.02는 영향 확인을 위한 민감도 조건이지 실측 신뢰구간이 아니다.</li><li>전력 이동으로 방전시간 수전량은 줄지만 충전·손실로 총 전력량은 증가할 수 있다. 전체 최대부하 감소와 특정 시각 감소를 분리했다. 본 시간표는 요금·계통 혼잡을 최적화하지 않으므로 비용 절감이나 피크저감을 보장하지 않는다.</li></ul>
<h2>6. 검증과 한계</h2><ul><li>기존 파일 {v['protected_file_count']}개 해시 보존: {v['protected_files_unchanged']}.</li><li>새 단위시험 및 기존 냉각/TES 시험: {v['unit_tests']['passed']}.</li><li>모든 시나리오의 명시된 에너지·출력·SOC·종료상태 검사: {v['all_case_checks_passed']}.</li><li>5분 기준 TES 미사용 전력의 기존 결과 재현 오차: {v['no_tes_baseline_max_difference_mw']:.3g} MW.</li><li>5분 고정 명령을 기존 10초 냉각수요에 대입한 냉동기 용량 검사: {v['held_commands_10s_chiller_check']['passed']} (최소 여유 {v['held_commands_10s_chiller_check']['minimum_headroom_mw_th']:.4f} MWth). 이는 빠른 제어나 공급수온을 검증한 것이 아님.</li><li>탱크 수온·성층·유량·응답속도는 검증하지 않았다. 적절한 냉수 온도창에서 정격 냉열 공급이 가능하다는 가정하의 운영 결과다.</li><li>NLR 기반 입력은 등가 IT 연구 입력이며 전체 서버 AC 실측으로 확인된 값은 아니다. 기존 UPS97%·변압기99%·보조부하0.1MW 등의 가정은 그대로다.</li></ul>
<h2>7. 다음 연구 단계</h2><ul><li>이 운영 후보를 기준으로 ESS와 TES의 가용 조정량·지속시간·복원에너지를 동일 형식으로 정리.</li><li>PCC 허용범위(DOE)를 입력받아 저장상태와 냉동기 제약을 지키는 스케줄링 문제 정식화.</li><li>그 이후 DSO–AIDC 정보교환 및 일반 예측학습/DFL의 운영성과 비교로 진행. TES 하드웨어 상세설계는 연구 핵심 경로에서 제외.</li></ul>
<h2>References</h2><ul>{refs}</ul><p><a href="summary.json">수치 요약</a> · <a href="verification.json">검증 내역</a> · <a href="scenario_configs.json">시나리오 조건</a> · <a href="model_contract.json">모델·출처 계약</a> · <a href="profiles_5min.npz">결과 배열</a> · <a href="profile_columns.json">열 이름</a></p></body></html>'''
    (OUT / 'report_ko.html').write_text(text, encoding='utf-8')


def main():
    c = read(CONFIG)
    legacy = read(ROOT / c['legacy_config'])
    protected = protected_files(c)
    before = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    OUT.mkdir(parents=True, exist_ok=True)
    cols = read(ROOT / c['baseline_columns'])
    with np.load(ROOT / c['baseline_profiles']) as raw:
        original = pd.DataFrame(raw['no_tes'], columns=cols)
    block = c['step_s'] // legacy['internal_step_s']
    if block * legacy['internal_step_s'] != c['step_s'] or len(original) % block:
        raise ValueError('Input requires complete 5-minute bins')
    base = original.groupby(np.arange(len(original)) // block).mean()
    frames, configs, result = {}, {}, {}
    cases = [('candidate_cop5_no_tes', 'no_tes', 5., {}), ('candidate_cop5_idle', 'idle', 5., {}),
             ('candidate_cop5_scheduled', 'scheduled', 5., {}),
             ('candidate_cop4_idle', 'idle', 4., {}), ('candidate_cop4_scheduled', 'scheduled', 4., {}),
             ('candidate_cop6_idle', 'idle', 6., {}), ('candidate_cop6_scheduled', 'scheduled', 6., {}),
             ('reference_size_scheduled', 'scheduled', 5., {'energy_capacity_mwh_th': 1., 'charge_limit_mw_th': .3, 'discharge_limit_mw_th': .3}),
             ('capacity5_scheduled', 'scheduled', 5., {'energy_capacity_mwh_th': 5.}),
             ('loss0_scheduled', 'scheduled', 5., {'capacity_loss_fraction_per_hour': 0.}),
             ('loss5x_scheduled', 'scheduled', 5., {'capacity_loss_fraction_per_hour': .002}),
             ('pump0_scheduled', 'scheduled', 5., {'pump_mw_per_mw_th': 0.}),
             ('pump2pct_scheduled', 'scheduled', 5., {'pump_mw_per_mw_th': .02})]
    for name, mode, cop, changes in cases:
        alt = copy.deepcopy(c)
        alt['tes'].update(changes)
        if name == 'reference_size_scheduled':
            alt['dispatch']['charge_command_mw_th'] = .3
            alt['dispatch']['discharge_command_mw_th'] = .3
        f = simulate_operating(base, legacy, alt, mode, cop)
        frames[name], configs[name] = f, {'mode': mode, 'cop': cop, 'config': alt}
        result[name] = metrics(f, alt, mode != 'no_tes')
    comparisons = {f'candidate_cop{cop}_scheduled_vs_idle': comparison(frames[f'candidate_cop{cop}_scheduled'], frames[f'candidate_cop{cop}_idle'], c) for cop in (4, 5, 6)}
    comparisons['candidate_cop5_scheduled_vs_no_tes'] = comparison(frames['candidate_cop5_scheduled'], frames['candidate_cop5_no_tes'], c)
    no_diff = float(np.abs(frames['candidate_cop5_no_tes'].pcc_mw.to_numpy() - base.pcc_mw.to_numpy()).max())
    f = frames['candidate_cop5_scheduled']
    # Recompute from exported fields, independently of the kernel's error field.
    dt = c['step_s'] / 3600
    thermal_residual = f.tes_energy_end_mwh_th - f.tes_energy_start_mwh_th - (
        c['tes']['charge_efficiency'] * f.tes_charge_mw_th
        - f.tes_discharge_mw_th / c['tes']['discharge_efficiency']
        - c['tes']['capacity_loss_fraction_per_hour'] * c['tes']['energy_capacity_mwh_th']) * dt
    continuity = float(np.abs(f.tes_energy_start_mwh_th.to_numpy()[1:] - f.tes_energy_end_mwh_th.to_numpy()[:-1]).max())
    independent = {'max_energy_residual_mwh_th': float(thermal_residual.abs().max()),
                   'max_state_continuity_error_mwh_th': continuity,
                   'passed': bool(thermal_residual.abs().max() < 1e-9 and continuity < 1e-9)}
    held_q = original.chiller_mw_th.to_numpy() + np.repeat((f.tes_pump_mw + f.tes_charge_mw_th - f.tes_discharge_mw_th).to_numpy(), block)
    held = {'passed': bool(np.all((held_q >= -1e-9) & (held_q <= legacy['plant']['chiller_capacity_mw_th'] + 1e-9))),
            'minimum_headroom_mw_th': float(legacy['plant']['chiller_capacity_mw_th'] - held_q.max())}
    command = [sys.executable, '-X', 'utf8', '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*tes*.py', '-v']
    tests = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    after = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    v = {'protected_file_count': len(before), 'protected_files_unchanged': before == after,
         'protected_sha256': before, 'unit_tests': {'passed': tests.returncode == 0, 'output': tests.stdout + tests.stderr},
         'all_case_checks_passed': all(all(value is None or value for value in m['checks'].values()) for m in result.values()),
         'no_tes_baseline_max_difference_mw': no_diff, 'held_commands_10s_chiller_check': held,
         'independent_energy_recalculation': independent}
    s = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': c['status'], 'cases': result, 'comparisons': comparisons,
         'rated_discharge_duration_from_full_h': c['tes']['energy_capacity_mwh_th'] * (c['tes']['max_soc'] - c['tes']['min_soc']) / (c['tes']['discharge_limit_mw_th'] / c['tes']['discharge_efficiency'] + c['tes']['capacity_loss_fraction_per_hour'] * c['tes']['energy_capacity_mwh_th']),
         'interpretation': 'Numerical energy-operation verification only; no optimized capacity, site hardware validation, or fast thermal control guarantee.'}
    dump(OUT / 'summary.json', s)
    dump(OUT / 'verification.json', v)
    dump(OUT / 'scenario_configs.json', configs)
    dump(OUT / 'model_contract.json', c)
    dump(OUT / 'profile_columns.json', list(f.columns))
    np.savez_compressed(OUT / 'profiles_5min.npz', **{k: v.to_numpy(float) for k, v in frames.items()})
    figure(frames)
    report(s, c, v)
    print(json.dumps({'verification': {k: val for k,val in v.items() if k not in ('unit_tests','protected_sha256')}, 'main': result['candidate_cop5_scheduled'], 'effect': comparisons['candidate_cop5_scheduled_vs_idle'], 'report': str(OUT/'report_ko.html')}, ensure_ascii=False, indent=2))
    if not (v['protected_files_unchanged'] and v['unit_tests']['passed'] and v['all_case_checks_passed'] and no_diff < 1e-9 and held['passed'] and independent['passed']):
        raise AssertionError('Verification failed; inspect verification.json')


if __name__ == '__main__':
    main()
