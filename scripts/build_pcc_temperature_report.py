"""Document temperature semantics without changing the legacy plant model."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import html
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/pcc_temperature_conditions_v1"


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def table(headers, rows):
    def cell(value):
        return html.escape(str(value))
    return ('<div class="scroll"><table><thead><tr>'
            + ''.join('<th>' + cell(v) + '</th>' for v in headers)
            + '</tr></thead><tbody>'
            + ''.join('<tr>' + ''.join('<td>' + cell(v) + '</td>' for v in row) + '</tr>' for row in rows)
            + '</tbody></table></div>')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = json.loads((ROOT / 'configs/pcc_cooling_tes_candidate_v1.json').read_text(encoding='utf-8'))
    numeric = json.loads((OUT / 'numeric_checks.json').read_text(encoding='utf-8'))
    verification = json.loads((OUT / 'verification.json').read_text(encoding='utf-8'))
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        ('room.initial_c / room.setpoint_c', '등가 실내 초기·목표온도', '24 ℃',
         '서버실을 하나로 묶은 열 상태와 제어 목표. 서버 흡입구·칩 온도가 아님.', '시나리오 가정 유지'),
        ('room.min_c / room.max_c', '등가 실내 운전범위', '22–27 ℃',
         '등가 온도에 적용하는 검사용 범위. ASHRAE 장비 적합성 판정이 아님.', '시나리오 가정 유지'),
        ('room.ambient_c', '등가 실내 외기 경계', '30 ℃',
         '실내 UA 열유입 계산에만 사용. 실제 기상자료나 냉동기 외기 성능곡선이 아님.', '고정 경계조건'),
        ('tes.source_chilled_c', 'TES 충전 저온원', '6 ℃',
         '충전 열교환기 저온원 경계. 냉동기에서 계산된 출구온도가 아님.', '고정 경계조건'),
        ('tes.initial_c / tes.charge_target_c', 'TES 초기·충전목표', '7.5 ℃',
         '초기 상태, 시간표 충전 목표, 종료 저장상태 복원 목표.', '시나리오 가정 유지'),
        ('tes.minimum_c', 'TES 최저 허용온도', '7 ℃',
         '충전 하한. 정상 충전목표 7.5 ℃와 구분하며 결빙점이나 표준 권고값이 아님.', '시나리오 제약'),
        ('tes.load_return_c', 'TES 부하측 유입·환수', '12 ℃',
         '방전 열교환기에 들어오는 고정 환수. 공조 코일에서 계산한 환수온도가 아님.', '고정 경계조건'),
        ('tes.maximum_discharge_supply_c', 'TES 부하측 출구 상한', '10 ℃',
         '방전 열교환기 출구온도 제한. 탱크 수온·전체 코일 입수온도와 동일하지 않음.', '조건부 시나리오 제약'),
        ('tes.reference_warm_c', '저장냉열 계산 기준', '12 ℃',
         'E=C(12−T) 계산용 기준. 실제 방전 가능 끝점이 아님. 환수와 값만 같고 역할은 다름.', '회계적 기준온도'),
        ('tes.ambient_c', '탱크 주변온도', '25 ℃',
         '탱크 UA 열유입 계산에 사용하는 경계. 서버실 외기 경계 30 ℃와 구분되는 가상 탱크 설치환경.', '고정 경계조건'),
        ('tes.hx_effectiveness / maximum_flow_kg_s', 'TES 유효도·유량', '0.95 / 500 kg/s',
         '단순화한 TES 분기 HX 능력. 왕복효율이나 전체 냉수계통 유량이 아님.', '시나리오 가정 유지'),
        ('room.effective_capacity_mwh_per_k / ua_mw_per_k', '실내 열용량·열관류', '1 MWhth/K / 0.02 MWth/K',
         '등가 열 응답 가정. 논문의 공기 열용량 계수를 실측 식별한 값이 아님.', '미식별 파라미터'),
    ]
    refs = [
        {'id': 'R1', 'title': 'ASHRAE Handbook 2023, Chapter 20, Table 1 and note o',
         'url': 'https://handbook.ashrae.org/Handbooks/A23/SI/A23_Ch20/a23_ch20_si.aspx',
         'supports': '장비 흡입구 권고온도: A1–A4 18–27 ℃, H1 18–22 ℃. H1 적용은 제조사 판단.',
         'does_not_support': '우리 등가 실내 24 ℃·22–27 ℃의 직접 출처 또는 실제 AIDC 적합성 검증'},
        {'id': 'R2', 'title': 'EnergyPlus 9.5 Engineering Reference: Water Thermal Tanks',
         'url': 'https://bigladdersoftware.com/epx/docs/9-5/engineering-reference/water-thermal-tanks-includes-water-heaters.html',
         'supports': '혼합 탱크의 열수지, 주변 열유입, 유효도·유량·온도차를 이용한 열교환 형태.',
         'does_not_support': '6/7/7.5/10/12 ℃, 유효도 0.95, 유량 500 kg/s라는 구체적 설계값'},
        {'id': 'R3', 'title': 'EnergyPlus 9.5 Engineering Reference: Chilled-Water-Based Air Cooling Coil',
         'url': 'https://bigladdersoftware.com/epx/docs/9-5/engineering-reference/coils.html',
         'supports': '입수온도, 공기·물 유량, 공기 온습도 및 UA에 따른 코일 냉각능력. 설계점 기반 축약 가능.',
         'does_not_support': 'TES 출구 10 ℃가 현재 가상 시설 냉각부하를 충분히 처리한다는 보장'},
        {'id': 'R4', 'title': 'DOE FEMP: Cooling Water Efficiency Opportunities for Federal Data Centers',
         'url': 'https://www.energy.gov/cmei/femp/cooling-water-efficiency-opportunities-federal-data-centers',
         'supports': '공랭 IT에서 공조기·냉수·냉동기·냉각탑으로 이어지는 열 경로와 공기 관리의 중요성.',
         'does_not_support': '우리 고정 냉수 온도나 냉동기 COP의 검증'},
        {'id': 'PL16', 'title': 'Short-term data center load forecasting method based on data-physics hybrid driven framework',
         'path': 'D:/DC_modeling/outputs/literature_review_example/paper_list/[16] Short-term data center load forecasting method based on data-physics hybrid driven framework.pdf',
         'doi': '10.1016/j.egyr.2025.108953', 'checked_pdf_pages': [5],
         'supports': 'Eq. 8–9의 실내·외기 열수지 형태. Table 1은 해당 시설의 관측 통계.',
         'does_not_support': '현재 온도범위, 등가 열용량 1 MWh/K, UA 0.02 MW/K 또는 10초 응답 검증'},
        {'id': 'PL4', 'title': 'Dynamic coupling real-time energy consumption modeling for data centers',
         'path': 'D:/DC_modeling/outputs/literature_review_example/paper_list/[4] Dynamic coupling real-time energy consumption modeling for data centers.pdf',
         'doi': '10.1016/j.egyr.2022.06.085', 'checked_pdf_pages': [5],
         'supports': 'Eq. 8,10,12–15의 설비별 전력과 열 전달 경로.',
         'does_not_support': '현재 냉수 온도·코일 정격값·전체 온도 연계 검증'},
    ]
    contract = {
        'version': 'pcc_temperature_conditions_v1', 'created_utc': now,
        'status': 'temperature_definitions_and_consistency_review_complete_values_provisional',
        'legacy_model_or_parameters_changed': False, 'physical_temperature_validation_complete': False,
        'reference_config': 'configs/pcc_cooling_tes_candidate_v1.json',
        'definitions': [dict(zip(('key', 'label_ko', 'current_value', 'meaning_ko', 'evidence_status_ko'), row)) for row in rows],
        'documented_decisions': [
            'Separate equivalent room state, equipment inlet temperature, TES tank temperature and HX outlet temperature.',
            'Retain all existing numbers solely for reproducible scenario comparisons; do not label them manufacturer-selected or universally recommended.',
            'Treat 6 C source and 12 C return as TES boundary assumptions, not simulated whole-loop temperatures.',
            'Do not equate TES branch outlet with coil inlet: series/parallel connection and mixing are not explicitly modeled.',
            'Report reference inventory, temperature-window usable energy and output-dependent usable energy separately.',
            'Do not use equivalent room thermal inertia as a validated flexibility resource.',
            'No finite-reserve claim at 2 MWh usable energy: current reserve applies to reference inventory.',
        ],
        'equations': {
            'C_tank_MWh_per_K': 'rho_kg_m3 * V_m3 * cp_J_kg_K / 3.6e9',
            'T_TES_out_C': 'T_return - epsilon * (T_return - T_tank)',
            'T_tank_supply_cutoff_C': 'T_return - (T_return - T_out_max) / epsilon',
            'Q_charge_flow_limit_MWth': 'epsilon * mdot_max * cp / 1e6 * (T_tank - T_source)',
            'Q_discharge_flow_limit_MWth': 'epsilon * mdot_max * cp / 1e6 * (T_return - T_tank)',
            'E_reference_MWhth': 'C_tank * (T_reference - T_tank)',
            'E_usable_at_Q_MWhth': 'C_tank * max(0, min(T_supply_cutoff, T_return-Q/K_HX, T_reference-E_reserve/C_tank)-T_initial)',
        },
        'connection_limits': {
            'room_temperature_is_server_inlet': False,
            'coil_capacity_depends_on_supply_temperature': False,
            'whole_loop_water_flow_or_mixing_solved': False,
            'chiller_cop_depends_on_temperature': False,
            'humidity_latent_heat_condensation_solved': False,
            'tank_stratification_solved': False,
        },
        'next_minimal_step': {
            'task': 'Define the chiller-TES-coil connection and a design-point/UA-based aggregate coil capability before claiming a physically justified coil supply-temperature ceiling.',
            'not_required': ['CFD', 'refrigerant cycle dynamics', 'pipe-by-pipe hydraulic design', 'server chip hotspot dynamics'],
            'sizing_sequence': 'connection and temperature range -> requested PCC adjustment/duration -> usable thermal energy and volume -> joint feasibility check',
        },
        'reporting_correction': 'A mixed tank is a simple topology assumption, not universally conservative relative to stratified storage.',
    }
    dump('temperature_contract.json', contract)
    dump('reference_register.json', {'created_utc': now, 'references': refs})
    cap = c['tes']['volume_m3']*c['tes']['water_density_kg_m3']*c['tes']['water_cp_j_kg_k']/3.6e9
    t = c['tes']; k = t['hx_effectiveness']*t['maximum_flow_kg_s']*t['water_cp_j_kg_k']/1e6
    cutoff = t['load_return_c']-(t['load_return_c']-t['maximum_discharge_supply_c'])/t['hx_effectiveness']
    sensitivity = []
    for limit in (8., 9., 10.):
        top = t['load_return_c']-(t['load_return_c']-limit)/t['hx_effectiveness']
        sensitivity.append((f'{limit:.0f}', f'{top:.3f}', f'{max(0., cap*(top-t["initial_c"])):.3f}'))
    reference_html = ''.join('<li><b>'+html.escape(r['id']+' — '+r['title'])+'</b><br>'
        +html.escape(r['supports'])+'<br><span class="muted">직접 뒷받침하지 않는 부분: '
        +html.escape(r['does_not_support'])+'</span><br><a href="'
        +html.escape(r.get('url', 'https://doi.org/'+r.get('doi','')), quote=True)+'">원문·공식 자료</a></li>' for r in refs)
    assert verification['original_files_unchanged']
    assert all(row['passed'] for row in verification['independent_assertions'])
    assert abs(cap - numeric['tank_heat_capacity_mwh_per_k']) < 1e-10
    assert abs(cutoff - numeric['maximum_tank_c_from_discharge_supply']) < 1e-10
    for row, computed in zip(sensitivity, numeric['maximum_supply_temperature_sensitivity']):
        assert abs(float(row[2]) - computed['ideal_temperature_window_energy_mwh_th']) < 0.00051
    audit = numeric['existing_enabled_tes_profile_audit']
    audit_rows = []
    for label, key, unit in [
        ('탱크 수온', 'tank_temperature_c_including_start_and_end', '℃'),
        ('방전 중 TES 출구온도', 'discharge_supply_c_active_discharge_only', '℃'),
        ('충전 중 TES 분기 유량', 'tes_branch_flow_kg_s_active_charge_only', 'kg/s'),
        ('방전 중 TES 분기 유량', 'tes_branch_flow_kg_s_active_discharge_only', 'kg/s'),
        ('기준온도 대비 저장량', 'stored_inventory_mwh_th_including_start_and_end', 'MWhth'),
    ]:
        audit_rows.append((label, f'{audit[key]["minimum"]:.3f}', f'{audit[key]["maximum"]:.3f}', unit))
    page = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>서버실·냉수·TES 온도 조건 정리</title><style>
body{{font-family:"Malgun Gothic",Arial,sans-serif;line-height:1.75;color:#20374a;max-width:1180px;margin:32px auto;padding:0 24px}}h1{{font-size:28px}}h2{{font-size:22px;margin-top:32px;border-bottom:1px solid #cadbe3;padding-bottom:7px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;border-bottom:1px solid #dce5eb;text-align:left;vertical-align:top}}th{{background:#eaf4f6}}.scroll{{overflow:auto}}.note{{padding:16px;border-left:4px solid #ae7b24;background:#fff6e4}}.eq{{padding:12px;background:#eef4f8;font-family:Cambria,serif}}.muted{{color:#657887}}a{{color:#076b89}}li{{margin:7px 0}}.flow{{display:flex;gap:10px;flex-wrap:wrap;align-items:center}}.box{{padding:12px;border:1px solid #86b5c4;border-radius:6px;background:#f1f8fa;max-width:300px}}code{{overflow-wrap:anywhere}}
</style></head><body><h1>서버실·냉수·TES 온도 조건 정리</h1>
<p class="muted">정의·근거·정합성 검토 / 기존 수치 유지 / 실제 설비 검증 및 최종 온도 선정 아님</p>
<div class="note"><b>핵심 결론:</b> 온도마다 적용 위치와 역할을 분리했다. 기존 온도값은 시나리오 가정으로 유지한다. TES 출구 온도·유량 제약은 계산하지만, 그 냉수가 실제 공조 코일에서 필요한 열을 제거하는지까지는 아직 연결하지 않았다.</div>
<h2>1. 무엇을 수행했는가</h2><ul><li>기존 코드에서 실내·TES 온도의 입력, 상태, 제약, 계산 기준을 추적.</li><li>ASHRAE 장비 흡입구 기준과 등가 실내온도를 구분하고, 기존 문헌의 실제 지원 범위를 재확인.</li><li>충·방전 온도차·유량 제약과 사용 가능 냉열량을 독립 계산.</li><li>저장된 기존 10초 실행 배열의 실제 온도·유량을 감사. 전 기간 시뮬레이션은 새로 수행하지 않음.</li><li>별도 정리·검증 파일만 생성. 원본 모델·설정·이전 결과는 보존.</li></ul>
<h2>2. 온도 정의표</h2>{table(['변수','항목','현재 값','의미','판정'],rows)}
<h2>3. 현재 계산되는 연결과 계산되지 않는 연결</h2>
<div class="flow"><div class="box">IT 발열 → 등가 실내 열수지<br>24 ℃ 목표 / 22–27 ℃ 검사</div><span>← 열량 수지 →</span><div class="box">공조 코일 냉각량<br>열부하·용량 한계로 산정</div><span>← 열량 수지 →</span><div class="box">냉동기 + TES<br>TES 온도·유량·출구 제한</div></div>
<p><b>물의 온도와 코일 냉각능력의 연계는 별도 가정이다.</b> TES에서 계산한 열량은 전체 열수지에 연결되지만, 코일 UA·입수온도·공기/물 유량에 따른 냉각능력은 없다. 10 ℃는 TES 분기 HX 출구 상한이지, 혼합 이후 전체 코일 입구를 계산한 결과가 아니다. 500 kg/s도 전체 계통 유량이 아니다.</p>
<p>그러므로 기존 ‘온도 위반 없음’은 선언된 모델 내 제약 통과이며, 서버 흡입구·핫스폿·제습·결로 검증이 아니다. 6 ℃ 저온원에서 COP 5가 유지되는 장비 성능도 별도 미검증이다.</p>
<h2>4. 문헌으로 확인한 것과 가정인 것</h2><ul><li>ASHRAE의 권고범위는 장비 흡입구 기준: A1–A4 18–27 ℃, H1 18–22 ℃. H1 적용은 제조사 판단이다. 현 실내 상태값을 이 기준에 직접 대입하지 않는다.</li><li>현재 24 ℃·22–27 ℃·냉수 6/12 ℃·TES 7.5/10 ℃는 논문에서 일괄 차용한 값이 아니다.</li><li>PL16 Eq. 8–9는 열수지 형태의 근거이며, PL4는 설비별 전력·열 경로의 근거다. 두 문헌이 현재 파라미터나 전체 온도 적합성을 검증하지 않는다.</li><li>EnergyPlus는 혼합 탱크 및 코일 계산의 근거를 제공한다. 구체적인 장비 설정값은 별도 선택·가정이다.</li></ul>
<h2>5. TES 온도차와 공급 가능량</h2>
<div class="eq">T<sub>TES,out</sub> = 12 − 0.95(12 − T<sub>tank</sub>)<br>T<sub>tank,cutoff</sub> = 12 − (12 − 10)/0.95 = {cutoff:.6f} ℃<br>K<sub>HX</sub> = ε ṁ<sub>max</sub> c<sub>p</sub> = {k:.4f} MWth/K</div>
<ul><li>탱크 초기 7.5 ℃일 때 부하측 출구는 7.725 ℃이다. 탱크와 출구 온도를 같게 놓지 않는다.</li><li>충전 유량한계: Q<sub>ch</sub> ≤ K<sub>HX</sub>(T<sub>tank</sub>−6). 7.5 ℃에서 최대 <b>{k*(7.5-6):.5f} MWth</b>이다. 냉동기 여유·정격 제한은 추가로 적용된다.</li><li>5 MWth 충전을 전량 전달하려면 탱크가 최소 {6+5/k:.6f} ℃이어야 한다. 충전 후반에는 온도차가 작아져 출력이 감소한다.</li><li>5 MWth 일정 방전은 유량 기준 탱크 {12-5/k:.6f} ℃까지만 가능하다. 공급온도 상한의 {cutoff:.6f} ℃보다 먼저 감발될 수 있다.</li><li>위 수치는 정적 한계이며, 코드의 10초 구간 끝점 검사·열손실·펌프·냉동기 조건을 모두 대체하지 않는다.</li></ul>
<h2>6. 저장량과 사용 가능한 냉열 분리</h2>
<ul><li>3,000 m³의 열용량: <b>{cap:.6f} MWhth/K</b>.</li><li>12 ℃ 기준 초기 저장량: <b>{cap*(12-7.5):.3f} MWhth</b>.</li><li>초기 7.5 ℃에서 공급온도 한계까지의 무손실 온도창: <b>{cap*(cutoff-7.5):.6f} MWhth</b>.</li><li>5 MWth 정출력에 필요한 유량 한계까지의 무손실 온도창: <b>{cap*(12-5/k-7.5):.6f} MWhth</b>. 출력이 커지면 동일 탱크라도 정출력으로 쓸 수 있는 양이 줄어든다.</li><li>공급온도 한계에 도달해도 기준 저장량은 {cap*(12-cutoff):.6f} MWhth 남는다. 기존 reserve 2 MWhth는 이 조건에서 먼저 걸리는 제한이 아니며, 사용 가능한 냉열 2 MWhth 예비량을 보장하지 않는다.</li></ul>
<p>TES 출구 허용온도만 바꾼 정적 비교(다른 입력 유지, 새 온도 선정 아님):</p>
{table(['TES 출구 상한 (℃)','방전가능 탱크온도 상한 (℃)','초기 7.5 ℃부터 무손실 온도창 (MWhth)'],sensitivity)}
<p class="note">이 표에서 10 ℃가 가장 큰 저장량을 주더라도, 이를 이유로 10 ℃를 선택할 수는 없다. 실제 코일 냉각능력이 허용하는 온도여야 한다. 표는 고정 출력 지속시간이나 현실의 가용 에너지를 보장하지 않는다.</p>
<h2>7. 현재 결정과 다음 최소 보완</h2><ul><li><b>이번에 정리한 것:</b> 각 온도의 정의·적용 경계·수치적 관계·근거 수준.</li><li><b>유지한 것:</b> 기존 온도값과 26 MWth/COP 5, 3,000 m³, 3 MWth 시간표. 최종 시설값으로 선정한 것은 아님.</li><li><b>다음 우선순위:</b> 냉동기–TES–공조기 연결방식과 최소 코일 냉각능력 관계를 정의. 설계점 또는 UA 기반 축약 모델로 충분하며, CFD·냉매·배관 상세설계는 요구하지 않음.</li><li>그 후 사용 가능한 온도 범위를 기준으로 PCC 조정량·지속시간에 필요한 TES 체적을 역산.</li><li>근거 없이 등가 실내 열용량을 이용한 온도 유연성을 추가하지 않음. 혼합형이 항상 성층형보다 보수적이라고 단정하지 않음.</li></ul>
<h2>8. 연구 진행 보고용 개조식</h2><ul><li><b>서버실·냉수·TES 온도 조건 정비</b><ul><li>등가 실내온도, TES 저장수 온도, 열교환기 출구온도의 역할을 구분하고 기존 설정의 근거 수준 정리</li><li>ASHRAE 장비 흡입구 기준과 모델의 실내온도 제약을 구분하여 적용 범위 명확화</li><li>온도차·유량 제한을 반영하여 TES 정격출력과 실제 전달 가능 출력의 차이 확인</li><li>기준온도 대비 저장량과 냉각에 사용할 수 있는 냉열량을 구분하여 용량 산정 기준 보완</li><li>냉수 온도와 공조 코일 냉각능력의 연계가 별도 가정임을 확인하고 최소 보완 항목으로 정리</li></ul></li></ul>
<h2>9. 근거 자료</h2><ol>{reference_html}</ol>
<h2>10. 기존 실행 배열의 온도·유량 확인</h2>
{table(['항목','최소','최대','단위'],audit_rows)}
<p>기존 약 {audit['duration_h']:.3f}시간, {audit['sample_count']:,}개 10초 배열을 확인했다. 실내 등가 온도는 수치오차 수준에서 24 ℃로 유지됐고, 탱크는 종료 시 {audit['final_tank_c']:.1f} ℃로 복원됐다. 온도·TES 유량 위반은 0이다. 실제 서버 흡입구 온도나 공급수에 따른 코일 성능을 검증한 것은 아니다.</p>
<h2>11. 재현성과 검증</h2><p><a href="temperature_contract.json">온도 정의·적용범위</a> · <a href="numeric_checks.json">독립 계산 및 기존 배열 감사</a> · <a href="verification.json">검증 결과</a> · <a href="source_manifest.json">원본 해시</a> · <a href="reference_register.json">출처별 지원 범위</a></p>
<p>독립 검사 {verification['independent_assertion_count']}개 및 기존 단위시험 14개 통과. 보호한 기존 파일 {verification['original_file_count']}개 해시 동일. 수치 검사 통과는 실제 시설 검증과 다르다.</p></body></html>'''
    (OUT / 'report_ko.html').write_text(page, encoding='utf-8')
    dump('report_manifest.json', {
        'created_utc': now,
        'numeric_checks_sha256': hashlib.sha256((OUT/'numeric_checks.json').read_bytes()).hexdigest(),
        'verification_sha256': hashlib.sha256((OUT/'verification.json').read_bytes()).hexdigest(),
        'temperature_contract_sha256': hashlib.sha256((OUT/'temperature_contract.json').read_bytes()).hexdigest(),
        'report_sha256': hashlib.sha256((OUT/'report_ko.html').read_bytes()).hexdigest(),
        'source_reference_count': len(refs),
        'numeric_data_present': bool(numeric), 'verification_present': bool(verification),
    })
    print(str(OUT / 'report_ko.html'))


if __name__ == '__main__':
    main()
