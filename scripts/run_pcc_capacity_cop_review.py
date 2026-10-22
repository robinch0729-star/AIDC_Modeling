"""Read-only-to-existing-data capacity / COP / fair TES comparison review.

Only this script and outputs/pcc_capacity_cop_review_v1 are new artifacts.
No IT scaling, fitted calibration, hardware selection or source-model changes.
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
import time

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aidc_facility.pcc_cooling_tes import simulate, summarize, aggregate, tank_capacity, maximum_discharge_tank_temperature

OUT = ROOT / "outputs/pcc_capacity_cop_review_v1"
CONFIG = ROOT / "configs/pcc_cooling_tes_candidate_v1.json"
ORIGINAL = ROOT / "outputs/pcc_cooling_tes_candidate_v1"


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
    paths = {CONFIG, ROOT / c["it_input"], ROOT / "src/aidc_facility/pcc_cooling_tes.py",
             ROOT / "tests/test_pcc_cooling_tes.py"}
    for folder in (ORIGINAL, ROOT / "outputs/it_5min_operating_baseline_v1"):
        paths.update(path for path in folder.rglob("*") if path.is_file())
    return sorted(paths)


def case_metrics(frame, c, enabled):
    m = summarize(frame, c, enabled)
    h = c["internal_step_s"] / 3600
    requested_charge = np.maximum(-frame.tes_command_mw_th.to_numpy(), 0.)
    requested_discharge = np.maximum(frame.tes_command_mw_th.to_numpy(), 0.)
    charge_shortfall = np.maximum(requested_charge - frame.tes_charge_mw_th.to_numpy(), 0.)
    discharge_shortfall = np.maximum(requested_discharge - frame.tes_discharge_mw_th.to_numpy(), 0.)
    m.update({
        "cop": c["plant"]["cop"],
        "chiller_max_thermal_mw_th": float(frame.chiller_mw_th.max()),
        "chiller_max_utilization_fraction": float(frame.chiller_mw_th.max() / c["plant"]["chiller_capacity_mw_th"]),
        "chiller_min_headroom_mw_th": float(c["plant"]["chiller_capacity_mw_th"] - frame.chiller_mw_th.max()),
        "tes_requested_charge_mwh_th": float(requested_charge.sum() * h),
        "tes_requested_discharge_mwh_th": float(requested_discharge.sum() * h),
        "tes_curtailed_charge_mwh_th": float(charge_shortfall.sum() * h),
        "tes_curtailed_discharge_mwh_th": float(discharge_shortfall.sum() * h),
        "tes_charge_curtailed_internal_steps": int((charge_shortfall > 1e-6).sum()),
        "tes_discharge_curtailed_internal_steps": int((discharge_shortfall > 1e-6).sum()),
        "tes_command_definition": "Actual controller request after charge-to-target logic; not a constant charge nameplate multiplied by the entire clock window.",
    })
    return m


def compare(case, reference, c, storage_state_comparable=True):
    case_agg = aggregate(case, c)
    case5 = case_agg.pcc_mw.to_numpy()
    reference5 = aggregate(reference, c).pcc_mw.to_numpy()
    delta = case5 - reference5
    discharge = case_agg.tes_discharge_mw_th.to_numpy() > 1e-6
    h = c["internal_step_s"] / 3600
    return {
        "max_same_time_5min_pcc_reduction_mw": float(max(0., -delta.min())),
        "max_same_time_5min_pcc_increase_mw": float(max(0., delta.max())),
        "reduction_at_elapsed_h": float(np.argmin(delta) * c["report_step_s"] / 3600),
        "increase_at_elapsed_h": float(np.argmax(delta) * c["report_step_s"] / 3600),
        "pcc_energy_change_mwh": float((case.pcc_mw.sum() - reference.pcc_mw.sum()) * h),
        "pcc_5min_peak_change_mw": float(case5.max() - reference5.max()),
        "max_same_time_pcc_reduction_during_case_discharge_mw": float(max(0., -delta[discharge].min())) if discharge.any() else None,
        "storage_state_comparison_applicable": storage_state_comparable,
        "equal_initial_tank_energy": bool(np.isclose(case.tes_energy_start_mwh_th.iloc[0], reference.tes_energy_start_mwh_th.iloc[0], atol=1e-9, rtol=0)) if storage_state_comparable else None,
        "final_tank_energy_difference_mwh_th": float(case.tes_energy_end_mwh_th.iloc[-1] - reference.tes_energy_end_mwh_th.iloc[-1]) if storage_state_comparable else None,
        "storage_state_note": "Compare storage states only when both cases have an installed tank. No-TES arrays retain computational placeholder tank values, not a physically installed storage state.",
        "peak_note": "Difference of global peaks is separate from maximum same-time demand reduction; no optimal peak shaving is claimed.",
        "terminal_recovery_note": "A whole-period maximum difference against connected idle may occur during its terminal recharge, not during scheduled discharge. Use the discharge-specific metric when discussing delivered flexibility.",
    }


def headroom_metrics(base, c):
    hours = base.time_s.to_numpy() / 3600
    phase = hours % 24
    start, end = c["dispatch"]["charge_hours_elapsed_mod24"]
    ordinary = (phase >= start) & (phase < end)
    recovery = hours >= len(base) * c["internal_step_s"] / 3600 - c["dispatch"]["terminal_recovery_hours"]
    masks = {"all_internal_steps": np.ones(len(base), dtype=bool),
             "ordinary_charge_clock_windows": ordinary,
             "terminal_recovery_window": recovery,
             "ordinary_or_terminal_charge_windows": ordinary | recovery}
    q = base.chiller_mw_th.to_numpy()
    cap = c["plant"]["chiller_capacity_mw_th"]
    t = c["tes"]
    pump_bound = t["maximum_flow_kg_s"] * 9.81 * t["pump_head_m"] / t["pump_efficiency"] / 1e6
    scopes = {}
    for name, mask in masks.items():
        head = cap - q[mask]
        scopes[name] = {
            "internal_steps": int(mask.sum()),
            "duration_h": float(mask.sum() * c["internal_step_s"] / 3600),
            "base_chiller_thermal_max_mw_th": float(q[mask].max()),
            "minimum_headroom_mw_th": float(head.min()),
            "mean_headroom_mw_th": float(head.mean()),
            "extra_charge_tests": [{
                "hypothetical_charge_mw_th": charge,
                "tes_pump_heat_conservative_bound_mw": pump_bound,
                "minimum_remaining_capacity_mw_th": float((head - charge - pump_bound).min()),
                "all_steps_pass_chiller_headroom_test": bool(np.all(head - charge - pump_bound >= -1e-8)),
                "capacity_shortfall_steps": int((head - charge - pump_bound < -1e-8).sum()),
            } for charge in (3., 5.)],
        }
    return {"chiller_rating_mw_th": cap, "tes_pump_upper_bound_mw": pump_bound, "scopes": scopes,
            "interpretation": "Chiller thermal-capacity-only sufficient-condition check using the maximum TES pump heat. Not proof of deliverable TES charge: tank temperature, flow, HX approach, storage bounds and duration still limit actual operation. Uses no-TES coil demand as the common counterfactual."}


def independently_check(frame, c, enabled):
    e, r, p, t = [c[key] for key in ("electrical", "room", "plant", "tes")]
    h = c["internal_step_s"] / 3600
    cooling = frame.fan_mw + frame.water_pump_mw + frame.tes_pump_mw + frame.chiller_mw + frame.condenser_pump_mw + frame.tower_fan_mw
    equations = {
        "ups_mw": frame.ups_loss_mw - frame.it_mw * (1 / e["ups_efficiency"] - 1),
        "cooling_sum_mw": frame.cooling_mw - cooling,
        "pcc_mw": frame.pcc_mw - (frame.it_mw / e["ups_efficiency"] + cooling + e["auxiliary_mw"]) / e["transformer_efficiency"],
        "cop_mw": frame.chiller_mw - frame.chiller_mw_th / p["cop"],
        "water_mw_th": frame.chiller_mw_th + frame.tes_discharge_mw_th - frame.tes_charge_mw_th - frame.coil_mw_th - frame.water_pump_mw - frame.tes_pump_mw,
        "room_mwh_th": r["effective_capacity_mwh_per_k"] * (frame.room_end_c - frame.room_start_c) - (frame.it_mw + e["ups_heat_to_room_fraction"] * frame.ups_loss_mw + e["auxiliary_mw"] + frame.fan_mw + frame.room_ambient_heat_mw_th - frame.coil_mw_th) * h,
        "tank_state_mwh_th": frame.tes_energy_end_mwh_th - tank_capacity(t) * (t["reference_warm_c"] - frame.tank_end_c),
    }
    if enabled:
        equations["tank_balance_mwh_th"] = frame.tes_energy_end_mwh_th - frame.tes_energy_start_mwh_th - (frame.tes_charge_mw_th - frame.tes_discharge_mw_th - frame.tes_heat_leak_mw_th) * h
    return {name: float(np.abs(value).max()) for name, value in equations.items()}


def table(rows):
    if not rows:
        return ""
    keys = list(rows[0])
    def fmt(v):
        if v is None:
            return "N/A"
        return f"{v:.6f}" if isinstance(v, float) else str(v)
    return "<table><thead><tr>" + "".join(f"<th>{html.escape(k)}</th>" for k in keys) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(fmt(row[k]))}</td>" for k in keys) + "</tr>" for row in rows) + "</tbody></table>"


def report(s, v):
    cases = s["cases"]
    tank = s["mixed_tank_capacity_interpretation"]
    case_rows = [{"시나리오": name, "COP": row["cop"], "평균 PCC(MW)": row["pcc_mean_mw"],
                  "5분 최대 PCC(MW)": row["pcc_5min_peak_mw"], "총 전력량(MWh)": row["pcc_energy_mwh"],
                  "최종 저장량 변화(MWh_th)": row["tes_energy_change_mwh_th"] if row["with_tes"] else None, "제약 통과": row["feasible_within_declared_checks"]}
                 for name, row in cases.items()]
    charge_rows = [{"시나리오": name, "요청 충전(MWh_th)": row["tes_requested_charge_mwh_th"],
                    "실제 충전(MWh_th)": row["tes_charge_mwh_th"], "제한된 충전(MWh_th)": row["tes_curtailed_charge_mwh_th"],
                    "요청 방전(MWh_th)": row["tes_requested_discharge_mwh_th"], "실제 방전(MWh_th)": row["tes_discharge_mwh_th"],
                    "제한된 방전(MWh_th)": row["tes_curtailed_discharge_mwh_th"]}
                   for name, row in cases.items() if row["with_tes"]]
    head_rows = []
    for name, scope in s["chiller_capacity_headroom"]["scopes"].items():
        for test in scope["extra_charge_tests"]:
            head_rows.append({"평가 시간": name, "기준 냉각 여유(MW_th)": scope["minimum_headroom_mw_th"],
                              "가상 추가 충전(MW_th)": test["hypothetical_charge_mw_th"],
                              "최대 TES 펌프열 포함 후 최소 여유(MW_th)": test["minimum_remaining_capacity_mw_th"],
                              "냉동기 용량만 검사": test["all_steps_pass_chiller_headroom_test"]})
    comparison_rows = [{"비교": name, "동시각 최대 PCC 감소(MW)": row["max_same_time_5min_pcc_reduction_mw"],
                        "동시각 최대 PCC 증가(MW)": row["max_same_time_5min_pcc_increase_mw"],
                        "총 전력량 증가(MWh)": row["pcc_energy_change_mwh"], "전체 최대값 변화(MW)": row["pcc_5min_peak_change_mw"],
                        "두 경우 최종 저장량 차이(MWh_th)": row["final_tank_energy_difference_mwh_th"]}
                       for name, row in s["comparisons"].items()]
    doc = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PCC 냉동기 용량·COP·TES 비교조건 검토</title><style>body{{font-family:"Malgun Gothic",Arial,sans-serif;line-height:1.75;color:#22384a;max-width:1300px;margin:32px auto;padding:0 24px}}h2{{margin-top:32px;border-bottom:1px solid #cbd6df;padding-bottom:8px}}table{{border-collapse:collapse;font-size:13px;width:100%}}th,td{{padding:8px;border-bottom:1px solid #d7e1e8;text-align:left;vertical-align:top}}th{{background:#eaf2f6}}.note{{border-left:4px solid #b98631;background:#fff7e8;padding:16px}}.scroll{{overflow:auto}}a{{color:#006893}}code{{overflow-wrap:anywhere}}</style></head><body>
<h1>PCC 냉동기 용량·COP·TES 비교조건 검토</h1>
<div class="note">기존 모델·설정·IT 프로파일은 변경하지 않았다. 기존 IT 입력은 NLR 기반 등가 연구 시나리오이며 서버 전체 AC 실측이 아니다. 아래 결과는 가정한 설비의 수치적 운전 가능성과 비교조건 검토이며 실측 검증·최적화·최종 설비 선정 결과가 아니다.</div>
<h2>1. 이번에 실행한 범위</h2><ul><li>동일한 약 {cases['cop5_no_tes']['duration_h']:.3f}시간 IT 입력을 10초 평균으로 계산, 출력은 5분 평균.</li><li>COP 4·5·6 각각에서 TES 미설치와 기존 시간표 TES 운전을 전 기간 실행.</li><li>COP 5에서 같은 TES 설비를 유지하되 일일 충·방전 시간표를 끄고 마지막 4시간에 초기 저장상태로 복원하는 기준운전 추가.</li><li>냉동기 26 MW_th의 여유를 평가하고 3 및 5 MW_th 추가 충전 가능성을 냉동기 용량 관점에서 점검.</li><li>기존 COP 5 배열 재현, 전기·열수지 재계산, 기존 단위시험 수행 및 기존 파일 SHA-256 보존 확인.</li></ul>
<h2>2. COP 민감도와 시나리오별 결과</h2><div class="scroll">{table(case_rows)}</div><p>COP는 냉동기 열출력/소비전력의 고정 비율만 변경했다. 온도·부분부하 성능곡선이나 COP별 냉동기 용량 변화는 가정하지 않았다. 따라서 이 민감도는 실측 오차범위나 제조사별 성능 비교가 아니다. 실내온도 추종은 이상적으로 연속 조절되는 냉동기/코일 제어의 결과다.</p>
<h2>3. 냉동기 여유용량</h2><div class="scroll">{table(head_rows)}</div><p>추가 충전 시 최대 유량 500 kg/s, 양정 20 m, 효율 0.75로 계산한 TES 펌프전력 상한 {s['chiller_capacity_headroom']['tes_pump_upper_bound_mw']:.6f} MW를 열부하로 포함했다. 이 값은 보수적 용량 검사다. 통과하더라도 탱크 온도·열교환 온도차·유량·저장량 때문에 실제 충전 명령을 전부 수행하지 못할 수 있다.</p>
<h2>4. TES 요청과 실제 동작</h2><div class="scroll">{table(charge_rows)}</div><p>충전 요청량은 충전 시간대 내 목표온도에 도달하기 위한 실제 제어명령이다. 정격 3 MW_th를 충전 시간대 전체에 곱한 값이 아니다. 방전 제한량은 냉각 미공급과 구분한다. 부족한 TES 방전을 냉동기가 보충하면 서버실 냉각은 유지된다.</p><p>탱크 3,000 m³의 열용량은 {tank['effective_heat_capacity_mwh_th_per_k']:.6f} MWh_th/K이다. 초기 7.5°C에서 12°C 기준 저장량은 {tank['initial_reference_inventory_mwh_th']:.6f} MWh_th이나, 10°C 이하 공급수 및 HX 효율 0.95 조건에서 허용되는 탱크 온도는 {tank['maximum_tank_temperature_during_discharge_c']:.6f}°C이다. 따라서 초기 상태에서 이 온도까지의 사용가능 온도창은 열손실 제외 {tank['initial_temperature_usable_window_mwh_th_before_leaks']:.6f} MWh_th이다. 전체 기준 저장량과 실제 공급 가능량을 혼동하면 안 된다.</p>
<h2>5. 공정한 비교조건</h2><div class="scroll">{table(comparison_rows)}</div><ul><li><b>설비 효과:</b> TES 미설치 vs TES 설치·시간표 운전. 탱크 열손실·추가 펌프 부하까지 포함한 설비 도입 비교.</li><li><b>운영전략 효과:</b> 같은 TES 설비의 비시프트 기준운전 vs 시간표 운전. 동일 IT·외기·설비·초기 상태와 최종 저장량을 사용한 비교.</li><li>비시프트 기준도 최종 저장량 복원을 위해 전력을 사용한다. 따라서 연결된 탱크가 있어도 손실 없이 영구 대기하는 모델이 아니다.</li><li>동시각 5분 PCC 감소와 전체 기간 최대부하 감소를 분리했다. 사전 시간표가 최적 피크저감 또는 DSO 지령을 의미하지 않는다.</li><li>UPS 97%, 변압기 99%, 시설 보조부하 0.1 MW는 이번에 변경하지 않았다. 문헌/장비 자료로 선정해야 할 가정이며 이 비교만으로 검증된 값이 되지는 않는다.</li></ul>
<h2>6. 검증 및 한계</h2><ul><li>기존 파일 보존: {v['protected_files_unchanged']} ({v['protected_file_count']}개)</li><li>기존 COP 5 배열 재현: {v['baseline_arrays_reproduced']}</li><li>새 비교의 명시된 물리 제약 통과: {v['all_declared_constraints_passed']}</li><li>독립 수식 재계산 오차 1e-8 이하: {v['independent_equations_passed']}</li><li>기존 단위시험 성공: {v['unit_tests']['passed']}</li></ul><p>완전혼합 단일 탱크, 고정 COP·외기, 단일 실내온도 및 선형/세제곱 보조설비 모델에 한정된다. 실제 AI 데이터센터 설비 적합성, 서버 전체 전력 경계, grid code, 수리학 상세설계 및 최적 운영은 검증하지 않았다.</p>
<p>같은 탱크의 대기·복원 기준과 비교할 때 전체 기간 최대 감소 {s['comparisons']['cop5_scheduled_vs_connected_idle']['max_same_time_5min_pcc_reduction_mw']:.6f} MW는 기준운전의 마지막 복원충전 시점에 발생한다. 이를 TES 방전 성능으로 해석하지 않는다. <b>실제 방전시간만 비교한 PCC 최대 감소는 {s['comparisons']['cop5_scheduled_vs_connected_idle']['max_same_time_pcc_reduction_during_case_discharge_mw']:.6f} MW</b>이다. 탱크가 없는 비교에서 저장상태 차이는 N/A이며, 동일 저장상태 검사는 탱크가 설치된 두 경우 사이에만 적용한다.</p>
<h2>7. 손실 가정의 근거와 미확인 부분</h2><ul><li><a href="https://www.energystar.gov/products/uninterruptible_power_supplies/key_product_criteria">ENERGY STAR UPS 평가 기준</a>: 부하율별 효율 평가를 참고한다. 고정 97%를 입증하는 자료는 아니다.</li><li><a href="https://www.eaton.com/us/en-us/catalog/backup-power-ups-surge-it-power-distribution/eaton-93pm-480v-ups.html">Eaton 93PM 480 V</a>: 이중변환 효율 최대 97%를 제시하는 장비 사례. 우리 시설의 선정 장비도 아니고 모든 부하율에서 97%라는 뜻도 아니다.</li><li><a href="https://www.energy.gov/cmei/buildings/articles/gm0060-distribution-and-building-level-transformers">U.S. DOE 변압기 자료</a>: 무부하손과 부하 관련 손실을 구분하는 근거. 고정 효율 99%를 확인한 것은 아니다.</li></ul>
<p><a href="summary.json">전체 수치 요약</a> · <a href="comparison_contract.json">비교조건 계약 및 손실 reference</a> · <a href="verification.json">검증 상세와 시험 출력</a> · <a href="scenario_configs.json">독립 시나리오 설정</a> · <a href="profiles_5min.npz">5분 결과 배열</a> · <a href="profile_columns.json">배열 열 이름</a></p></body></html>'''
    (OUT / "report_ko.html").write_text(doc, encoding="utf-8")


def main():
    started = time.perf_counter()
    c = read(CONFIG)
    OUT.mkdir(parents=True, exist_ok=True)
    protected = protected_files(c)
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in protected}
    raw = np.load(ROOT / c["it_input"])["it_power_mw"][c["it_seed_index"]]
    it = raw.reshape(-1, c["internal_step_s"]).mean(axis=1)
    configs, frames, metrics = {}, {}, {}
    for cop in (4, 5, 6):
        for label, enabled in (("no_tes", False), ("scheduled_tes", True)):
            name = f"cop{cop}_{label}"
            alt = copy.deepcopy(c)
            alt["plant"]["cop"] = float(cop)
            configs[name] = alt
            frames[name] = simulate(it, alt, enabled)
            metrics[name] = case_metrics(frames[name], alt, enabled)
            print(name, "mean PCC MW", metrics[name]["pcc_mean_mw"], flush=True)
    idle = copy.deepcopy(c)
    idle["dispatch"]["charge_hours_elapsed_mod24"] = [0., 0.]
    idle["dispatch"]["discharge_hours_elapsed_mod24"] = [0., 0.]
    name = "cop5_connected_tes_idle_with_terminal_recovery"
    configs[name] = idle
    frames[name] = simulate(it, idle, True)
    metrics[name] = case_metrics(frames[name], idle, True)
    print(name, "mean PCC MW", metrics[name]["pcc_mean_mw"], flush=True)
    comparisons = {f"cop{cop}_scheduled_vs_no_tes": compare(frames[f"cop{cop}_scheduled_tes"], frames[f"cop{cop}_no_tes"], c, storage_state_comparable=False) for cop in (4, 5, 6)}
    comparisons["cop5_scheduled_vs_connected_idle"] = compare(frames["cop5_scheduled_tes"], frames[name], c)
    comparisons["cop5_connected_idle_vs_no_tes"] = compare(frames[name], frames["cop5_no_tes"], c, storage_state_comparable=False)
    old_arrays = np.load(ORIGINAL / "internal_profiles.npz")
    old_columns = read(ORIGINAL / "internal_columns.json")
    reproduction = {}
    for old, new in (("no_tes", "cop5_no_tes"), ("with_tes", "cop5_scheduled_tes")):
        actual = frames[new][old_columns].to_numpy(float)
        expected = old_arrays[old]
        errors = np.max(np.abs(actual - expected), axis=0)
        reproduction[new] = {"same_shape": actual.shape == expected.shape, "max_absolute_error": float(errors.max()),
                             "per_column_max_absolute_error": dict(zip(old_columns, [float(v) for v in errors])),
                             "allclose_atol_1e8_rtol_1e10": bool(np.allclose(actual, expected, atol=1e-8, rtol=1e-10))}
    independent = {name: independently_check(f, configs[name], metrics[name]["with_tes"]) for name, f in frames.items()}
    tests = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_pcc_cooling_tes.py", "-v"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    after = {str(path.relative_to(ROOT)): digest(path) for path in protected}
    verification = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "protected_file_count": len(hashes),
        "protected_files_unchanged": hashes == after, "protected_sha256_before": hashes, "protected_sha256_after": after,
        "baseline_arrays_reproduced": all(v["allclose_atol_1e8_rtol_1e10"] for v in reproduction.values()), "baseline_array_checks": reproduction,
        "all_declared_constraints_passed": all(v["feasible_within_declared_checks"] for v in metrics.values()),
        "independent_equations_passed": all(error < 1e-8 for check in independent.values() for error in check.values()),
        "independent_equation_errors": independent,
        "unit_tests": {"passed": tests.returncode == 0, "exit_code": tests.returncode, "stdout": tests.stdout, "stderr": tests.stderr},
        "it_energy_mwh_source": float(raw.sum() / 3600), "it_energy_mwh_10s": float(it.sum() * c["internal_step_s"] / 3600),
        "terminal_energy_recovery_passed": all(abs(v["tes_energy_change_mwh_th"]) < 1e-5 for v in metrics.values() if v["with_tes"]),
    }
    verification["all_checks_passed"] = all(verification[key] for key in ("protected_files_unchanged", "baseline_arrays_reproduced", "all_declared_constraints_passed", "independent_equations_passed", "terminal_energy_recovery_passed")) and verification["unit_tests"]["passed"]
    summary = {"created_utc": verification["created_utc"], "status": "research_candidate_sensitivity_and_fair_comparison_not_site_validation",
               "it_boundary": "Preserved NLR CPU/GPU-derived equivalent IT input; complete server AC measurement bridge remains unresolved.",
               "cases": metrics, "comparisons": comparisons, "chiller_capacity_headroom": headroom_metrics(frames["cop5_no_tes"], c),
               "comparison_definitions": {"no_tes": "Storage absent/disabled; no tank heat leakage.", "scheduled_tes": "Same original daily clock commands and terminal recovery.",
                                          "connected_idle": "Same TES hardware and initial state, no daily charge/discharge, heat leakage retained, final four-hour recovery to initial target."},
               "runtime_s": time.perf_counter() - started, "physical_facility_validated": False, "new_parameter_selection": False,
               "uncertainty_note": "COP 4/5/6 are research sensitivity settings, not empirical bounds or a confidence interval.",
               "unchanged_loss_assumptions": c["electrical"]}
    t = c["tes"]
    cap = tank_capacity(t)
    t_limit = maximum_discharge_tank_temperature(t)
    summary["mixed_tank_capacity_interpretation"] = {
        "volume_m3": t["volume_m3"], "effective_heat_capacity_mwh_th_per_k": cap,
        "initial_temperature_c": t["initial_c"], "maximum_tank_temperature_during_discharge_c": t_limit,
        "initial_reference_inventory_mwh_th": cap * (t["reference_warm_c"] - t["initial_c"]),
        "initial_temperature_usable_window_mwh_th_before_leaks": cap * (t_limit - t["initial_c"]),
        "minimum_temperature_usable_window_mwh_th_before_leaks": cap * (t_limit - t["minimum_c"]),
        "note": "Temperature-window energy is not a guaranteed continuous-output service; heat leakage, HX flow and requested discharge power can reduce deliverable energy. Reference inventory measured against 12 C is not all serviceable at a 10 C supply ceiling.",
    }
    contract = {
        "status": "proposed_comparison_protocol_executed_for_existing_candidate_not_final_hardware_selection",
        "operational_strategy_comparison": {"reference": "cop5_connected_tes_idle_with_terminal_recovery", "test": "cop5_scheduled_tes",
            "definition": "Same installed tank, chiller, fan, pumps, electrical losses and thermal parameters. Reference disables daily shift commands but retains heat leakage and the same final four-hour recovery to initial storage temperature.",
            "same_initial_tank_energy": comparisons["cop5_scheduled_vs_connected_idle"]["equal_initial_tank_energy"],
            "terminal_tank_energy_difference_mwh_th": comparisons["cop5_scheduled_vs_connected_idle"]["final_tank_energy_difference_mwh_th"],
            "interpretation": "Tests this preset shift schedule against this explicit no-shift recovery baseline; not an optimal no-shift or tariff-cost baseline."},
        "installation_comparison": {"reference": "cop5_no_tes", "test": "cop5_scheduled_tes",
            "definition": "No tank or storage heat leakage in reference, so differences include installation overhead as well as operation.",
            "storage_state_comparison_applicable": False,
            "storage_state_note": "No tank is installed in the reference. Placeholder tank values retained by the numerical simulator are not physical storage states; initial/final storage comparisons are N/A."},
        "held_identical": ["unscaled IT series and selected residual seed", "external and tank ambient temperatures", "thermal coefficients and temperature constraints", "electrical loss assumptions", "equipment sizes and pump/HX parameters", "10-second internal step and 300-second reporting", "initial room and tank state", "terminal recovery target and window for installed-TES cases"],
        "loss_assumptions": {"values": c["electrical"], "identified_for_this_facility": False,
            "meaning": "Fixed scenario efficiencies and auxiliary demand retained, not measured or vendor-calibrated. Literature supports efficiency definition, part-load evaluation and topology, not these exact constant values.",
            "references": [
                {"title": "ENERGY STAR UPS key product criteria", "url": "https://www.energystar.gov/products/uninterruptible_power_supplies/key_product_criteria", "use": "Part-load UPS efficiency evaluation; not proof of constant 97%."},
                {"title": "Eaton 93PM 480 V UPS", "url": "https://www.eaton.com/us/en-us/catalog/backup-power-ups-surge-it-power-distribution/eaton-93pm-480v-ups.html", "use": "Up-to-97% double-conversion efficiency as an equipment example only; not selected hardware or every-load efficiency."},
                {"title": "U.S. DOE: Distribution and Building-Level Transformers", "url": "https://www.energy.gov/cmei/buildings/articles/gm0060-distribution-and-building-level-transformers", "use": "Separate no-load and load-related transformer losses; not proof of constant 99%."},
            ]},
        "not_claimed": ["optimal dispatch", "operating cost reduction without tariff model", "measured facility validation", "server AC boundary correction", "all grid-code compliance", "full physical design of tank or cooling system"],
    }
    arrays = {name: aggregate(f, c).to_numpy(float) for name, f in frames.items()}
    np.savez_compressed(OUT / "profiles_5min.npz", **arrays)
    dump(OUT / "profile_columns.json", list(aggregate(frames["cop5_no_tes"], c).columns))
    dump(OUT / "scenario_configs.json", configs)
    dump(OUT / "comparison_contract.json", contract)
    dump(OUT / "summary.json", summary)
    dump(OUT / "verification.json", verification)
    report(summary, verification)
    print(json.dumps({"output": str(OUT), "all_checks_passed": verification["all_checks_passed"], "runtime_s": summary["runtime_s"], "comparisons": comparisons, "headroom": summary["chiller_capacity_headroom"]}, ensure_ascii=False), flush=True)
    if not verification["all_checks_passed"]:
        raise SystemExit("Review generated but one or more checks failed; inspect verification.json")


def refresh_reports():
    """Clarify report semantics using existing results, without simulation."""
    summary = read(OUT / "summary.json")
    verification = read(OUT / "verification.json")
    contract = read(OUT / "comparison_contract.json")
    for name, comparison in summary["comparisons"].items():
        applicable = not name.endswith("_vs_no_tes")
        comparison["storage_state_comparison_applicable"] = applicable
        comparison["storage_state_note"] = "Compare storage states only when both cases have an installed tank. No-TES arrays retain computational placeholder tank values, not a physically installed storage state."
        if not applicable:
            comparison["equal_initial_tank_energy"] = None
            comparison["final_tank_energy_difference_mwh_th"] = None
    contract["installation_comparison"].update({
        "storage_state_comparison_applicable": False,
        "storage_state_note": "No tank is installed in the reference. Placeholder tank values retained by the numerical simulator are not physical storage states; initial/final storage comparisons are N/A.",
    })
    before = verification["protected_sha256_before"]
    after = {name: digest(ROOT / name) for name in before}
    assert before == after, "A protected existing file has changed"
    stamp = datetime.now(timezone.utc).isoformat()
    summary["report_semantics_refreshed_utc"] = stamp
    verification["report_refresh"] = {"created_utc": stamp, "simulation_rerun": False, "protected_files_unchanged": before == after,
                                       "change": "No-TES storage comparison fields changed to null/N/A; installed-to-installed comparisons retained. Discharge-only PCC reduction explicitly shown in HTML."}
    dump(OUT / "summary.json", summary)
    dump(OUT / "verification.json", verification)
    dump(OUT / "comparison_contract.json", contract)
    report(summary, verification)
    print(json.dumps({"report_refresh": verification["report_refresh"], "comparisons": summary["comparisons"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    refresh_reports() if "--refresh-reports" in sys.argv else main()
