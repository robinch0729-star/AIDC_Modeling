"""Audit source-to-PCC accounting without changing any legacy model or data.

Raw-channel diagnostics use native sample means in the 200 published windows.
They do not reproduce NLR's interpolated 1-ms output or recalibrate M3/R3.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "NLR_dataset"
OUT = ROOT / "outputs/it_pcc_boundary_audit_v1"
PCC = ROOT / "outputs/pcc_cooling_tes_candidate_v1"
IT = ROOT / "outputs/it_5min_operating_baseline_v1"
RAW = DATA / "00_raw_datasets/inference_online_rate_llama3_70b"
AGG = DATA / "01_aggregated_datasets/inference_online_rate_llama3_70b"
CPU_PATH = RAW / "rapl_wattameter_12146821-x3104c0s13b0n0.log"
GPU_PATH = RAW / "nvml_wattameter_12146821-x3104c0s13b0n0.log"
CONFIG = ROOT / "configs/pcc_cooling_tes_candidate_v1.json"
SOURCE_CODE = ROOT / "src/aidc_facility/pcc_cooling_tes.py"
CPU_COLS = ["cpu-0[W]", "cpu-0-core[W]", "cpu-1[W]", "cpu-1-core[W]"]
GPU_COLS = [f"gpu-{i}[mW]" for i in range(4)]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_raw(path, columns):
    with path.open(encoding="utf-8") as stream:
        next(stream)
        next(stream)
        names = next(stream).lstrip("# ").split()
    frame = pd.read_csv(path, sep=r"\s+", skiprows=3, names=names,
                        usecols=["timestamp", *columns], comment="#")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], format="%Y-%m-%d_%H:%M:%S.%f")
    frame = frame.set_index("timestamp")
    assert frame.index.is_monotonic_increasing
    return frame, names


def raw_diagnostics():
    cpu, cpu_header = read_raw(CPU_PATH, CPU_COLS)
    gpu, gpu_header = read_raw(GPU_PATH, GPU_COLS)
    gpu = gpu / 1000.0
    metadata = pd.read_csv(AGG / "metadata.csv", dtype={"id": str})
    assert len(metadata) == 200
    records = []
    for _, row in metadata.iterrows():
        a, b = pd.Timestamp(row["start_time"]), pd.Timestamp(row["end_time"])
        c, g = cpu.loc[a:b], gpu.loc[a:b]
        # Missing readings are unavailable, never zero. These diagnostic means
        # intentionally do not substitute for upstream interpolation/cleaning.
        cc, gg = c.dropna(), g.dropna()
        assert len(cc) and len(gg)
        cm, gm = cc.mean(), gg.mean()
        core = float(cm["cpu-0-core[W]"] + cm["cpu-1-core[W]"])
        parent = float(cm["cpu-0[W]"] + cm["cpu-1[W]"])
        gp = float(gm.sum())
        official = float(row["mean_power[W]"])
        records.append({
            "run_id": row["id"], "dataset": row["dataset-path"],
            "request_rate_rps": float(row["request_rate"]),
            "start": str(a), "end": str(b),
            "cpu_sample_count": len(c), "cpu_missing_power_rows": int(c.isna().any(axis=1).sum()),
            "gpu_sample_count": len(g), "gpu_missing_power_rows": int(g.isna().any(axis=1).sum()),
            "cpu_above_800w_values": int((cc > 800).to_numpy().sum()),
            "gpu_above_800w_values": int((gg > 800).to_numpy().sum()),
            "gpu_native_mean_w": gp, "cpu_parent_native_mean_w": parent,
            "cpu_core_native_mean_w": core, "upstream_total_mean_w": official,
            "core_share_of_upstream_mean_percent": 100.0 * core / official,
            "native_total_minus_upstream_mean_w": gp + parent + core - official,
        })
    largest = max(records, key=lambda row: row["core_share_of_upstream_mean_percent"])
    return {
        "method": "Native sample arithmetic means inside official 200 windows. Missing power rows excluded, not zero-filled. GPU mW converted to W. No upstream data changed. Diagnostic means do not reproduce the upstream 1-ms interpolation/backfill.",
        "cpu_header": cpu_header, "gpu_header": gpu_header,
        "runs": records,
        "run_count": len(records),
        "cpu_missing_power_rows_in_windows": sum(x["cpu_missing_power_rows"] for x in records),
        "gpu_above_800w_values_in_windows": sum(x["gpu_above_800w_values"] for x in records),
        "mean_of_run_core_means_w": float(np.mean([x["cpu_core_native_mean_w"] for x in records])),
        "maximum_run_mean_core_share_percent": largest["core_share_of_upstream_mean_percent"],
        "maximum_core_share_run": largest,
        "physical_core_overlap_resolved": False,
        "ac_server_measurement_available": False,
    }


def saved_array_checks(config):
    columns = json.loads((PCC / "internal_columns.json").read_text(encoding="utf-8"))
    data = np.load(PCC / "internal_profiles.npz", allow_pickle=False)
    cool_cols = ["chiller_mw", "fan_mw", "water_pump_mw", "tes_pump_mw", "condenser_pump_mw", "tower_fan_mw"]
    checks = {}
    means = {}
    for name in data.files:
        f = pd.DataFrame(data[name], columns=columns)
        e = config["electrical"]
        expected_ups = f.it_mw * (1 / e["ups_efficiency"] - 1)
        expected_cooling = f[cool_cols].sum(axis=1)
        expected_pcc = (f.it_mw / e["ups_efficiency"] + expected_cooling + f.auxiliary_mw) / e["transformer_efficiency"]
        expected_qch = f.coil_mw_th + f.water_pump_mw + f.tes_pump_mw + f.tes_charge_mw_th - f.tes_discharge_mw_th
        h = config["internal_step_s"] / 3600
        room_in = (f.it_mw + e["ups_heat_to_room_fraction"] * f.ups_loss_mw + f.auxiliary_mw + f.fan_mw + f.room_ambient_heat_mw_th)
        room_storage = config["room"]["effective_capacity_mwh_per_k"] * (f.room_end_c - f.room_start_c)
        cooling_sum_error = float(np.max(np.abs(f.cooling_mw - expected_cooling)))
        errors = {
            "pcc_formula_max_error_mw": float(np.max(np.abs(f.pcc_mw - expected_pcc))),
            "ups_branch_max_error_mw": float(np.max(np.abs(f.ups_loss_mw - expected_ups))),
            "cooling_sum_max_error_mw": cooling_sum_error,
            "water_balance_max_error_mw_th": float(np.max(np.abs(f.chiller_mw_th - expected_qch))),
            "chiller_cop_max_error_mw": float(np.max(np.abs(f.chiller_mw - f.chiller_mw_th / config["plant"]["cop"]))),
            "room_balance_max_error_mwh_th": float(np.max(np.abs(room_storage - h * (room_in - f.coil_mw_th)))),
            "condenser_rejection_max_error_mw_th": float(np.max(np.abs(f.heat_rejection_mw_th - f.chiller_mw_th - f.chiller_mw - f.condenser_pump_mw))),
        }
        assert all(value < 1e-8 for value in errors.values()), (name, errors)
        checks[name] = errors
        fields = ["it_mw", *cool_cols, "cooling_mw", "ups_loss_mw", "transformer_loss_mw", "auxiliary_mw", "pcc_mw", "coil_mw_th", "chiller_mw_th", "room_ambient_heat_mw_th", "heat_rejection_mw_th"]
        means[name] = {key: float(f[key].mean()) for key in fields}
        means[name]["pcc_to_it_proxy_energy_ratio"] = float(f.pcc_mw.sum() / f.it_mw.sum())
    return checks, means


def boundary_contract():
    return {
        "version": "it_pcc_boundary_audit_v1",
        "status": "boundary_audit_complete_legacy_scenario_retained_not_physical_ac_conversion",
        "legacy_models_modified": False,
        "source_signal": "NLR upstream-reported CPU/GPU aggregate; four NVML GPU fields and four RAPL parent/core fields",
        "source_signal_not": "Independently metered full-server AC input or facility PCC",
        "legacy_mapping": {
            "mode": "Identity numerical equivalence to modeled IT AC branch",
            "meaning": "Scenario assumption only; server fan/PSU inclusion is not a verified measurement fact",
            "physical_ac_conversion_validated": False,
            "missing_server_other_power_mw": None,
            "server_psu_efficiency": None,
            "rapl_disjoint_domain_correction": None,
        },
        "electrical_topology": {
            "transformer_supplies": ["UPS input", "cooling equipment", "facility auxiliary"],
            "ups_supplies": ["modeled IT branch only"],
            "cooling_bypasses_ups": True,
            "auxiliary_bypasses_ups": True,
            "pdu_cable_losses_explicitly_modeled": False,
            "ess_present": False,
        },
        "pcc_equation": "P_PCC=(P_IT_proxy/eta_UPS + P_cooling + P_aux)/eta_transformer",
        "cooling_equation": "P_cooling=P_chiller+P_CRAH_fan+P_CHW_pump+P_TES_pump+P_condenser_pump+P_tower_fan",
        "cop_boundary": "Q_chiller/P_chiller only; external fans/pumps/tower not included in denominator",
        "heat_destinations": {
            "IT_proxy": "equivalent room, fraction 1 by assumption",
            "UPS_losses": "equivalent room, configured fraction currently 1",
            "facility_auxiliary": "equivalent room, fraction 1 by assumption",
            "CRAH_fan": "equivalent room",
            "CHW_pump": "chilled-water loop, all electrical input assumed deposited",
            "TES_pump": "plant chilled-water loop, not separately duplicated in tank state",
            "chiller_electricity": "condenser-loop heat rejection",
            "condenser_pump": "condenser-loop heat rejection",
            "tower_fan": "ambient outside room/water states (implicit)",
            "transformer_losses": "ambient outside room/water states (implicit)",
        },
        "interpretation_corrections": {
            "legacy_pue_energy_ratio": "Keep numeric legacy field; interpret as PCC-to-IT-input-proxy energy ratio, not verified facility PUE",
            "legacy_heat_rejection_mw_th": "Condenser-loop rejection, not all facility heat including tower motor and transformer",
            "legacy_no_tes": "Storage subsystem absent/disabled; not a connected idle tank with standby leakage",
            "auxiliary_0_1_mw": "Facility auxiliary assumption; not identified missing-server-overhead correction",
        },
        "future_ac_adapter": "If supported component boundaries/overhead/PSU data are obtained, add a separately parameterized mapping before the existing thermal/PCC plant and rerun. Do not rescale final PCC or invent a percent uplift.",
        "allowed_claim": "NLR-based equivalent IT demand combined with a literature-informed cooling/storage and loss scenario",
        "not_claimed": ["measured full-server AC profile", "validated site PCC", "measured PUE", "proven lower bound on real demand", "resolved CPU parent/core overlap"],
    }


def report_html(raw, checks, means, verification):
    esc = html.escape
    labels = {
        "it_mw": "NLR 기반 IT 등가 입력", "chiller_mw": "냉동기", "fan_mw": "서버실 외부 공조 팬",
        "water_pump_mw": "냉수 펌프", "tes_pump_mw": "TES 펌프", "condenser_pump_mw": "냉각수 펌프",
        "tower_fan_mw": "냉각탑 팬", "cooling_mw": "냉각 전력 소계", "ups_loss_mw": "UPS 손실",
        "transformer_loss_mw": "변압기 손실", "auxiliary_mw": "기타 시설 보조부하", "pcc_mw": "PCC 합계",
    }
    component_rows = "".join(f"<tr><td>{label}</td><td>{means['no_tes'][key]:.6f}</td><td>{means['with_tes'][key]:.6f}</td></tr>" for key, label in labels.items())
    ledger = [
        ("GPU 4개 NVML 채널", "포함 확인", "GPU 계측 영역이며 전체 서버 AC는 아님. GPU 메모리 등 주변회로를 포함할 수 있으므로 HBM을 다시 일괄 추가하지 않음."),
        ("CPU 2개 parent 및 core 4채널", "포함 확인 / 중첩 미확정", "NLR 후처리는 4개 열을 모두 합산. 물리적 CPU 4개라는 뜻은 아님."),
        ("호스트 DRAM·NIC·보드·스토리지", "별도 포함 확인 불가", "별도 열과 소비전력 보정이 없음. 센서별 포함 범위가 달라 모두 완전히 누락됐다고 단정하지 않음."),
        ("서버 내부 팬·PSU 손실", "전체 포함 근거 없음", "후처리에 별도 보정 없음. 현재 IT=서버 AC라는 연결은 등가 시나리오 가정."),
        ("외부 냉각 팬·냉동기·펌프·냉각탑", "PCC 모델에서 추가", "서버 내부 팬과 다른 설비. COP는 냉동기만의 경계."),
        ("UPS 손실", "PCC 모델에서 추가", "IT 분기에만 적용. 냉각·시설 보조부하는 UPS를 우회하는 가정."),
        ("변압기 손실", "PCC 모델에서 추가", "IT+UPS 손실+냉각+시설 보조부하를 공급. UPS 손실과 중복이 아님."),
        ("시설 보조부하 0.1 MW", "연구 가정", "서버 누락분을 확인하여 채운 값이 아님. 그 용도로 재해석하지 않음."),
        ("PDU·케이블·배전반 손실", "독립 항목 없음", "현재 단순화 범위로 명시. 이미 포함됐다는 근거 없음."),
    ]
    ledger_rows = "".join("<tr>" + "".join(f"<td>{esc(x)}</td>" for x in row) + "</tr>" for row in ledger)
    core_rows = "".join(f"<tr><td>{'Code' if 'InstructCoder' in r['dataset'] else 'Conversation'}</td><td>{r['request_rate_rps']:.0f}</td><td>{r['gpu_native_mean_w']:.3f}</td><td>{r['cpu_parent_native_mean_w']:.3f}</td><td>{r['cpu_core_native_mean_w']:.4f}</td><td>{r['core_share_of_upstream_mean_percent']:.5f}%</td></tr>" for r in raw["runs"] if r["request_rate_rps"] in (1, 30, 50, 100))
    sources = [
        ("NLR 원자료 GPU 헤더", GPU_PATH, "3행"), ("NLR 원자료 CPU 헤더", CPU_PATH, "3행"),
        ("NLR 공식 후처리", AGG / "postprocess.py", "164–180행 합산, 160행 재표본화, 267행 이후 구간"),
        ("NLR 공통 재표본화 함수", DATA / "01_aggregated_datasets/utilities.py", "create_dataframe_multinode"),
        ("기존 IT 계측 경계 감사", ROOT / "docs/time_varying_it_model_audit.md", "1–2절"),
        ("PCC 설정", CONFIG, "electrical 및 assumptions"), ("PCC 실행 수식", SOURCE_CODE, "142–167행"),
    ]
    source_rows = "".join(f'<li><a href="{path.as_uri()}">{esc(label)}</a>: {esc(where)}</li>' for label, path, where in sources)
    no = means["no_tes"]
    return f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IT–PCC 계산 경계 검토 결과</title><style>
body{{font-family:"Malgun Gothic",Arial,sans-serif;color:#253447;max-width:1080px;margin:32px auto;padding:0 24px;line-height:1.75}}
h1{{font-size:29px;color:#153e62}}h2{{font-size:22px;margin-top:36px;border-bottom:1px solid #ccd5dd;padding-bottom:8px}}h3{{font-size:18px}}
table{{border-collapse:collapse;width:100%;font-size:14px;margin:14px 0}}th,td{{padding:9px 11px;border-bottom:1px solid #d9dfe4;text-align:left;vertical-align:top}}th{{background:#eef3f7}}
.number td:not(:first-child){{text-align:right;font-variant-numeric:tabular-nums}}.note{{background:#fff5de;border-left:4px solid #ba8618;padding:16px 20px}}.eq,pre{{background:#f3f6f8;padding:16px;overflow-wrap:anywhere;white-space:pre-wrap}}.muted{{color:#5f6e7d}}a{{color:#005fae}}.scroll{{overflow-x:auto}}@media(max-width:650px){{body{{padding:0 14px}}table{{font-size:13px}}}}
</style></head><body>
<h1>IT–PCC 계산 경계 검토 결과</h1>
<p>현재 NLR 기반 IT 생성기와 PCC·냉각·TES 후보모델의 포함 항목, 전기 공급 경로 및 열 발생 위치를 대조했다. 기존 데이터·모델·보정계수는 변경하지 않았다.</p>
<div class="note"><strong>핵심 결론</strong><br>기존 IT 프로파일은 CPU/GPU 계측 채널 기반의 등가 입력이다. 서버 전체 AC 입력전력으로 검증된 값은 아니다. PCC 수식은 선언된 가정 안에서 일관되지만, 두 경계를 동일시한 해석은 분리해야 한다. 이번 결과는 경계 감사 완료이며, 전체 서버 AC 변환이나 실측 시설 검증 완료를 뜻하지 않는다.</div>
<h2>1. 무엇이 포함되어 있는가</h2><div class="scroll"><table><tr><th>항목</th><th>판단</th><th>해석</th></tr>{ledger_rows}</table></div>
<p>CPU/GPU 외 서버 구성요소의 소비전력은 현재 자료만으로 수치화하지 않았다. 미확인 값을 0으로 확정하거나 임의의 5–10% 가산율을 적용하지 않았다.</p>
<h2>2. 실제 구현된 전기 공급 경로</h2>
<pre>PCC → 변압기(효율 0.99) → 시설 모선
                              ├─ UPS(효율 0.97) → IT 등가 입력
                              ├─ 냉각설비 (UPS 우회)
                              └─ 기타 시설 보조부하 0.1 MW (UPS 우회)</pre>
<div class="eq">P<sub>UPS,loss</sub> = P<sub>IT,proxy</sub> (1/η<sub>UPS</sub> − 1)<br>
P<sub>cool</sub> = P<sub>chiller</sub> + P<sub>CRAH fan</sub> + P<sub>CHW pump</sub> + P<sub>TES pump</sub> + P<sub>condenser pump</sub> + P<sub>tower fan</sub><br>
P<sub>PCC</sub> = (P<sub>IT,proxy</sub>/η<sub>UPS</sub> + P<sub>cool</sub> + P<sub>aux</sub>)/η<sub>tr</sub></div>
<p>변압기는 UPS 손실까지 공급하므로 변압기 손실 계산에 UPS 손실을 포함하는 것이 맞다. 최종 PCC에 PUE를 다시 곱하지 않는다. 위 η 값은 계측값이 아닌 현재 연구용 설정값이다.</p>
<h2>3. 전력 소비와 열 발생은 다른 장부이다</h2>
<table><tr><th>소비전력·손실</th><th>현재 열 전달 위치</th></tr>
<tr><td>IT, UPS 손실, 시설 보조부하, 외부 공조 팬</td><td>등가 서버실. UPS 손실과 보조부하의 100% 실내 유입은 연구 가정.</td></tr>
<tr><td>냉수·TES 펌프</td><td>냉수 계통. 전기입력 전체가 물의 열부하로 들어간다고 가정. TES 펌프 열을 탱크 상태에 다시 더하지 않음.</td></tr>
<tr><td>냉동기 전력, 냉각수 펌프</td><td>응축기·냉각탑 루프의 방출 열.</td></tr>
<tr><td>냉각탑 팬, 변압기 손실</td><td>모델의 실내·냉수 상태 바깥에서 방열하는 것으로 암묵적 처리.</td></tr></table>
<p><strong>펌프 전력을 PCC에 더하고 그 발열을 냉각부하에 더하는 것은 중복 계산이 아니다.</strong> 하나는 전력 소비이고 다른 하나는 그 결과로 생기는 열부하다. 단, 같은 열을 탱크와 냉수 루프 양쪽에 다시 넣어서는 안 된다.</p>
<div class="eq">Q<sub>chiller</sub> = Q<sub>coil</sub> + P<sub>CHW pump</sub> + P<sub>TES pump</sub> + Q<sub>ch</sub> − Q<sub>dis</sub><br>P<sub>chiller</sub> = Q<sub>chiller</sub>/COP</div>
<p>COP 5는 냉동기 단독의 경계다. 팬·펌프·냉각탑을 포함한 전체 냉각계통 COP가 아니다. <code>heat_rejection_mw_th</code>는 응축기 루프 방출 열이며 변압기·냉각탑 팬의 직접 방열까지 포함한 시설 전체 방출 열이 아니다.</p>
<h2>4. 기존 결과를 구성요소별로 다시 확인</h2>
<p>약 119.83시간의 기존 내부 출력 배열을 직접 읽어 아래 평균과 수식을 재확인했다. 새로운 보정 부하나 운전 시나리오를 만든 결과가 아니다.</p>
<div class="scroll"><table class="number"><tr><th>항목</th><th>TES 미사용 평균 MW</th><th>기존 TES 시간표 평균 MW</th></tr>{component_rows}</table></div>
<p>냉각 소계는 위의 냉각설비 행을 합한 값이므로, 총합을 계산할 때 소계와 세부 행을 중복해서 더하지 않는다.</p>
<p>TES 미사용 시 서버실 코일 제거 열은 {no['coil_mw_th']:.6f} MWth, 펌프 열을 포함한 냉동기 열부하는 {no['chiller_mw_th']:.6f} MWth다. 이를 COP 5로 나누면 냉동기 전력 {no['chiller_mw']:.6f} MW가 된다.</p>
<div class="note">기존 평균 PCC {no['pcc_mw']:.3f} MW는 현재 등가 입력·설비 가정에 따른 시나리오 값이다. 기존 PUE 필드 {no['pcc_to_it_proxy_energy_ratio']:.6f}는 이 단계에서는 <strong>PCC 에너지 / IT 입력 프록시 에너지 비율</strong>로 해석한다. 실제 서버 AC 경계가 정리되기 전에는 검증된 시설 PUE라고 주장하지 않는다. 숫자는 재현성을 위해 원본 그대로 보존했다.</div>
<h2>5. CPU parent/core 채널의 중첩 가능성</h2>
<p>NLR 후처리는 CPU parent와 core 채널을 모두 합한다. 현재 공개 WattAMeter 코드는 package 이름을 cpu로 바꾸고 하위 도메인 이름을 붙인다. 그러나 실험 당시 버전·sysfs 정의가 확보되지 않아, 이 기록의 물리적 중복량을 확정하거나 core 열을 임의 삭제하지 않았다.</p>
<p>공식 200개 분석 구간의 원시 채널을 직접 계산한 결과, 실행별 평균 core 합계가 공식 총전력 평균에서 차지하는 비중은 최대 <strong>{raw['maximum_run_mean_core_share_percent']:.4f}%</strong>였다. 이 수치는 관측 구간의 채널 크기 비교이며, 모델 전체 오차 상한이나 서버 AC 보정계수가 아니다.</p>
<div class="scroll"><table class="number"><tr><th>서비스</th><th>req/s</th><th>GPU W</th><th>CPU parent W</th><th>CPU core W</th><th>core / 공식 평균</th></tr>{core_rows}</table></div>
<p class="muted">원시 표본의 산술평균이며 1 ms 보간 후 공식 결과와 비트 단위로 동일하지 않다. 선택 구간에 RAPL 전력값이 없는 행 {raw['cpu_missing_power_rows_in_windows']}개를 발견하여 0이 아닌 결측으로 제외했다. GPU 800 W 초과 값 {raw['gpu_above_800w_values_in_windows']}개는 이 원시 진단에서 별도 기록했으며, 공식 후처리의 보간·backfill과 구분했다.</p>
<h2>6. 이번 작업에서 정리한 처리 원칙</h2>
<ol><li>기존 M3/R3·IT 파형·instance 수·PCC 결과는 그대로 보존한다.</li>
<li>NLR 측정 영역, 등가 IT 입력 가정, 냉각·손실 모델을 명시적으로 구분한다.</li>
<li>미확인 서버 부하·PSU 효율·RAPL 중첩 보정량은 <code>null</code>로 기록한다. 0 또는 임의 비율로 채우지 않는다.</li>
<li>현재 계산을 유지하려면 ‘NLR 기반 등가 IT 부하를 사용한 운영 연구 시나리오’로 설명한다.</li>
<li>실제 전체 서버 AC 전력으로 연결하려면 근거가 있는 별도 변환층을 추가한다. 이 경우 서버 발열·냉각·UPS·변압기 손실까지 다시 계산한다.</li>
<li>보정량이 미확정이므로 현재 PCC가 실제 시설 전력의 상한·하한이라고도 단정하지 않는다.</li></ol>
<p><strong>권고:</strong> 현재 연구 목적에서는 등가 시나리오임을 명시하고 기존 기본안을 보존할 수 있다. 절대 시설 전력과 PUE를 핵심 정량 주장으로 사용하려면 서버 AC 경계의 보정 근거가 추가로 필요하다. 다음 단계의 20 MW 규모 조정은 이 해석 선택 뒤에 진행한다.</p>
<h2>7. 이번에 직접 수행한 확인</h2>
<ul><li>공식 분석 구간 {raw['run_count']}개에 대한 원자료 채널 통계 재계산.</li>
<li>저장된 no_tes/with_tes/overcommand 배열에서 UPS·냉각·PCC 합산, 서버실·냉수 열수지, 냉동기 COP를 독립 재계산. 오차는 모두 10<sup>−8</sup> 미만(항목별 MW 또는 MWh).</li>
<li>기존 PCC 단위시험 재실행: {'통과' if verification['unit_test_return_code']==0 else '실패'}.</li>
<li>보호 대상 원본 {verification['protected_file_count']}개 SHA-256 전후 비교: 변경 없음.</li>
<li>실제 설비의 측정 검증 또는 서버 AC 보정모델 식별은 수행하지 않았다.</li></ul>
<h2>8. 근거와 저장 파일</h2><ul>{source_rows}
<li><a href="https://github.com/NatLabRockies/WattAMeter/blob/main/src/wattameter/readers/rapl.py">공식 WattAMeter RAPL 소스</a>: 현재 공개 소스, 실험 버전과 동일하다고 확정하지 않음.</li>
<li><a href="https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html">NVIDIA NVML 공식 문서</a>: GPU 및 연관 회로 전력의 API 경계.</li>
<li><a href="boundary_contract.json">경계 정의·해석 및 미확정 항목</a></li><li><a href="summary.json">수치 대조 결과</a></li><li><a href="verification.json">재검사 및 원본 보존 기록</a></li><li><a href="raw_window_channel_summary.json">200개 구간 채널 통계</a></li></ul>
</body></html>'''


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    calibration = ROOT / "outputs/r3_high_rate_benchmark"
    protected = [CPU_PATH, GPU_PATH, AGG / "postprocess.py", AGG / "metadata.csv", CONFIG, SOURCE_CODE,
                 ROOT / "src/nlr_data/final_it_profile_generator.py",
                 calibration / "train_validation_m3_parameters.csv", calibration / "train_validation_residual_parameters.csv"]
    protected += [p for folder in (IT, PCC) for p in folder.rglob("*") if p.is_file()]
    protected = sorted(set(protected))
    before = {str(p): digest(p) for p in protected}
    raw = raw_diagnostics()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    checks, means = saved_array_checks(config)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    tests = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_pcc_cooling_tes.py", "-v"],
                           cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
    assert tests.returncode == 0, tests.stdout + tests.stderr
    after = {str(p): digest(p) for p in protected}
    assert before == after, "Protected original content changed"
    verification = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protected_file_count": len(protected), "all_protected_files_unchanged": before == after,
        "sha256": before, "fresh_saved_array_checks": checks,
        "unit_test_return_code": tests.returncode, "unit_test_output": tests.stdout + tests.stderr,
        "full_new_facility_simulation_run": False,
        "scope": "Fresh audit of raw channels and stored arrays plus unit tests; no AC-boundary calibration or site validation",
    }
    summary = {"status": "audit_complete_unresolved_ac_conversion", "component_means": means,
               "raw_channel_summary": {k: v for k, v in raw.items() if k != "runs"},
               "legacy_models_modified": False, "physical_server_ac_correction_identified": False}
    save_json("raw_window_channel_summary.json", raw)
    save_json("boundary_contract.json", boundary_contract())
    save_json("summary.json", summary)
    save_json("verification.json", verification)
    (OUT / "report_ko.html").write_text(report_html(raw, checks, means, verification), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "runs": raw["run_count"], "max_core_share_percent": raw["maximum_run_mean_core_share_percent"],
                      "baseline_pcc_mw": means["no_tes"]["pcc_mw"], "protected_files": len(protected), "tests_passed": tests.returncode == 0}, indent=2))


if __name__ == "__main__":
    main()
