"""Korean report and scientific figures for the original-voltage IEEE33 test."""
from __future__ import annotations

import html
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ieee33_1266kv_screen_v1"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def fmt(value, n=3):
    return "미확인" if value is None else f"{value:.{n}f}"


def figure(d):
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.0), facecolor="white")
    colors = {"no_aidc": "#718497", "bus2_mean": "#087f8c", "bus2_peak": "#ba5d31"}
    labels = {"no_aidc": "Original background only", "bus2_mean": "AIDC mean load at Bus 2", "bus2_peak": "AIDC peak load at Bus 2"}
    for key, values in d["voltage_profiles"].items():
        axes[0].plot(range(1, 34), values, color=colors[key], label=labels[key], lw=1.8)
    axes[0].axhline(.95, c="#b63f4d", ls="--", label="Retained study lower bound: 0.95 p.u.")
    axes[0].set(xlabel="Bus number", ylabel="Voltage (p.u.)", xlim=(1, 33), ylim=(.87, 1.015))
    axes[0].set_title("Baseline undervoltage is not caused by the AIDC", loc="left", fontsize=11, weight="bold")
    for filename, label, color, ls in (("no_storage_replay.json", "Without storage", "#ba5d31", "-"),
                                        ("existing_joint_replay.json", "Existing ESS-TES schedule", "#087f8c", "--")):
        rows = read(filename)
        axes[1].plot([r["time_h"] for r in rows], [r.get("max_line_loading_percent", np.nan) for r in rows],
                     color=color, label=label, lw=1.5, ls=ls)
    axes[1].axhline(100, c="#b63f4d", ls=":", label="600 A study limit (not a source rating)")
    axes[1].set(xlabel="Elapsed time (h)", ylabel="Maximum line loading (%)", xlim=(0, 120), ylim=(85, 260))
    axes[1].set_title("Frozen five-minute PCC profiles at Bus 2", loc="left", fontsize=11, weight="bold")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.2)
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("IEEE 33-bus at 12.66 kV: unchanged AIDC demand and storage", fontsize=16, weight="bold", color="#163f59")
    fig.text(.5, .012, "Original topology / R-X / background retained | Constant background | No network-aware rescheduling or reinforcement", ha="center", fontsize=9, color="#536777")
    fig.tight_layout(rect=(0, .04, 1, .94))
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"network_test_summary.{ext}", dpi=190, bbox_inches="tight")
    plt.close(fig)


def main():
    d = read("results.json")
    cfg = read("model_contract.json")
    figure(d)
    b, s0, sj, bound = (d[k] for k in ("no_aidc_study_bounds", "no_storage_replay", "existing_joint_replay", "necessary_power_bound"))
    peak_rows = []
    for row in d["snapshots"]:
        if row["demand"] != "peak":
            continue
        result = "전압·선로 제약 위반" if row["converged"] and not row["feasible"] else "NR 미수렴: 정상상태 해 미확인"
        if row["feasible"]:
            result = "설정 제약 만족"
        peak_rows.append(f'<tr><td>Bus {row["bus"]}</td><td>{fmt(row.get("pcc_voltage_pu"),4)}</td><td>{fmt(row.get("min_voltage_pu"),4)}</td><td>{fmt(row.get("max_line_current_ka"),3)}</td><td>{result}</td></tr>')
    rows = "".join(f'<tr><td>{label}</td><td>{s["converged_intervals"]}/{s["intervals"]}</td><td>{s["converged_violation_intervals"]}</td><td>{s["unresolved_intervals"]}</td><td>{fmt(s["pcc_peak_mw"])}</td><td>{fmt(s["min_voltage_pu"],4)}</td><td>{fmt(s["max_line_loading_percent"],2)}%</td></tr>'
                   for label, s in (("무저장 기준", s0), ("기존 ESS-TES 공동계획", sj)))
    checks = {"protected_file_count": len(d["protected_input_hashes"]),
              "all_protected_unchanged": all(d["protected_inputs_unchanged"].values()),
              "nr_bfsw_max_voltage_difference_pu": d["independent_solver_check"]["maximum_voltage_difference_pu"],
              "maximum_power_balance_error_mw": max(s0["max_balance_error_mw"], sj["max_balance_error_mw"])}
    xml = OUT / "test_results.xml"
    if xml.exists():
        tree = ET.parse(xml).getroot()
        checks["pytest"] = tree.find("testsuite").attrib if tree.tag == "testsuites" else tree.attrib
    (OUT / "verification.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>IEEE 33-bus 12.66 kV 테스트 결과</title><style>
body{{font-family:'Malgun Gothic',Arial,sans-serif;color:#213d50;max-width:1150px;margin:32px auto;padding:0 24px;line-height:1.8}}h1{{font-size:29px}}h2{{font-size:22px;border-bottom:1px solid #ccdce5;margin-top:30px}}.note{{background:#edf6f7;padding:18px;border-left:5px solid #087f8c}}.warn{{background:#fff4e8;padding:18px;border-left:5px solid #ba5d31}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #dce6eb}}th{{background:#edf3f7}}img{{max-width:100%;height:auto}}pre{{background:#eef4f6;padding:16px;white-space:pre-wrap}}a{{color:#166b8b}}li{{margin:7px 0}}.small{{font-size:13px;color:#526b79}}
</style><h1>IEEE 33-bus: 12.66 kV 원형 전압 유지 테스트</h1>
<div class="note"><b>사용자 선택 반영: 주 테스트 계통은 IEEE 33-bus, 전압은 12.66 kV.</b><br>
기존 33 kV 후보는 보존하고 새로운 조건에서 AC 조류계산을 실행했다. AIDC 부하·ESS·TES·배경부하를 축소하거나 선로를 증설하지 않았다. <b>12.66 kV 선택과 기존 AIDC의 접속 가능성은 별개</b>이며, 이번 결과는 현재 조건의 부적합을 보여준다.</div>
<h2>1. 변경 및 유지 조건</h2><ul>
<li>전압: 직전 후보 33 kV에서 <b>원형 12.66 kV</b>로 변경. 33개 모선·32개 폐로 선로·5개 개방 tie, 물리적 R/X는 원형 유지.</li>
<li>원형 배경부하: <b>3.715 MW + 2.3 MVAr</b>, 시간불변. Bus1 전압은 1.0 p.u.</li>
<li>AIDC: 기본 PCC Bus2, 비교 Bus6·18. 무저장 평균 {s0['pcc_mean_mw']:.3f} MW, 최대 {s0['pcc_peak_mw']:.3f} MW. PF=0.98 지상은 연구 가정.</li>
<li>ESS: 1 MW / 4 MWh, TES: 3 MWth / 10 MWhth, 각 충전·방전 효율 0.9 유지.</li>
<li>전압 0.95–1.05 p.u., 선로 600 A, 상위 공급한계 40 MVA는 통제된 비교를 위해 유지한 <b>기존 연구 가정</b>. 600 A는 IEEE33 원문 선로 정격이 아니며 40 MVA는 모델링된 변압기 사양이 아님.</li>
<li>25.5 MW 외생 상한을 대상으로 만든 기존 공동운전 계획도 그대로 재생. 이 계통에 맞춘 재최적화가 아님.</li></ul>
<h2>2. AIDC를 넣기 전부터 존재하는 전압 문제</h2>
<p>원형 배경부하만으로 최저전압은 <b>{b['min_voltage_pu']:.4f} p.u. (Bus {b['min_voltage_bus']})</b>. 기존 0.95 하한보다 낮으므로 이 위반을 AIDC가 만든 것으로 계산하지 않는다. 정적 허용전력은 <b>기준운전 부적합으로 산정 보류(null)</b>이며, ‘수용량 0 MW’라고 단정하지 않는다.</p>
<p>원본 case의 0.9–1.1 전압 범위를 별도로 적용한 무AIDC 진단 결과: 제약 만족 {d['no_aidc_source_voltage_bounds_diagnostic']['feasible']}. 이는 출처 조건 비교일 뿐, 본 연구의 0.95 기준을 몰래 완화한 것이 아니다.</p>
<h2>3. AIDC 최대부하 연결 결과</h2>
<table><tr><th>위치</th><th>PCC 전압(p.u.)</th><th>계통 최저전압(p.u.)</th><th>최대 전류(kA)</th><th>판정</th></tr>{''.join(peak_rows)}</table>
<p>PCC 자체 전압만 보면 전체 계통의 저전압을 놓칠 수 있다. Bus6·18의 NR 미수렴은 물리적 전압붕괴 증명이 아니다. 전체 비슬랙 32개 모선에서도 같은 최대부하를 검사했으며, 설정 제약을 만족한 위치는 {sum(r['feasible'] for r in d['all_non_slack_bus_peak_screen'])}개였다.</p>
<h2>4. 약 5일·5분 간격 연속 운전점 테스트</h2>
<img src="network_test_summary.png" alt="배경부하와 AIDC 접속 시 모선 전압, 무저장 및 기존 공동운전의 선로 부하율">
<table><tr><th>운전</th><th>수렴 구간</th><th>수렴한 해의 제약위반 구간</th><th>미수렴 구간</th><th>최대 PCC(MW)</th><th>최저전압(p.u.)</th><th>최대 선로 부하율</th></tr>{rows}</table>
<p>각 {s0['intervals']}개 운전점, {s0['duration_h']:.3f}시간. 계통 기준의 신규 최적화가 아닌 <b>기존 프로파일 적합성 검사</b>다. 상위 공급한계 40 MVA 이내여도 하위 선로전류·모선전압 제약은 위반할 수 있다.</p>
<pre>P_PCC = P_facility_after_TES + P_ESS,ch - P_ESS,dis
Q_PCC = (P_PCC - P_ESS,ch + P_ESS,dis) * tan(acos(0.98))
Q_ESS = 0</pre>
<p>ESS의 유효전력 방전을 가상의 무효전력 지원으로 오인하지 않도록 시설 부분에만 역률을 적용했다. 기존 시설 손실은 PCC 부하에 포함되며, 계통 선로손실만 상위에서 추가한다.</p>
<h2>5. 저장장치 재최적화만으로 해결할 수 있는가?</h2>
<div class="warn"><b>현재 600 A 가정에서는 기존 저장장치로 해결하기 어려운 정도가 아니라, 다음 필요조건을 만족하지 못한다.</b>
<ul><li>12.66 kV, Bus1=1.0 p.u., 600 A 첫 선로의 피상전력 한계: <b>{bound['root_line_mva_bound']:.3f} MVA</b>.</li>
<li>무효전력과 선로손실을 아예 무시해도, 배경부하를 제외한 AIDC 유효전력 상한은 최대 <b>{bound['optimistic_aidc_active_power_upper_bound_mw']:.3f} MW</b>. 이것은 느슨한 필요조건이지 실제 AC 허용전력이 아님.</li>
<li>ESS·TES가 동시에 최대 방전하고 SOC·지속시간 제약까지 무시한 낙관적 순간 감축량: <b>{bound['instantaneous_storage_reduction_upper_bound_mw']:.3f} MW</b>.</li>
<li>그만큼 매 순간 줄여도 최소 AIDC 수요는 <b>{bound['optimistic_after_storage_pcc_min_mw']:.3f} MW</b>. 전체 {bound['intervals_failing_necessary_bound_even_with_max_discharge']}개 구간이 위 필요조건을 초과.</li></ul>
따라서 현재 부하 크기·600 A 공급구조를 유지한 채 충방전 시간표만 바꾸는 것으로는 전체 구간의 계통 제약을 만족시킬 수 없다. 이는 현재 연구 가정에서의 결론이며, 모든 12.66 kV 계통에 대한 일반적 결론이 아니다.</div>
<h2>6. 후속 결정 — 이번 실행에서 변경하지 않은 사항</h2><ul>
<li><b>12.66 kV·33-bus는 유지</b>하되, 배경부하의 기준 운전점과 전압 운용 조건을 명확히 정할 필요가 있다. 전압 기준·탭·무효보상을 임의로 조정하지 않았다.</li>
<li>AIDC의 현재 20 MW대 규모와 원형 계통 규모의 정합성을 선택: 원형 benchmark에 맞춘 별도 소규모 AIDC 시나리오, 또는 현재 AIDC 규모에 대한 명시적 병렬 공급/계통 수정 시나리오.</li>
<li>연구 조건을 사전 고정한 뒤 계통 기반 PCC 허용범위와 ESS–TES 재최적화를 진행. 원하는 성과가 나오도록 한계를 사후 튜닝하지 않는다.</li></ul>
<h2>7. 검증 및 출처</h2><ul>
<li>이전 33 kV 결과·기존 시설/저장장치 입력 {checks['protected_file_count']}개 해시 변경 없음: {checks['all_protected_unchanged']}.</li>
<li>Bus2 최대부하 NR/BFSW 독립 조류계산 최대 전압차: {checks['nr_bfsw_max_voltage_difference_pu']:.2e} p.u.</li>
<li>최대 유효전력 수지오차: {checks['maximum_power_balance_error_mw']:.2e} MW.</li>
<li><a href="https://github.com/MATPOWER/matpower/blob/master/data/case33bw.m">MATPOWER case33bw / Baran–Wu</a>: 원형 전압·계통 구조·R/X·배경부하.</li>
<li><a href="https://www.nlr.gov/docs/fy25osti/90266.pdf">Song, Seo and Won, IEEE Access, 2025</a>: Section IV-A·Fig.9의 12.66 kV modified33 활용 선행사례. 우리 부하 크기나 600 A 정격의 근거는 아님.</li></ul>
<p><a href="results.json">전체 결과</a> · <a href="model_contract.json">계산 조건</a> · <a href="verification.json">검증</a> · <a href="no_storage_replay.json">무저장 시계열</a> · <a href="existing_joint_replay.json">기존 공동운전 시계열</a></p>
<p class="small">평형 3상·5분 준정상상태 검토. 동특성·불평형·고장·보호·N−1·grid-code 전체·HIL 미포함. 실제 계통 접속 설계가 아니다.</p></html>'''
    (OUT / "report_ko.html").write_text(page, encoding="utf-8")
    print(OUT / "report_ko.html")


if __name__ == "__main__":
    main()
