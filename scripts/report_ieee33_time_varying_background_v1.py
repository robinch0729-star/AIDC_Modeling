"""Render the stored assessment; never rerun or tune the electrical model."""
from __future__ import annotations

import html
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/ieee33_time_varying_background_v1"


def load(name):
    return json.loads((OUT/name).read_text(encoding="utf-8"))


def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{html.escape(str(v))}</th>' for v in headers)+\
        '</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in row)+'</tr>' for row in rows)+\
        '</tbody></table></div>'


def plot(r,profiles):
    x=profiles.time_h
    fig,axs=plt.subplots(4,1,figsize=(13,12),sharex=True,layout="constrained")
    fig.suptitle("Time-varying background: adopted modified IEEE 33-bus reference",fontsize=16,fontweight="bold")
    colors={"mean_preserved":"#126d80","peak_preserved":"#749438","mean_shift_6h":"#c77739"}
    labels={"mean_preserved":"Mean preserved", "peak_preserved":"Peak preserved", "mean_shift_6h":"Mean preserved, +6 h phase"}
    for case in colors:
        axs[0].plot(x,profiles[case+"_background_p_mw"],color=colors[case],lw=1.3,label=labels[case])
    axs[0].axhline(3.715,color="#666666",ls="--",lw=1,label="Previous constant background")
    axs[0].set_ylabel("Background demand\n(MW)")
    axs[0].legend(ncol=2,fontsize=9,loc="upper left")
    base=pd.DataFrame(load("reference_mean_preserved_no_storage.json"))
    joint=pd.DataFrame(load("reference_mean_preserved_existing_joint.json"))
    axs[1].plot(x,base.pcc_p_mw,color="#126d80",lw=1.2,label="Frozen PCC without storage")
    axs[1].plot(x,joint.pcc_p_mw,color="#c77739",lw=1.1,label="Existing 25.5 MW-cap schedule replay")
    axs[1].set_ylabel("AIDC PCC demand\n(MW)");axs[1].legend(fontsize=9,loc="upper left")
    off=pd.DataFrame(load("reference_mean_preserved_aidc_off_diagnostic.json"))
    peak=pd.DataFrame(load("reference_peak_preserved_no_storage.json"))
    axs[2].plot(x,base.min_voltage_pu,color="#126d80",lw=1.2,label="Mean-preserved + AIDC")
    axs[2].plot(x,off.min_voltage_pu,color="#7e668a",lw=1,ls="--",label="Mean-preserved, AIDC removed")
    axs[2].plot(x,peak.min_voltage_pu,color="#749438",lw=1.1,label="Peak-preserved + AIDC")
    axs[2].axhline(.95,color="#b33d31",lw=1,ls=":",label="Study lower bound")
    axs[2].set_ylabel("Minimum bus voltage\n(p.u.)");axs[2].legend(ncol=2,fontsize=9,loc="lower left")
    minimum=pd.DataFrame(load("comparison_mean_preserved_no_storage.json"))
    axs[3].plot(x,base.max_line_loading_percent,color="#126d80",lw=1.2,label="Reference: allowance configuration")
    axs[3].plot(x,minimum.max_line_loading_percent,color="#c77739",lw=1.1,label="Comparison: minimum configuration")
    axs[3].axhline(100,color="#b33d31",lw=1,ls=":",label="Current limit")
    axs[3].set_ylabel("Maximum line loading\n(%)");axs[3].set_xlabel("Elapsed time (h); 1,438 five-minute intervals")
    axs[3].legend(ncol=2,fontsize=9,loc="lower left")
    for ax in axs:
        ax.grid(alpha=.15);ax.set_xlim(0,120)
        ax.spines[["top","right"]].set_visible(False)
    fig.savefig(OUT/"background_grid_assessment.png",dpi=170)
    fig.savefig(OUT/"background_grid_assessment.svg")
    plt.close(fig)


def main():
    r=load("results.json"); cfg=load("model_contract.json"); ref=load("adopted_network.json")
    profiles=pd.read_csv(OUT/"background_profiles_5min.csv")
    plot(r,profiles)
    audit=r["background_profile_audit"]
    labels={s["id"]:s["label_ko"] for s in cfg["scenarios"]}
    modes={"no_storage":"저장장치 미사용","existing_joint":"기존 ESS–TES 스케줄 재생","aidc_off_diagnostic":"AIDC 제거 진단"}
    demand_rows=[];rows=[];diag=[];windows=[]
    for case,a in audit["scenarios"].items():
        demand_rows.append([labels[case],f'{a["p_min_mw"]:.3f}',f'{a["p_mean_mw"]:.3f}',f'{a["p_max_mw"]:.3f}',
                            f'{a["energy_mwh"]:.3f}',f'{a["pearson_with_aidc_no_storage"]:.3f}'])
    for role in ("reference","comparison"):
        for case in labels:
            e=r["evaluations"][role][case]
            for mode,a in e["replays"].items():
                rows.append([ref[role]["label_ko"],labels[case],modes[mode],f'{a["min_voltage_pu"]:.4f}',
                             f'{a["max_line_loading_percent"]:.2f}%',f'{a["max_head_s_mva"]:.3f}',
                             f'{a["converged_violation_intervals"]} / {a["converged_violation_hours"]:.2f}',
                             a["under_voltage_intervals"],a["line_overload_intervals"],a["head_overload_intervals"]])
                if role=="reference" and case=="mean_preserved" and mode=="no_storage":
                    windows=a["violation_windows"]
            a=e["violation_attribution"]
            diag.append([ref[role]["label_ko"],labels[case],a["background_infeasible_intervals"],
                         a["additional_no_storage_violations_with_background_feasible"]])
    mean=r["evaluations"]["reference"]["mean_preserved"]
    peak=r["evaluations"]["reference"]["peak_preserved"]
    off=mean["replays"]["aidc_off_diagnostic"]
    no=mean["replays"]["no_storage"]
    joint=mean["replays"]["existing_joint"]
    envelope=mean["optimistic_discharge_diagnostic"]
    checks=[c for role in ("reference","comparison") for case in labels
            for a in r["evaluations"][role][case]["replays"].values() for c in a["solver_crosschecks"]]
    verification=dict(protected_files=len(r["protected_inputs_unchanged"]),
        protected_files_unchanged=all(r["protected_inputs_unchanged"].values()),
        profile_energy_preserved_before_trim=abs(audit["original_15min_energy_mwh"]-audit["expanded_full_5min_energy_mwh"])<1e-8,
        independent_solver_checks=len(checks),maximum_nr_bfsw_voltage_difference_pu=max(c["max_voltage_difference_pu"] for c in checks),
        tests=None)
    if (OUT/"test_results.xml").exists():
        suite=ET.parse(OUT/"test_results.xml").getroot()
        suites=[suite] if suite.tag=="testsuite" else list(suite)
        verification["tests"]={key:sum(int(x.attrib.get(key,0)) for x in suites) for key in ("tests","failures","errors","skipped")}
    (OUT/"verification.json").write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding="utf-8")
    references=''.join(f'<li>[{html.escape(a["id"])}] <a href="{html.escape(a["url"])}">{html.escape(a["title"])}</a></li>' for a in cfg["references"])
    h=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>기준 계통 확정 및 시간가변 배경부하 검토</title>
<style>body{{font-family:'Malgun Gothic',Arial,sans-serif;max-width:1250px;margin:30px auto;padding:0 22px;line-height:1.8;color:#243d50}}h1{{font-size:29px}}h2{{margin-top:32px;font-size:23px;border-bottom:1px solid #cad6df}}.note,.warn{{padding:17px;border-left:5px solid #137b85;background:#eef8f7}}.warn{{border-color:#bb673e;background:#fff5ea}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid #d5e1e8;text-align:left;vertical-align:top}}th{{background:#eef4f8}}.scroll{{overflow:auto}}img{{max-width:100%;height:auto}}a{{color:#126a92}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f8;padding:16px}}.small{{font-size:13px;color:#576c7b}}li{{margin:5px 0}}</style>
<h1>기준 계통 확정 및 시간가변 배경부하 검토</h1>
<div class="note"><b>사용자 결정 반영:</b> 여유 보강안을 기준 계통, 최소 보강안을 비교 시나리오로 등록했다.
시간가변 배경부하는 사전 정의한 3개 조건을 평가했으며, 실측 계통으로 보정되거나 최종 부하조건이 확정된 것은 아니다.
계통 정격·전압·AIDC 크기·ESS/TES 사양을 결과에 맞춰 재조정하지 않았다.</div>
<h2>1. 이번에 확정한 것</h2>
{table(['항목','기준 계통: 여유 보강안','비교 계통: 최소 보강안'],[
['공칭전압 / 접속점','12.66 kV / Bus 2','동일'],['변경 구간 / 전원 전압','Bus 1–2 / 1.04 pu','동일'],
['등가 R / X (Ω)','0.021333 / 0.0396','0.0215 / 0.0558'],['등가 C / 허용전류','1170 nF / 1.785 kA','980 nF / 1.524 kA'],
['나머지 선로 / 상위 공급한계','기존 R/X, 600 A / 40 MVA','동일']])}
<p>위 정격은 연구용 등가값이며 실제 케이블·변압기 설계 승인이 아니다. 허용전류 −10%·저항 +10%는 이전에 통과한 민감도 조건이며 기준값에 몰래 중복 적용하지 않았다.
기존 후보 보고서의 ‘확정 전’ 표기는 당시 이력으로 보존하며, 현재 상태는 <a href="../../configs/ieee33_operating_reference_v1.json">기준 계통 설정</a>과 <a href="../../configs/grid_testbed_registry.json">registry</a>가 우선한다.</p>
<h2>2. 배경부하를 어떻게 만들었는가?</h2>
<ul><li>배경부하란 AIDC를 제외한 기존 IEEE33의 32개 모선 부하이다. 원래 합계는 3.715 MW + 2.3 MVAr이다.</li>
<li>SimBench 1.6.3의 <b>1-MV-urban--0-sw</b>에 배정된 수요 프로파일을 활성전력 크기로 가중합해 시간 형태만 가져왔다. SimBench의 계통·발전기·전압수준은 가져오지 않았다.</li>
<li>2016년 5월 둘째 일요일(5월 8일)부터 목요일까지 5일을 선택했다. 기존 Azure의 2024년 5월 둘째 일요일 시작 구간과 계절·요일을 맞춘 <b>합성 결합</b>이며 동일 지역·동일 시각의 실제 관측쌍이 아니다. 시차도 실측으로 보정하지 않았다.</li>
<li>15분 자료를 구간 대표 평균전력으로 해석하고 각 값을 세 번 유지해 5분 입력으로 만들었다. 변환 자체는 에너지를 보존하나 새로운 5분 단기변동을 만들어내지는 않는다. 처음·마지막 5분을 잘라 기존 1,438개 구간(119시간 50분)에 맞췄다.</li>
<li>원자료 연간 시각에는 봄의 건너뜀·가을의 중복 4개가 있다. 선택한 5월 구간은 중복·누락 없이 480개이며, 연간 자료를 임의 수정하지 않았다.</li>
<li>기존 모선 모두에 같은 계수를 적용해 모선별 비중과 역률을 유지했다. <b>AIDC에는 배경부하 계수를 적용하지 않았다.</b> 공간적으로 서로 다른 패턴이나 측정된 상관관계를 재현하지 않는다.</li></ul>
<pre>P_bg,i(k) = α(k) P_bg,i,0     Q_bg,i(k) = α(k) Q_bg,i,0
평균 유지: α(k) = g(k) / mean(g)      최대 유지: α(k) = g(k) / max(g)
g(k): SimBench에서 가져온 배경부하 시간 형태
P_PCC·ESS·TES 입력은 기존 값 그대로 유지</pre>
{table(['배경 시나리오','최소 MW','평균 MW','최대 MW','기간 에너지 MWh','AIDC와 표본 상관계수'],demand_rows)}
<p class="small">상관계수는 임의 결합한 이 구간의 수치일 뿐 실제 시설-계통 상관성의 근거가 아니다. 6시간 이동은 1,438개 값의 원형 이동으로 에너지·분포를 동일하게 유지한 민감도이다. 경계 연결과 요일 의미는 물리적 시계열이 아니므로 예측학습 자료로 그대로 사용하지 않는다.</p>
<h2>3. 전체 시간대 AC 조류계산 결과</h2>
<p>고정 전원 전압·같은 설비를 사용하여 두 계통 × 배경 3조건 × 운전 3조건을 계산했다. 기존 공동운전은 외생 25.5 MW 상한을 만족하도록 만들었던 스케줄의 <b>재생</b>이며, 새로운 계통 제약에 맞춰 최적화한 것이 아니다.</p>
{table(['계통','배경','운전','최저전압 pu','최대 선로 부하율','최대 공급 MVA','위반 구간 / 시간 h','저전압 구간','선로과부하 구간','공급한계 구간'],rows)}
<img src="background_grid_assessment.png" alt="시간가변 배경부하와 PCC 수요, 최저전압 및 선로 부하율">
<h2>4. 결과 해석: 배경부하 정규화와 원인 구분</h2>
<div class="warn"><b>평균 유지가 작은 변동을 뜻하지 않는다.</b> 평균 3.715 MW를 그대로 유지했지만 이 자료의 최대값은 {audit['scenarios']['mean_preserved']['p_max_mw']:.3f} MW이다.
기준 계통에서 AIDC 미사용 진단만으로도 {off['converged_violation_intervals']}개 구간({off['converged_violation_hours']:.2f}시간)이 위반했다.
이를 전부 AIDC 때문이거나 저장장치 용량 부족 때문이라고 해석해서는 안 된다.</div>
{table(['계통','배경','AIDC 제거 후에도 위반한 구간','배경 단독은 가능하지만 AIDC 추가 시 위반한 구간'],diag)}
<ul><li>평균 유지·기준 계통에서 무저장 위반은 {no['converged_violation_intervals']}구간, 기존 공동운전 재생은 {joint['converged_violation_intervals']}구간이다. 기존 외생 상한 준수가 이번 계통의 전압 제약 준수를 보장하지 않음을 보여준다.</li>
<li>최대 유지 시나리오는 기준 계통에서 무저장 위반 {peak['replays']['no_storage']['converged_violation_intervals']}구간이다. 다만 평균 배경부하가 {audit['scenarios']['peak_preserved']['p_mean_mw']:.3f} MW로 낮아진 조건이므로, 평균 유지 대비 차이를 제어 알고리즘의 개선효과로 해석할 수 없다.</li>
<li>무저장 위반 시점 {envelope['tested_violating_intervals']}개에서 SOC·지속시간을 무시하고 ESS/TES를 모두 정격 방전하는 낙관적 점검을 추가했다. {envelope['still_violates']}개는 여전히 위반하고, {envelope['passes_instantaneous_test']}개는 순간 전기적 검사만 통과했다. 통과한 경우도 실제 충방전 계획의 존재를 보장하지 않는다.</li>
<li>기준 계통 확정과 배경부하 정규화의 확정은 별개이다. 원래 IEEE33 부하를 이 기간의 평균으로 볼지, 최대 기준으로 볼지는 자료·연구 시나리오로 정해야 하며 결과를 통과시키기 위해 은밀히 선택하지 않는다.</li></ul>
<details><summary>기준 계통·평균 유지·무저장 위반 시간대 (AIDC UTC 라벨 기준)</summary>
{table(['시작','끝 (미포함)','지속시간 h','저전압 모선'],[[w['start_utc'],w['end_exclusive_utc'],f"{w['duration_h']:.3f}",str(w['buses'])] for w in windows])}</details>
<h2>5. 다음 단계와 이번 단계의 범위</h2>
<ul><li>배경부하의 기준 크기 해석을 선택한다. 최대 유지 조건은 기존 정적 부하범위 안의 시간가변 기준 후보, 평균 유지 조건은 현재 계통의 적용 한계를 드러내는 고피크 비교조건으로 사용할 수 있다. 이 역할 변경은 아직 설정에 자동 반영하지 않았다.</li>
<li>그다음 선택한 조건에서 시간별 PCC 허용범위를 계산한다. 배경 단독으로도 불가능한 시점은 AIDC에 ‘음수 상한’을 주어 해결하지 않고 계통측 불가능 상태로 별도 표시해야 한다.</li>
<li>실현 가능한 허용범위를 바탕으로 ESS–TES를 새로 스케줄링하고, 저장상태·종료 복원과 AC 계통 제약을 다시 검증한다. 필요하면 불가능 원인을 전력·에너지·무효전력·배경계통으로 나눈다.</li>
<li>이번 단계에서 DOE/DFL, 전압제어·무효전력 최적화, 추가 보강, 저장장치 재설계, 한국 실계통 재현, N−1 또는 HIL 검증은 수행하지 않았다.</li></ul>
<h2>6. 검증 및 재현</h2>
<pre>{html.escape(json.dumps(verification,ensure_ascii=False,indent=2))}</pre>
<p>5분 정상상태 AC 결과이며 15분보다 빠른 실제 변동이나 과도 안정성을 보장하지 않는다. 이전 시설·저장장치·보강 후보 파일의 해시를 대조하여 변경되지 않았음을 확인했다.</p>
<pre>&amp; .venv-grid/Scripts/python.exe scripts/assess_ieee33_time_varying_background_v1.py
&amp; .venv-grid/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_ieee33_time_varying_background_v1.py --junitxml=outputs/ieee33_time_varying_background_v1/test_results.xml
&amp; .venv-grid/Scripts/python.exe scripts/report_ieee33_time_varying_background_v1.py</pre>
<p><a href="background_profiles_5min.csv">5분 배경부하 입력 CSV</a> · <a href="profile_audit.json">원자료·시간축·에너지 감사</a> · <a href="results.json">전체 요약 JSON</a> · <a href="adopted_network.json">확정 계통 정의</a></p>
<h2>참고자료 및 출처</h2><ul>{references}</ul>
<p>SimBench 수요 형태의 파생자료: database ODbL / individual contents DbCL. 원자료·사용 버전·해시는 profile_audit.json에 명시했다.
계통 자체와 정규화·모선 배분·시간 정렬은 본 연구의 가정이며 SimBench 원형 계통의 실행 결과가 아니다.</p>
<p class="small">생성 결과 UTC: {r['created_utc']}</p></html>'''
    (OUT/"report_ko.html").write_text(h,encoding="utf-8")
    print(json.dumps(verification,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
