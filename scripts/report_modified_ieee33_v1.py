"""Render a traceable Korean review report for the IEEE33-derived candidate."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandapower as pp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/modified_ieee33_candidate_v1"
BLUE, GREEN, ORANGE, RED = "#174d78", "#087f80", "#bf731b", "#b53e42"


def topology():
    net=pp.from_json(str(OUT/"original_case33bw.json"))
    pos={i:(i-1,2.1) for i in range(1,19)}
    pos.update({i:(i-18,0.6) for i in range(19,23)})
    pos.update({i:(i-20,3.7) for i in range(23,26)})
    pos.update({i:(i-20,0.6) for i in range(26,34)})
    fig,ax=plt.subplots(figsize=(15,5.1),facecolor="white")
    for row in net.line.loc[net.line.in_service].itertuples():
        a,b=int(row.from_bus)+1,int(row.to_bus)+1
        ax.plot([pos[a][0],pos[b][0]],[pos[a][1],pos[b][1]],c="#9aacb8",lw=2,zorder=1)
    for b,(x,y) in pos.items():
        col={2:GREEN,6:ORANGE,18:RED}.get(b,BLUE)
        ax.scatter(x,y,s=210 if b in (2,6,18) else 125,c=col,zorder=2)
        ax.text(x,y+.26,str(b),ha="center",fontsize=10,color=col,weight="bold")
    ax.annotate("AIDC PCC\nBase connection: Bus 2",xy=pos[2],xytext=(.7,4.4),ha="center",fontsize=11,
                color=GREEN,weight="bold",arrowprops={"arrowstyle":"->","color":GREEN,"lw":1.7})
    ax.text(7.3,4.2,"Location comparisons: Bus 6 / Bus 18\nAll 32 non-slack buses screened",color=BLUE,fontsize=12)
    ax.text(-.15,1.1,"Bus 1\nFixed source\n1.0 p.u.",ha="center",fontsize=10,color=BLUE)
    ax.text(7.1,-.12,"33 buses | 32 closed branches | 5 ties remain open (not drawn)",ha="center",fontsize=11,color="#566d7b")
    ax.text(7.1,-.5,"33 kV, 600 A and 40 MVA supply cap: declared study assumptions, not an installed feeder design",ha="center",fontsize=10,color="#566d7b")
    ax.set(xlim=(-.8,17.6),ylim=(-.7,5.0));ax.axis("off")
    ax.set_title("Modified IEEE 33-bus candidate — topology retained from Baran–Wu",loc="left",fontsize=16,weight="bold",color=BLUE,pad=15)
    fig.tight_layout()
    for ext in ("png","svg"):
        fig.savefig(OUT/f"network_topology.{ext}",dpi=180,bbox_inches="tight")
    plt.close(fig)


def results_figure(data):
    fig,axs=plt.subplots(1,2,figsize=(15,5.6),facecolor="white")
    limits=data["static_limits_not_doe"]
    buses=[r["bus"] for r in limits]
    vals=[r["certified_p_mw"] or 0 for r in limits]
    axs[0].bar(buses,vals,color=[{2:GREEN,6:ORANGE,18:RED}.get(b,"#7cabbf") for b in buses],width=.8)
    axs[0].axhline(data["aidc_peak_pcc_mw"],ls="--",color=RED,lw=1.5,label="Current AIDC peak: 26.55 MW")
    axs[0].axhline(data["aidc_mean_pcc_mw"],ls=":",color=BLUE,lw=1.3,label="Current AIDC mean: 22.89 MW")
    axs[0].set(xlabel="AIDC connection bus (IEEE labels)",ylabel="Static feasible AIDC import (MW)",xticks=[2,6,10,14,18,22,26,30,33],ylim=(0,38))
    axs[0].set_title("Location matters",loc="left",weight="bold",color=BLUE)
    axs[0].legend(fontsize=9,loc="upper right")
    rows=json.loads((OUT/"replay_5min.json").read_text(encoding="utf-8"))
    t=np.arange(len(rows))/12
    axs[1].plot(t,[r["aidc_p_mw"] for r in rows],c=GREEN,lw=1.1,label="Frozen AIDC PCC demand")
    lim=next(r["certified_p_mw"] for r in limits if r["bus"]==2)
    axs[1].axhline(lim,c=BLUE,ls="--",lw=1.5,label=f"Bus 2 static boundary: {lim:.2f} MW")
    axs[1].set(xlabel="Elapsed time (h)",ylabel="AIDC PCC import (MW)",xlim=(0,120),ylim=(20,32))
    axs[1].set_title("Bus 2: five-minute baseline replay",loc="left",weight="bold",color=BLUE)
    axs[1].legend(fontsize=9,loc="upper right")
    for ax in axs:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("Fixed background, PF = 0.98, voltage 0.95–1.05 p.u., branch limit 600 A",fontsize=13,color=BLUE,y=.99)
    fig.text(.5,.01,"Single-period static feasibility is NOT a time-varying DOE, a storage schedule, or a grid-code certificate.",ha="center",fontsize=10,color="#596d78")
    fig.tight_layout(rect=(0,.045,1,.96))
    for ext in ("png","svg"):
        fig.savefig(OUT/f"connection_feasibility.{ext}",dpi=180,bbox_inches="tight")
    plt.close(fig)


def fmt(v,d=3):
    return "—" if v is None else f"{v:.{d}f}"


def verdict(row):
    if not row["converged"]:
        return "미수렴·가능성 미확인"
    if row["feasible"]:
        return "설정 제약 만족"
    reasons=[]
    for key,label in (("under_voltage_buses","전압 하한 위반"),
                      ("over_voltage_buses","전압 상한 위반"),
                      ("overloaded_branches","선로 과부하"),
                      ("head_overloaded","상위 공급한계 초과")):
        if row.get(key):
            reasons.append(label)
    return " / ".join(reasons) or row.get("failure","가능성 미확인")


def main():
    cfg=json.loads((OUT/"model_contract.json").read_text(encoding="utf-8"))
    d=json.loads((OUT/"results.json").read_text(encoding="utf-8"))
    topology();results_figure(d)
    peak=[r for r in d["snapshots"] if r["voltage_kv"]==33 and r["demand"]=="peak"]
    rows="".join(f'<tr><td>Bus {r["bus"]}</td><td>{fmt(r.get("min_voltage_pu"),4)}</td><td>{fmt(r.get("max_line_loading_percent"),2)}%</td><td>{fmt(r.get("head_s_mva"),2)}</td><td>{verdict(r)}</td></tr>' for r in peak)
    limits="".join(f'<tr><td>Bus {r["bus"]}</td><td>{fmt(r["certified_p_mw"],2)} MW</td><td>{html.escape(str(r["boundary"]))}</td></tr>' for r in d["static_limits_not_doe"] if r["bus"] in (2,6,18))
    sens=[r for r in d["sensitivities"] if r["bus"]==2]
    rating_rows="".join(f'<tr><td>{r["branch_limit_ka"]*1000:.0f} A</td><td>{r["head_limit_mva"]:.0f} MVA</td><td>{r["max_line_loading_percent"]:.1f}%</td><td>{"만족" if r["feasible"] else " / ".join((["선로 과부하"] if r["overloaded_branches"] else [])+(["공급한계 초과"] if r["head_overloaded"] else []))}</td></tr>' for r in sens if r["kind"]=="current_and_head_rating")
    refs="".join(f'<li><b>{html.escape(r["id"])}</b> <a href="{r["url"]}">{html.escape(r["title"])}</a><br>{html.escape(r["use"])}</li>' for r in cfg["references"])
    base=d["baseline_replay"]
    pfrows=[r for r in sens if r["kind"]=="background_and_pf"]
    underoriginal=d["original_voltage_under_study_criteria"]["min_voltage_pu"]
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Modified IEEE 33-bus 접속 조건 후보 검토</title><style>
body{{font-family:'Malgun Gothic',Arial,sans-serif;max-width:1160px;margin:32px auto;padding:0 26px;line-height:1.8;color:#213d50}}h1{{font-size:30px}}h2{{font-size:23px;margin-top:36px;border-bottom:2px solid #dce8eb;padding-bottom:6px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{text-align:left;vertical-align:top;border-bottom:1px solid #dbe6ec;padding:10px}}th{{background:#edf4f7}}.note,.warn{{padding:17px 20px;border-left:5px solid #087f80;background:#eef8f6}}.warn{{background:#fff6e7;border-color:#bf731b}}img{{width:100%;height:auto}}code,pre{{white-space:pre-wrap;background:#f1f5f7;padding:10px;overflow-wrap:anywhere}}a{{color:#176391}}li{{margin:7px 0}}.small{{font-size:13px;color:#587080}}
</style><h1>Modified IEEE 33-bus 접속 조건 후보 검토</h1>
<div class="note"><b>기존 연구계획의 Modified IEEE 33-bus를 주 테스트베드로 유지한다.</b><br>33 kV 연구용 후보를 구현하고 원형 비교·접속 위치·정격 민감도·기존 5분 PCC 부하의 연속 운전점을 점검했다. <b>수치 조건은 사용자 검토 전 후보</b>이며, 실제 케이블·변압기 설계나 DSO 접속 승인이 아니다.</div>
<h2>1. 기존 계획에서 이어받은 것과 이번에 정한 것</h2>
<ul><li>기존 기록: Modified IEEE 33-bus 사용, 계통 조건을 연계한계 계산 전에 고정, 원하는 AIDC 용량에 맞춘 반복 보강 금지.</li>
<li>미정이었던 항목: 전압·변압기/상위공급 규모·구체 접속 방식. 실행 가능한 수치 모델은 기존 기록에서 확인되지 않았다.</li>
<li>직전 SimBench 검토는 <b>대안 검토 이력</b>으로만 보존. 현재 후보와 혼용하지 않는다.</li></ul>
<h2>2. 후보 계통과 설정 근거</h2><img src="network_topology.png" alt="33개 모선 구조 및 AIDC Bus2 접속">
<table><tr><th>항목</th><th>현재 후보</th><th>출처 또는 가정</th></tr>
<tr><td>토폴로지·선로 R/X</td><td>33 buses, 32 closed branches, 5 open ties<br>Ω 단위 R/X 원형 유지</td><td>Baran–Wu / MATPOWER case33bw [R1]. pandapower의 길이 1 km는 등가 입력표현이지 실측 길이가 아님.</td></tr>
<tr><td>배경부하</td><td>3.715 MW + 2.3 MVAr</td><td>원형 총량·모선별 분포 유지 [R1]. 기본 계산에서는 시간불변.</td></tr>
<tr><td>전압</td><td>12.66 → 33 kV</td><td><b>연구용 수정 가정.</b> 33 kV의 배전망 사용 사례는 [R4]; 이 계통이 독일/ACS 실제망이라는 뜻은 아님.</td></tr>
<tr><td>계산 기준·상위전압</td><td>10 MVA base, Bus1 = 1.0 p.u.</td><td>계산 baseMVA를 바꾸는 것으로 설비 용량을 늘리지 않음. 고정 이상 전압원 경계.</td></tr>
<tr><td>전압 제약</td><td>0.95–1.05 p.u.</td><td>Enhanced33 논문의 연구 운전범위 [R2] 차용. 그 논문의 DG·무효보상·tie 수정은 복제하지 않음.</td></tr>
<tr><td>선로 전류 한계</td><td>모든 폐로 선로 600 A<br>450 / 600 / 750 A 민감도</td><td><b>합성 시험계통 가정.</b> 원형은 정격 미정의이며 변환본의 99999 kA는 placeholder. 실제 케이블 허용전류 인증 아님.</td></tr>
<tr><td>상위 공급 제약</td><td>40 MVA<br>30 / 40 / 50 MVA 민감도</td><td><b>연구 가정.</b> 변압기를 모델링한 것이 아니므로 변압기 정격이라고 부르지 않음. 상위계통 임피던스·변압기 손실/온도는 제외.</td></tr>
<tr><td>AIDC 접속</td><td>기본 Bus2, 비교 Bus6·18<br>전체 비슬랙 모선 32개도 검사</td><td>Bus2는 전원 인근의 기본 사례로 사전 지정. 위치별 전압·전류 제약 분석은 paper list[20] [R3]와 연결.</td></tr>
<tr><td>AIDC 입력</td><td>평균 {d['aidc_mean_pcc_mw']:.3f} MW<br>5분 피크 {d['aidc_peak_pcc_mw']:.3f} MW<br>PF=0.98 지상</td><td>P는 기존 PCC 결과 그대로, 재규모화 없음. PF는 별도 연구 가정이며 0.95/1.0도 검사.</td></tr></table>
<div class="warn"><b>33 kV로 바꾼다는 뜻:</b> R/X Ω를 유지한 채 실제 운전전압을 변경하므로 p.u. 임피던스가 (12.66/33)² ≈ 0.147배가 된다. 전압강하가 작아지는 <b>다른 합성 계통</b>을 정의한 것이지, 원형 모델의 단위를 바꾼 것 또는 원형 배전선을 그대로 승압해도 된다는 주장이 아니다. 정격 600 A·33 kV의 명목 용량은 약 34.3 MVA이므로 기본 조건에서는 첫 선로 한계가 40 MVA 공급한계보다 먼저 작동한다.</div>
<h2>3. 기존 PCC 및 ESS–TES 계산 경계</h2>
<pre>P_PCC = P_facility(after TES) + P_ESS,ch − P_ESS,dis
Q_PCC = P_facility(after TES) × tan(acos(PF)) + Q_ESS,  Q_ESS = 0
S_head = sqrt(P_head² + Q_head²) ≤ 40 MVA
P_head = P_background + P_PCC + P_network_loss</pre>
<ul><li>시설 UPS·시설 변압기 손실은 기존 PCC 입력에 이미 포함. 다시 더하지 않음.</li>
<li>기존 모선 자체를 PCC로 보는 등가 접속. 별도 인입선 임피던스는 이번 후보에 없음.</li>
<li>ESS 1 MW / 4 MWh와 TES 3 MWth / 10 MWhth 및 각 충전·방전 효율 0.9 설정 유지. 이번 계산에서 저장장치 운전은 재최적화하지 않음.</li>
<li>ESS 운전 후 순수전 P에 고정 PF를 다시 적용하면 존재하지 않는 ESS 무효전력 지원을 만들 수 있으므로, 추후 연계 시 위 Q 경계를 유지.</li></ul>
<h2>4. AC 조류계산 결과</h2>
<p>원형 12.66 kV는 AIDC를 넣기 전에도 최소전압 {underoriginal:.4f} p.u.로 이번 0.95 하한에 미달한다. 이를 AIDC가 새로 일으킨 위반으로 계산하지 않는다. 원형의 0.9–1.1 허용범위를 쓴 결과도 따로 보존했다.</p>
<p>수정 후보의 무AIDC 최소전압은 {d['modified_no_aidc_baseline']['min_voltage_pu']:.4f} p.u.이다. 아래는 원형 배경부하와 동일한 AIDC 피크 {d['aidc_peak_pcc_mw']:.3f} MW, PF=0.98 조건이다.</p>
<table><tr><th>접속 위치</th><th>계통 전체 최저전압</th><th>최대 선로 부하율</th><th>상위 공급 MVA</th><th>판정</th></tr>{rows}</table>
<p class="small">미수렴은 이 수치해석에서 정상상태 해를 확인하지 못했다는 뜻이며, 물리적 전압붕괴를 증명한 결과는 아니다.</p>
<img src="connection_feasibility.png" alt="모선별 정적 허용전력 비교와 Bus2 기존5분프로파일 검토">
<table><tr><th>위치</th><th>명목 조건의 단일시점 P 경계</th><th>먼저 나타난 제약</th></tr>{limits}</table>
<p>이 값은 배경부하와 PF를 고정한 <b>단일시점 AC 허용전력 경계</b>이다. 다기간 ESS–TES 제약·불확실성·정보교환을 포함하지 않으므로 DOE라고 부르지 않는다. 0–50 MW 범위의 거친 단조성 검사 후 경계 오차 0.002 MW 이내로 탐색했다.</p>
<div class="note"><b>기존 약 5일 전체 프로파일 검토:</b> Bus2에서 {base['intervals']}개 5분 운전점({base['duration_hours']:.3f} h)을 계산했다. 모든 운전점 수렴: {base['all_converged']}. 설정 제약 위반 {base['violation_intervals']}개. 최저전압 {base['min_voltage_pu']:.4f} p.u., 최대 선로 부하율 {base['max_line_loading_percent']:.2f}%, 최대 상위 공급 {base['max_head_s_mva']:.2f} MVA. <b>배경부하는 고정</b>이며 시간대별 실제 배경수요와 결합한 검증은 아니다.</div>
<h2>5. 가정의 민감도와 해석</h2>
<p>Bus2 피크에서 배경부하 0.5 / 1.0 / 1.5배 및 PF 0.95 / 0.98 / 1.0의 9개 조합 중 {sum(r['feasible'] for r in pfrows)}개가 기본 정격 제약을 만족했다. 정격 가정을 바꾼 결과는 다음과 같다.</p>
<table><tr><th>선로 한계</th><th>상위 공급한계</th><th>최대 선로 부하율</th><th>판정</th></tr>{rating_rows}</table>
<ul><li>Bus2 기본 사례에는 저장장치 없이도 여유가 있다. 이 조건만으로 ESS–TES가 반드시 필요하거나 운영성과가 개선된다고 주장하지 않음.</li>
<li>Bus6·18의 문제를 저장장치가 자동 해결한다고 해석하지 않음. 요구 조정량과 기존 저장장치의 전력·에너지 한계를 별도로 비교해야 함.</li>
<li>혼잡 사례는 정격·배경부하·접속 위치 등 조건을 사전에 명시한 별도 시나리오로 구성. 기존 25.5 MW 외생 상한이 나오도록 계통을 맞추지 않음.</li></ul>
<h2>6. 다음 단계</h2>
<ol><li><b>후보 조건 검토:</b> 33 kV 합성 계통, Bus2 기본 접속, 전류·상위 공급한계 가정을 연구 조건으로 채택할지 확인.</li>
<li><b>배경수요 시간축:</b> 공통 비교에 사용할 시간대별 배경부하와 혼잡 시나리오를 정하고 고정.</li>
<li><b>DSO 허용범위 계산:</b> 해당 시점의 전체 모선 전압·선로 전류·상위 공급제약으로 PCC 허용범위를 계산. 필요 시 AC 검증을 붙인 선형 OPF로 확장.</li>
<li><b>ESS–TES 연계:</b> 계통에서 계산된 범위에 대해 기존 공동 운전모델이 준수 가능한지 평가. 그다음 예측 불확실성 및 DFL 비교로 진행.</li></ol>
<h2>7. 연구보고서에 넣을 개조식 요약</h2>
<ul><li><b>대표 배전계통 구성</b><ul><li>기존 계획에 따라 Modified IEEE 33-bus를 테스트베드로 유지</li><li>원형 토폴로지·선로 임피던스·배경부하를 보존한 33 kV 연구용 후보 구성</li><li>문헌 차용 조건과 정격·역률 등 연구 가정을 구분하고 민감도 평가</li></ul></li>
<li><b>AIDC 접속 조건 검토</b><ul><li>기존 PCC 부하를 재규모화하지 않고 접속 위치별 AC 조류계산 수행</li><li>전원 인근 Bus2에서는 전체 5분 기준 프로파일이 설정 제약을 만족</li><li>Bus6에서는 전압 하한 위반, Bus18 피크에서는 정상상태 해 미확인</li></ul></li>
<li><b>후속 연구 연결</b><ul><li>접속 위치와 계통 조건에 따라 AIDC 허용전력이 달라짐을 확인</li><li>후보 조건 확정 후 시간대별 계통 허용범위와 ESS–TES 공동 운전을 연결</li></ul></li></ul>
<h2>References</h2><ol>{refs}</ol>
<p><a href="results.json">전체 수치·민감도·입력해시</a> · <a href="model_contract.json">설정과 출처 구분</a> · <a href="candidate_bus2_peak_network.json">재현 가능한 계통 JSON</a> · <a href="verification.json">검증 결과</a></p>
<p class="small">검증 범위: 3상 평형 준정상상태 AC 조류계산. 실제 설비 설계·N−1·불평형·동특성·grid-code 전체·HIL·다기간 DOE는 포함하지 않음.</p></html>'''
    (OUT/"report_ko.html").write_text(page,encoding="utf-8")
    junit=OUT/"test_results.xml"
    verification={"protected_inputs_unchanged":d["protected_inputs_unchanged"],"config_sha256":d["config_sha256"]}
    if junit.exists():
        verification["pytest"]=ET.parse(junit).getroot().find("testsuite").attrib
    (OUT/"verification.json").write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding="utf-8")
    print(str(OUT/"report_ko.html"))


if __name__=="__main__":
    main()
