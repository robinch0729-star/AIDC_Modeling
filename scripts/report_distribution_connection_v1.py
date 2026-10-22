"""Report the selected benchmark and its observed limitations; never solve OPF/DOE."""
from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/distribution_connection_v1"
SCREEN = ROOT / "outputs/grid_connection_screen_v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def schematic():
    fig, ax = plt.subplots(figsize=(14, 5.8), facecolor="white")
    ax.set(xlim=(0, 14), ylim=(0, 5.8))
    ax.axis("off")
    blue, green, grey = "#155c91", "#087d80", "#586875"
    def box(x, y, w, h, title, body, color=blue):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", linewidth=1.4,
                                   edgecolor=color, facecolor="#f5f9fb"))
        ax.text(x+w/2, y+h*0.69, title, ha="center", va="center", fontsize=12, weight="bold", color=color)
        ax.text(x+w/2, y+h*0.3, body, ha="center", va="center", fontsize=10.5, color="#233b48")
    ax.text(.15, 5.5, "Selected distribution-network connection", fontsize=20, weight="bold", color="#183f54")
    ax.text(.15, 5.13, "SimBench 1-MV-comm--0-sw with an explicitly added AIDC branch", fontsize=12, color=grey)
    box(.2, 2.15, 1.65, 1.2, "110 kV grid", "External-grid\nboundary")
    box(2.5, 3.0, 2.1, 1.05, "Transformer 1", "110/20 kV · 40 MVA")
    box(2.5, 1.55, 2.1, 1.05, "Transformer 2", "110/20 kV · 40 MVA")
    ax.plot([1.85,2.13,2.13,2.5],[2.75,2.75,3.52,3.52], color=blue, lw=2)
    ax.plot([2.13,2.13,2.5],[2.75,2.075,2.075],color=blue,lw=2)
    ax.plot([4.6,5.25],[3.52,3.52],color=blue,lw=2)
    ax.plot([4.6,5.25],[2.075,2.075],color=blue,lw=2)
    ax.plot([5.25,5.25],[1.75,4.35],color=blue,lw=4)
    ax.text(5.25,4.52,"20 kV busbar",ha="center",fontsize=12,color=blue,weight="bold")
    ax.text(3.55,1.4,"Native couplers closed\nSynchronized fixed tap: −2",ha="center",va="top",fontsize=10.5,color=grey)
    box(6.7, 3.2, 4.1, 1.15, "Original MV feeders", "98 loads + distributed generation\nOriginal topology and ratings retained")
    ax.annotate("", xy=(6.65,3.75), xytext=(5.25,3.75), arrowprops={"arrowstyle":"->","lw":2,"color":blue})
    for off in (-.10,0,.10):
        ax.plot([5.25,9.65],[2.075+off,2.075+off],color=green,lw=1.8)
    ax.text(7.38,2.65,"New dedicated connection",ha="center",fontsize=12,color=green,weight="bold")
    ax.text(7.38,1.58,"1 km · 3 parallel circuits\n185 mm² · 362 A per circuit",ha="center",fontsize=10.5,color=green)
    ax.plot(9.68,2.075,"o",ms=9,color=green)
    ax.text(9.68,2.6,"PCC",ha="center",fontsize=12,weight="bold",color=green)
    ax.plot([9.68,10.28],[2.075,2.075],color=green,lw=2)
    box(10.35,1.35,3.25,1.5,"AIDC facility + ESS–TES","Storage-inactive baseline\n22.89 MW mean · 26.55 MW peak\nExisting PCC accounting retained",green)
    ax.text(.2,.52,"Connection length/circuit count and facility PF=0.98 are research assumptions. N−0 snapshot screening only.",fontsize=10.5,color=grey)
    ax.text(.2,.15,"The DSO transformers above are separate from the site-transformer losses already included inside the AIDC PCC model.",fontsize=10,color=grey)
    fig.savefig(OUT/"connection_schematic.png",dpi=180,bbox_inches="tight")
    fig.savefig(OUT/"connection_schematic.svg",bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=read(ROOT/"configs/distribution_connection_v1.json")
    selected=read(SCREEN/"selection.json")
    screen=read(SCREEN/"screening_results.json")
    assert selected["selected_synchronized_ratio_tap"] == cfg["tap_operating_assumption"]["selected_synchronized_tap"]
    rows=[r for r in selected["selected_case_checks"] if r["demand"]=="selected_peak"]
    mean=next(r for r in selected["selected_case_checks"] if r["demand"]=="selected_mean" and r["background_case"]=="native")
    peak=next(r for r in rows if r["background_case"]=="native")
    neutral=next(r for r in screen["cases"] if r["network"]==cfg["benchmark"]["code"] and r["background_case"]=="native" and r["demand"]=="peak" and r["main_bus"]==2)
    current=cfg["aidc_interface"]["pcc_peak_5min_no_storage_mw"]/(math.sqrt(3)*20*.98)
    summary={
        "status":"alternative_exploratory_case_not_the_main_research_testbed",
        "config":"configs/distribution_connection_v1.json",
        "network_code":selected["network_code"],"selected_tap":selected["selected_synchronized_ratio_tap"],
        "connection_current_at_nominal_voltage_ka":current,
        "mean_snapshot":mean,"peak_snapshot":peak,"peak_background_checks":rows,
        "neutral_tap_peak_snapshot":neutral,
        "not_all_backgrounds_feasible":any(not r["all_native_mv_constraints_satisfied"] for r in rows),
        "network_dispatch_or_doe_implemented":False,
        "existing_facility_and_storage_models_modified":False,
        "protected_input_hash_check":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
                                      for name,sha in screen["source_files"].items() if not name.startswith("scripts/")},
        "sources":cfg["references"],"versions":screen["versions"],
        "next_step":"Define a common DSO voltage-control baseline and align background profiles before calculating time-varying PCC feasibility/DOE. Do not force old25.5MW cap into the network.",
    }
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"model_contract.json").write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding="utf-8")
    junit=OUT/"test_results.xml"
    if junit.exists():
        suite=ET.parse(junit).getroot().find("testsuite")
        checks={"pytest":dict(suite.attrib),"protected_inputs_unchanged":summary["protected_input_hash_check"],
                "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (ROOT/"configs/distribution_connection_v1.json",Path(__file__),ROOT/"scripts/screen_grid_connection_v1.py",OUT/"model_contract.json",SCREEN/"selected_network_mean.json")}}
        (OUT/"verification.json").write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding="utf-8")
    schematic()
    labels={"native":"원본 명목 상태","hL":"고부하·저발전 hL","hW":"고부하·풍력 hW","hPV":"고부하·태양광 hPV","lW":"저부하·풍력 lW","lPV":"저부하·태양광 lPV"}
    table="".join(f'<tr><td>{labels.get(r["background_case"],r["background_case"])}</td><td>{r["native_mv_voltage_min_pu"]:.4f}</td><td>{r["native_mv_voltage_max_pu"]:.4f}</td><td>{r["max_line_loading_percent"]:.2f}%</td><td>{r["max_trafo_loading_percent"]:.2f}%</td><td>{"통과" if r["all_native_mv_constraints_satisfied"] else "전압 위반"}</td></tr>' for r in rows)
    refs="".join(f'<li><b>{html.escape(r["id"])}</b>: <a href="{html.escape(r["url"])}">{html.escape(r.get("title",r["id"]))}</a><br>{html.escape(r["used"])}</li>' for r in cfg["references"])
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>대표 배전계통·AIDC 접속 조건 선정 v1</title><style>
body{{font-family:'Malgun Gothic',Arial,sans-serif;max-width:1180px;margin:32px auto;padding:0 24px;color:#203e50;line-height:1.75}}h1{{font-size:29px}}h2{{margin-top:32px;border-bottom:2px solid #d7e4ea;padding-bottom:8px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #dce6eb;text-align:left;vertical-align:top}}th{{background:#eaf3f7}}.note{{padding:16px;border-left:5px solid #16808a;background:#eef8f7}}.warn{{padding:16px;border-left:5px solid #cb8c24;background:#fff7e8}}img{{width:100%;height:auto}}code,pre{{background:#f2f5f7;overflow-wrap:anywhere}}a{{color:#146d9a}}li{{margin-bottom:5px}}
</style><h1>SimBench 대안 계통 검토 이력</h1>
<div class="warn"><b>선정 정정: 본 연구의 주 테스트베드는 기존 계획의 Modified IEEE 33-bus이다.</b> 아래 SimBench 선정 표현과 계산은 과거 대안 검토 이력으로만 보존한다. <a href="../modified_ieee33_candidate_v1/report_ko.html">현재 Modified IEEE 33-bus 후보 검토</a></div>
<div class="note"><b>선정: SimBench 1-MV-comm--0-sw + 변전소 20 kV 모선의 AIDC 전용 접속 분기.</b><br>기존 IT·PCC·ESS·TES 규모는 유지. 이번 단계는 계통/접속 조건과 정상 운전점 선정 및 AC 스냅샷 점검이다. DSO-DOE 계산, 네트워크 연계 ESS–TES 최적화, 실제 접속 승인 또는 전 시계열 검증은 아직 수행하지 않았다.</div>
<h2>1. 왜 이 계통인가?</h2><table><tr><th>후보</th><th>검토 내용</th><th>판단</th></tr>
<tr><td>Baran–Wu 33-bus</td><td>12.66 kV·원형 총부하3.715 MW. MATPOWER 제공본은 branch rateA=0으로 열정격 미정의.</td><td>약27 MW 시설을 위해 대폭 확대·정격 가정 필요. 주 사례에서 제외.</td></tr>
<tr><td>CIGRE European MV</td><td>20 kV·25 MVA 변압기2대. 기존 배전선 전류정격이 작음.</td><td>현재 AIDC의 임의 말단 접속에 부적합. 변전소/선로 변경량이 큼.</td></tr>
<tr><td>SimBench commercial</td><td>20 kV·110/20 kV40 MVA2대. 독일 기반 부하·발전·스위치·연간 프로파일 제공.</td><td>원형 정격 유지 + 신규 전용 분기라는 명확한 수정으로 현재 규모 수용 검토 가능.</td></tr>
<tr><td>SimBench semiurban / urban / HV</td><td>semiurban도 비교 계산. urban 원형은10 kV. 110 kV 직접접속은 별도 상위계통 연구로 가능.</td><td>상업지역20 kV를 1차 사례로 선택. 실제 시설은 DSO 협의에 따라110 kV 등 다른접속 가능.</td></tr></table>
<p>상업지역은 현재 AIDC 규모를 줄이지 않으면서 원형 계통을 보존할 수 있다는 연구적 선택이다. RWTH/ACS의 실제 계통이라는 뜻이 아니며, 실제 독일 데이터센터의 보편적 접속전압을 주장하지 않는다.</p>
<h2>2. 선정 접속 구성과 출처 구분</h2><img src="connection_schematic.png" alt="SimBench 배전계통과 신규 AIDC 접속 분기">
<table><tr><th>항목</th><th>설정</th><th>근거</th></tr>
<tr><td>기본계통</td><td>107 buses,109 lines,2 transformers,98 loads,89 aggregated/direct generators</td><td>설치한 SimBench1.6.3 원형 데이터. 공급지점98개와 bus107개를 혼동하지 않음.</td></tr>
<tr><td>변압기/토폴로지</td><td>110/20 kV,2×40 MVA; 원본 HV·MV 모선연결기 닫힘</td><td>원본 유지. 두 변압기가 부하를 분담하나80 MVA 전체가 AIDC 전용인 것은 아님.</td></tr>
<tr><td>접속점</td><td>bus2:MV4.101 busbar1.1 → 신규PCC bus107</td><td>새 분기 가정. 원본bus2/4는 닫힌모선으로 동전위여서 성능최적화로 고른위치가 아님.</td></tr>
<tr><td>신규 선로</td><td>1 km,185 mm²,3병렬,회선당362 A</td><td>케이블R=.161Ω/km,X=.117Ω/km,C≈273nF/km는원형; 길이·병렬수는연구가정.</td></tr>
<tr><td>AIDC전력</td><td>평균22.893 MW,5분피크26.553 MW</td><td>기존PCC결과유지.20 MW로축소하지않음.</td></tr>
<tr><td>전압/열정격</td><td>MV0.965–1.055 pu; 선로·변압기부하율≤100%</td><td>벤치마크의정상상태 연구조건. grid code 전체 준수주장 아님.</td></tr>
<tr><td>시설역률</td><td>0.98지상;0.95/1.0민감도</td><td>측정값이나보편규정이아닌가정.ESS무효전력은0.</td></tr></table>
<p>정격전압에서 피크전류 I=P/(√3 V PF)≈<b>{current*1000:.0f} A</b>. 따라서원래의수백A급단일배전선말단에연결하지않고전용선을추가하였다.3×362=1086 A는포설·열적상호영향미검증모델정격이다.1회선고장시724 A이므로N−1안전접속을주장하지않는다.</p>
<h2>3. 탭 설정과 기준 운전점</h2><ul>
<li>원본에는 autoTap=1,LV목표1.025 pu가 있으나 가져온네트워크에실행컨트롤러가없음.</li>
<li>해당패키지조합은 tap_changer_type=None로가져오므로tap_pos만바꾸면효과가없음을확인.원본1.5%전압비탭을Ratio로명시하여해석.</li>
<li><b>원본명목배경+평균AIDC부하+저장장치미동작</b>에서원본탭범위−9…+9를동시에탐색.목표1.025 pu와전압오차가가장작은<b>−2</b>를두변압기에동일적용.</li>
<li>보정점LV전압1.018317 pu.ESS의효과나제약통과를목적함수로탭을고르지않음.이후조건별검토는선정탭을고정.</li>
<li>이는명목운전점의고정탭가정이며자동OLTC동작·지연·데드밴드검증은아님.탭0원형결과와모든위반사례를보존.</li></ul>
<h2>4. 실제 AC 조류계산 결과</h2><p>아래는모두AIDC피크26.553 MW·선정탭−2·시설역률0.98조건.공식SimBench배경별부하·발전계수와상위전압설정을그대로적용했으며,표본은동일날짜시계열이아니다.</p>
<table><tr><th>배경조건</th><th>최소MV전압</th><th>최대MV전압</th><th>최대선로부하율</th><th>최대변압기부하율</th><th>판정</th></tr>{table}</table>
<div class="warn"><b>모든 운전조건이 통과한 것은 아님.</b>저부하·고발전lW/lPV에서동일한승압탭−2를유지하면과전압발생.이상태를ESS필요성으로바로해석하지않고,DSO의기본전압운전/탭제어를먼저구분해야함.고발전상태에서는AIDC수요감소가오히려전압상승을키울수있으므로PCC상한하나만으로모든문제를표현할수없음.</div>
<p>중성탭0에서는AIDC피크추가시최소전압<b>{neutral['native_mv_voltage_min_pu']:.6f} pu</b>로하한미달이나PCC자체전압은{neutral['pcc_vm_pu']:.6f} pu이다.따라서PCC전압뿐아니라<b>계통전체모선의전압</b>을검사해야한다.선정탭조건에서원본배경최대선로{peak['max_line_loading_percent']:.2f}%,변압기{peak['max_trafo_loading_percent']:.2f}%로기존설비정격증대없이피크스냅샷이가능하다.</p>
<h2>5. PCC 경계·무효전력·시간축</h2><pre>P_PCC = P_facility(TES) + P_ESS,ch − P_ESS,dis
Q_PCC = P_facility(TES) × tan(acos(0.98)) + Q_ESS; Q_ESS = 0</pre>
<ul><li>Q는ESS유효전력적용전시설부하로산정.순수전P에역률을일괄적용하여ESS가무효전력을공급하는것처럼만들지않음.</li>
<li>기존시설PCC전력에포함된UPS·시설변압기손실을다시더하지않음.이번DSO변압기와외부접속선손실은PCC상류의별도전력.</li>
<li>후속5분운영평가에는SimBench15분평균을3개5분구간에동일배분.새로운5분실측변동확보가아님.기간·요일·시간대정렬은이번단계에서확정하지않음.</li>
<li>3상평형정상상태AC모델.N−1,불평형,고장전류,보호,고조파,FRT,실시간제어는이번범위밖.</li></ul>
<h2>6. 다음 작업</h2><ol><li>대표기간의배경부하·DER프로파일과AIDC프로파일의정렬방식확정.</li><li>모든비교에공통으로적용할DSO전압운전/탭기준정리.원형태생위반과AIDC추가영향을분리.</li><li>계통전체전압·선로·변압기제약으로PCC허용범위를계산.기존25.5 MW시험상한을결과에맞춰강제로재현하지않음.</li><li>그다음해당범위에서기존ESS–TES운영계획과연결.아직네트워크연계운영개선효과를주장하지않음.</li></ol>
<h2>References / 역할</h2><ol>{refs}</ol>
<p><a href="summary.json">요약수치</a> · <a href="model_contract.json">모델선정조건</a> · <a href="../grid_connection_screen_v1/selected_network_mean.json">재로딩가능계통JSON</a> · <a href="../grid_connection_screen_v1/selection.json">탭선정·전체검토</a></p></html>'''
    (OUT/"report_ko.html").write_text(page,encoding="utf-8")
    print(json.dumps({"report":str(OUT/"report_ko.html"),"tap":selected["selected_synchronized_ratio_tap"],"protected_inputs_unchanged":all(summary["protected_input_hash_check"].values())},ensure_ascii=False))


if __name__=="__main__":
    main()
