"""Readable report and scientific figures for a finite reinforcement screen."""
from __future__ import annotations
import html
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/ieee33_reinforcement_v1"


def read(name,folder=OUT):
    return json.loads((folder/name).read_text(encoding="utf-8"))


def designation(c):
    return f'{c["circuits"]}회선 × {c["area_mm2"]} mm², 전원 {c["source_voltage_pu"]:.3f} p.u.'


def fmt(x,n=3):
    return "—" if x is None else f"{x:.{n}f}"


def failure(r):
    if not r.get("converged"): return "조류해 미확인"
    reasons=[]
    if r.get("under_voltage_buses"): reasons.append("저전압")
    if r.get("over_voltage_buses"): reasons.append("과전압")
    if r.get("overloaded_branches"): reasons.append("선로 과부하")
    if r.get("head_overloaded"): reasons.append("상위 공급한계")
    return ", ".join(reasons) if reasons else "만족"


def figures(d,old):
    colors={"nominal_minimum":"#087f8c","allowance_alternative":"#6a63a1"}
    labels={"nominal_minimum":"Nominal minimum","allowance_alternative":"Allowance alternative"}
    fig,axes=plt.subplots(1,2,figsize=(14,5.3))
    axes[0].plot(range(1,34),old["voltage_profiles"]["bus2_peak"],color="#c26a3e",lw=2,label="Unreinforced: source 1.000 p.u.")
    before=read("no_storage_replay.json",ROOT/"outputs/ieee33_1266kv_screen_v1")
    axes[1].plot([r["time_h"] for r in before],[r["max_line_loading_percent"] for r in before],color="#c26a3e",lw=1.6,label="Unreinforced: 600 A root limit")
    for label,e in d["evaluations"].items():
        c=e["candidate"]
        axes[0].plot(range(1,34),e["peak_voltage_profile_pu"],color=colors[label],lw=2,
                     label=f'{labels[label]}: {c["circuits"]} x {c["area_mm2"]} mm2, {c["source_voltage_pu"]:.3f} p.u.')
        rows=read(e["replays"]["no_storage"]["file"])
        axes[1].plot([r["time_h"] for r in rows],[r["root_loading_percent"] for r in rows],color=colors[label],lw=1.6,label=labels[label]+" (catalog rating)")
    for y in (.95,1.05): axes[0].axhline(y,color="#b34549",ls="--",lw=1)
    axes[1].axhline(100,color="#b34549",ls="--",lw=1,label="Continuous-current limit")
    axes[0].set(title="Bus voltages at the unchanged AIDC peak",xlabel="Bus number",ylabel="Voltage (p.u.)",xlim=(1,33),ylim=(.885,1.06))
    axes[1].set(title="Root-corridor loading across the frozen profile",xlabel="Elapsed time (h)",ylabel="Current / candidate current limit (%)",xlim=(0,120),ylim=(50,265))
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False); ax.grid(alpha=.2); ax.legend(fontsize=8,loc="best")
    fig.suptitle("12.66 kV IEEE 33-bus: preserve AIDC scale, reinforce corridor 1-2",fontsize=16,fontweight="bold",color="#183f56")
    fig.text(.5,.01,"Same 22.893 MW mean / 26.553 MW peak PCC input | 1 km equivalent route assumed | Steady-state AC study, not connection approval",ha="center",fontsize=9,color="#536b78")
    fig.tight_layout(rect=(0,.035,1,.92))
    for ext in ("png","svg"): fig.savefig(OUT/f"reinforcement_summary.{ext}",dpi=190,bbox_inches="tight")
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12.5,4.5),sharey=True)
    bgs=[0,.5,1,1.1,1.2,1.5]; pfs=[.95,.98,1]
    for ax,(label,e) in zip(axes,d["evaluations"].items()):
        grid=np.zeros((len(pfs),len(bgs))); text={}
        for r in e["sensitivity"]:
            if r["kind"]!="background_pf" or r["mode"]!="peak_charge_envelope": continue
            i=pfs.index(r["value"]["pf"]); j=bgs.index(r["value"]["background"])
            grid[i,j]=r["feasible"]
            text[i,j]=f'{"PASS" if r["feasible"] else "FAIL"}\nVmin {r["min_voltage_pu"]:.3f}'
        from matplotlib.colors import ListedColormap
        ax.imshow(grid,cmap=ListedColormap(["#fae1d5","#d9efea"]),vmin=0,vmax=1,aspect="auto")
        for (i,j),t in text.items(): ax.text(j,i,t,ha="center",va="center",fontsize=8,color="#1e3c4d")
        ax.set(xticks=range(len(bgs)),xticklabels=bgs,yticks=range(len(pfs)),yticklabels=pfs,xlabel="Original background-load multiplier",title=labels[label])
    axes[0].set_ylabel("Facility lagging power factor")
    fig.suptitle("Applicability check: AIDC peak + maximum charging envelope",fontsize=14,fontweight="bold",color="#183f56")
    fig.text(.5,.01,"Source voltage and reinforcement fixed; catalog current limits. PASS is conditional, not an uncertainty guarantee.",ha="center",fontsize=9,color="#536b78")
    fig.tight_layout(rect=(0,.04,1,.9))
    for ext in ("png","svg"): fig.savefig(OUT/f"applicability_sensitivity.{ext}",dpi=190,bbox_inches="tight")
    plt.close(fig)


def main():
    d=read("results.json"); cfg=read("model_contract.json"); records=read("candidate_screen.json")
    old=read("results.json",ROOT/"outputs/ieee33_1266kv_screen_v1")
    figures(d,old)
    names={"nominal_minimum":"정격 기준 최소안","allowance_alternative":"허용전류 −10%·저항 +10% 고려 대안"}
    evals=d["evaluations"]; n=evals["nominal_minimum"]; a=evals["allowance_alternative"]
    selected={"status":"screened_candidates_pending_user_review_not_adopted_design",
              "generation_config":"configs/ieee33_reinforcement_v1.json",
              "config_sha256":d["config_sha256"],"fixed":cfg["fixed"],
              "assumed_route_length_km":cfg["search"]["assumed_route_length_km"],
              "candidates":{k:{"selection":e["candidate"],"root_parameters":e["root_parameters"]} for k,e in evals.items()},
              "preferred_for_next_review":"allowance_alternative",
              "note":"No automatic adoption or dispatch reoptimization. Both configurations require the stated assumptions."}
    (OUT/"selected_candidates.json").write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding="utf-8")
    checks={"protected_file_count":len(d["protected_input_hashes"]),"unchanged":all(d["protected_inputs_unchanged"].values()),
            "maximum_nr_bfsw_voltage_difference_pu":max(e["solver_crosscheck"]["max_voltage_difference_pu"] for e in evals.values()),
            "maximum_active_balance_error_mw":max(v["max_balance_error_mw"] for e in evals.values() for v in e["replays"].values()),
            "maximum_reactive_balance_error_mvar":max(v["max_q_balance_error_mvar"] for e in evals.values() for v in e["replays"].values())}
    if (OUT/"test_results.xml").exists():
        tree=ET.parse(OUT/"test_results.xml").getroot()
        checks["pytest"]=(tree.find("testsuite") if tree.tag=="testsuites" else tree).attrib
    (OUT/"verification.json").write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding="utf-8")
    cases=[("보강 전",old["no_storage_replay"]),("정격 기준 최소안",n["replays"]["no_storage"]),("여유 대안",a["replays"]["no_storage"])]
    comparison="".join(f'<tr><td>{name}</td><td>{r["min_voltage_pu"]:.4f}</td><td>{r["max_line_loading_percent"]:.2f}%</td><td>{r["max_head_s_mva"]:.3f}</td><td>{r["feasible_intervals"]}/{r["intervals"]}</td></tr>' for name,r in cases)
    cable_rows="".join(f'<tr><td>{c["area_mm2"]}</td><td>{c["r_ohm_per_km"]:.3f}</td><td>{c["x_ohm_per_km_at_50hz"]:.3f}</td><td>{c["c_nf_per_km"]}</td><td>{c["buried_trefoil_current_ka"]*1000:.0f}</td><td>[{c["reference"]}]</td></tr>' for c in cfg["cables"])
    frontier=[]
    for circuits in cfg["search"]["parallel_circuit_candidates"]:
        for cable in cfg["cables"]:
            subset=[r for r in records if r["candidate"]["circuits"]==circuits and r["candidate"]["area_mm2"]==cable["area_mm2"]]
            values=[]
            for key in ("nominal","allowance"):
                v=[r["candidate"]["source_voltage_pu"] for r in subset if r[key]["all_feasible"]]
                values.append(f'{min(v):.3f}' if v else "통과 없음")
            frontier.append(f'<tr><td>{circuits} × {cable["area_mm2"]}</td><td>{values[0]}</td><td>{values[1]}</td></tr>')
    detailed=[]
    mode_names={"no_storage":"저장장치 미사용","existing_joint":"기존 ESS–TES 공동운전","max_charge_envelope":"매 구간 최대 충전 전력 포락선","derated_charge_envelope":"포락선 + 허용전류 −10%·저항 +10%"}
    for label,e in evals.items():
        for mode,r in e["replays"].items():
            detailed.append(f'<tr><td>{names[label]}</td><td>{mode_names[mode]}</td><td>{r["pcc_peak_mw"]:.3f}</td><td>{r["min_voltage_pu"]:.4f}–{r["max_voltage_pu"]:.4f}</td><td>{r["max_root_loading_percent"]:.2f}%</td><td>{r["max_head_s_mva"]:.3f}</td><td>{r["feasible_intervals"]}/{r["intervals"]}</td></tr>')
    off=[]
    for label,e in evals.items():
        for r in e["selection_points"]["points"]:
            if r["name"].startswith("aidc_off"):
                off.append(f'<tr><td>{names[label]}</td><td>{r["background"]:.1f}</td><td>{r["min_voltage_pu"]:.4f}–{r["max_voltage_pu"]:.4f}</td><td>{failure(r)}</td></tr>')
    stress=[]
    for label,e in evals.items():
        for r in e["sensitivity"]:
            if r["mode"]!="peak_charge_envelope" and r["kind"]!="one_parallel_circuit_out_of_service": continue
            if r["kind"]=="background_pf": continue
            stress.append(f'<tr><td>{names[label]}</td><td>{r["kind"]}</td><td>{html.escape(str(r["value"]))}</td><td>{r["min_voltage_pu"]:.4f}</td><td>{r["root_loading_percent"]:.2f}%</td><td>{failure(r)}</td></tr>')
    references="".join(f'<li id="{r["id"]}">[{r["id"]}] <a href="{html.escape(r["url"],quote=True)}">{html.escape(r["title"])}</a> {html.escape(r.get("publication",""))}<br>{html.escape(r.get("use","제조사 공개 전기적 특성·허용전류 값. 실제 포설 설계 조건의 인증을 대신하지 않음."))}</li>' for r in cfg["references"])
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>12.66 kV · 현재 AIDC 규모 유지: 최소 보강 후보 검토</title><style>
body{{font-family:'Malgun Gothic',Arial,sans-serif;max-width:1220px;margin:30px auto;padding:0 24px;line-height:1.8;color:#213e51}}h1{{font-size:29px}}h2{{font-size:23px;margin-top:32px;border-bottom:1px solid #cddce5}}h3{{font-size:19px}}.note,.warn{{padding:17px;border-left:5px solid #087f8c;background:#edf7f6}}.warn{{border-color:#ba623b;background:#fff4e8}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #d6e2e8;text-align:left;vertical-align:top}}th{{background:#eff4f7}}img{{width:100%;height:auto}}.scroll{{overflow-x:auto}}pre{{padding:15px;background:#eff4f7;white-space:pre-wrap;overflow-wrap:anywhere}}li{{margin:6px 0}}a{{color:#126c92}}.small{{font-size:13px;color:#577181}}
</style><h1>12.66 kV와 현재 AIDC 규모를 유지한 최소 계통 보강 후보</h1>
<div class="note"><b>연구용 후보 선정·정상상태 AC 검증 완료, 사용자 확정 전.</b><br>
정격 기준 최소안: <b>{designation(n['candidate'])}</b>.<br>
허용전류 −10%·저항 +10%를 동시에 고려한 대안: <b>{designation(a['candidate'])}</b>.<br>
두 안 모두 Bus 1–2 한 구간만 변경한다. 계통 공칭전압, AIDC 부하, ESS–TES 사양은 유지했다.</div>
<p><b>‘최소’의 범위:</b> 1–2 공급구간 보강 + 전원 전압 조정이라는 제한된 후보군에서, 변경 구간 수 → 병렬 회선 수 → 도체 단면적 → 전원 전압 순으로 선정했다. 전체 투자비 최소화, 모든 접속 위치·보강 기술에 대한 전역 최적해, 실제 케이블 시공안은 아니다.</p>
<h2>1. 무엇을 고정했고 무엇을 바꿨는가?</h2>
<ul><li>공칭전압 <b>12.66 kV</b>, packaged case의 <b>60 Hz</b>, IEEE33 모선·방사상 접속 구조 유지. AIDC는 기존 기본 접속점 <b>Bus 2</b>.</li>
<li>AIDC 무저장 PCC: 평균 <b>{d['frozen_profile']['mean_mw']:.3f} MW</b>, 최대 <b>{d['frozen_profile']['peak_mw']:.3f} MW</b>. <b>1,438개 5분 평균값</b>(119시간 50분)을 그대로 사용. 전력 프로파일 크기 조정 없음.</li>
<li>ESS <b>1 MW / 4 MWh</b>, TES <b>3 MWth / 10 MWhth</b>, 기존 효율·열모델·스케줄 유지. ESS 무효전력은 0, 시설 PF=0.98 지상은 연구 가정.</li>
<li>배경부하 3.715 MW + 2.3 MVAr 유지. 상위 공급한계 40 MVA와 나머지 선로 600 A는 기존의 <b>연구용 한계값</b>이지 원형 계통의 실제 설비 정격이 아니다.</li>
<li>변경: <b>1–2 구간의 케이블 등가 R/X/C·허용전류</b>, <b>Bus 1 고정 전압 설정값</b>. 실물 변압기·OLTC 제어기는 추가하지 않음.</li></ul>
<h2>2. 왜 이 보강을 검토했는가?</h2>
<p>보강 전 AIDC 최대부하에서는 공급구간 1–2의 전류가 약 <b>1.464 kA</b>로 기존 600 A 가정을 초과했다. 다른 구간을 통과하는 AIDC 전력이 거의 없는 Bus 2 접속이므로 우선 이 공급구간만 강화했다. 전압도 부족하므로 계통 전체 선로를 임의 축소하지 않고 전원 전압 조정과 공급구간 임피던스를 함께 검사했다.</p>
<ul><li>[ENH33]은 IEEE33 연구에서 전원 전압 제어와 0.95–1.05 p.u. 제약을 사용하는 선행 근거. 이번 0.005 p.u. 탐색 간격은 자체 계산 격자이며 특정 OLTC의 실제 탭 사양이 아니다.</li>
<li>[COPLAN]은 데이터센터–배전계통 공동계획에서 선로 유형·상위 설비 확장을 고려한 선행 사례. 해당 논문의 투자최적화 전체를 재현한 것은 아니다.</li>
<li>[C300–C630] 제조사 공개자료로 물리적으로 연결된 R/X/C·허용전류 후보를 구성. 허용전류만 올리고 전기적 특성을 그대로 두는 방법은 쓰지 않았다.</li></ul>
<div class="scroll"><table><tr><th>동 도체(mm²/상/회선)</th><th>R₉₀℃(Ω/km)</th><th>X₅₀Hz(Ω/km)</th><th>C(nF/km)</th><th>지중 trefoil 허용전류(A/회선)</th><th>근거</th></tr>{cable_rows}</table></div>
<p>12/20(24) kV는 케이블 절연 등급이며 계통 공칭전압을 20 kV로 바꾼다는 뜻이 아니다. 한 회선은 3상 단심 케이블 3개로 구성되는 등가 회선이다.</p>
<pre>동일 케이블 n회선, 경로 길이 L:
R_eq = R₉₀℃ × L / n      X_eq = X₅₀Hz × (60/50) × L / n
C_eq = C × L × n         I_limit,total = n × I_catalog × current_factor
0.95 ≤ |V_bus| ≤ 1.05    I_branch ≤ I_limit    |S_head| ≤ 40 MVA</pre>
<div class="warn"><b>출처 값과 연구 가정의 구분:</b> 경로 길이 <b>1 km는 가정</b>이며 IEEE33 원자료에서 확인한 실제 노선 길이가 아니다. 제조사 50 Hz 특성 중 X는 주파수 비로 보정했고 R 및 허용전류는 등가 연구값으로 사용했다. 실제 60 Hz 포설·집합·토양 조건의 허용전류는 별도 설계가 필요하다. 현재 웹페이지와 오래된 2021 PDF의 허용전류가 달라, 이번에는 링크한 <b>현재 웹페이지의 더 작은 값</b>으로 통일했다. −10%/−20% 민감도는 인증된 집합 보정계수가 아니다.</div>
<h2>3. 후보 탐색 및 최소안의 의미</h2>
<p>전압 조정만 적용하는 11개 대조안 + 케이블 4종 × 1–4회선 × 전원 전압 11단계 = <b>{d['candidates_count']}개</b>. 각 후보를 AIDC 최소·평균·최대부하, 최대부하+정격 충전, 최소부하−정격 방전, AIDC 정지·배경 100/50/10/0%의 <b>9개 운전점</b>으로 검사했다. 정격 조건과 (허용전류 −10%, 저항 +10%) 조건을 각각 검사했다.</p>
<p>전압만 조정하고 600 A 공급선로를 유지한 대조안은 모두 탈락했다. 아래는 각 구성에서 9개 운전점을 모두 만족한 가장 낮은 전원 전압이다.</p>
<table><tr><th>회선 수 × 단면적(mm²)</th><th>정격 조건 최저 설정(p.u.)</th><th>허용전류 −10%·저항 +10%</th></tr>{''.join(frontier)}</table>
<p>정격 통과 {d['nominal_passing_count']}개, 여유 조건 통과 {d['allowance_passing_count']}개. 회선 수를 우선하는 선정 규칙이므로, 재료 총량이나 비용이 가장 작다고 해석해서는 안 된다. 보강된 병렬회선은 단일 등가 공급구간으로 모델링했다.</p>
<h2>4. 약 5일 전체 프로파일 검증 결과</h2>
<img src="reinforcement_summary.png" alt="보강 전후 모선전압 및 전체 프로파일 선로부하율 비교">
<table><tr><th>무저장 운전 비교</th><th>계통 최저전압(p.u.)</th><th>최대 선로 부하율</th><th>최대 상위 공급(MVA)</th><th>전체 제약 만족 구간</th></tr>{comparison}</table>
<p>부하율은 각 구성의 허용전류로 나눈 값이므로, 선로전류 자체가 60% 줄었다는 뜻이 아니다. 대개는 병렬회선 증가로 허용전류가 커지고 전압·손실도 개선된 효과이다.</p>
<div class="scroll"><table><tr><th>보강안</th><th>재생 조건</th><th>최대 PCC(MW)</th><th>전압 범위(p.u.)</th><th>최대 1–2 부하율</th><th>최대 상위 공급(MVA)</th><th>제약 만족 구간</th></tr>{''.join(detailed)}</table></div>
<p><b>충전 포락선</b>은 각 시점의 기존 PCC 수요에 ESS 1 MW와 TES의 전기적 충전 증가분 {d['storage']['tes_charge_e_mw']:.3f} MW를 더한 순간 전기적 스트레스다. <b>약 5일 내내 최대 충전하는 실행 가능한 SOC 스케줄을 의미하지 않는다.</b> 기존 공동운전은 25.5 MW 외생 PCC 상한에 대해 계산했던 계획을 그대로 재생한 것으로, 이 계통의 DOE나 새로운 최적운전 결과가 아니다.</p>
<h3>AIDC 정지 및 경부하 확인</h3><table><tr><th>보강안</th><th>기존 배경부하 배율</th><th>전압 범위(p.u.)</th><th>판정</th></tr>{''.join(off)}</table>
<p>AIDC가 연결된 상태를 기준으로 보강을 선택하고, 같은 설비·전압 설정에서 AIDC 정지 시 과전압 여부도 함께 확인했다. 별도의 무AIDC 보강안을 먼저 확정한 것이 아니다.</p>
<h2>5. 적용 범위와 민감도</h2><img src="applicability_sensitivity.png" alt="배경부하 및 시설역률 변화에 따른 제약 만족 여부">
<p>공칭 최소안은 경계 가까이 운전하므로 조건 변화에 강하다는 뜻이 아니다. 아래 민감도는 선정한 전압·구성을 다시 최적화하지 않고 실시했다. 전기적 순간 정격 충전 조건 기준이며, 회선 고장은 무저장 최대부하로 별도 검사했다.</p>
<div class="warn"><b>중요한 적용 한계:</b> 두 후보 모두 배경부하를 100%에서 <b>110%</b>로 늘리거나 공급경로를 1 km에서 <b>2 km</b>로 늘리면 0.95 p.u. 미만 저전압이 발생했다. 전원 설정이 1.040에서 1.035 p.u.로 내려가도 저전압이 발생했다. 3회선 대안은 허용전류·저항 오차에 대한 대안이지, 배경부하·전원 전압의 모든 변동을 흡수하는 강건 설계가 아니다. 후속 시간가변 배경부하를 정하면 그 조건에서 기준계통 및 DSO 운전 가능성을 다시 검사해야 한다.</div>
<details><summary>길이·허용전류·공급한계·전압 설정·회선 고장 결과 펼치기</summary><div class="scroll"><table><tr><th>보강안</th><th>변경 변수</th><th>값</th><th>최저전압</th><th>공급구간 부하율</th><th>판정</th></tr>{''.join(stress)}</table></div></details>
<ul><li>고정 배경·고정 PF에서 계산한 순간 PCC 허용 경계: 정격 최소안 <b>{n['static_limit_not_doe']['capacity_mw']:.3f} MW</b>, 여유 대안 <b>{a['static_limit_not_doe']['capacity_mw']:.3f} MW</b>. <b>시간가변 DOE가 아니고</b>, 저장장치의 지속시간을 보장하지 않는다.</li>
<li>병렬회선 1개 정지 조건은 두 후보 모두 전체 제약을 만족하지 못했다. <b>N−1 설계가 아니다.</b></li>
<li>선로길이·토양·포설·상위 임피던스·실제 배경부하가 바뀌면 재검토해야 한다. 단락·보호·고조파·불평형·과도현상 및 grid-code 종합 준수는 이번 검증에 포함하지 않았다.</li></ul>
<h2>6. 이번 단계의 결론과 다음 연구 단계</h2>
<ul><li><b>해결한 것:</b> 기존 AIDC 규모가 연결된 12.66 kV IEEE33 등가 테스트베드에서, 전압·열적 선로·상위 공급 제약을 만족하는 제한된 보강 후보를 찾았다. AIDC나 저장장치 규모를 축소할 필요 없이 후속 협조운영 연구를 위한 조건부 기준계통을 만들 수 있다.</li>
<li><b>권장 사용 구분:</b> 정격 최소안은 ‘이 정도까지가 후보군의 최소 변경’이라는 비교 근거, 여유 대안은 허용전류·저항의 단순 변화까지 고려한 <b>후속 연구용 우선 검토 후보</b>. 여유 대안도 실제 공학적 설계 확정은 아니다.</li>
<li><b>다음:</b> 사용자가 기준 후보를 검토한 후 배경부하·접속점 조건을 고정하고, DSO의 시간별 PCC 허용범위 계산 → AIDC ESS–TES 스케줄링 → AC 조류 재검증으로 연결. 계통을 임의로 약화시켜 개선효과를 만들지 말고, 혼잡 시나리오를 사전에 명시해야 한다.</li></ul>
<h2>7. 재현성과 품질 확인</h2><pre>{html.escape(json.dumps(checks,ensure_ascii=False,indent=2))}</pre>
<p>원본 시설·프로파일·직전 계통 결과 파일의 해시를 보존했다. peak 상태에서 Newton–Raphson과 backward/forward sweep 조류해를 교차 비교했다. 5분 정상상태 검증으로 초단위 전압 응답이나 HIL 성능을 주장하지 않는다.</p>
<pre>&amp; .venv-grid/Scripts/python.exe scripts/screen_ieee33_reinforcement_v1.py
&amp; .venv-grid/Scripts/python.exe -m pytest -q tests/test_ieee33_reinforcement_v1.py --junitxml=outputs/ieee33_reinforcement_v1/test_results.xml
&amp; .venv-grid/Scripts/python.exe scripts/report_ieee33_reinforcement_v1.py</pre>
<h2>참고자료</h2><ol>{references}</ol>
<p class="small">생성 UTC: {d['created_utc']} · 모든 수치는 첨부 JSON 결과에서 생성 · 검토 전 후보이며 기존 시설 모델을 교체하지 않음.</p></html>'''
    (OUT/"report_ko.html").write_text(page,encoding="utf-8")
    print(OUT/"report_ko.html")


if __name__=="__main__": main()
