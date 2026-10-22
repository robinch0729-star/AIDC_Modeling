"""Run the audited fixed-node, trace-shaped M3+R3 prototype (no refitting).

Prepare traffic_bins.csv/provenance.json using fetch_inference_traffic_sample.py
first. No synthetic traffic fallback is permitted by this runner.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import savemat

from aidc_facility.time_varying_it import (
    FixedService, make_dispatch, profile_metrics, reference_design_scale,
    simulate_case, validate_dispatch,
)
from nlr_data.final_it_profile_generator import FinalITProfileGenerator
from aidc_facility.arrival_trace import verify_saved_sample


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md_table(frame: pd.DataFrame, columns: list[str]) -> str:
    rows = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for values in frame[columns].itertuples(index=False, name=None):
        rows.append("| " + " | ".join(f"{v:.6f}" if isinstance(v, (float, np.floating)) else str(v) for v in values) + " |")
    return "\n".join(rows)


def plots(output: Path, profiles: pd.DataFrame, nodes: pd.DataFrame, dispatch: pd.DataFrame, metrics: pd.DataFrame, traffic_bin: float) -> None:
    colors = {"B_constant": "#2878a3", "C_trace_shaped": "#d25d27"}
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    for service, part in dispatch.loc[dispatch.case.eq("C_trace_shaped")].groupby("service", sort=False):
        axes[0].step(np.r_[part.time_s, part.time_s.iloc[-1] + part.bin_s.iloc[-1]],
                     np.r_[part.node_offered_arrival_rps, part.node_offered_arrival_rps.iloc[-1]], where="post", label=service)
    axes[0].set_ylabel("Arrival rate per node\n(request/s)")
    axes[0].set_title(f"Measured arrival counts -> scaled scenario demand ({traffic_bin:g} s bins, no time compression)")
    axes[0].legend()
    for case in colors:
        part = profiles.loc[profiles.case.eq(case) & profiles.scenario_id.eq(0)]
        axes[1].plot(part.time_s, part.total_it_mw, color=colors[case], lw=1, label=case)
        axes[1].plot(part.time_s, part.m3_total_mw, color=colors[case], lw=1.3, ls="--", label=f"{case}: M3 sum")
        axes[2].plot(part.time_s, part.residual_total_mw * 1000, color=colors[case], lw=.8, label=case)
    axes[1].set_ylabel("Reported IT power (MW)")
    axes[1].set_title("Quasi-steady prototype: traffic-bin edges are NOT validated fast power transitions")
    axes[1].legend(ncol=2, fontsize=8)
    axes[2].set_ylabel("Aggregate residual (kW)")
    axes[2].set_xlabel("Elapsed time (s)")
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(output / "traffic_to_power.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    for case, ax in zip(colors, axes):
        part = profiles.loc[profiles.case.eq(case)]
        for sid, curve in part.groupby("scenario_id"):
            ax.plot(curve.time_s, curve.total_it_mw, lw=.75, alpha=.7, label=f"residual seed {sid}")
        ax.set_title(case + " | same traffic; different residual realizations")
        ax.set_ylabel("Reported IT power (MW)")
        ax.grid(alpha=.2)
        ax.legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("Elapsed time (s)")
    fig.savefig(output / "matched_profiles.png", dpi=160)
    plt.close(fig)

    chosen = metrics.loc[metrics.case.isin(colors)].groupby("case", sort=False).mean(numeric_only=True)
    columns = ["max_abs_ramp_0p1s_mw_per_s", "max_within_bin_abs_ramp_0p1s_mw_per_s", "p95_abs_traffic_bin_mean_ramp_mw_per_s"]
    labels = ["Raw 0.1 s maximum\n(includes bin-edge artifacts)", "Within-bin 0.1 s maximum\n(excludes rate-transition edges)", f"P95 {traffic_bin:g} s bin-mean ramp\n(coarse scenario metric)"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, column, label in zip(axes, columns, labels):
        vals = [chosen.loc[case, column] for case in colors]
        ax.bar(["B", "C"], vals, color=list(colors.values()))
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("MW/s")
        ax.grid(axis="y", alpha=.2)
    fig.savefig(output / "boundary_ramp_diagnostic.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    c = nodes.loc[nodes.case.eq("C_trace_shaped")]
    for ax, (service, part) in zip(axes, c.groupby("service", sort=False)):
        ax.plot(part.time_s, part.example_node_power_kw, lw=.8, label="single node realization")
        ax.set_title(service + " | continuous R3 state; quasi-steady M3 input")
        ax.set_ylabel("NLR-reported power (kW)")
        ax.grid(alpha=.2)
        ax.legend()
    axes[-1].set_xlabel("Elapsed time (s)")
    fig.savefig(output / "example_node_profiles.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traffic-dir", type=Path, default=ROOT / "data/raw/azure_inference_sample")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/time_varying_it_v1")
    parser.add_argument("--scenarios", type=int, default=5)
    parser.add_argument("--seed", type=int, default=29092026)
    parser.add_argument("--burn-in-s", type=float, default=120)
    args = parser.parse_args()
    traffic_path = args.traffic_dir / "traffic_bins.csv"
    provenance_path = args.traffic_dir / "provenance.json"
    if not traffic_path.is_file() or not provenance_path.is_file():
        parser.error("Real traffic sample and provenance are required; run fetch_inference_traffic_sample.py first")
    started = time.perf_counter()
    traffic, provenance = verify_saved_sample(args.traffic_dir)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline_dir = ROOT / "outputs/heterogeneous_equivalent_facility_17mw"
    baseline_config = json.loads((baseline_dir / "generation_config.json").read_text(encoding="utf-8"))
    protected = [baseline_dir / name for name in ("generation_config.json", "facility_profiles_long.csv", "scenario_summary.csv")]
    hashes_before = {str(p): sha256(p) for p in protected}
    generator = FinalITProfileGenerator.from_benchmark_artifact(ROOT / "outputs/r3_high_rate_benchmark")
    services = (
        FixedService("code", "InstructCoder", int(baseline_config["unit_counts"]["code_generation"]), 55.0),
        FixedService("conversation", "mlperf-inference-llama2-data", int(baseline_config["unit_counts"]["general_conversation"]), 42.5),
    )
    target = 20.0 / 1.15
    scale = reference_design_scale(generator, services, target)
    count_columns = {"code": "code_count", "conversation": "conversation_count"}
    dispatches, profiles, nodes, audits, statuses = [], [], [], [], {}
    for case, variable in (("B_constant", False), ("C_trace_shaped", True)):
        dispatch = make_dispatch(traffic, services, count_columns, scale, variable=variable)
        dispatch.insert(0, "case", case)
        statuses[case] = validate_dispatch(generator, dispatch, 0.1)
        dispatches.append(dispatch)
        print(f"Simulating {case}: {args.scenarios} residual realizations, {sum(s.node_count for s in services)} fixed units", flush=True)
        p, n, a = simulate_case(generator, dispatch, services, case=case, scenario_count=args.scenarios, seed=args.seed, burn_in_s=args.burn_in_s)
        profiles.append(p)
        nodes.append(n)
        audits.append(a)
    profile = pd.concat(profiles, ignore_index=True)
    node = pd.concat(nodes, ignore_index=True)
    dispatch = pd.concat(dispatches, ignore_index=True)
    node_audit = pd.concat(audits, ignore_index=True)
    traffic_bin = float(traffic.bin_s.iloc[0])
    metric_rows = []
    for (case, sid), frame in profile.groupby(["case", "scenario_id"], sort=False):
        metric_rows.append({"case": case, "scenario_id": sid, **profile_metrics(frame, 0.1, traffic_bin)})
    # Historical baseline: no rerun or modification; not the matched causal control.
    old = pd.read_csv(baseline_dir / "facility_profiles_long.csv")
    for sid, frame in old.groupby("scenario_id"):
        metric_rows.append({"case": "A_historical", "scenario_id": sid, **profile_metrics(frame, 0.1, traffic_bin)})
    metrics = pd.DataFrame(metric_rows)
    summary = metrics.groupby("case", sort=False).mean(numeric_only=True).reset_index().drop(columns="scenario_id")
    profile.to_csv(output / "facility_profiles.csv", index=False)
    node.to_csv(output / "example_node_profiles.csv", index=False)
    dispatch.to_csv(output / "request_dispatch.csv", index=False)
    node_audit.to_csv(output / "node_power_checks.csv", index=False)
    metrics.to_csv(output / "scenario_metrics.csv", index=False)
    summary.to_csv(output / "comparison_summary.csv", index=False)
    source_bins = traffic.copy()
    source_bins.to_csv(output / "source_traffic_bins.csv", index=False)
    mat = {"time_s": profile.loc[profile.case.eq("B_constant") & profile.scenario_id.eq(0), "time_s"].to_numpy()}
    for case in ("B_constant", "C_trace_shaped"):
        mat[case + "_it_mw"] = np.stack([g.total_it_mw.to_numpy() for _, g in profile.loc[profile.case.eq(case)].groupby("scenario_id")])
    savemat(output / "it_profiles.mat", mat, do_compression=True)
    demand = dispatch.groupby(["case", "service"]).apply(lambda g: pd.Series({"offered_requests": float((g.offered_rps*g.bin_s).sum()), "admitted_requests": float((g.admitted_rps*g.bin_s).sum()), "excess_requests": float((g.excess_rps*g.bin_s).sum())}), include_groups=False).reset_index()
    demand.to_csv(output / "request_accounting.csv", index=False)
    equal_offered = all(np.isclose(*demand.loc[demand.service.eq(s.key), "offered_requests"].to_numpy()) for s in services)
    config = {
        "stage": "trace-shaped fixed-node M3+R3 prototype", "services": [asdict(s) for s in services],
        "reference_input_rate_scale": scale, "target_reference_it_mw": target,
        "pue_1p15_role": "reference sizing arithmetic only; no cooling/PCC profile generated",
        "scenario_count": args.scenarios, "seed": args.seed, "interval_s": 0.1,
        "traffic_bin_s": traffic_bin, "duration_s": float(traffic.bin_s.sum()), "burn_in_s": args.burn_in_s,
        "traffic_provenance": provenance, "traffic_file_sha256": sha256(traffic_path),
        "traffic_hashes_and_raw_rebin_verified": True,
        "node_count_interpretation": "6487 NLR-equivalent node/instance scenario units; not an observed physical fleet",
        "locked_parameters": {name: sha256(generator.source_dir / name) for name in ("train_validation_m3_parameters.csv", "train_validation_residual_parameters.csv")},
        "support_checks": statuses, "same_mean_offered_demand_B_C": equal_offered,
        "post_generation_scaling": False, "finite_window_residual_demeaning": False,
        "residual_state_reset_at_traffic_boundary": False,
        "physical_facility_power_validated": False, "subsecond_rate_transition_validated": False,
        "idle_or_autoscaling_modeled": False, "queue_or_completion_throughput_modeled": False,
        "parameter_refit_or_model_reselection": False,
        "nominal_reference_tail_check": {
            "reference_kw": 3.52, "not_a_hard_cap": True,
            "samples_above_reference": int(node_audit.samples_above_nominal_reference.sum()),
            "node_time_samples": int(node_audit.node_time_samples.sum()),
            "maximum_generated_node_kw": float(node_audit.maximum_node_reported_power_kw.max()),
            "note": "unbounded Gaussian-mixture tails are not a hard-constrained hardware power model; no clipping applied",
        },
        "power_boundary": "NLR upstream-reported CPU/GPU aggregate; CPU domain overlap unresolved, not full-server/PCC meter",
        "traffic_mapping": "Azure service temporal shape -> NLR prompt class is a scenario assumption; amplitudes resized, time unchanged",
        "R3_extension": "same innovation law; continuous state; coefficients updated at traffic bins; no global or block demeaning",
        "baseline_hashes": hashes_before, "historical_baseline_unchanged": all(sha256(Path(p)) == h for p, h in hashes_before.items()),
        "runtime_s": time.perf_counter() - started,
        "references": {"PL3": "https://arxiv.org/html/2604.07345v1#S2.SS3.SSS2", "PL43": "https://arxiv.org/html/2603.18383v1", "DynamoLLM": "https://doi.org/10.1109/HPCA61900.2025.00102"},
    }
    (output / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    plots(output, profile, node, dispatch, metrics, traffic_bin)
    columns = ["case", "mean_it_mw", "std_it_mw", "cv_percent", "p95_abs_ramp_0p1s_mw_per_s", "max_abs_ramp_0p1s_mw_per_s", "max_within_bin_abs_ramp_0p1s_mw_per_s"]
    display = summary[["case", "mean_it_mw", "std_it_mw", "cv_percent"]].copy()
    display["std_it_mw"] *= 1000
    display.columns = ["Case", "평균전력_MW", "시간축_표준편차_kW", "변동계수_%"]
    bv = summary.set_index("case").loc["B_constant"]
    cv = summary.set_index("case").loc["C_trace_shaped"]
    totals = provenance["sources"]
    reference_rates = ", ".join(f"{s.key}: {s.reference_rate_rps*scale:.3f} request/s/node" for s in services)
    report = rf"""# 시간가변 inference 요청 기반 IT 부하 합산 — 실행 결과

## 0. 결과의 지위

- **실제 Azure 요청 도착 자료의 시간적 모양을 이용한 연구 시나리오**이며 실제 Azure/NLR 시설전력 재현 검증은 아님.
- M3+R3 계수·모델선정은 고정. 요청 분배층과 연속 residual 상태 연결만 추가.
- 전력 경계는 NLR upstream-reported CPU/GPU 합산. CPU package/core domain 중복 가능성은 미검증; full-server/PCC meter 아님.
- 관측 traffic을 {traffic_bin:g}초로 집계하고 각 구간의 운전점을 유지하는 **준정상 근사**. 계단 경계의 0.1초 전력 급변은 실제 ramp로 해석하면 안 됨.

## 1. 단계별 완료 내용

1. 측정 단위·설정 offered request rate 감사: [감사 문서](../../docs/time_varying_it_model_audit.md).
2. 기존 A 결과 보존 및 hash 검사: {config['historical_baseline_unchanged']}.
3. 실제 요청 자료 일부 구간 취득·출처 저장: `run_config.json`의 traffic_provenance.
4. NLR-equivalent 노드/instance {sum(s.node_count for s in services):,}개를 고정한 시나리오(30/70 근사 단위 구성비)에 균등 분배. 실제 관측 시설의 서버 대수나 처리완료율/SLO 모델은 아님.
5. rate 변화에도 residual 상태를 유지. 구간별 재시작·재중심화·전력 smoothing 미적용.
6. B/C 생성·CSV·MATLAB MAT 출력.
7. 산술·지원범위·인공 경계 ramp 점검 완료. 외부 facility 및 전이응답 검증은 미완료.

## 2. 시나리오와 공정한 비교

- A_historical: 기존 이질적 고정 rate 독립 합산. 과거 비교용이며 직접적인 matched control 아님.
- B_constant: 새 고정노드 균등분배 구조, 서비스별 평균 traffic 유지.
- C_trace_shaped: B와 같은 노드 수·분배·총 offered request량, 시간적 분포만 원자료에 따라 변화.
- B/C 평균 offered demand 일치: {equal_offered}. 초과 요청 수는 `request_accounting.csv`에서 별도 확인.
- 기준점에서만 input 수요 크기를 설계: {reference_rates}. 기준 M3 합 {target:.6f} MW.
- 이 input 크기와 서비스 매핑은 연구 시나리오 가정. 원자료의 절대 시설규모·모델·token 분포와 동일하다는 주장 없음.
- 생성 후 평균을 17.39 MW로 강제 보정하지 않음. PUE 1.15는 설계 기준 산술에만 사용.
- 각 B/C는 동일 traffic에 residual seed {args.scenarios}개. 새로운 관측 traffic {args.scenarios}개나 독립 검증자료가 아님.
- 잔차는 기존 window demeaning을 제거했으므로 A와의 차이에 이 변경도 포함. B/C는 동일 처리이므로 직접 비교 가능.

## 3. 결과

{md_table(display, list(display.columns))}

- 실제 입력 관측창: `{provenance['start_in_source_clock']}` ~ `{provenance['end_exclusive_in_source_clock']}`(끝점 제외).
- 원자료 요청 수: code {totals['code']['window_request_count']:,}건, conversation {totals['conversation']['window_request_count']:,}건. 이 도착률의 시간적 모양을 시설 시나리오 크기로 재설정했으며, 요청 기록을 그대로 6천 노드에 재생한 것은 아님. 원본 byte range와 sample hash 및 재집계 bin의 일치를 자동 검사함.
- 시간가변 C의 표준편차 {cv['std_it_mw']*1000:.3f} kW 중 대부분은 평균 운전점 변화(M3 합의 표준편차 {cv['m3_std_mw']*1000:.3f} kW)에서 발생. 합산 residual 표준편차는 {cv['residual_std_mw']*1000:.3f} kW로 여전히 작음.
- B/C 평균전력 차이는 {(cv['mean_it_mw']-bv['mean_it_mw'])*1000:.3f} kW. 이번에는 admission guard 초과가 없어 요청 총량은 같으며, 비선형 M3와 유한 residual 실현으로 출력 평균이 달라짐.
- **요청량 변화가 시설 평균 운전점을 움직이는 연결은 구현했지만, 고속 전이응답의 물리 검증은 아직 완료하지 않음.**

상세 수치:

{md_table(summary, columns)}

![요청량–운전점–residual](traffic_to_power.png)

![Matched profiles](matched_profiles.png)

### Ramp 해석

- raw 0.1초 ramp에는 traffic-bin 경계에서 M3가 즉시 이동하는 인공 성분 포함.
- within-bin ramp는 해당 경계를 가로지르는 구간을 제외한 지표이며, 실제 경계 ramp를 복원한 값이 아님.
- {traffic_bin:g}초 bin-mean ramp는 구간 평균전력의 변화율. 0.1초 실측 ramp와 비교하지 않음.
- **큰 raw ramp가 나왔다고 모델이 더 현실적이라는 결론을 내리지 않음.**
- C의 raw 최대 0.1초 ramp는 scenario 평균 {cv['max_abs_ramp_0p1s_mw_per_s']:.3f} MW/s인 반면, bin 경계 제외 최대는 {cv['max_within_bin_abs_ramp_0p1s_mw_per_s']:.3f} MW/s. 큰 급변은 주로 구간별 M3 즉시 이동에서 발생함.

![경계 ramp 진단](boundary_ramp_diagnostic.png)

## 4. 핵심 수식과 차용 범위

$$r_{{s,j}}(t)=R_s^{{admitted}}(t)/N_s$$

- *[3]의 균등 요청 분배 원리를 적용. N_s 고정은 이번 최소 실험의 가정.*
- scenario offered = admitted + excess. admitted는 serving 입력 arrival이며 completed throughput이 아님.
- 상한은 모델 입력 지원범위 guard이며 실제 처리성능·latency 보장 용량이 아님.

$$P_j(t)=P_{{M3}}(r_j(t),d_j)+e_j(t),\qquad P_{{IT}}(t)=\sum_j P_j(t)$$

- *[3]의 실측 P(r) 배정을 우리 M3+R3로 교체한 구성. [43]은 요청 기반 bottom-up 구조의 보조 근거이며 이 식을 그대로 사용하지 않음.*

$$e_{{k}}=\phi(r_k,d)e_{{k-1}}+\eta_k$$

- *R3 signed-mixture innovation은 기존 잠금 계수를 그대로 재사용. [2] Jump-OU 골격의 우리 이산 근사이며, 시간가변 계수 연결은 추가 가정.*
- 창별 평균을 빼는 후처리를 제거하고 state를 유지. 물리적 rate 전환 응답의 실측 검증을 대신하지 않음.

## 5. 남은 검증

- 같은 하드웨어의 time-varying arrival–power 실측으로 전이응답 검증.
- 실제 전체시설 IT power와 같은 측정 경계·시간해상도로 비교.
- CPU package/core 전력계측 domain 검증. 현재 NLR 표본을 조용히 재보정하지 않았음.
- 생성된 단위전력 최대는 {config['nominal_reference_tail_check']['maximum_generated_node_kw']:.3f} kW. [3]의 CPU/GPU 명판 합계 참고값3.52 kW를 넘은 표본은 {config['nominal_reference_tail_check']['samples_above_reference']} / {config['nominal_reference_tail_check']['node_time_samples']:,}개. 이는 엄밀한 hard cap 판정은 아니지만 R3의 비유계 Gaussian tail 및 측정경계 검토가 필요함. 임의 clipping은 하지 않았음.
- 장시간 traffic 사용 시 180초 보정 통계의 장시간 유효성 검증.
- 지원범위 밖 저율·무요청·과부하는 별도 idle/queue/운영모델 필요. 필요성 확인 전 자동 확장하지 않음.
- cooling·UPS·기타 부하·ESS–TES·PCC·DFL은 이번 작업에 포함하지 않음.

## References

- *[3] Vercellino et al., Measurement of Generative AI Workload Power Profiles for Whole-Facility Data Center Infrastructure Planning, 2026, §2.3.2/§4.2: https://arxiv.org/html/2604.07345v1
- *[43] Wilkins et al., From Servers to Sites: Compositional Power Trace Generation of LLM Inference for Infrastructure Planning, arXiv, 2026, §3: https://arxiv.org/html/2603.18383v1
- *DynamoLLM, IEEE HPCA 2025: https://doi.org/10.1109/HPCA61900.2025.00102 — traffic 출처·운영 구조 근거; 해당 운영 최적화 알고리즘을 구현한 것은 아님.
- *M3/R3 원 계수·방법은 `outputs/r3_high_rate_benchmark/protocol.json`과 소스 감사 문서에 기록.
"""
    (output / "report_ko.md").write_text(report, encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"Output: {output}\nRuntime: {time.perf_counter()-started:.2f} s", flush=True)


if __name__ == "__main__":
    main()
