"""Combine executed bin diagnosis and raw-log audit; never manufacture a tau fit."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from aidc_facility.transition_model_gate import decide_transition_model


def table(frame):
    header = "| " + " | ".join(frame.columns) + " |"
    lines = [header, "|" + "---|" * len(frame.columns)]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |")
    return "\n".join(lines)


def main():
    bin_dir, log_dir = ROOT / "outputs/traffic_bin_diagnostic_v1", ROOT / "outputs/nlr_transition_audit_v1"
    output = ROOT / "outputs/it_transition_diagnosis_v1"
    output.mkdir(parents=True, exist_ok=True)
    bins = json.loads((bin_dir / "run_summary.json").read_text(encoding="utf-8"))
    audit = json.loads((log_dir / "audit_summary.json").read_text(encoding="utf-8"))
    decision = decide_transition_model(audit)
    decision["source_audit"] = "../nlr_transition_audit_v1/audit_summary.json"
    decision["source_bin_diagnostic"] = "../traffic_bin_diagnostic_v1/run_summary.json"
    (output / "model_selection_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    if decision["status"] != "no_identifiable_operating_transition_in_audited_material":
        raise RuntimeError("Evidence gate differs from current report scope; inspect rather than emit a stale conclusion")
    metrics = pd.read_csv(bin_dir / "power_metrics.csv")
    arrival = pd.read_csv(bin_dir / "arrival_summary.csv")
    phase = pd.read_csv(bin_dir / "phase_sensitivity.csv")
    m3 = metrics.loc[(metrics.component == "M3") & (metrics.phase_fraction == 0)].copy()
    m3["std_kw"] = m3.std_mw * 1000
    m3_table = m3[["requested_bin_s", "mean_mw", "std_kw", "max_abs_ramp_0p1s_mw_per_s"]].rename(columns={
        "requested_bin_s": "요청 집계 간격 초", "mean_mw": "M3 평균 MW", "std_kw": "M3 표준편차 kW",
        "max_abs_ramp_0p1s_mw_per_s": "M3 최대 0.1초 ramp MW/s"})
    unsupported = arrival.loc[(arrival.requested_bin_s == .1) & (arrival.phase_fraction == 0)].copy()
    unsupported["unsupported_percent"] = unsupported.unsupported_time_fraction * 100
    unsupported["zero_percent"] = unsupported.zero_time_fraction * 100
    unsupported_table = unsupported[["service", "node_min_rps", "node_max_rps", "zero_percent", "unsupported_percent"]].rename(columns={
        "service": "서비스", "node_min_rps": "최소 요청률", "node_max_rps": "최대 요청률", "zero_percent": "무요청 시간 비율 %",
        "unsupported_percent": "현재 지원범위 밖 시간 비율 %"})
    phases = phase[["requested_bin_s", "M3_phase_difference_rmse_mw", "M3_phase_max_abs_difference_mw"]].rename(columns={
        "requested_bin_s": "요청 집계 간격 초", "M3_phase_difference_rmse_mw": "경계 변경 차이 RMSE MW",
        "M3_phase_max_abs_difference_mw": "경계 변경 최대 절대 차이 MW"})
    family_rows = []
    for name, item in audit["families"].items():
        family_rows.append({"workload": name.replace("inference_", "").replace("_llama3_70b", ""),
                           "실행 수": item["run_count"], "서버 상태 기록 수": item["server_metrics_count"],
                           "서버 상태 기록 중앙 간격 초": item["server_metric_interval_median_s"] if item["server_metric_interval_median_s"] is not None else "해당 없음",
                           "가동 중 요청 변경과 시각 확인": "미확인"})
    # Exact historical case replication is checked separately from scientific validation.
    old = pd.read_csv(ROOT / "outputs/time_varying_it_v1/facility_profiles.csv")
    new = pd.read_csv(bin_dir / "facility_profiles.csv")
    old = old.loc[old.case == "C_trace_shaped"].sort_values(["scenario_id", "time_s"])
    new = new.loc[new.case == "bin_10s_phase_0"].sort_values(["scenario_id", "time_s"])
    if len(old) != len(new) or not np.array_equal(old[["scenario_id", "time_s"]].to_numpy(), new[["scenario_id", "time_s"]].to_numpy()):
        raise AssertionError("Historical reproduction is not on the same samples")
    max_difference = float(np.max(np.abs(old.total_it_mw.to_numpy() - new.total_it_mw.to_numpy())))
    if max_difference > 1e-10:
        raise AssertionError("10 s baseline changed")
    quality = {"request_counts_conserved": bins["all_request_counts_conserved"],
               "historical_inputs_unchanged": bins["historical_and_input_hashes_unchanged"],
               "historical_10s_case_max_difference_mw": max_difference,
               "all_audited_logs_scanned_to_eof": audit["all_files_eof_complete"],
               "new_tau_fitted": False, "new_physical_profile_claim": False}
    (output / "integration_checks.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    report = rf"""# IT 부하 계단형 파형 진단 결과

## 결론

동일 요청자료의 집계 간격과 경계 위치를 바꾸어 실험하고 NLR 원시 로그를 전수 점검했다. **집계 간격을 줄이는 것만으로 계단 문제가 해결되지 않았으며, 현재 로컬 자료로는 일반적인 가동 중 요청률 변화에 대한 전이시간을 식별할 근거가 부족하다.** 새로운 시간상수나 지연을 임의로 넣지 않았다.

① 집계 진단 완료, ② 원시 로그 가용성 감사 완료, ③ 최소 전이모델 선정의 근거 심사 완료 및 선정 보류다. 계단이 제거된 검증 모델을 완성했다는 뜻이 아니다. 기존 M3+R3와 과거 결과는 보존했다.

## 동일 요청자료의 집계 간격 진단

- 원자료는 Azure coding {bins['source_count']['code']:,}건, conversation {bins['source_count']['conversation']:,}건의 같은 180초 관측창이다.*[44]
- 단위 수 6,487개, 서비스 구성, 수요 배율, M3/R3 계수 및 출력 간격 0.1초를 고정했다.
- 요청 집계 10·5·1·0.1초 및 집계 경계 0·반 구간 이동의 총 8개 입력 표현을 비교했다. 처음과 마지막 부분 구간은 실제 길이를 사용했다.
- 1·5·10초의 6개 지원 조건에서 각각 residual 5회, 총 30개 시설 profile을 생성했다. 서로 다른 실제 traffic 30개가 아니라 같은 관측창을 다시 표현한 결과다.
- 모든 경우 원본 요청 총량과 시나리오 offered 요청 적분값을 보존했다. 완료량·SLO가 같다는 검증은 아니다.
- 0.1초 두 조건은 입력 지원범위 이탈 때문에 전력 생성에서 제외했다. clipping, idle 대체, 출력 평균 재정규화 및 smoothing은 적용하지 않았다.

### 평균 운전점만 비교한 결과

아래 표는 경계를 이동하지 않은 조건 phase=0이다.

{table(m3_table)}

이 표는 확률적 residual을 제외한 **M3 합산**이다. 최대 ramp는 인공적인 구간 경계 변화까지 포함한 진단값이며 실제 시설 ramp의 추정치가 아니다. 1초 집계가 더 정확하다는 결론을 낼 수 없다. 집계가 짧아질수록 원자료의 요청 개수 요동이 확대되고 정적 M3에 즉시 반영된다. 비선형 M3 때문에 평균 요청량이 같아도 평균전력은 달라지며 이를 에너지 절감으로 해석하지 않는다.

![집계 간격별 전력과 ramp](../traffic_bin_diagnostic_v1/power_and_ramp_by_bin.png)

### 집계 경계 위치에 대한 민감도

{table(phases)}

같은 요청 기록에서 집계 경계를 반 구간 옮긴 두 M3 profile의 차이다. 예측 정확도 RMSE가 아니라 **입력 표현 민감도**다. 시작·끝 부분 구간 변경도 포함한다. 경계 위치에 따라 파형과 최대 ramp가 달라져, 현재 빠른 전력 변화가 원자료만으로 고유하게 결정되지 않음을 보여준다.

![집계 경계별 180초 IT 전력](../traffic_bin_diagnostic_v1/bin_phase_power_comparison.png)

### 0.1초 요청 집계를 바로 적용하지 않은 이유

아래 표는 경계를 이동하지 않은 조건 phase=0이다.

{table(unsupported_table)}

0.1초 출력 간격과 0.1초 요청 개수 집계는 다른 설정이다. 매우 짧은 요청 집계에서 0건인 구간과 높은 요청률이 늘어난다. 이 요동을 수천 단위에 공통으로 복제하는 것이 실제 시설의 처리상태라는 근거는 없다. 위 지원범위 판정은 기존 artifact 정책을 그대로 따른다. 1~80 보정과 81~100 benchmark 사이 비정수 구간의 정책도 임의로 바꾸지 않았다.

![요청 집계 해상도 비교](../traffic_bin_diagnostic_v1/arrival_resolution_comparison.png)

## NLR 원시 로그 감사

- 범위: 로컬 online rate·online finite·offline inference 폴더의 {audit['files_scanned']}개 파일, {audit['bytes_scanned']:,} bytes를 EOF까지 읽고 SHA-256과 검사 범위를 기록했다.*[3]
- 원시 전력은 약 0.1초 간격이지만 서버 상태 Running/Waiting 및 token throughput 로그는 대체로 10초 간격이다.
- online rate의 200개 benchmark 설정은 각각 하나의 request rate를 가지며 ramp 설정은 활성화되지 않았다. 요청별 HTTP access 행에 개별 timestamp가 없어 세밀한 실제 arrival 경로를 복원할 수 없었다.
- 실행 경계·시작 전 단일 요청 시험·처리 종료 구간은 확인되지만, 이것을 교란요인이 없는 가동 중 요청률 step으로 간주하지 않았다. 서버가 계속 켜져 있어도 benchmark별 시험·설정·drain 영향을 별도로 배제해야 한다.
- naive wall-clock 시각의 범위가 겹친다는 사실만으로 시계 동기화나 물리적 지연을 확정하지 않았다.

{table(pd.DataFrame(family_rows))}

![online rate 실행 경계의 원시 GPU 전력 진단](../nlr_transition_audit_v1/inference_online_rate_llama3_70b_boundary_diagnostics.png)

위 그림의 전력은 시각 점검을 위해 1초 평균한 **GPU 채널 합**이며 full-server/PCC 전력이 아니다. 1초 도착-전력 인과 쌍을 복원한 것도 아니다. 첫 실행들의 경계를 보여주는 예시이지 전체 전이를 대표하도록 통계적으로 선정한 표본은 아니다.

자료 판정은 **B 실행 시작·종료 분석 후보와 C 고정 요청률 분석 자료는 존재, A 검증된 가동 중 요청 변경과 정밀 arrival 시각은 미확인**이다. 이것은 로컬 감사 범위의 결론이며 NLR 전체 또는 다른 외부 자료에 그런 데이터가 없다는 뜻은 아니다.

## 최소 전이모델 선정 결과

- 새로운 전이모델 선정: **보류**.
- 추정한 시간상수와 지연: **없음**.
- 현재 M3+R3: 고정 운전점 및 명시적인 준정상 비교 기준으로 보존.
- R3 상관시간을 평균전력 전이시간으로 대체하지 않았다.
- 서로 다른 고정 요청률 실험을 이어 붙여 step 응답을 만들지 않았다.

근거가 확보되면 순수 지연, 1차 평균 응답, 필요 시 지연을 포함한 1차 응답을 비교할 수 있다. 아래 식은 **향후 검토할 일반 시스템 식별 구조를 M3에 연결한 후보이며, 이번에 채택·보정한 식이 아니다.**

$$\tau\frac{{d\bar P}}{{dt}}+\bar P=P_{{M3}}(r(t-L),d),\qquad P=\bar P+e.$$

일반적인 1차 응답 및 지연 구조의 근거는 MathWorks의 process model 설명이다. 이는 AIDC에 특정 \(\tau,L\)이 적합하다는 증거가 아니다.*[EXT1] R3를 추가 결합하기 전에 평균 전이오차가 residual에 중복 포함되는지도 검토해야 한다.

대기열·동시 실행 요청을 사용해야 할 경우 [43]의 처리상태 기반 접근을 검토할 수 있다. 다만 그 논문의 입력과 현재 M3 request rate는 의미가 다르므로 단순 치환할 수 없다. 현재 확인한 문헌은 arXiv preprint다.*[43]

## 다음에 필요한 최소 자료

1. 같은 warm serving instance에서 요청률을 낮춤·높임하는 실험. 모델·prompt 구성·token 조건·클록·power cap을 통제한다.
2. 실제 요청 송신·처리 시작·완료 시각, 입력/출력 토큰, running/queue 정보를 기록한다.
3. 같은 시간 기준의 원시 GPU·CPU 전력을 기록하고 계측 적분시간·시계오차를 문서화한다.
4. 여러 상승·하강 전이를 반복하고, 다른 실행 전체를 검증용으로 남긴다. 반복 횟수와 유지시간은 pilot의 잡음·응답시간을 보고 정하며 임의로 고정하지 않는다.
5. 자료가 없으면 장시간 운영용 준정상 모델과 빠른 응답 불확실성을 분리해 사용한다. 고속 ramp 재현을 검증했다고 주장하지 않는다.

## 재현성과 검증

- 집계 진단 실행시간 {bins['run_seconds']:.1f}초. 이 값은 이번 컴퓨터에서의 실행 기록이다.
- 기존 10초 조건의 전체 5개 profile 재생 차이 최대 {max_difference:.3g} MW.
- 보호한 과거 결과·계수·원자료 hash 일치: {bins['historical_and_input_hashes_unchanged']}.
- 같은 seed 번호를 사용했지만 기존 generator의 bin별 난수 배열 크기가 달라 정확히 같은 innovation을 짝지은 것은 아니다. 인과 분리는 M3 결정론적 비교로 수행했다.
- boundary 지표는 경계에 변화가 얼마나 집중되는지를 나타내며 '몇 %가 가짜 변동'을 추정한 값이 아니다. 1초 bin의 1초 ramp에 within-bin 값이 없는 경우 NaN은 정상이다.
- 코드 테스트 결과는 별도 `test_results.txt`에 기록한다. 테스트 통과는 물리적 모델 검증을 대신하지 않는다.

재실행 순서:

```powershell
Set-Location 'D:/DC_modeling/aidc-dfl-doe-research'
& 'D:/DC_modeling/NLR_dataset/.venv/Scripts/python.exe' scripts/run_traffic_bin_diagnostic.py
& 'D:/DC_modeling/NLR_dataset/.venv/Scripts/python.exe' scripts/run_nlr_transition_audit.py
& 'D:/DC_modeling/NLR_dataset/.venv/Scripts/python.exe' scripts/summarize_it_transition_diagnosis.py
```

## 근거와 차용 범위

- *[3] Vercellino et al., Measurement of Generative AI Workload Power Profiles for Whole-Facility Data Center Infrastructure Planning, 2026. [원문](https://arxiv.org/html/2604.07345v1). NLR 측정·실험 구조의 출처. 이번 bin 민감도 시험과 전이 식별 gate를 직접 제공한 논문은 아님.
- *[44] Stojkovic et al., DynamoLLM, IEEE HPCA 2025. [DOI](https://doi.org/10.1109/HPCA61900.2025.00102), [Azure 공식 자료](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md). 요청 원자료의 출처이며 실제 power trace가 아님.
- *[43] Wilkins et al., From Servers to Sites, arXiv:2603.18383v1, 2026. [원문](https://arxiv.org/html/2603.18383v1). 요청→처리상태→전력이라는 후속 대안의 근거. 이번 작업에서 해당 classifier·queue 모델을 구현하지 않음.
- *[EXT1] MathWorks, [What Is a Process Model?](https://www.mathworks.com/help/ident/ug/what-is-a-process-model.html). 일반적인 1차 동특성·지연 구조의 정의이며 AIDC 전이시간의 경험적 근거는 아님.
- 집계 간격·위상 민감도 실험, 입력 지원범위 gate 및 자료 적합성 gate는 이번 연구에서 구성한 진단 절차다. 새로운 임의 가중치 종합점수는 사용하지 않았다.

## 상세 결과 파일

- [요청 집계 진단표](../traffic_bin_diagnostic_v1/arrival_summary.csv)
- [전력 지표](../traffic_bin_diagnostic_v1/power_metrics.csv)
- [집계 경계 민감도](../traffic_bin_diagnostic_v1/phase_sensitivity.csv)
- [NLR 감사 요약](../nlr_transition_audit_v1/audit_summary.json)
- [파일별 읽기 범위와 hash](../nlr_transition_audit_v1/coverage.json)
- [원문 근거 위치](../nlr_transition_audit_v1/evidence_ledger.csv)
- [모델 선정 판단](model_selection_decision.json)
"""
    (output / "report_ko.md").write_text(report, encoding="utf-8")
    print(json.dumps({"decision": decision["status"], "checks": quality, "report": str(output / "report_ko.md")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
