"""CSV/MAT/figure/report exports for the bottom-up IT facility run."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from .simulation import FacilityCaseResult


def _summary(result: FacilityCaseResult) -> dict[str, float | int | str | bool]:
    profile = result.aggregation.profile
    operations = result.aggregation.operations
    values = profile["it_power_mw"].to_numpy(float)
    target = result.boundary.target_it_mean_mw
    return {
        "status": "complete",
        "method": "NLR_DIPLOEE_flow_with_M3_R3_node_power_replacement",
        "target_pcc_mean_mw_for_future_complete_aidc": result.boundary.target_pcc_mean_mw,
        "assumed_pue_for_mean_bookkeeping": result.boundary.pue,
        "target_it_mean_mw": target,
        "realized_it_mean_mw": float(np.mean(values)),
        "mean_target_error_pct": float(100.0 * (np.mean(values) - target) / target),
        "it_min_mw": float(np.min(values)),
        "it_max_mw": float(np.max(values)),
        "it_std_mw": float(np.std(values, ddof=0)),
        "total_nodes": result.sizing.total_nodes,
        "mean_active_nodes": float(operations["active_nodes"].mean()),
        "mean_utilization": float(operations["utilization"].mean()),
        "mean_total_request_rate_rps": result.sizing.mean_total_request_rate_rps,
        "mean_incomplete_request_rate_rps": float(
            operations["incomplete_request_rate_rps"].mean()
        ),
        "micro_interval_s": result.config.micro_interval_s,
        "macro_interval_s": result.config.macro_interval_s,
        "duration_s": float(profile["time_s"].iloc[-1] + result.config.micro_interval_s),
        "cooling_ups_auxiliary_included": False,
        "unmeasured_it_peripheral_overhead_included": False,
        "instantaneous_pue_multiplier_applied": False,
        "equivalent_mw_scaling_applied": False,
        "independent_node_residual_assumption": True,
    }


def save_facility_case(result: FacilityCaseResult, output_dir: str | Path) -> Path:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    profile = result.aggregation.profile
    operations = result.aggregation.operations
    plan = result.sizing.plan
    allocation = result.sizing.node_allocation
    summary = _summary(result)

    profile.to_csv(output / "facility_it_profile.csv", index=False, encoding="utf-8-sig")
    operations.to_csv(output / "facility_operations.csv", index=False, encoding="utf-8-sig")
    plan.to_csv(output / "service_dispatch_plan.csv", index=False, encoding="utf-8-sig")
    allocation.to_csv(output / "service_node_allocation.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    run_config = {
        "config": asdict(result.config),
        "node_assumptions": {
            "rated_compute_power_kw": 3.520,
            "idle_power_kw": 0.420,
        },
        "provenance": {
            "traffic_to_instance_and_node_allocation": "Paper list [3], Eqs. (2)-(7)",
            "service_mix_rate_limits_one_instance_per_node": "Paper list [3], Section 4.2",
            "node_rated_and_idle_power": "Paper list [3], Sections 3.1 and 4",
            "node_power_mapping": "this study: NLR-calibrated M3+R3 replaces empirical P(r)",
            "short_horizon_traffic_shape": "general deterministic test driver, not Azure reproduction",
            "pue_use": "general energy bookkeeping; sets mean IT target only",
        },
    }
    (output / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    savemat(
        output / "facility_it_profile.mat",
        {
            "time_s": profile["time_s"].to_numpy(float),
            "it_power_mw": profile["it_power_mw"].to_numpy(float),
            "active_nodes": profile["active_nodes"].to_numpy(int),
            "idle_nodes": profile["idle_nodes"].to_numpy(int),
            "macro_time_s": operations["macro_time_s"].to_numpy(float),
            "utilization": operations["utilization"].to_numpy(float),
            "incoming_request_rate_rps": operations[
                "incoming_request_rate_rps"
            ].to_numpy(float),
        },
        do_compression=True,
    )

    figure, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=160, sharex=False)
    axes[0].plot(profile["time_s"] / 60.0, profile["it_power_mw"], lw=0.8, color="#2563eb")
    axes[0].axhline(
        result.boundary.target_it_mean_mw,
        color="#dc2626",
        ls="--",
        lw=1.2,
        label=f"target mean {result.boundary.target_it_mean_mw:.3f} MW",
    )
    axes[0].set(ylabel="IT power (MW)", title="Bottom-up facility IT power (M3+R3 per active node)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].step(
        operations["macro_time_s"] / 60.0,
        operations["utilization"],
        where="post",
        color="#059669",
        label="node utilization",
    )
    axes[1].axhline(
        result.config.target_utilization,
        color="#111827",
        ls="--",
        lw=1.0,
        label="target utilization",
    )
    axes[1].set(xlabel="Time (min)", ylabel="Utilization (p.u.)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output / "bottom_up_facility_it_profile.png", bbox_inches="tight")
    plt.close(figure)

    allocation_text = allocation.to_string(index=False)
    report = f"""# NLR [3] 기반 bottom-up AIDC IT 합산 결과

## 결과

- 향후 완성 AIDC 목표 평균 PCC 전력: {result.boundary.target_pcc_mean_mw:.3f} MW
- 평균 산정용 PUE: {result.boundary.pue:.3f}
- 목표 평균 IT 전력: {result.boundary.target_it_mean_mw:.6f} MW
- 생성 평균 IT 전력: {summary['realized_it_mean_mw']:.6f} MW
- 평균 오차: {summary['mean_target_error_pct']:.4f}%
- 전체 node: {result.sizing.total_nodes:,}
- 평균 node utilization: {summary['mean_utilization']:.4f}
- IT 전력 범위: {summary['it_min_mw']:.4f}-{summary['it_max_mw']:.4f} MW

## Node allocation

```text
{allocation_text}
```

## 적용한 [3]의 흐름

1. 전체 inference request를 coding/conversation으로 38.1/61.9% 배분
2. request rate와 최대 instance rate 100/50 request/s로 active instance 수 계산
3. 각 service의 node pool 한도 내에서 request를 균등 배분
4. 초과 request를 incomplete request로 기록
5. 한 model instance를 한 node에서 실행
6. active node에는 M3+R3, inactive node에는 0.420 kW를 적용하고 물리적으로 합산

## [3]에서 변경한 부분

- [3]의 empirical power-sample assignment `P(r)`를 본 연구의 M3+R3 node generator로 교체함.
- 공개된 DIPLOEE 생성 코드가 없으므로 short-horizon traffic shape는 명시적인 일반 test driver를 사용함.
- node residual은 서로 독립이라는 baseline 가정을 사용함. 이는 [3]에서 명시한 상관모델이 아님.

## 현재 모델 경계

- 결과는 CPU/GPU telemetry 기반의 compute-equivalent IT 전력이다.
- 냉각, UPS 손실, 메모리·팬·NIC 등 미계측 IT 주변부 및 기타 보조부하는 포함하지 않았다.
- PUE 1.15는 목표 평균 IT 전력 `{result.boundary.target_pcc_mean_mw}/1.15`를 계산할 때만 사용했다.
- 순간 전력에 1.15를 곱하지 않았으며, PCC profile을 생성했다고 주장하지 않는다.
- 추후 냉각·기타 부하 모델을 결합한 뒤 에너지 기준 PUE를 재검증해야 한다.

![Bottom-up result](bottom_up_facility_it_profile.png)
"""
    (output / "bottom_up_facility_report_ko.md").write_text(report, encoding="utf-8")
    return output
