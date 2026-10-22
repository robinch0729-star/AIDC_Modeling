"""Run the preliminary 30/70 heterogeneous equivalent-facility experiment."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlr_data.final_it_profile_generator import FinalITProfileGenerator  # noqa: E402
from nlr_data.heterogeneous_facility_aggregation import (  # noqa: E402
    FacilityAggregationRequest,
    WorkloadSpec,
    run_heterogeneous_aggregation,
)


def main() -> None:
    generator = FinalITProfileGenerator.from_benchmark_artifact(
        PROJECT_ROOT / "outputs" / "r3_high_rate_benchmark"
    )
    workloads = (
        WorkloadSpec(
            key="code_generation",
            label="Code generation / InstructCoder",
            prompt_dataset="InstructCoder",
            share=0.30,
            request_rate_low=10,
            request_rate_high=100,
        ),
        WorkloadSpec(
            key="general_conversation",
            label="General conversation / Llama2-data",
            prompt_dataset="mlperf-inference-llama2-data",
            share=0.70,
            request_rate_low=5,
            request_rate_high=80,
        ),
    )
    request = FacilityAggregationRequest(
        target_pcc_mean_mw=20.0,
        pue=1.15,
        duration_s=180.0,
        interval_s=0.1,
        facility_scenarios=30,
        master_seed=20260909,
    )
    output = run_heterogeneous_aggregation(
        generator,
        workloads,
        request,
        PROJECT_ROOT / "outputs" / "heterogeneous_equivalent_facility_17mw",
    )
    print(output)


if __name__ == "__main__":
    main()

