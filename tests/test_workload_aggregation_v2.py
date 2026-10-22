from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlr_data.dynamic_identifiability import M3Parameters  # noqa: E402
from nlr_data.residual_model_selection import ResidualParameterLibrary  # noqa: E402
from nlr_data.workload_aggregation_v2 import (  # noqa: E402
    WorkloadAggregationGenerator,
    profile_metrics,
)


def _generator() -> WorkloadAggregationGenerator:
    residual_parameters = pd.DataFrame(
        [
            {
                "profile_id": "p1",
                "prompt_dataset": "d",
                "request_rate": 50.0,
                "residual_std_kw": 0.25,
                "phi": 0.8,
                "innovation_std_kw": 0.15,
                "jump_evidence": 1.0,
                "negative_weight": 0.05,
                "negative_mean_kw": -0.8,
                "negative_std_kw": 0.1,
                "background_weight": 0.90,
                "background_mean_kw": 0.0,
                "background_std_kw": 0.08,
                "positive_weight": 0.05,
                "positive_mean_kw": 0.7,
                "positive_std_kw": 0.1,
            }
        ]
    )
    return WorkloadAggregationGenerator(
        m3_parameters={"d": M3Parameters(2.0, 1.0, 3.0, 2.0)},
        residual_library=ResidualParameterLibrary(
            residual_parameters, {"p1": np.linspace(-0.2, 0.2, 1000)}
        ),
        calibration_splits=("train", "validation"),
    )


def test_aggregate_profile_is_finite_and_has_exact_target_mean() -> None:
    generator = _generator()
    for model in ("R2_ou", "R3_jump_ou"):
        profile = generator.generate_aggregate_profile(
            residual_model=model,
            prompt_dataset="d",
            request_rate=50.0,
            target_it_mean_mw=20.0,
            workload_group_count=10,
            synchronization_scenario="partial_25pct",
            duration_s=30.0,
            interval_s=0.1,
            seed=7,
        )
        assert np.isfinite(profile["it_power_mw"]).all()
        assert abs(float(profile["it_power_mw"].mean()) - 20.0) < 1e-10
        assert int(profile["synchronized_group_count"].iloc[0]) == 2
        metrics = profile_metrics(profile)
        assert metrics["p99_it_power_mw"] >= metrics["p01_it_power_mw"]


def test_independent_group_aggregation_reduces_variability() -> None:
    generator = _generator()
    one = generator.generate_aggregate_profile(
        residual_model="R2_ou",
        prompt_dataset="d",
        request_rate=50.0,
        target_it_mean_mw=20.0,
        workload_group_count=1,
        synchronization_scenario="independent",
        duration_s=180.0,
        interval_s=0.1,
        seed=9,
    )
    fifty = generator.generate_aggregate_profile(
        residual_model="R2_ou",
        prompt_dataset="d",
        request_rate=50.0,
        target_it_mean_mw=20.0,
        workload_group_count=50,
        synchronization_scenario="independent",
        duration_s=180.0,
        interval_s=0.1,
        seed=9,
    )
    assert float(fifty["it_power_mw"].std()) < float(one["it_power_mw"].std())
