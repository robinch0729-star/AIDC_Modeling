"""Tests for heterogeneous equivalent-workload aggregation helpers."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlr_data.final_it_profile_generator import FinalITProfileGenerator  # noqa: E402
from nlr_data.heterogeneous_facility_aggregation import (  # noqa: E402
    FacilityAggregationRequest,
    WorkloadSpec,
    allocate_unit_counts,
    run_heterogeneous_aggregation,
)
from test_final_it_profile_generator import _artifact  # noqa: E402


class HeterogeneousAggregationTests(unittest.TestCase):
    def test_largest_remainder_allocation_preserves_total(self) -> None:
        workloads = (
            WorkloadSpec("a", "A", "d", 0.3, 10, 20),
            WorkloadSpec("b", "B", "d", 0.7, 10, 20),
        )
        counts = allocate_unit_counts(101, workloads)
        self.assertEqual(sum(counts.values()), 101)
        self.assertEqual(counts, {"a": 30, "b": 71})

    def test_small_aggregation_exports_and_does_not_post_scale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generator = FinalITProfileGenerator.from_benchmark_artifact(
                _artifact(root / "artifact")
            )
            workloads = (
                WorkloadSpec("a", "A", "d", 0.3, 10, 20),
                WorkloadSpec("b", "B", "d", 0.7, 10, 20),
            )
            output = run_heterogeneous_aggregation(
                generator,
                workloads,
                FacilityAggregationRequest(
                    target_pcc_mean_mw=0.1,
                    pue=1.0,
                    duration_s=2.0,
                    facility_scenarios=2,
                    master_seed=7,
                ),
                root / "output",
            )
            self.assertTrue((output / "facility_profiles_long.csv").is_file())
            self.assertTrue((output / "heterogeneous_facility_it_profiles.png").is_file())
            self.assertTrue((output / "generation_config.json").is_file())
            profiles = np.genfromtxt(
                output / "scenario_summary.csv", delimiter=",", names=True
            )
            self.assertEqual(len(np.atleast_1d(profiles)), 2)


if __name__ == "__main__":
    unittest.main()
