"""Tests for the NLR [3]-inspired bottom-up facility modules."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aidc_facility.allocation import InstanceAllocator, calibrate_mean_request_rate  # noqa: E402
from aidc_facility.boundary import FixedPUEBoundary  # noqa: E402
from aidc_facility.traffic import (  # noqa: E402
    NLR_INFERENCE_SERVICES,
    TrafficDistribution,
    representative_short_horizon_shape,
)


class BottomUpFacilityTests(unittest.TestCase):
    def test_pue_sets_mean_it_target_without_instantaneous_scaling(self) -> None:
        boundary = FixedPUEBoundary(target_pcc_mean_mw=20.0, pue=1.15)
        self.assertAlmostEqual(boundary.target_it_mean_mw, 20.0 / 1.15)
        self.assertAlmostEqual(boundary.implied_non_it_mean_mw, 20.0 - 20.0 / 1.15)

    def test_paper_three_node_pool_allocation(self) -> None:
        allocator = InstanceAllocator(285, NLR_INFERENCE_SERVICES)
        allocation = allocator.node_allocation.set_index("service")
        self.assertEqual(int(allocation["node_pool"].sum()), 285)
        self.assertEqual(int(allocation.loc["coding", "node_pool"]), 67)
        self.assertEqual(int(allocation.loc["conversation", "node_pool"]), 218)

    def test_dispatch_tracks_incomplete_requests(self) -> None:
        allocator = InstanceAllocator(285, NLR_INFERENCE_SERVICES)
        traffic = TrafficDistribution.from_normalized_shape(
            np.ones(2),
            mean_total_request_rate_rps=1_000_000.0,
            macro_interval_s=60.0,
            services=NLR_INFERENCE_SERVICES,
        )
        plan = allocator.plan(traffic)
        self.assertTrue((plan["incomplete_request_rate_rps"] > 0).all())
        active = plan.groupby("macro_index")["active_nodes"].sum()
        self.assertTrue((active == 285).all())

    def test_request_count_to_rate_identity_is_explicit(self) -> None:
        traffic = TrafficDistribution.from_normalized_shape(
            np.ones(2),
            mean_total_request_rate_rps=1000.0,
            macro_interval_s=60.0,
            services=NLR_INFERENCE_SERVICES,
        )
        rows = traffic.by_service()
        np.testing.assert_allclose(
            rows["requests_in_macro_interval"],
            rows["incoming_request_rate_rps"] * 60.0,
        )

    def test_bisection_matches_target_utilization(self) -> None:
        allocator = InstanceAllocator(1000, NLR_INFERENCE_SERVICES)
        traffic, plan = calibrate_mean_request_rate(
            representative_short_horizon_shape(15),
            allocator=allocator,
            target_utilization=0.8,
            macro_interval_s=60.0,
        )
        self.assertGreater(float(np.mean(traffic.total_request_rate_rps)), 0.0)
        utilization = (
            plan.groupby("macro_index")["active_nodes"].sum().mean() / 1000.0
        )
        self.assertLessEqual(abs(utilization - 0.8), 5e-4)


if __name__ == "__main__":
    unittest.main()
