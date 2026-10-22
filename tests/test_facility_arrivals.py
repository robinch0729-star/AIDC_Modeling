"""Tests for arrival construction, NOT physical validation of an AIDC."""
import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
import pandas as pd

from aidc_facility.facility_arrivals import (aggregate_counts, scale_arrivals,
    arrival_metrics, relative_within_block_rms, build_source_seconds, sha256)


class FacilityArrivalTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(77).poisson(20, 600)

    def test_replay_exact_and_relative_changes_invariant(self):
        y = scale_arrivals(self.x, 100, "replay")
        np.testing.assert_array_equal(y, self.x * 100)
        self.assertAlmostEqual(arrival_metrics(y, 1)["cv"], arrival_metrics(self.x, 1)["cv"])

    def test_every_parent_block_conserved(self):
        for method, block in (("conditional_60s", 60), ("conditional_300s", 300)):
            y = scale_arrivals(self.x, 100, method, 4)
            np.testing.assert_array_equal(aggregate_counts(y, block), aggregate_counts(self.x, block)*100)
            np.testing.assert_array_equal(aggregate_counts(y, 300), aggregate_counts(self.x, 300)*100)
            self.assertTrue(np.all(y >= 0))

    def test_seed_is_reproducible_not_shared(self):
        a = scale_arrivals(self.x, 10, "conditional_60s", 7)
        np.testing.assert_array_equal(a, scale_arrivals(self.x, 10, "conditional_60s", 7))
        self.assertFalse(np.array_equal(a, scale_arrivals(self.x, 10, "conditional_60s", 8)))

    def test_zero_block_stays_zero(self):
        x = np.concatenate([np.zeros(300, dtype=np.int64), self.x[:300]])
        for method in ("replay", "conditional_60s", "conditional_300s"):
            self.assertEqual(scale_arrivals(x, 10, method, 8)[:300].sum(), 0)

    def test_no_counts_move_between_parent_blocks(self):
        x = np.zeros(600, dtype=np.int64)
        x[59], x[300] = 100, 200
        y = scale_arrivals(x, 5, "conditional_60s", 1)
        self.assertEqual(y[60:300].sum(), 0)
        self.assertEqual(y[:60].sum(), 500)
        self.assertEqual(y[300:360].sum(), 1000)

    def test_shifted_bin_edges_explicit(self):
        x = np.arange(20, dtype=np.int64)
        np.testing.assert_array_equal(aggregate_counts(x, 5, 2), [20, 45, 70])
        self.assertEqual(arrival_metrics(x, 5, 2)["dropped_edge_seconds"], 5)

    def test_invalid_inputs(self):
        for bad in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                scale_arrivals(self.x, bad, "replay")
        for bad in (np.array([1., 2.]), np.array([-1, 2]), np.array([], dtype=int)):
            with self.assertRaises(ValueError):
                scale_arrivals(bad, 1, "replay")
        with self.assertRaises(ValueError):
            scale_arrivals(self.x[:-1], 10, "conditional_300s")
        with self.assertRaises(ValueError):
            scale_arrivals(self.x, 1, "unknown")

    def test_conditional_variance_scaling(self):
        x = np.full(30000, 20, dtype=np.int64)
        y1 = scale_arrivals(x, 1, "conditional_60s", 10)
        y100 = scale_arrivals(x, 100, "conditional_60s", 10)
        ratio = relative_within_block_rms(y100) / relative_within_block_rms(y1)
        self.assertTrue(.09 < ratio < .11)

    def test_single_bin_metrics(self):
        result = arrival_metrics(np.zeros(300, dtype=np.int64), 300)
        self.assertEqual(result["adjacent_change_p95_rps"], 0)
        self.assertEqual(result["mean_rps"], 0)

    def test_timestamp_units_boundaries_and_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw, audit, out = [root/x for x in ("raw", "audit", "output")]
            for path in (raw, audit, out):
                path.mkdir()
            timestamps = ["2024-05-12T00:04:59.999999+00:00", "2024-05-12T00:05:00+00:00",
                          "2024-05-12T00:09:59.999999+00:00", "2024-05-12T00:10:00+00:00",
                          "2024-05-12T00:15:00+00:00"]
            service_records, five_rows = [], []
            for service, short in (("code", "code"), ("conversation", "conv")):
                path = raw/f"AzureLLMInferenceTrace_{short}_1week.csv"
                pd.DataFrame({"TIMESTAMP": timestamps}).to_csv(path, index=False)
                service_records.append({"service": service, "raw_sha256": sha256(path)})
                for t, count in (("2024-05-12 00:05:00+00:00", 2), ("2024-05-12 00:10:00+00:00", 1)):
                    five_rows.append({"service": service, "bin_start_utc": t, "request_count": count})
            data = {"services": service_records, "common_window": {
                "start_inclusive_utc": "2024-05-12T00:05:00+00:00",
                "end_exclusive_utc": "2024-05-12T00:15:00+00:00"}}
            (audit/"audit_summary.json").write_text(json.dumps(data))
            pd.DataFrame(five_rows).to_csv(audit/"common_5min_requests.csv", index=False)
            result, _, _ = build_source_seconds(raw, audit, out, 2)
            self.assertEqual(result["code"].sum(), 3)
            np.testing.assert_array_equal(np.flatnonzero(result["code"]), [0, 299, 300])
            cached, _, _ = build_source_seconds(raw, audit, out, 2)
            np.testing.assert_array_equal(cached["code"], result["code"])


if __name__ == "__main__":
    unittest.main()
