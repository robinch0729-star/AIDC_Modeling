import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from aidc_facility.nlr_processing_diagnostic import (parse_metric_line, linear_slope,
    time_average, diagnose_window, residence_screen, diagnose_runs, finite_diagnostics)


def metrics(running, waiting=None, times=None):
    n = len(running)
    times = np.arange(n)*10 if times is None else np.asarray(times)
    return pd.DataFrame(dict(timestamp=pd.Timestamp("2026-01-05")+pd.to_timedelta(times, unit="s"),
        running_requests=running, waiting_requests=np.zeros(n) if waiting is None else waiting,
        prompt_tokens_per_s=np.repeat(1000., n), generation_tokens_per_s=np.repeat(2000., n),
        source_line=np.arange(n)+1))


class ProcessingDiagnosticTests(unittest.TestCase):
    def test_raw_metric_is_snapshot_not_interval_max(self):
        line = "\x1b[0;32mINFO\x1b[0m 01-05 06:12:10 [x] Avg prompt throughput: 100.1 tokens/s, Avg generation throughput: 200.2 tokens/s, Running: 37 reqs, Waiting: 15 reqs, GPU KV cache usage: 3.2%, Prefix cache hit rate: 4.1%"
        r = parse_metric_line(line, 2026, 48)
        self.assertEqual(r["running_requests"], 37)
        self.assertEqual(r["source_line"], 48)
        self.assertEqual(r["timestamp"], "2026-01-05 06:12:10")
        self.assertEqual(r["gpu_kv_cache_pct"], 3.2)

    def test_reject_missing_timestamp(self):
        self.assertIsNone(parse_metric_line("Running: 5 reqs, Waiting: 0 reqs", 2026, 1))

    def test_slope_irregular_timestamps(self):
        t = np.array([0, 9, 21, 35])
        self.assertAlmostEqual(linear_slope(t, 3*t+2), 3)

    def test_weighted_mean_not_snapshot_mean(self):
        self.assertAlmostEqual(time_average([0, 1, 10], [0, 10, 10]), 9.5)

    def test_duplicate_times_rejected(self):
        with self.assertRaises(ValueError):
            time_average([0, 0, 10], [1, 1, 1])

    def test_constant_stock_little_proxy(self):
        r = diagnose_window(metrics([120]*18), 30)
        self.assertEqual(r["conditional_departure_rps"], 30)
        self.assertEqual(r["conditional_residence_proxy_s"], 4)
        self.assertTrue(r["low_drift_screen_2pct"])

    def test_running_growth_matters_without_waiting(self):
        r = diagnose_window(metrics(100+np.arange(18)*20), 40)
        self.assertEqual(r["waiting_max"], 0)
        self.assertAlmostEqual(r["unfinished_slope_rps"], 2)
        self.assertAlmostEqual(r["conditional_departure_rps"], 38)
        self.assertFalse(residence_screen(r))
        self.assertTrue(np.isnan(r["conditional_residence_proxy_s"]))

    def test_waiting_blocks_residence_proxy(self):
        r = diagnose_window(metrics([1000]*18, [3]*18), 40)
        self.assertTrue(r["waiting_positive_all_snapshots"])
        self.assertFalse(r["low_drift_screen_2pct"])

    def test_token_rate_never_becomes_request_rate(self):
        r = diagnose_window(metrics([1000]*18, np.arange(18)*80), 50)
        self.assertEqual(r["generation_tps_snapshot_mean"], 2000)
        self.assertAlmostEqual(r["conditional_departure_rps"], 42)

    def test_window_left_inclusive_right_exclusive(self):
        m = metrics(np.arange(33), times=np.arange(33)*10)
        runs = pd.DataFrame([dict(id="000001", run_start="2026-01-05", run_end="2026-01-05 00:05:20",
                                 request_rate=10, **{"dataset-path":"likaixin/InstructCoder"})])
        p = pd.DataFrame(dict(timestamp=pd.date_range("2026-01-05", periods=320, freq="s"),
                              gpu_w=np.repeat(2000,320), cpu_reported_domains_w=np.repeat(400,320)))
        w, b = diagnose_runs(runs, m, p)
        main = w[w.window.eq("primary_60_240")].iloc[0]
        self.assertEqual(main.metric_count, 18)
        self.assertEqual(main.unfinished_first, 6)
        self.assertEqual(main.unfinished_last, 23)
        self.assertEqual(main.power_seconds, 180)
        self.assertEqual(len(b), 1)

    def test_finite_completion_identity_and_join(self):
        meta = pd.DataFrame([{"id":"1", "dataset-path":"likaixin/InstructCoder", "hf-output-len":256,"request_rate":10},
                             {"id":"2", "dataset-path":"likaixin/InstructCoder", "hf-output-len":256,"request_rate":10}])
        result = pd.DataFrame([dict(id="1", request_rate=10, completed=1000, duration=125., request_throughput=8.,
                                   mean_e2el_ms=3200, total_output_tokens=50000)])
        rows, grouped, summary = finite_diagnostics(meta, result)
        self.assertEqual(rows.mean_e2el_s.iloc[0], 3.2)
        self.assertEqual(summary["missing_result_ids"], ["2"])
        self.assertEqual(summary["max_throughput_identity_error"], 0)
        self.assertEqual(len(grouped), 1)


if __name__ == "__main__":
    unittest.main()
