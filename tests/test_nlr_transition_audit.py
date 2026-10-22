from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aidc_facility.nlr_transition_audit import (
    Evidence, classify_gate, namespace_fields, scan_json, scan_power, scan_text,
)


class TransitionAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.evidence = Evidence()

    def tearDown(self):
        self.temp.cleanup()

    def test_namespace_safe_scalar_fields(self):
        fields = namespace_fields("Namespace(request_rate=20.0, save_detailed=False, ramp_up_strategy=None, dataset_path='a/b', metadata=['id=000002'])")
        self.assertEqual(fields["request_rate"], 20)
        self.assertFalse(fields["save_detailed"])
        self.assertIsNone(fields["ramp_up_strategy"])
        self.assertEqual(fields["id"], "000002")

    def test_power_full_scan_header_and_interval(self):
        path = self.root / "nvml.log"
        path.write_text("# timestamp reading-time[ns] gpu-0[mW] gpu-1[mW] gpu-0[C]\n"
                        "2026-01-01_00:00:00.000000 1 1000 2000 40\n"
                        "2026-01-01_00:00:00.100000 1 2000 2000 40\n"
                        "2026-01-01_00:00:01.000000 1 3000 2000 40\n", encoding="utf-8")
        c, frame = scan_power(path, self.root, self.evidence)
        self.assertTrue(c["eof_complete"])
        self.assertEqual(c["records"], 3)
        self.assertEqual(frame.iloc[0].mean_power_w, 3.5)
        self.assertEqual(len(frame), 2)
        self.assertEqual(c["above_0p2s_count"], 1)

    def test_access_lines_not_invented_arrivals(self):
        path = self.root / "server.log"
        path.write_text('INFO: 127.0.0.1:11 - "POST /v1/completions HTTP/1.1" 200 OK\n'
                        'INFO 01-05 05:31:01 [x] Avg prompt throughput: 91.0 tokens/s, Avg generation throughput: 36.3 tokens/s, Running: 3 reqs, Waiting: 0 reqs\n', encoding="utf-8")
        c, _, m = scan_text(path, self.root, 2026, self.evidence)
        self.assertEqual(c["untimestamped_request_http_access_lines"], 1)
        self.assertNotIn("timestamped_request_http_access_lines", c)
        self.assertEqual(m.iloc[0].running_requests, 3)
        self.assertEqual(m.iloc[0].timestamp.year, 2026)

    def test_jsonl_all_keys_not_only_first_record(self):
        path = self.root / "results.json"
        path.write_text('{"id": "1", "mean_ttft_ms": 3}\n{"id": "2", "arrival_times": [1, 2]}\n', encoding="utf-8")
        c, rows = scan_json(path, self.root, self.evidence)
        self.assertEqual(c["records"], 2)
        self.assertEqual(c["request_detail_keys_present"], ["arrival_times"])

    def test_gate_does_not_fit_tau_from_fixed_runs(self):
        ns = pd.DataFrame({"request_rate": [1., 2.], "ramp_up_strategy": [None, None]})
        g = classify_gate(ns, [], True, True)
        self.assertFalse(g["class_A_verified_rate_change_and_arrival_time"])
        self.assertTrue(g["class_B_start_stop_and_power_candidate"])
        self.assertEqual(g["non_null_ramp_configuration_fields"], 0)

    def test_evidence_keeps_first_two_last(self):
        for i in range(1, 10):
            self.evidence.add("a", i, "k", str(i))
        self.assertEqual([r["line"] for r in self.evidence.rows()], [1, 2, 9])
        self.assertEqual(self.evidence.rows()[0]["full_file_category_occurrences"], 9)


if __name__ == "__main__":
    unittest.main()
