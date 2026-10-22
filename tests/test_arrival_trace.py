from pathlib import Path
import sys
import unittest
import tempfile
import hashlib
import json
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aidc_facility.arrival_trace import parse_prefix, common_window, bin_arrivals, verify_saved_sample


class ArrivalTraceTests(unittest.TestCase):
    def test_partial_last_line_discarded(self):
        data = b"TIMESTAMP,ContextTokens,GeneratedTokens\n2024-05-10 00:00:00.100,2,3\n2024-05-10"
        self.assertEqual(len(parse_prefix(data)), 1)

    def test_nonmonotonic_prefix_rejected(self):
        data = b"TIMESTAMP,ContextTokens,GeneratedTokens\n2024-05-10 00:00:02,2,3\n2024-05-10 00:00:01,2,3\n"
        with self.assertRaises(ValueError):
            parse_prefix(data)

    def test_bins_preserve_all_requests_and_exclude_end(self):
        base = pd.Timestamp("2024-05-10")
        f = pd.DataFrame({"TIMESTAMP": [base+pd.Timedelta(seconds=t) for t in (0, .1, .99, 1, 2)]})
        b = bin_arrivals({"code": f}, base, 2, 1)
        np.testing.assert_array_equal(b.code_count, [3, 1])

    def test_first_common_window_is_deterministic_and_covered(self):
        b = pd.Timestamp("2024-05-10")
        frames = {k: pd.DataFrame({"TIMESTAMP": [b+pd.Timedelta(seconds=t), b+pd.Timedelta(seconds=100)]}) for k,t in (("code", .1), ("conversation", .3))}
        start, end = common_window(frames, 20, 10)
        self.assertEqual(start, b+pd.Timedelta(seconds=1))
        self.assertEqual(end-start, pd.Timedelta(seconds=20))
        with self.assertRaises(ValueError):
            common_window(frames, 200, 10)

    def test_cached_sample_is_rebinned_and_tampering_is_detected(self):
        data = b"TIMESTAMP,ContextTokens,GeneratedTokens\n2024-05-10 00:00:00,2,3\n2024-05-10 00:00:01,3,4\n2024-05-10 00:00:02,3,4\n"
        frame = parse_prefix(data)
        start = pd.Timestamp("2024-05-10")
        selected = frame.iloc[:2]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            meta = {"start_in_source_clock": start.isoformat(), "end_exclusive_in_source_clock": (start+pd.Timedelta(seconds=2)).isoformat(), "duration_s": 2, "bin_s": 1, "sources": {}}
            for service in ("code", "conversation"):
                selected.to_csv(path / f"{service}_arrival_sample.csv", index=False)
                sample = (path / f"{service}_arrival_sample.csv").read_bytes()
                (path / f"{service}_downloaded_prefix.csv.part").write_bytes(data)
                meta["sources"][service] = {"start_byte": 0, "window_request_count": 2,
                    "window_sample_sha256": hashlib.sha256(sample).hexdigest(), "prefix_sha256": hashlib.sha256(data).hexdigest()}
            bins = bin_arrivals({"code": frame, "conversation": frame}, start, 2, 1)
            bins.to_csv(path / "traffic_bins.csv", index=False)
            (path / "provenance.json").write_text(json.dumps(meta), encoding="utf-8")
            actual, _ = verify_saved_sample(path)
            self.assertEqual(actual.code_count.sum(), 2)
            bins.loc[0, "code_count"] = 99
            bins.to_csv(path / "traffic_bins.csv", index=False)
            with self.assertRaises(AssertionError):
                verify_saved_sample(path)
            bins.loc[0, "code_count"] = 1
            bins.to_csv(path / "traffic_bins.csv", index=False)
            (path / "code_arrival_sample.csv").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_saved_sample(path)


if __name__ == "__main__":
    unittest.main()
