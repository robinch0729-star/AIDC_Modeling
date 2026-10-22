"""Small synthetic fixtures test mechanics, never substitute real trace data."""
import tempfile
from pathlib import Path
import unittest

from aidc_facility.azure_trace_audit import audit_file, common_five_minutes, histogram_summary


class AzureTraceAuditTests(unittest.TestCase):
    def test_exact_histogram_quantiles(self):
        result = histogram_summary({0: 1, 2: 2, 9: 1})
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["sum"], 13)
        self.assertEqual(result["p50"], 2)
        self.assertEqual(result["p99"], 9)

    def test_cross_chunk_duplicates_and_conservation(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            raw = p / "fixture.csv"
            raw.write_text("TIMESTAMP,ContextTokens,GeneratedTokens\n"
                "2024-05-12 00:00:00+00:00,1,2\n"
                "2024-05-12 00:05:00+00:00,3,4\n"
                "2024-05-12 00:05:00+00:00,3,4\n"
                "2024-05-12 00:05:00+00:00,5,6\n"
                "2024-05-12 00:10:00+00:00,0,0\n", encoding="utf-8")
            summary, minutes, *_ = audit_file(raw, "fixture", p, chunksize=2)
            self.assertEqual(summary["duplicate_rows_detected"], 1)
            self.assertEqual(summary["timestamp_duplicate_count"], 2)
            self.assertTrue(summary["duplicate_check_global_exact"])
            self.assertTrue(summary["minute_count_reconciles"])
            self.assertEqual(minutes.request_count.sum(), 5)
            common, window = common_five_minutes({"fixture": minutes}, [summary])
            self.assertEqual(common.request_count.tolist(), [1, 3])
            self.assertEqual(window["services"]["fixture"]["five_minute_requests"], 4)

    def test_invalid_values_and_order_not_silently_fixed(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            raw = p / "fixture.csv"
            raw.write_text("TIMESTAMP,ContextTokens,GeneratedTokens\n"
                "2024-05-12 00:01:00+00:00,-1,2.5\n"
                "2024-05-12 00:00:00+00:00,1,2\n"
                "bad,3,4\n", encoding="utf-8")
            summary, minutes, *_ = audit_file(raw, "fixture", p, chunksize=2)
            self.assertEqual(summary["rows"], 3)
            self.assertEqual(summary["invalid_timestamp"], 1)
            self.assertEqual(summary["invalid_ContextTokens"], 1)
            self.assertEqual(summary["invalid_GeneratedTokens"], 1)
            self.assertFalse(summary["chronological"])
            self.assertFalse(summary["duplicate_check_global_exact"])
            self.assertEqual(minutes.request_count.sum(), 2)

    def test_partial_edges_are_not_treated_as_complete_bins(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            raw = p / "fixture.csv"
            raw.write_text("TIMESTAMP,ContextTokens,GeneratedTokens\n"
                "2024-05-12 00:00:00.001+00:00,1,2\n"
                "2024-05-12 00:05:00+00:00,3,4\n"
                "2024-05-12 00:10:59.999+00:00,5,6\n", encoding="utf-8")
            summary, minutes, *_ = audit_file(raw, "fixture", p, chunksize=2)
            common, window = common_five_minutes({"fixture": minutes}, [summary])
            self.assertEqual(len(common), 1)
            self.assertEqual(common.request_count.tolist(), [1])
            self.assertEqual(window["start_inclusive_utc"], "2024-05-12T00:05:00+00:00")
            self.assertEqual(window["end_exclusive_utc"], "2024-05-12T00:10:00+00:00")

    def test_duplicate_timestamp_is_not_duplicate_record(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            raw = p / "fixture.csv"
            raw.write_text("TIMESTAMP,ContextTokens,GeneratedTokens\n"
                "2024-05-12 00:00:00+00:00,1,2\n"
                "2024-05-12 00:00:00+00:00,3,4\n", encoding="utf-8")
            summary, _, *_ = audit_file(raw, "fixture", p, chunksize=1)
            self.assertEqual(summary["timestamp_duplicate_count"], 1)
            self.assertEqual(summary["duplicate_rows_detected"], 0)


if __name__ == "__main__":
    unittest.main()
