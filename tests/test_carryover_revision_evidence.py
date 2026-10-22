import tempfile
from pathlib import Path
import unittest
from aidc_facility.carryover_revision_evidence import parse_namespace, scan_bench, scan_server, work_scale_counterexample


class RevisionEvidenceTests(unittest.TestCase):
    def test_namespace_literals_not_function(self):
        x = parse_namespace("Namespace(dispatch_function=<function f at 0x123>, seed=0, hf_output_len=None, save_result=False, metadata=['id=000040'])")
        self.assertEqual(x['id'], '000040')
        self.assertEqual(x['seed'], 0)
        self.assertIsNone(x['hf_output_len'])

    def test_reject_executable_values(self):
        with self.assertRaises((ValueError, TypeError)):
            parse_namespace("Namespace(seed=__import__('os'), metadata=['id=x'])")

    def test_missing_id(self):
        with self.assertRaises(ValueError):
            parse_namespace('Namespace(seed=0)')

    def test_duplicate_bench_id(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'bench.log'
            p.write_text("Namespace(seed=0, metadata=['id=x'])\n"*2)
            with self.assertRaises(ValueError):
                scan_bench(p)

    def test_http_does_not_establish_completion(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'server.log'
            p.write_text('INFO: 127.0.0.1:123 - "POST /v1/completions HTTP/1.1" 200 OK\n')
            x = scan_server(p)
            self.assertEqual(x['http_access_rows'], 1)
            self.assertEqual(x['http_rows_with_clock_pattern'], 0)
            self.assertFalse(x['http_200_is_request_completion'])
            self.assertFalse(x['absence_of_reset_markers_proves_no_reset'])

    def test_work_normalization(self):
        x = work_scale_counterexample()
        self.assertTrue(x['same_completion_time'])
        self.assertEqual(x['case_a']['completion_time_s_in_fixed_state'], 2.5)


if __name__ == '__main__':
    unittest.main()
