import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from aidc_facility.carryover_parameter_audit import validate_windows, slot_consistency, saturated_rows
from aidc_facility.nlr_processing_diagnostic import SERVICES, WINDOWS


def fixture():
    return pd.DataFrame([dict(run_id=str(i), service=s, configured_rps=20., window=w,
        running_mean=60., running_max=65., waiting_max=0., unfinished_slope_rps=0.,
        unfinished_endpoint_slope_rps=0., conditional_departure_rps=20., metric_count=18,
        max_metric_gap_s=10., waiting_positive_all_snapshots=False,
        low_drift_all_windows_2pct=True) for i,s in enumerate(SERVICES) for w in WINDOWS])


class ParameterAuditTests(unittest.TestCase):
    def test_valid(self):
        validate_windows(fixture())

    def test_missing_window(self):
        with self.assertRaises(ValueError): validate_windows(fixture().iloc[1:])

    def test_duplicate(self):
        f=fixture()
        with self.assertRaises(ValueError): validate_windows(pd.concat([f,f.iloc[:1]]))

    def test_stale_screen(self):
        f=fixture(); f.loc[0,"waiting_max"]=1
        with self.assertRaises(ValueError): validate_windows(f)

    def test_unknown_service(self):
        f=fixture(); f.loc[0,"service"]="unknown"
        with self.assertRaises(ValueError): validate_windows(f)

    def test_bad_numeric(self):
        f=fixture(); f.loc[0,"running_mean"]=np.nan
        with self.assertRaises(ValueError): validate_windows(f)

    def test_inconsistent_rate(self):
        f=fixture(); f.loc[0,"configured_rps"]=21
        with self.assertRaises(ValueError): validate_windows(f)

    def test_consistency_identity(self):
        r=slot_consistency(1000, 4, 50)
        self.assertEqual(r["implied_full_pool_rps"],250)
        self.assertEqual(r["lifetime_needed_at_full_pool_s"],20)
        self.assertEqual(r["effective_slots_needed_with_low_load_lifetime"],200)

    def test_reject_invalid_identity(self):
        for values in [(0,3,40),(1024,0,40),(1024,3,-1),(1024,np.nan,40)]:
            with self.assertRaises(ValueError): slot_consistency(*values)

    def test_high_rate_requires_persistent_waiting(self):
        f=fixture(); f["configured_rps"]=80
        self.assertEqual(len(saturated_rows(f)),0)
        f["waiting_positive_all_snapshots"]=True
        self.assertEqual(len(saturated_rows(f)),len(f))
        f.loc[0,"max_metric_gap_s"]=21
        self.assertEqual(len(saturated_rows(f)),len(f)-1)


if __name__ == "__main__": unittest.main()
