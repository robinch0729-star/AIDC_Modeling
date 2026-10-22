"""Representative integration tests for the local NLR dataset readers."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nlr_data.readers import (  # noqa: E402
    METADATA_COLUMNS,
    SIGNAL_COLUMNS,
    UnsupportedDataRoleError,
    read_timeseries,
    summarize_timeseries,
)
from nlr_data.schema_registry import identify_schema  # noqa: E402


class NLRReaderIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        default_root = PROJECT_ROOT.parent / "NLR_dataset"
        cls.data_root = Path(os.environ.get("NLR_DATA_ROOT", default_root)).resolve()
        if not cls.data_root.is_dir():
            raise unittest.SkipTest(f"NLR dataset not found: {cls.data_root}")

    def _first(self, pattern: str) -> Path:
        matches = sorted(self.data_root.glob(pattern))
        self.assertTrue(matches, f"No representative file matches {pattern}")
        return matches[0]

    def _assert_canonical(self, frame: pd.DataFrame, expected_rows: int = 256) -> None:
        self.assertEqual(frame.columns.tolist(), SIGNAL_COLUMNS + METADATA_COLUMNS)
        self.assertEqual(len(frame), expected_rows)
        self.assertAlmostEqual(float(frame["time_s"].iloc[0]), 0.0)
        self.assertTrue(frame["time_s"].is_monotonic_increasing)
        self.assertTrue(frame["power_w"].notna().any())
        self.assertGreater(float(frame["power_w"].max()), 0.0)

    def test_raw_training_nvml_and_rapl(self) -> None:
        nvml = self._first(
            "00_raw_datasets/training_llama2_70b_lora/2node/nvml*.log"
        )
        rapl = self._first(
            "00_raw_datasets/training_llama2_70b_lora/2node/rapl*.log"
        )

        nvml_frame = read_timeseries(
            nvml, dataset_root=self.data_root, max_rows=256
        )
        rapl_frame = read_timeseries(
            rapl, dataset_root=self.data_root, max_rows=256
        )
        self._assert_canonical(nvml_frame)
        self._assert_canonical(rapl_frame)
        self.assertEqual(nvml_frame.attrs["measurement_scope"], "GPU_power_only")
        self.assertEqual(rapl_frame.attrs["measurement_scope"], "CPU_power_only")

    def test_raw_inference_multiline_header(self) -> None:
        path = self._first(
            "00_raw_datasets/inference_offline_llama3_70b/nvml_wattameter*.log"
        )
        frame = read_timeseries(path, dataset_root=self.data_root, max_rows=256)
        self._assert_canonical(frame)
        self.assertEqual(frame.attrs["schema_name"], "raw_nvml_power_log")
        self.assertEqual(frame.attrs["workload_type"], "offline_inference")

    def test_all_aggregated_workload_groups(self) -> None:
        patterns = [
            "01_aggregated_datasets/training/results/*.parquet",
            "01_aggregated_datasets/inference_offline_llama3_70b/results/*.parquet",
            "01_aggregated_datasets/inference_online_finite_llama3_70b/results/*.parquet",
            "01_aggregated_datasets/inference_online_rate_llama3_70b/results/*.parquet",
        ]
        for pattern in patterns:
            with self.subTest(pattern=pattern):
                path = self._first(pattern)
                frame = read_timeseries(
                    path, dataset_root=self.data_root, max_rows=256
                )
                self._assert_canonical(frame)
                self.assertTrue(frame["timestamp"].isna().all())
                self.assertEqual(frame.attrs["measurement_scope"], "job_total_IT_power")

    def test_whole_facility_inference_and_colocation(self) -> None:
        patterns = [
            "03_whole-facility_profiles/inference/simulated_data/*_power.csv",
            "03_whole-facility_profiles/colocation/simulated_data/*_power.csv",
        ]
        for pattern in patterns:
            with self.subTest(pattern=pattern):
                path = self._first(pattern)
                frame = read_timeseries(
                    path, dataset_root=self.data_root, max_rows=256
                )
                self._assert_canonical(frame)
                self.assertTrue(frame["timestamp"].notna().all())
                self.assertTrue(frame["utilization"].notna().all())
                self.assertEqual(frame.attrs["measurement_scope"], "whole_facility_power")
                self.assertAlmostEqual(
                    summarize_timeseries(frame)["median_interval_s"], 60.0
                )

    def test_metadata_file_is_not_read_as_timeseries(self) -> None:
        metadata_path = self._first(
            "01_aggregated_datasets/inference_offline_llama3_70b/metadata.csv"
        )
        self.assertEqual(
            identify_schema(metadata_path, self.data_root).role,
            "metadata",
        )
        with self.assertRaises(UnsupportedDataRoleError):
            read_timeseries(metadata_path, dataset_root=self.data_root)


if __name__ == "__main__":
    unittest.main(verbosity=2)

