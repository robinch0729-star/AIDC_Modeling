"""Tests for the locked final M3+R3 IT profile generator."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlr_data.final_it_profile_generator import (  # noqa: E402
    FinalITProfileGenerator,
    GenerationRequest,
)


def _artifact(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"prompt_dataset": "d", "p0_kw": 2.0, "delta_p_kw": 0.8, "k": 4.0, "x0": 3.0}]
    ).to_csv(root / "train_validation_m3_parameters.csv", index=False)
    rows = []
    for split, rate in (("train", 10.0), ("validation", 20.0)):
        rows.append(
            {
                "profile_id": f"{split}-{rate:g}",
                "prompt_dataset": "d",
                "request_rate": rate,
                "split": split,
                "interval_s": 0.1,
                "residual_std_kw": 0.1,
                "phi": 0.85,
                "innovation_std_kw": 0.05,
                "jump_evidence": 1.0,
                "negative_weight": 0.08,
                "negative_mean_kw": -0.25,
                "negative_std_kw": 0.04,
                "background_weight": 0.84,
                "background_mean_kw": 0.0,
                "background_std_kw": 0.03,
                "positive_weight": 0.08,
                "positive_mean_kw": 0.25,
                "positive_std_kw": 0.04,
            }
        )
    pd.DataFrame(rows).to_csv(
        root / "train_validation_residual_parameters.csv", index=False
    )
    (root / "protocol.json").write_text(
        json.dumps(
            {
                "fixed_model_before_benchmark": "R3_jump_ou",
                "selection_locked": True,
                "request_rate_range": [21.0, 30.0],
            }
        ),
        encoding="utf-8",
    )
    (root / "status.json").write_text(
        json.dumps({"fixed_model": "R3_jump_ou", "model_reselected": False}),
        encoding="utf-8",
    )
    return root


class FinalITProfileGeneratorTests(unittest.TestCase):
    def test_generation_is_reproducible_and_has_expected_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            generator = FinalITProfileGenerator.from_benchmark_artifact(
                _artifact(Path(temporary) / "artifact")
            )
            request = GenerationRequest(
                prompt_dataset="d",
                request_rate=15.0,
                duration_s=10.0,
                scenario_count=4,
                seed=7,
            )
            first = generator.generate(request)
            second = generator.generate(request)
            self.assertEqual(len(first), 400)
            np.testing.assert_allclose(first["it_power_kw"], second["it_power_kw"])
            self.assertEqual(
                first["request_rate_status"].iloc[0],
                "train_validation_interpolation",
            )

    def test_scaled_profiles_preserve_target_mean(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            generator = FinalITProfileGenerator.from_benchmark_artifact(
                _artifact(Path(temporary) / "artifact")
            )
            request = GenerationRequest(
                prompt_dataset="d",
                request_rate=25.0,
                duration_s=10.0,
                scenario_count=3,
                seed=8,
                target_it_mean_mw=20.0,
            )
            frame = generator.generate(request)
            means = frame.groupby("scenario_id")["it_power_mw"].mean()
            np.testing.assert_allclose(means, 20.0, atol=1e-12)
            self.assertEqual(
                frame["request_rate_status"].iloc[0],
                "validated_high_rate_extrapolation",
            )

    def test_unsupported_rate_and_wrong_dt_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            generator = FinalITProfileGenerator.from_benchmark_artifact(
                _artifact(Path(temporary) / "artifact")
            )
            with self.assertRaises(ValueError):
                generator.generate(GenerationRequest(prompt_dataset="d", request_rate=40.0))
            with self.assertRaises(ValueError):
                generator.generate(
                    GenerationRequest(
                        prompt_dataset="d", request_rate=10.0, interval_s=1.0
                    )
                )

    def test_csv_and_matlab_exports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generator = FinalITProfileGenerator.from_benchmark_artifact(
                _artifact(root / "artifact")
            )
            request = GenerationRequest(
                prompt_dataset="d",
                request_rate=10.0,
                duration_s=2.0,
                scenario_count=2,
                seed=3,
            )
            frame = generator.generate(request)
            output = generator.save_generation(frame, request, root / "output")
            self.assertTrue((output / "it_profiles_long.csv").is_file())
            self.assertTrue((output / "generation_config.json").is_file())
            matrix = loadmat(output / "it_profiles.mat")
            self.assertEqual(matrix["it_power_kw"].shape, (2, 20))


if __name__ == "__main__":
    unittest.main()
