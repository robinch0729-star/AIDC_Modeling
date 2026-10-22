from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlr_data.residual_model_selection import (  # noqa: E402
    ResidualParameterLibrary,
    feature_errors,
    simulate_residual,
    simulate_residual_ensemble,
)


def _library() -> ResidualParameterLibrary:
    table = pd.DataFrame(
        [
            {
                "profile_id": "p1",
                "prompt_dataset": "d",
                "request_rate": 10.0,
                "residual_std_kw": 0.25,
                "phi": 0.8,
                "innovation_std_kw": 0.15,
                "jump_evidence": 1.0,
                "negative_weight": 0.05,
                "negative_mean_kw": -0.8,
                "negative_std_kw": 0.1,
                "background_weight": 0.90,
                "background_mean_kw": 0.0,
                "background_std_kw": 0.08,
                "positive_weight": 0.05,
                "positive_mean_kw": 0.7,
                "positive_std_kw": 0.1,
            }
        ]
    )
    return ResidualParameterLibrary(table, {"p1": np.linspace(-0.2, 0.2, 1000)})


def test_all_residual_generators_return_finite_zero_mean_series() -> None:
    library = _library()
    for model in (
        "R0_m3_only",
        "R1_white_noise",
        "R2_ou",
        "R3_jump_ou",
        "R4_empirical_replay",
    ):
        values = simulate_residual(
            model,
            length=1000,
            prompt_dataset="d",
            request_rate=10.0,
            library=library,
            rng=np.random.default_rng(3),
        )
        assert len(values) == 1000
        assert np.isfinite(values).all()
        assert abs(float(np.mean(values))) < 1e-10


def test_feature_error_prefers_identical_trace() -> None:
    rng = np.random.default_rng(5)
    observed = np.zeros(4000)
    for index in range(1, len(observed)):
        observed[index] = 0.85 * observed[index - 1] + rng.normal(0.0, 0.1)
    same = feature_errors(observed, observed.copy(), 0.1)
    shuffled = feature_errors(observed, rng.permutation(observed), 0.1)
    assert same["overall_score"] < 1e-10
    assert shuffled["overall_score"] > same["overall_score"]


def test_vectorized_r0_to_r3_ensembles_are_finite_and_row_centered() -> None:
    library = _library()
    for model in ("R0_m3_only", "R1_white_noise", "R2_ou", "R3_jump_ou"):
        values = simulate_residual_ensemble(
            model,
            draws=8,
            length=250,
            prompt_dataset="d",
            request_rate=10.0,
            library=library,
            rng=np.random.default_rng(11),
        )
        assert values.shape == (8, 250)
        assert np.isfinite(values).all()
        assert np.max(np.abs(np.mean(values, axis=1))) < 1e-10
