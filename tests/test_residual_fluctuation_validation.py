import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nlr_data.residual_fluctuation_validation import (
    fluctuation_calibration, scenario_interval_metrics, aidc_temporal_diagnostics,
)


def test_fluctuation_calibration_exact_match_is_zero():
    y = np.array([0., 1., 3., 2.])
    x = np.tile(y, (4, 1))
    result = fluctuation_calibration(y, x, 1)
    assert all(value == pytest.approx(0.) for value in result.values())


def test_interval_metrics_definition():
    y = np.array([0., 3., 2.])
    x = np.array([[-1., 0., 1.], [1., 2., 3.]])
    result = scenario_interval_metrics(y, x, lambda_siw=10.)
    assert result["aiw_kw"] == pytest.approx(2.)
    assert result["awd_kw"] == pytest.approx(1/3)
    assert result["siw_kw"] == pytest.approx(2 + 10/3)
    assert result["cp"] == pytest.approx(2/3)


def test_temporal_diagnostics_identical_ensemble():
    rng = np.random.default_rng(1)
    y = rng.normal(size=100)
    x = np.tile(y, (5, 1))
    result = aidc_temporal_diagnostics(y, x, interval_s=.1)
    assert result["acf_mae"] == pytest.approx(0.)
    assert result["ramp_w1_0p1s_kw_per_s"] == pytest.approx(0.)
    assert result["q01_abs_error_kw"] == pytest.approx(0.)
