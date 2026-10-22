import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nlr_data.r3_high_rate_benchmark import regression_metrics


def test_regression_metrics_exact_prediction():
    y = np.array([1., 2., 4.])
    result = regression_metrics(y, y)
    assert result["mae_kw"] == pytest.approx(0.)
    assert result["rmse_kw"] == pytest.approx(0.)
    assert result["mape_pct"] == pytest.approx(0.)
    assert result["r2"] == pytest.approx(1.)


def test_regression_metrics_known_error():
    y = np.array([1., 2.])
    p = np.array([2., 2.])
    result = regression_metrics(y, p)
    assert result["mae_kw"] == pytest.approx(.5)
    assert result["rmse_kw"] == pytest.approx(np.sqrt(.5))
    assert result["mean_bias_kw"] == pytest.approx(.5)
