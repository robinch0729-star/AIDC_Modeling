"""Prediction-error and physical-shape diagnostics for saturation models."""

from __future__ import annotations

import numpy as np


def regression_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    parameter_count: int,
) -> dict[str, float]:
    y = np.asarray(observed, dtype=float)
    y_hat = np.asarray(predicted, dtype=float)
    valid = np.isfinite(y) & np.isfinite(y_hat)
    y = y[valid]
    y_hat = y_hat[valid]
    if y.size == 0:
        raise ValueError("No finite observations for evaluation")
    residual = y_hat - y
    sse = float(np.sum(np.square(residual)))
    mse = sse / len(y)
    variance = float(np.sum(np.square(y - np.mean(y))))
    r2 = 1.0 - sse / variance if variance > np.finfo(float).eps else np.nan
    nonzero = np.abs(y) > np.finfo(float).eps
    mape = float(np.mean(np.abs(residual[nonzero] / y[nonzero])) * 100.0)
    likelihood_term = len(y) * np.log(max(mse, np.finfo(float).tiny))
    return {
        "sample_count": int(len(y)),
        "rmse_kw": float(np.sqrt(mse)),
        "mae_kw": float(np.mean(np.abs(residual))),
        "mape_pct": mape,
        "r2": float(r2),
        "bias_kw": float(np.mean(residual)),
        "aic": float(likelihood_term + 2.0 * parameter_count),
        "bic": float(likelihood_term + parameter_count * np.log(len(y))),
    }


def shape_diagnostics(
    request_rate: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    x = np.asarray(request_rate, dtype=float)
    y = np.asarray(observed, dtype=float)
    y_hat = np.asarray(predicted, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(y_hat)
    x, y, y_hat = x[valid], y[valid], y_hat[valid]
    if x.size == 0:
        raise ValueError("No finite cases for shape diagnostics")

    grouped = {}
    for rate in np.unique(x):
        grouped[rate] = float(np.mean(y_hat[x == rate]))
    ordered_predictions = np.array([grouped[rate] for rate in sorted(grouped)])
    differences = np.diff(ordered_predictions)
    tolerance = 1e-9
    violation_count = int(np.sum(differences < -tolerance))

    threshold = float(np.quantile(x, 0.90))
    high = x >= threshold
    observed_high = float(np.mean(y[high]))
    predicted_high = float(np.mean(y_hat[high]))
    saturation_error_pct = (
        100.0 * (predicted_high - observed_high) / observed_high
        if abs(observed_high) > np.finfo(float).eps
        else np.nan
    )
    return {
        "monotonicity_violation_count": violation_count,
        "minimum_prediction_step_kw": float(np.min(differences)) if differences.size else 0.0,
        "high_rate_bias_kw": float(predicted_high - observed_high),
        "high_rate_saturation_error_pct": float(saturation_error_pct),
    }

