from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlr_data.residual_diagnostics import (  # noqa: E402
    _ljung_box,
    fit_ar1_ou,
    fit_gaussian_mixture_1d,
)


def test_ar1_ou_recovers_positive_persistence() -> None:
    rng = np.random.default_rng(7)
    phi = 0.82
    values = np.zeros(20_000)
    noise = rng.normal(0.0, 0.2, len(values) - 1)
    for index in range(1, len(values)):
        values[index] = phi * values[index - 1] + noise[index - 1]
    fitted = fit_ar1_ou(values, interval_s=0.1)
    assert abs(fitted["phi"] - phi) < 0.02
    assert fitted["theta_per_s"] > 0.0
    assert fitted["correlation_time_s"] > 0.0


def test_ar1_innovations_are_substantially_whiter() -> None:
    rng = np.random.default_rng(11)
    values = np.zeros(10_000)
    for index in range(1, len(values)):
        values[index] = 0.9 * values[index - 1] + rng.normal(0.0, 0.15)
    fitted = fit_ar1_ou(values, interval_s=0.1)
    innovations = fitted["innovations"]
    _, residual_p = _ljung_box(values, 20)
    _, innovation_p = _ljung_box(innovations, 20)
    assert residual_p < 1e-6
    assert innovation_p > residual_p


def test_gmm_bic_detects_signed_tail_mixture() -> None:
    rng = np.random.default_rng(19)
    background = rng.normal(0.0, 0.08, 18_000)
    negative = rng.normal(-0.8, 0.12, 1_000)
    positive = rng.normal(0.7, 0.10, 1_000)
    values = np.concatenate([background, negative, positive])
    rng.shuffle(values)
    mixture = fit_gaussian_mixture_1d(values)
    mean = float(np.mean(values))
    std = float(np.std(values))
    gaussian_ll = float(
        np.sum(-0.5 * np.square((values - mean) / std) - np.log(std) - 0.5 * np.log(2.0 * np.pi))
    )
    gaussian_bic = 2 * np.log(len(values)) - 2.0 * gaussian_ll
    assert mixture.converged
    assert gaussian_bic - mixture.bic > 10.0
    assert mixture.means[0] < 0.0 < mixture.means[-1]

