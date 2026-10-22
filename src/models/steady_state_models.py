"""Candidate monotone saturation maps from request rate to IT active power."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


Array = np.ndarray


@dataclass(frozen=True)
class ModelSpec:
    name: str
    parameter_names: tuple[str, ...]
    equation: str
    interpretation: str
    caveat: str
    function: Callable[[Array, Array], Array]
    initializer: Callable[[Array, Array], Array]
    bounder: Callable[[Array, Array], tuple[Array, Array]]

    def predict(self, request_rate: Array, parameters: Array) -> Array:
        values = self.function(
            np.asarray(request_rate, dtype=float), np.asarray(parameters, dtype=float)
        )
        return np.asarray(values, dtype=float)


def _response_guesses(x: Array, y: Array) -> tuple[float, float, float, float]:
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    tail = max(1, int(np.ceil(len(y_sorted) * 0.10)))
    low = float(np.median(y_sorted[:tail]))
    high = float(np.median(y_sorted[-tail:]))
    delta = max(high - low, float(np.ptp(y_sorted)), max(abs(low), 1.0) * 0.01)
    target = low + 0.5 * delta
    half_rate = float(x_sorted[np.argmin(np.abs(y_sorted - target))])
    return low, delta, max(half_rate, 1e-3), max(float(np.max(x_sorted)), 1.0)


def _common_bounds(x: Array, y: Array) -> tuple[float, float, float]:
    y_max = max(float(np.max(y)), 1e-3)
    y_range = max(float(np.ptp(y)), y_max * 0.05)
    x_max = max(float(np.max(x)), 1.0)
    return y_max, y_range, x_max


def _exponential(x: Array, p: Array) -> Array:
    p_low, delta_p, rate_scale = p
    return p_low + delta_p * (1.0 - np.exp(-np.maximum(x, 0.0) / rate_scale))


def _exponential_init(x: Array, y: Array) -> Array:
    low, delta, half_rate, _ = _response_guesses(x, y)
    return np.array([low, delta, max(half_rate / np.log(2.0), 1e-3)])


def _exponential_bounds(x: Array, y: Array) -> tuple[Array, Array]:
    y_max, y_range, x_max = _common_bounds(x, y)
    return (
        np.array([0.0, 0.0, 1e-3]),
        np.array([1.5 * y_max, max(4.0 * y_range, y_max), 20.0 * x_max]),
    )


def _hill(x: Array, p: Array) -> Array:
    p_low, delta_p, rate_half, hill_n = p
    nonnegative = np.maximum(x, 0.0)
    ratio = np.power(nonnegative / rate_half, hill_n)
    return p_low + delta_p * ratio / (1.0 + ratio)


def _hill_init(x: Array, y: Array) -> Array:
    low, delta, half_rate, _ = _response_guesses(x, y)
    return np.array([low, delta, half_rate, 1.0])


def _hill_bounds(x: Array, y: Array) -> tuple[Array, Array]:
    y_max, y_range, x_max = _common_bounds(x, y)
    return (
        np.array([0.0, 0.0, 1e-3, 0.2]),
        np.array([1.5 * y_max, max(4.0 * y_range, y_max), 20.0 * x_max, 8.0]),
    )


def _piecewise_capped_linear(x: Array, p: Array) -> Array:
    p_low, slope, knee_rate = p
    return p_low + slope * np.minimum(np.maximum(x, 0.0), knee_rate)


def _piecewise_init(x: Array, y: Array) -> Array:
    low, delta, half_rate, x_max = _response_guesses(x, y)
    knee = min(max(2.0 * half_rate, 1e-3), x_max)
    return np.array([low, delta / max(knee, 1e-3), knee])


def _piecewise_bounds(x: Array, y: Array) -> tuple[Array, Array]:
    y_max, y_range, x_max = _common_bounds(x, y)
    slope_upper = max(4.0 * y_range, y_max) / max(float(np.min(x[x > 0])), 1e-3)
    return (
        np.array([0.0, 0.0, 1e-3]),
        np.array([1.5 * y_max, slope_upper, 20.0 * x_max]),
    )


MODEL_SPECS = {
    "exponential": ModelSpec(
        name="exponential",
        parameter_names=("p_low_kw", "delta_p_kw", "rate_scale_req_s"),
        equation="P=P_low+DeltaP(1-exp(-r/r_scale))",
        interpretation="Smooth monotone saturation with one characteristic request-rate scale",
        caveat="Candidate synthesis; the exact exponential form is not prescribed by NLR",
        function=_exponential,
        initializer=_exponential_init,
        bounder=_exponential_bounds,
    ),
    "hill": ModelSpec(
        name="hill",
        parameter_names=("p_low_kw", "delta_p_kw", "rate_half_req_s", "hill_n"),
        equation="P=P_low+DeltaP*r^n/(r_half^n+r^n)",
        interpretation="Flexible sigmoidal saturation with an identifiable half-response rate",
        caveat="Extra shape parameter can overfit and should be judged on held-out request rates",
        function=_hill,
        initializer=_hill_init,
        bounder=_hill_bounds,
    ),
    "piecewise_capped_linear": ModelSpec(
        name="piecewise_capped_linear",
        parameter_names=("p_low_kw", "slope_kw_per_req_s", "knee_rate_req_s"),
        equation="P=P_low+slope*min(r,r_knee)",
        interpretation="Transparent linear loading followed by a hard saturation plateau",
        caveat="Non-smooth knee is interpretable but may underrepresent a gradual transition",
        function=_piecewise_capped_linear,
        initializer=_piecewise_init,
        bounder=_piecewise_bounds,
    ),
}


def get_model_spec(name: str) -> ModelSpec:
    try:
        return MODEL_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown steady-state model: {name}") from exc


def derived_parameters(model_name: str, parameters: Array) -> dict[str, float]:
    """Return common physical quantities for cross-model comparison."""

    values = np.asarray(parameters, dtype=float)
    if model_name in {"exponential", "hill"}:
        p_low, delta_p = values[:2]
        p_sat = p_low + delta_p
    elif model_name == "piecewise_capped_linear":
        p_low, slope, knee_rate = values
        p_sat = p_low + slope * knee_rate
    else:
        raise ValueError(f"Unknown steady-state model: {model_name}")
    return {
        "derived_p_low_kw": float(p_low),
        "derived_p_sat_kw": float(p_sat),
        "derived_dynamic_range_kw": float(p_sat - p_low),
    }


def model_registry_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_name": spec.name,
                "parameter_names": ";".join(spec.parameter_names),
                "equation": spec.equation,
                "physical_interpretation": spec.interpretation,
                "research_caveat": spec.caveat,
            }
            for spec in MODEL_SPECS.values()
        ]
    )

