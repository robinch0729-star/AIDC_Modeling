"""Controlled aggregation diagnostics, not a fitted transient power model.

Same event timestamps and observation window are retained for every bin width
and phase. Partial endpoint bins use their actual duration. Unsupported M3
inputs are exposed, never clipped or replaced by an inferred idle value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def bin_edges(duration_s: float, bin_s: float, phase_fraction: float = 0.0) -> np.ndarray:
    """Return half-open bins clipped to [0, duration], using nanosecond ticks."""
    if not np.isfinite([duration_s, bin_s, phase_fraction]).all():
        raise ValueError("Finite bin settings required")
    if duration_s <= 0 or bin_s <= 0 or not 0 <= phase_fraction < 1:
        raise ValueError("Positive duration/width and phase in [0, 1) required")
    end, width = int(round(duration_s * 1e9)), int(round(bin_s * 1e9))
    offset = int(round(bin_s * phase_fraction * 1e9))
    if width < 1 or end < 1:
        raise ValueError("Nanosecond resolution is the minimum")
    interior = np.arange(offset, end, width, dtype=np.int64)
    return np.unique(np.r_[0, interior, end]).astype(float) / 1e9


def aggregate_arrivals(times_s, duration_s: float, bin_s: float, phase_fraction: float = 0.0) -> pd.DataFrame:
    times = np.asarray(times_s, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("Finite one-dimensional arrival times required")
    if np.any(times < 0) or np.any(times >= duration_s):
        raise ValueError("Arrivals must lie in the end-exclusive observation window")
    edges = bin_edges(duration_s, bin_s, phase_fraction)
    counts, _ = np.histogram(times, bins=edges)
    widths = np.diff(edges)
    if counts.sum() != len(times):
        raise AssertionError("Source request count changed")
    return pd.DataFrame({"time_s": edges[:-1], "bin_s": widths,
                         "source_count": counts, "source_rate_rps": counts / widths})


def weighted_quantile(values, weights, probability: float) -> float:
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if len(values) == 0 or np.any(weights <= 0) or not 0 <= probability <= 1:
        raise ValueError("Nonempty values, positive weights and valid probability required")
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    index = min(np.searchsorted(cumulative, probability, side="left"), len(values) - 1)
    return float(values[order[index]])


def describe_bins(part: pd.DataFrame) -> dict:
    weights = part.bin_s.to_numpy(float)
    source = part.source_rate_rps.to_numpy(float)
    rate = part.node_rate_rps.to_numpy(float)
    mean = float(np.average(source, weights=weights))
    unsupported = part.support_status.eq("unsupported_extrapolation").to_numpy()
    return {
        "bin_count": len(part), "source_request_count": int(part.source_count.sum()),
        "duration_s": float(weights.sum()), "source_mean_rps": mean,
        "source_std_rps_time_weighted": float(np.sqrt(np.average((source - mean)**2, weights=weights))),
        "zero_bin_fraction": float(np.mean(source == 0)),
        "zero_time_fraction": float(np.average(source == 0, weights=weights)),
        "node_mean_rps": float(np.average(rate, weights=weights)),
        "node_min_rps": float(rate.min()), "node_max_rps": float(rate.max()),
        "node_p95_rps_time_weighted": weighted_quantile(rate, weights, .95),
        "unsupported_bin_count": int(unsupported.sum()),
        "unsupported_time_fraction": float(np.average(unsupported, weights=weights)),
        "above_100_time_fraction": float(np.average(rate > 100, weights=weights)),
        "below_1_time_fraction": float(np.average(rate < 1, weights=weights)),
        "scenario_offered_requests": float(np.sum(part.offered_rps * weights)),
    }


def project_bins(part: pd.DataFrame, column: str, interval_s: float = .1) -> tuple[np.ndarray, np.ndarray]:
    """Project ONLY grid-aligned bin edges; no accidental nearest interpolation."""
    times, widths = part.time_s.to_numpy(float), part.bin_s.to_numpy(float)
    if not np.isclose(times[0], 0) or not np.allclose(times[1:], times[:-1] + widths[:-1]):
        raise ValueError("Contiguous bins starting at zero required")
    if not np.allclose(times / interval_s, np.rint(times / interval_s)) or not np.allclose(widths / interval_s, np.rint(widths / interval_s)):
        raise ValueError("Bin boundaries must align with output grid")
    samples = np.rint(widths / interval_s).astype(int)
    if np.any(samples < 1):
        raise ValueError("Every bin needs an output sample")
    values = np.repeat(part[column].to_numpy(float), samples)
    boundary = np.zeros(len(values), bool)
    boundary[np.cumsum(samples)[:-1]] = True
    return values, boundary


def ramp_metrics(values, boundaries, interval_s: float = .1) -> dict:
    values, boundaries = np.asarray(values, float), np.asarray(boundaries, bool)
    if len(values) != len(boundaries) or not np.isfinite(values).all():
        raise ValueError("Finite aligned power and boundary series required")
    result = {"mean_mw": float(values.mean()), "std_mw": float(values.std()),
              "energy_kwh": float(values.sum() * interval_s / 3.6),
              "minimum_mw": float(values.min()), "maximum_mw": float(values.max())}
    prefix = np.cumsum(boundaries.astype(int))
    for horizon, label in ((.1, "0p1s"), (1., "1s")):
        lag = int(round(horizon / interval_s))
        if lag < 1 or lag >= len(values) or not np.isclose(lag * interval_s, horizon):
            continue
        ramp = np.abs(values[lag:] - values[:-lag]) / horizon
        crosses = (prefix[lag:] - prefix[:-lag]) > 0
        result[f"p95_abs_ramp_{label}_mw_per_s"] = float(np.quantile(ramp, .95))
        result[f"max_abs_ramp_{label}_mw_per_s"] = float(ramp.max())
        result[f"boundary_window_fraction_{label}"] = float(crosses.mean())
        result[f"within_bin_max_abs_ramp_{label}_mw_per_s"] = float(ramp[~crosses].max()) if (~crosses).any() else np.nan
        variation = np.sum(np.abs(values[lag:] - values[:-lag]))
        result[f"boundary_abs_change_share_{label}"] = float(np.sum(ramp[crosses]) / ramp.sum()) if variation > 0 else np.nan
    return result


def to_dispatch(part: pd.DataFrame) -> pd.DataFrame:
    if part.support_status.eq("unsupported_extrapolation").any():
        raise ValueError("No simulation for unsupported arrivals; no implicit clipping")
    result = part.rename(columns={"node_rate_rps": "node_offered_arrival_rps"}).copy()
    result["admitted_rps"] = result.offered_rps
    result["excess_rps"] = 0.0
    result["completed_throughput_modeled"] = False
    return result
