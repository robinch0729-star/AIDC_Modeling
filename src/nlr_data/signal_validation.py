"""Signal-quality checks and analysis-resolution policies for NLR profiles."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .readers import SIGNAL_COLUMNS


def _longest_true_run(mask: np.ndarray) -> int:
    if mask.size == 0 or not mask.any():
        return 0
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return int((changes[1::2] - changes[::2]).max())


def resample_to_interval(
    frame: pd.DataFrame,
    interval_s: float,
    *,
    method: str = "mean_bin_downsample",
) -> pd.DataFrame:
    """Downsample a canonical frame using fixed elapsed-time bins.

    ``mean_bin_downsample`` prevents interpolated 1 ms points from being
    treated as independent measurements. It is intentionally not used for
    profiles whose native and analysis intervals are already equal.
    """

    if interval_s <= 0:
        raise ValueError("interval_s must be positive")
    if method == "none":
        return frame.copy()
    if method != "mean_bin_downsample":
        raise ValueError(f"Unsupported resampling method: {method}")

    working = frame[SIGNAL_COLUMNS].copy()
    working["time_s"] = pd.to_numeric(working["time_s"], errors="coerce")
    working = working.dropna(subset=["time_s"]).sort_values("time_s", kind="stable")
    if working.empty:
        return working

    start = float(working["time_s"].iloc[0])
    relative_time = working["time_s"] - start
    # A small relative tolerance keeps exact boundary samples in their intended bin.
    bins = np.floor((relative_time + interval_s * 1e-9) / interval_s).astype(int)
    working["_bin"] = bins

    grouped = working.groupby("_bin", sort=True, observed=True)
    result = grouped.agg(
        timestamp=("timestamp", "first"),
        power_w=("power_w", "mean"),
        utilization=("utilization", "mean"),
    ).reset_index()
    result["time_s"] = result["_bin"].astype(float) * interval_s
    result = result.drop(columns="_bin")[SIGNAL_COLUMNS]
    result.attrs.update(frame.attrs)
    result.attrs["resampling_method"] = method
    result.attrs["analysis_interval_s"] = interval_s
    return result


def apply_analysis_policy(
    frame: pd.DataFrame,
    manifest_record: Mapping[str, Any],
) -> pd.DataFrame:
    method = str(manifest_record.get("resampling_method", "none"))
    interval = float(manifest_record["analysis_interval_s"])
    return resample_to_interval(frame, interval, method=method)


def signal_quality_metrics(
    frame: pd.DataFrame,
    *,
    expected_interval_s: float | None = None,
) -> dict[str, Any]:
    """Calculate whole-profile quality metrics without silently repairing data."""

    metrics: dict[str, Any] = {
        "rows": len(frame),
        "quality_status": "pass",
        "quality_reasons": "",
    }
    reasons_fail: list[str] = []
    reasons_warning: list[str] = []

    missing_columns = [column for column in SIGNAL_COLUMNS if column not in frame]
    metrics["missing_required_column_count"] = len(missing_columns)
    if missing_columns:
        metrics["quality_status"] = "fail"
        metrics["quality_reasons"] = f"missing_columns={missing_columns}"
        return metrics
    if frame.empty:
        metrics["quality_status"] = "fail"
        metrics["quality_reasons"] = "empty_profile"
        return metrics

    time_s = pd.to_numeric(frame["time_s"], errors="coerce")
    power = pd.to_numeric(frame["power_w"], errors="coerce")
    finite_time = np.isfinite(time_s)
    finite_power = np.isfinite(power)

    metrics["missing_time_count"] = int((~finite_time).sum())
    metrics["missing_power_count"] = int((~finite_power).sum())
    metrics["negative_power_count"] = int((power < 0).sum())
    metrics["duplicate_time_count"] = int(time_s.duplicated().sum())
    metrics["nonmonotonic_time"] = int(not time_s.is_monotonic_increasing)
    metrics["zero_power_count"] = int(power.eq(0).sum())

    if metrics["missing_time_count"]:
        reasons_fail.append("invalid_time")
    if metrics["missing_power_count"] == len(frame):
        reasons_fail.append("all_power_invalid")
    elif metrics["missing_power_count"]:
        reasons_warning.append("partial_power_missing")
    if metrics["negative_power_count"]:
        reasons_warning.append("negative_power")
    if metrics["duplicate_time_count"]:
        reasons_fail.append("duplicate_time")
    if metrics["nonmonotonic_time"]:
        reasons_fail.append("nonmonotonic_time")

    valid = pd.DataFrame({"time_s": time_s, "power_w": power})
    valid = valid[np.isfinite(valid["time_s"]) & np.isfinite(valid["power_w"])]
    deltas_t = valid["time_s"].diff().dropna()
    positive_dt = deltas_t[deltas_t > 0]
    deltas_p = valid["power_w"].diff().dropna()

    metrics["duration_s"] = (
        float(valid["time_s"].iloc[-1] - valid["time_s"].iloc[0])
        if len(valid) >= 2
        else 0.0
    )
    metrics["median_interval_s"] = (
        float(positive_dt.median()) if not positive_dt.empty else None
    )
    metrics["min_interval_s"] = float(positive_dt.min()) if not positive_dt.empty else None
    metrics["max_interval_s"] = float(positive_dt.max()) if not positive_dt.empty else None
    metrics["std_interval_s"] = float(positive_dt.std(ddof=0)) if not positive_dt.empty else None

    reference_interval = expected_interval_s or metrics["median_interval_s"]
    if reference_interval and not positive_dt.empty:
        tolerance = max(1e-9, float(reference_interval) * 0.02)
        irregular = (positive_dt - float(reference_interval)).abs() > tolerance
        metrics["irregular_interval_count"] = int(irregular.sum())
        metrics["large_gap_count"] = int(
            (positive_dt > float(reference_interval) * 1.5).sum()
        )
        metrics["interval_relative_error"] = abs(
            float(metrics["median_interval_s"]) - float(reference_interval)
        ) / float(reference_interval)
        if metrics["large_gap_count"]:
            reasons_warning.append("large_time_gap")
    else:
        metrics["irregular_interval_count"] = 0
        metrics["large_gap_count"] = 0
        metrics["interval_relative_error"] = None

    if deltas_p.empty:
        flat_mask = np.array([], dtype=bool)
    else:
        scale = max(float(valid["power_w"].abs().max()), 1.0)
        flat_mask = deltas_p.abs().to_numpy() <= scale * 1e-10
    metrics["flat_step_fraction"] = (
        float(flat_mask.mean()) if flat_mask.size else None
    )
    metrics["longest_flat_run_samples"] = _longest_true_run(flat_mask)
    metrics["constant_signal"] = int(valid["power_w"].nunique(dropna=True) <= 1)
    if metrics["constant_signal"]:
        reasons_warning.append("constant_signal")

    if reasons_fail:
        metrics["quality_status"] = "fail"
    elif reasons_warning:
        metrics["quality_status"] = "warning"
    metrics["quality_reasons"] = ";".join(reasons_fail + reasons_warning)
    return metrics

