"""Extract run-level workload/power features from NLR power profiles."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.signal import periodogram
from tqdm.auto import tqdm

from .metadata_join import build_run_manifest, save_run_manifest
from .readers import read_timeseries
from .signal_validation import apply_analysis_policy, signal_quality_metrics


FEATURE_METADATA_COLUMNS = [
    "profile_id",
    "relative_path",
    "workload_type",
    "model_name",
    "source_kind",
    "measurement_scope",
    "measurement_origin",
    "condition_id",
    "run_id",
    "nodes",
    "scale_mw",
    "target_utilization",
    "request_rate",
    "num_prompts",
    "batch_size",
    "max_output_tokens",
    "burstiness",
    "repeat",
    "seed",
    "native_interval_s",
    "analysis_interval_s",
    "resampling_method",
]


def _segment_lengths(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0 or not mask.any():
        return np.array([], dtype=int)
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return changes[1::2] - changes[::2]


def _safe_quantile(values: np.ndarray, quantile: float) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, quantile)) if finite.size else None


def _autocorrelation(values: np.ndarray, lag_samples: int) -> float | None:
    if lag_samples <= 0 or values.size <= lag_samples + 2:
        return None
    left = values[:-lag_samples]
    right = values[lag_samples:]
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3:
        return None
    left = left[mask]
    right = right[mask]
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _dominant_period_seconds(values: np.ndarray, interval_s: float) -> float | None:
    finite = np.isfinite(values)
    if finite.sum() < 16 or interval_s <= 0:
        return None
    series = pd.Series(values).interpolate(limit_direction="both").to_numpy(dtype=float)
    centered = series - np.mean(series)
    if np.std(centered) == 0:
        return None
    frequencies, spectral_power = periodogram(
        centered,
        fs=1.0 / interval_s,
        detrend="linear",
        scaling="spectrum",
    )
    valid = frequencies > 0
    if not valid.any():
        return None
    dominant_frequency = frequencies[valid][np.argmax(spectral_power[valid])]
    return float(1.0 / dominant_frequency) if dominant_frequency > 0 else None


def compute_profile_features(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute level, energy, ramp, burst, and temporal features for one run."""

    working = frame[["time_s", "power_w"]].copy()
    working["time_s"] = pd.to_numeric(working["time_s"], errors="coerce")
    working["power_w"] = pd.to_numeric(working["power_w"], errors="coerce")
    working = working.replace([np.inf, -np.inf], np.nan).dropna()
    if len(working) < 2:
        raise ValueError("At least two valid samples are required for feature extraction")

    time_s = working["time_s"].to_numpy(dtype=float)
    power = working["power_w"].to_numpy(dtype=float)
    dt = np.diff(time_s)
    dp = np.diff(power)
    positive_dt = dt > 0
    ramp = dp[positive_dt] / dt[positive_dt]
    interval_s = float(np.median(dt[positive_dt]))

    mean_power = float(np.mean(power))
    std_power = float(np.std(power, ddof=0))
    duration_s = float(time_s[-1] - time_s[0])
    energy_wh = float(np.trapezoid(power, time_s) / 3600.0)

    upward = ramp[ramp > 0]
    downward_magnitude = -ramp[ramp < 0]
    absolute_ramp = np.abs(ramp)

    burst_threshold = mean_power + std_power
    burst_mask = power > burst_threshold
    burst_lengths = _segment_lengths(burst_mask)
    burst_durations = burst_lengths.astype(float) * interval_s

    if absolute_ramp.size:
        ramp_event_threshold = float(np.quantile(absolute_ramp, 0.95))
        ramp_event_mask = absolute_ramp >= ramp_event_threshold
        ramp_event_count = int(len(_segment_lengths(ramp_event_mask)))
    else:
        ramp_event_threshold = None
        ramp_event_count = 0

    lag_1s = max(1, int(round(1.0 / interval_s)))
    lag_10s = max(1, int(round(10.0 / interval_s)))

    return {
        "sample_count": len(power),
        "duration_s": duration_s,
        "sampling_interval_s": interval_s,
        "mean_power_w": mean_power,
        "std_power_w": std_power,
        "min_power_w": float(np.min(power)),
        "max_power_w": float(np.max(power)),
        "median_power_w": float(np.median(power)),
        "p05_power_w": _safe_quantile(power, 0.05),
        "p10_power_w": _safe_quantile(power, 0.10),
        "p90_power_w": _safe_quantile(power, 0.90),
        "p95_power_w": _safe_quantile(power, 0.95),
        "p99_power_w": _safe_quantile(power, 0.99),
        "peak_to_average_ratio": (
            float(np.max(power) / mean_power) if mean_power != 0 else None
        ),
        "coefficient_of_variation": (
            float(std_power / mean_power) if mean_power != 0 else None
        ),
        "energy_wh": energy_wh,
        "mean_abs_ramp_w_per_s": (
            float(np.mean(absolute_ramp)) if absolute_ramp.size else None
        ),
        "rms_ramp_w_per_s": (
            float(np.sqrt(np.mean(np.square(ramp)))) if ramp.size else None
        ),
        "max_ramp_up_w_per_s": float(np.max(upward)) if upward.size else 0.0,
        "max_ramp_down_w_per_s": (
            float(np.max(downward_magnitude)) if downward_magnitude.size else 0.0
        ),
        "p95_ramp_up_w_per_s": _safe_quantile(upward, 0.95),
        "p95_ramp_down_w_per_s": _safe_quantile(downward_magnitude, 0.95),
        "p95_abs_ramp_w_per_s": _safe_quantile(absolute_ramp, 0.95),
        "p99_abs_ramp_w_per_s": _safe_quantile(absolute_ramp, 0.99),
        "ramp_event_threshold_w_per_s": ramp_event_threshold,
        "ramp_event_count": ramp_event_count,
        "burst_threshold_w": burst_threshold,
        "burst_count": int(len(burst_lengths)),
        "burst_total_duration_s": (
            float(np.sum(burst_durations)) if burst_durations.size else 0.0
        ),
        "burst_mean_duration_s": (
            float(np.mean(burst_durations)) if burst_durations.size else 0.0
        ),
        "burst_max_duration_s": (
            float(np.max(burst_durations)) if burst_durations.size else 0.0
        ),
        "autocorrelation_1s": _autocorrelation(power, lag_1s),
        "autocorrelation_10s": _autocorrelation(power, lag_10s),
        "dominant_period_s": _dominant_period_seconds(power, interval_s),
    }


def _prefix_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def build_condition_uncertainty(features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated runs into condition-level empirical uncertainty."""

    if features.empty:
        return pd.DataFrame()
    metrics = [
        "mean_power_w",
        "max_power_w",
        "std_power_w",
        "p95_abs_ramp_w_per_s",
        "max_ramp_up_w_per_s",
        "max_ramp_down_w_per_s",
        "energy_wh",
    ]
    rows: list[dict[str, Any]] = []
    for (workload_type, condition_id), group in features.groupby(
        ["workload_type", "condition_id"], dropna=False, observed=True
    ):
        record: dict[str, Any] = {
            "workload_type": workload_type,
            "condition_id": condition_id,
            "run_count": len(group),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            record[f"{metric}_condition_mean"] = (
                float(values.mean()) if not values.empty else None
            )
            record[f"{metric}_condition_std"] = (
                float(values.std(ddof=1)) if len(values) >= 2 else None
            )
            record[f"{metric}_condition_p05"] = (
                float(values.quantile(0.05)) if not values.empty else None
            )
            record[f"{metric}_condition_p95"] = (
                float(values.quantile(0.95)) if not values.empty else None
            )
        rows.append(record)
    return pd.DataFrame(rows).sort_values(
        ["workload_type", "condition_id"], kind="stable"
    ).reset_index(drop=True)


def extract_manifest_features(
    manifest: pd.DataFrame,
    dataset_root: str | Path,
    *,
    max_profiles: int | None = None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate and extract all eligible profiles in one sequential pass."""

    root = Path(dataset_root).expanduser().resolve()
    selected = manifest.copy()
    if max_profiles is not None:
        eligible_index = selected.index[selected["eligible"].astype(bool)][:max_profiles]
        selected = selected.loc[
            selected.index.isin(eligible_index) | ~selected["eligible"].astype(bool)
        ]

    quality_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    iterator: Iterable[tuple[int, pd.Series]] = selected.iterrows()
    iterator = tqdm(
        iterator,
        total=len(selected),
        desc="Validating and extracting NLR profiles",
        disable=not show_progress,
    )

    for _, record in iterator:
        identity = {column: record.get(column) for column in FEATURE_METADATA_COLUMNS}
        if not bool(record["eligible"]):
            quality_rows.append(
                {
                    **identity,
                    "processing_status": "excluded",
                    "processing_error": str(record["exclusion_reason"]),
                }
            )
            continue

        try:
            frame = read_timeseries(
                root / str(record["relative_path"]),
                dataset_root=root,
                include_metadata_columns=False,
                validate=False,
            )
            raw_quality = signal_quality_metrics(
                frame,
                expected_interval_s=float(record["native_interval_s"]),
            )
            analysis_frame = apply_analysis_policy(frame, record)
            analysis_quality = signal_quality_metrics(
                analysis_frame,
                expected_interval_s=float(record["analysis_interval_s"]),
            )

            quality_row = {
                **identity,
                "processing_status": (
                    "failed_quality"
                    if analysis_quality["quality_status"] == "fail"
                    else "ok"
                ),
                "processing_error": "",
                **_prefix_metrics("raw_", raw_quality),
                **_prefix_metrics("analysis_", analysis_quality),
            }
            quality_rows.append(quality_row)

            if analysis_quality["quality_status"] == "fail":
                failure_rows.append(
                    {
                        **identity,
                        "failure_stage": "quality",
                        "failure_reason": analysis_quality["quality_reasons"],
                    }
                )
                continue

            features = compute_profile_features(analysis_frame)
            feature_row = {**identity, **features}
            metadata_mean = record.get("mean_power_metadata_w")
            metadata_peak = record.get("peak_power_metadata_w")
            feature_row["mean_power_metadata_w"] = metadata_mean
            feature_row["peak_power_metadata_w"] = metadata_peak
            feature_row["mean_power_metadata_relative_error"] = (
                abs(features["mean_power_w"] - float(metadata_mean)) / abs(float(metadata_mean))
                if pd.notna(metadata_mean) and float(metadata_mean) != 0
                else None
            )
            feature_row["peak_power_metadata_relative_error"] = (
                abs(features["max_power_w"] - float(metadata_peak)) / abs(float(metadata_peak))
                if pd.notna(metadata_peak) and float(metadata_peak) != 0
                else None
            )
            feature_rows.append(feature_row)
        except Exception as exc:
            failure_rows.append(
                {
                    **identity,
                    "failure_stage": "read_or_extract",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            )
            quality_rows.append(
                {
                    **identity,
                    "processing_status": "error",
                    "processing_error": f"{type(exc).__name__}: {exc}",
                }
            )

    return (
        pd.DataFrame(quality_rows),
        pd.DataFrame(feature_rows),
        pd.DataFrame(failure_rows),
    )


def feature_definitions() -> pd.DataFrame:
    definitions = {
        "mean_power_w": "Mean active power over the analysis profile [W]",
        "std_power_w": "Population standard deviation of active power [W]",
        "peak_to_average_ratio": "Maximum power divided by mean power [-]",
        "coefficient_of_variation": "Power standard deviation divided by mean [-]",
        "energy_wh": "Trapezoidal integral of active power [Wh]",
        "mean_abs_ramp_w_per_s": "Mean absolute dP/dt [W/s]",
        "max_ramp_up_w_per_s": "Maximum positive dP/dt [W/s]",
        "max_ramp_down_w_per_s": "Maximum magnitude of negative dP/dt [W/s]",
        "p95_abs_ramp_w_per_s": "95th percentile of absolute dP/dt [W/s]",
        "burst_threshold_w": "Per-profile burst threshold: mean plus one std [W]",
        "burst_count": "Number of contiguous periods above the burst threshold",
        "burst_total_duration_s": "Total time above burst threshold [s]",
        "autocorrelation_1s": "Power autocorrelation at approximately 1 second",
        "autocorrelation_10s": "Power autocorrelation at approximately 10 seconds",
        "dominant_period_s": "Largest non-DC periodogram component [s]",
    }
    return pd.DataFrame(
        [{"feature": feature, "definition": definition} for feature, definition in definitions.items()]
    )


def save_pipeline_outputs(
    output_dir: str | Path,
    *,
    quality: pd.DataFrame,
    features: pd.DataFrame,
    failures: pd.DataFrame,
    uncertainty: pd.DataFrame,
) -> dict[str, Path]:
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    tables = {
        "profile_quality_report": quality,
        "workload_features": features,
        "extraction_failures": failures,
        "condition_uncertainty_features": uncertainty,
        "feature_definitions": feature_definitions(),
    }
    paths: dict[str, Path] = {}
    for stem, table in tables.items():
        csv_path = directory / f"{stem}.csv"
        table.to_csv(csv_path, index=False, encoding="utf-8-sig")
        paths[f"{stem}_csv"] = csv_path
        if not table.empty:
            parquet_path = directory / f"{stem}.parquet"
            table.to_parquet(parquet_path, index=False)
            paths[f"{stem}_parquet"] = parquet_path
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/signal_validation")
    )
    parser.add_argument("--max-profiles", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_run_manifest(args.data_root)
    save_run_manifest(manifest, args.output_dir)
    quality, features, failures = extract_manifest_features(
        manifest,
        args.data_root,
        max_profiles=args.max_profiles,
        show_progress=not args.no_progress,
    )
    uncertainty = build_condition_uncertainty(features)
    paths = save_pipeline_outputs(
        args.output_dir,
        quality=quality,
        features=features,
        failures=failures,
        uncertainty=uncertainty,
    )

    print("\nProcessing status")
    print(quality["processing_status"].value_counts(dropna=False).to_string())
    print(f"\nExtracted profiles: {len(features):,}")
    print(f"Failures: {len(failures):,}")
    print("\nOutputs")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

