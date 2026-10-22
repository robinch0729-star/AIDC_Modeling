"""Analyse extracted NLR features for gray-box calibration and DFL inputs.

This module intentionally stops before fitting a prediction model.  It converts
run-level features into scale-normalised quantities, measures exploratory
metadata sensitivity inside each workload family, and quantifies repeat-run
uncertainty.  The resulting tables are the evidence used to choose the first
gray-box calibration targets and the information later passed to a DFL model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


CONDITION_COLUMNS = [
    "nodes",
    "target_utilization",
    "request_rate",
    "num_prompts",
    "batch_size",
    "max_output_tokens",
    "burstiness",
]

ANALYSIS_FEATURES = [
    "mean_power_w",
    "peak_to_average_ratio",
    "coefficient_of_variation",
    "p95_abs_ramp_pu_per_s",
    "max_ramp_up_pu_per_s",
    "max_ramp_down_pu_per_s",
    "burst_fraction",
    "burst_mean_duration_s",
    "autocorrelation_1s",
    "autocorrelation_10s",
    "dominant_period_s",
]

REPEATABILITY_FEATURES = [
    "mean_power_w",
    "peak_to_average_ratio",
    "coefficient_of_variation",
    "p95_abs_ramp_pu_per_s",
    "max_ramp_up_pu_per_s",
    "max_ramp_down_pu_per_s",
    "burst_fraction",
    "burst_mean_duration_s",
    "dominant_period_s",
]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return finite division results and use NaN for zero/invalid denominators."""

    left = pd.to_numeric(numerator, errors="coerce").astype(float)
    right = pd.to_numeric(denominator, errors="coerce").astype(float)
    result = left / right.where(right.abs() > np.finfo(float).eps)
    return result.replace([np.inf, -np.inf], np.nan)


def enrich_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add scale-normalised ramp, range, burst, and energy consistency features."""

    required = {
        "mean_power_w",
        "max_power_w",
        "min_power_w",
        "duration_s",
        "energy_wh",
        "mean_abs_ramp_w_per_s",
        "p95_abs_ramp_w_per_s",
        "p99_abs_ramp_w_per_s",
        "max_ramp_up_w_per_s",
        "max_ramp_down_w_per_s",
        "burst_total_duration_s",
    }
    missing = sorted(required.difference(features.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    enriched = features.copy()
    mean = enriched["mean_power_w"]
    enriched["power_range_w"] = enriched["max_power_w"] - enriched["min_power_w"]
    enriched["power_range_pu"] = _safe_divide(enriched["power_range_w"], mean)
    enriched["mean_abs_ramp_pu_per_s"] = _safe_divide(
        enriched["mean_abs_ramp_w_per_s"], mean
    )
    enriched["p95_abs_ramp_pu_per_s"] = _safe_divide(
        enriched["p95_abs_ramp_w_per_s"], mean
    )
    enriched["p99_abs_ramp_pu_per_s"] = _safe_divide(
        enriched["p99_abs_ramp_w_per_s"], mean
    )
    enriched["max_ramp_up_pu_per_s"] = _safe_divide(
        enriched["max_ramp_up_w_per_s"], mean
    )
    enriched["max_ramp_down_pu_per_s"] = _safe_divide(
        enriched["max_ramp_down_w_per_s"], mean
    )
    enriched["burst_fraction"] = _safe_divide(
        enriched["burst_total_duration_s"], enriched["duration_s"]
    ).clip(lower=0.0, upper=1.0)
    enriched["energy_implied_mean_power_w"] = _safe_divide(
        enriched["energy_wh"] * 3600.0, enriched["duration_s"]
    )
    enriched["energy_mean_relative_error"] = _safe_divide(
        (enriched["energy_implied_mean_power_w"] - mean).abs(), mean.abs()
    )
    return enriched


def build_workload_feature_summary(features: pd.DataFrame) -> pd.DataFrame:
    """Summarise central tendency and spread for each workload family."""

    rows: list[dict[str, Any]] = []
    for workload_type, group in features.groupby("workload_type", observed=True):
        for feature in ANALYSIS_FEATURES:
            if feature not in group:
                continue
            values = pd.to_numeric(group[feature], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            rows.append(
                {
                    "workload_type": workload_type,
                    "feature": feature,
                    "profile_count": len(group),
                    "valid_count": len(values),
                    "median": float(values.median()) if len(values) else np.nan,
                    "p05": float(values.quantile(0.05)) if len(values) else np.nan,
                    "p95": float(values.quantile(0.95)) if len(values) else np.nan,
                    "minimum": float(values.min()) if len(values) else np.nan,
                    "maximum": float(values.max()) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_condition_coverage(features: pd.DataFrame) -> pd.DataFrame:
    """Report which physical/workload condition variables exist per workload."""

    rows: list[dict[str, Any]] = []
    for workload_type, group in features.groupby("workload_type", observed=True):
        for condition in CONDITION_COLUMNS:
            values = pd.to_numeric(group[condition], errors="coerce").dropna()
            rows.append(
                {
                    "workload_type": workload_type,
                    "condition_variable": condition,
                    "profile_count": len(group),
                    "non_null_count": len(values),
                    "unique_count": int(values.nunique()),
                    "minimum": float(values.min()) if len(values) else np.nan,
                    "maximum": float(values.max()) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_metadata_sensitivity(
    features: pd.DataFrame,
    *,
    minimum_observations: int = 4,
    minimum_unique_inputs: int = 3,
) -> pd.DataFrame:
    """Compute exploratory within-workload Spearman correlations.

    These correlations identify useful conditional inputs; they are not causal
    estimates because some NLR condition variables are changed together.
    """

    rows: list[dict[str, Any]] = []
    for workload_type, group in features.groupby("workload_type", observed=True):
        for condition in CONDITION_COLUMNS:
            x_all = pd.to_numeric(group[condition], errors="coerce")
            for feature in ANALYSIS_FEATURES:
                y_all = pd.to_numeric(group[feature], errors="coerce")
                valid = x_all.notna() & y_all.notna() & np.isfinite(x_all) & np.isfinite(y_all)
                x = x_all.loc[valid]
                y = y_all.loc[valid]
                if len(x) < minimum_observations or x.nunique() < minimum_unique_inputs:
                    continue
                if y.nunique() < 2:
                    rho, p_value = np.nan, np.nan
                else:
                    result = spearmanr(x.to_numpy(), y.to_numpy(), nan_policy="omit")
                    rho, p_value = float(result.statistic), float(result.pvalue)
                rows.append(
                    {
                        "workload_type": workload_type,
                        "condition_variable": condition,
                        "feature": feature,
                        "observation_count": len(x),
                        "unique_input_count": int(x.nunique()),
                        "spearman_rho": rho,
                        "absolute_spearman_rho": abs(rho) if np.isfinite(rho) else np.nan,
                        "p_value": p_value,
                        "interpretation": "exploratory/non-causal",
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["workload_type", "absolute_spearman_rho"],
        ascending=[True, False],
        kind="stable",
    ).reset_index(drop=True)


def build_condition_information(features: pd.DataFrame) -> pd.DataFrame:
    """Build condition-level empirical targets and uncertainty intervals."""

    metadata_columns = [
        "workload_type",
        "condition_id",
        "model_name",
        "source_kind",
        "measurement_scope",
        "measurement_origin",
        *CONDITION_COLUMNS,
    ]
    target_features = [
        "mean_power_w",
        "peak_to_average_ratio",
        "coefficient_of_variation",
        "p95_abs_ramp_pu_per_s",
        "max_ramp_up_pu_per_s",
        "max_ramp_down_pu_per_s",
        "burst_fraction",
        "burst_mean_duration_s",
        "dominant_period_s",
    ]
    rows: list[dict[str, Any]] = []
    for (workload_type, condition_id), group in features.groupby(
        ["workload_type", "condition_id"], dropna=False, observed=True
    ):
        row: dict[str, Any] = {
            "workload_type": workload_type,
            "condition_id": condition_id,
            "run_count": len(group),
        }
        first = group.iloc[0]
        for column in metadata_columns[2:]:
            row[column] = first.get(column)
        for feature in target_features:
            values = pd.to_numeric(group[feature], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            row[f"{feature}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{feature}_p05"] = float(values.quantile(0.05)) if len(values) else np.nan
            row[f"{feature}_p50"] = float(values.quantile(0.50)) if len(values) else np.nan
            row[f"{feature}_p95"] = float(values.quantile(0.95)) if len(values) else np.nan
            row[f"{feature}_p90_interval_width"] = (
                float(values.quantile(0.95) - values.quantile(0.05))
                if len(values)
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["workload_type", "condition_id"], kind="stable"
    ).reset_index(drop=True)


def build_repeatability_summary(features: pd.DataFrame) -> pd.DataFrame:
    """Quantify within-condition run-to-run variation for repeated conditions."""

    condition_rows: list[dict[str, Any]] = []
    for (workload_type, condition_id), group in features.groupby(
        ["workload_type", "condition_id"], dropna=False, observed=True
    ):
        if len(group) < 2:
            continue
        for feature in REPEATABILITY_FEATURES:
            values = pd.to_numeric(group[feature], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            if len(values) < 2:
                continue
            mean = float(values.mean())
            standard_deviation = float(values.std(ddof=1))
            denominator = abs(mean)
            condition_rows.append(
                {
                    "workload_type": workload_type,
                    "condition_id": condition_id,
                    "feature": feature,
                    "run_count": len(values),
                    "condition_mean": mean,
                    "condition_std": standard_deviation,
                    "condition_cv": (
                        standard_deviation / denominator
                        if denominator > np.finfo(float).eps
                        else np.nan
                    ),
                    "condition_p05": float(values.quantile(0.05)),
                    "condition_p95": float(values.quantile(0.95)),
                }
            )

    condition_table = pd.DataFrame(condition_rows)
    if condition_table.empty:
        return condition_table
    rows: list[dict[str, Any]] = []
    for (workload_type, feature), group in condition_table.groupby(
        ["workload_type", "feature"], observed=True
    ):
        cv = group["condition_cv"].dropna()
        widths = group["condition_p95"] - group["condition_p05"]
        rows.append(
            {
                "workload_type": workload_type,
                "feature": feature,
                "repeated_condition_count": group["condition_id"].nunique(),
                "total_run_count": int(group["run_count"].sum()),
                "median_within_condition_cv": float(cv.median()) if len(cv) else np.nan,
                "p90_within_condition_cv": float(cv.quantile(0.90)) if len(cv) else np.nan,
                "median_p90_interval_width": float(widths.median()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["workload_type", "feature"], kind="stable"
    ).reset_index(drop=True)


def graybox_target_candidates() -> pd.DataFrame:
    """Return a research-facing candidate registry, not an automatic selection."""

    rows = [
        ("mean_power_w", "level", "primary", "Workload-conditioned IT base power", "Calibrate the static workload-to-power map", "NLR IT-level values must be scaled/mapped before PCC use"),
        ("coefficient_of_variation", "variability", "primary", "Relative short-term power variation", "Calibrate stochastic fluctuation amplitude", "Dimensionless but analysis-window dependent"),
        ("p95_abs_ramp_pu_per_s", "ramp", "primary", "Frequent high dP/dt relative to mean power", "Calibrate normal high-ramp behaviour", "Sensitive to sampling/resampling interval"),
        ("max_ramp_up_pu_per_s", "ramp", "primary", "Extreme upward workload ramp", "Set an upper-ramp parameter or scenario", "Maximum is sample-size sensitive"),
        ("max_ramp_down_pu_per_s", "ramp", "primary", "Extreme downward workload ramp", "Set a lower-ramp parameter or scenario", "Maximum is sample-size sensitive"),
        ("burst_fraction", "burst", "primary", "Fraction of time above mean plus one standard deviation", "Calibrate burst duty ratio", "Threshold is profile-relative"),
        ("burst_mean_duration_s", "burst", "secondary", "Typical contiguous high-load duration", "Calibrate burst duration/state dwell time", "Zero when no burst is detected"),
        ("autocorrelation_1s", "temporal", "secondary", "One-second workload persistence", "Calibrate coloured-noise or state persistence", "Unavailable for very short profiles"),
        ("dominant_period_s", "temporal", "secondary", "Dominant non-DC oscillation period", "Candidate periodic component", "Periodogram peak need not imply a physical cycle"),
        ("peak_to_average_ratio", "level", "secondary", "Peak headroom relative to average", "Validate or constrain peak scaling", "Does not preserve temporal ordering"),
        ("energy_wh", "energy", "validation_only", "Integrated energy over each run", "Check model energy consistency", "Not comparable across unequal durations without normalisation"),
    ]
    columns = [
        "feature",
        "role",
        "priority",
        "physical_interpretation",
        "graybox_use",
        "caveat",
    ]
    return pd.DataFrame(rows, columns=columns)


def dfl_information_candidates() -> pd.DataFrame:
    """Describe candidate AIDC-to-DSO information inferred from NLR/gray-box data."""

    rows = [
        ("conditional_expected_it_power", "mean_power_w_mean", "point estimate", "Expected IT demand for the declared workload condition"),
        ("conditional_power_interval", "mean_power_w_p05/p95", "uncertainty", "Empirical repeat-run interval for IT power"),
        ("conditional_ramp_interval", "max_ramp_up/down_pu_per_s_p05/p95", "uncertainty", "Directional workload ramp uncertainty"),
        ("variability_and_burst", "coefficient_of_variation; burst_fraction", "dynamic descriptor", "Short-term fluctuation and burst occupancy"),
        ("workload_condition", "nodes/utilization/request_rate/batch_size/tokens", "explanatory input", "Declared workload state used to condition estimates"),
        ("data_provenance", "measurement_scope; measurement_origin", "quality flag", "Separates measured IT data from simulated whole-facility traces"),
    ]
    table = pd.DataFrame(
        rows,
        columns=["information_item", "source_fields", "information_role", "meaning"],
    )
    table["system_boundary_note"] = (
        "NLR values are not automatically PCC demand; a gray-box IT/cooling/UPS/TES "
        "mapping and uncertainty propagation step is required before DSO/DOE use."
    )
    return table


def save_analysis_outputs(
    output_dir: str | Path,
    *,
    enriched: pd.DataFrame,
    workload_summary: pd.DataFrame,
    condition_coverage: pd.DataFrame,
    sensitivity: pd.DataFrame,
    condition_information: pd.DataFrame,
    repeatability: pd.DataFrame,
) -> dict[str, Path]:
    """Save analysis tables in CSV and Parquet where useful."""

    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    tables = {
        "analysis_ready_features": enriched,
        "workload_feature_summary": workload_summary,
        "condition_coverage": condition_coverage,
        "metadata_sensitivity": sensitivity,
        "condition_information": condition_information,
        "repeatability_summary": repeatability,
        "graybox_target_candidates": graybox_target_candidates(),
        "dfl_information_candidates": dfl_information_candidates(),
    }
    paths: dict[str, Path] = {}
    for stem, table in tables.items():
        csv_path = directory / f"{stem}.csv"
        table.to_csv(csv_path, index=False, encoding="utf-8-sig")
        paths[f"{stem}_csv"] = csv_path
        if not table.empty and stem not in {
            "graybox_target_candidates",
            "dfl_information_candidates",
        }:
            parquet_path = directory / f"{stem}.parquet"
            table.to_parquet(parquet_path, index=False)
            paths[f"{stem}_parquet"] = parquet_path
    return paths


def run_feature_analysis(input_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Run the complete post-extraction feature analysis."""

    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Feature table not found: {source}")
    features = pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_csv(source)
    enriched = enrich_features(features)
    workload_summary = build_workload_feature_summary(enriched)
    condition_coverage = build_condition_coverage(enriched)
    sensitivity = build_metadata_sensitivity(enriched)
    condition_information = build_condition_information(enriched)
    repeatability = build_repeatability_summary(enriched)
    paths = save_analysis_outputs(
        output_dir,
        enriched=enriched,
        workload_summary=workload_summary,
        condition_coverage=condition_coverage,
        sensitivity=sensitivity,
        condition_information=condition_information,
        repeatability=repeatability,
    )
    return {
        "features": enriched,
        "workload_summary": workload_summary,
        "condition_coverage": condition_coverage,
        "sensitivity": sensitivity,
        "condition_information": condition_information,
        "repeatability": repeatability,
        "paths": paths,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/signal_validation/workload_features.parquet"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/feature_analysis")
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_feature_analysis(args.input, args.output_dir)
    print(f"Analysed profiles: {len(result['features']):,}")
    print(f"Condition-level rows: {len(result['condition_information']):,}")
    print(f"Sensitivity rows: {len(result['sensitivity']):,}")
    print(f"Repeatability rows: {len(result['repeatability']):,}")
    print("\nOutputs")
    for name, path in result["paths"].items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
