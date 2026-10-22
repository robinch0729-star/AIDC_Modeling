"""Build leakage-safe condition targets for NLR gray-box calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


WORKLOAD_TYPE = "online_inference_rate"


def _read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Table not found: {source}")
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    return pd.read_csv(source)


def _as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def build_online_rate_targets(
    feature_path: str | Path,
    audited_manifest_path: str | Path,
) -> pd.DataFrame:
    """Return one measured IT-power target per prompt-dataset/request-rate case.

    ``online_inference_rate`` has no run-to-run repeats in the local NLR release.
    Consequently ``within_profile_std_kw`` describes temporal fluctuation inside
    one three-minute trace; it is not an empirical condition uncertainty.
    """

    features = _read_table(feature_path)
    audit = _read_table(audited_manifest_path)

    feature_required = {
        "profile_id",
        "relative_path",
        "workload_type",
        "measurement_scope",
        "measurement_origin",
        "request_rate",
        "mean_power_w",
        "std_power_w",
        "duration_s",
        "p05_power_w",
        "p95_power_w",
    }
    audit_required = {
        "relative_path",
        "dataset_path",
        "recommended_condition_id",
        "analysis_cohort",
        "eligible",
    }
    missing_features = sorted(feature_required.difference(features.columns))
    missing_audit = sorted(audit_required.difference(audit.columns))
    if missing_features:
        raise ValueError(f"Feature table is missing columns: {missing_features}")
    if missing_audit:
        raise ValueError(f"Audited manifest is missing columns: {missing_audit}")

    audit_subset = audit[
        [
            "relative_path",
            "dataset_path",
            "recommended_condition_id",
            "analysis_cohort",
            "eligible",
        ]
    ].copy()
    if audit_subset["relative_path"].duplicated().any():
        raise ValueError("Audited manifest contains duplicate relative paths")

    merged = features.merge(
        audit_subset,
        on="relative_path",
        how="left",
        validate="one_to_one",
    )
    selected = merged.loc[
        merged["workload_type"].eq(WORKLOAD_TYPE)
        & merged["analysis_cohort"].eq("measured_it_workload")
        & _as_bool(merged["eligible"])
    ].copy()
    if selected.empty:
        raise ValueError("No eligible measured online-inference-rate profiles found")

    numeric_columns = [
        "request_rate",
        "mean_power_w",
        "std_power_w",
        "duration_s",
        "p05_power_w",
        "p95_power_w",
    ]
    for column in numeric_columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    if selected[["request_rate", "mean_power_w"]].isna().any().any():
        raise ValueError("Selected targets contain invalid request rate or mean power")
    if (selected["request_rate"] < 0).any() or (selected["mean_power_w"] <= 0).any():
        raise ValueError("Request rate must be nonnegative and target power positive")
    if selected["dataset_path"].isna().any():
        raise ValueError("Prompt dataset is missing from one or more audited cases")

    selected["prompt_dataset"] = selected["dataset_path"].astype(str).str.rsplit(
        "/", n=1
    ).str[-1]
    selected["target_power_kw"] = selected["mean_power_w"] / 1000.0
    selected["within_profile_std_kw"] = selected["std_power_w"] / 1000.0
    selected["within_profile_p05_kw"] = selected["p05_power_w"] / 1000.0
    selected["within_profile_p95_kw"] = selected["p95_power_w"] / 1000.0
    selected["uncertainty_interpretation"] = (
        "within-profile temporal variation; not run-to-run condition uncertainty"
    )

    duplicate = selected.duplicated(["dataset_path", "request_rate"], keep=False)
    if duplicate.any():
        cases = selected.loc[duplicate, ["dataset_path", "request_rate"]]
        raise ValueError(
            "Expected one trace per prompt-dataset/request-rate case; duplicates: "
            f"{cases.drop_duplicates().to_dict(orient='records')[:5]}"
        )

    columns = [
        "profile_id",
        "relative_path",
        "recommended_condition_id",
        "dataset_path",
        "prompt_dataset",
        "request_rate",
        "target_power_kw",
        "within_profile_std_kw",
        "within_profile_p05_kw",
        "within_profile_p95_kw",
        "duration_s",
        "measurement_scope",
        "measurement_origin",
        "uncertainty_interpretation",
    ]
    result = selected[columns].sort_values(
        ["dataset_path", "request_rate"], kind="stable"
    ).reset_index(drop=True)
    if not np.isfinite(result[["request_rate", "target_power_kw"]]).all().all():
        raise ValueError("Targets contain non-finite numeric values")
    return result

