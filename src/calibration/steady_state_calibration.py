"""Calibrate and compare request-rate saturation models on NLR data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from evaluation.steady_state_evaluation import regression_metrics, shape_diagnostics
from modeling.request_rate_splits import attach_split, build_request_rate_splits
from models.steady_state_models import (
    MODEL_SPECS,
    derived_parameters,
    get_model_spec,
    model_registry_table,
)
from nlr_data.condition_targets import build_online_rate_targets


@dataclass(frozen=True)
class FitResult:
    model_name: str
    parameters: np.ndarray
    success: bool
    status: int
    message: str
    cost: float
    optimality: float
    function_evaluations: int


def fit_steady_state_model(
    model_name: str,
    request_rate: np.ndarray,
    target_power_kw: np.ndarray,
) -> FitResult:
    """Fit one physically bounded saturation model by nonlinear least squares."""

    spec = get_model_spec(model_name)
    x = np.asarray(request_rate, dtype=float)
    y = np.asarray(target_power_kw, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(y) <= len(spec.parameter_names):
        raise ValueError(f"Insufficient samples to fit {model_name}")
    initial = spec.initializer(x, y)
    lower, upper = spec.bounder(x, y)
    initial = np.clip(initial, lower + 1e-10, upper - 1e-10)
    residual_scale = max(float(np.std(y)), float(np.mean(np.abs(y))) * 1e-3, 1e-6)

    result = least_squares(
        lambda parameters: (spec.predict(x, parameters) - y) / residual_scale,
        x0=initial,
        bounds=(lower, upper),
        method="trf",
        max_nfev=20000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    return FitResult(
        model_name=model_name,
        parameters=result.x,
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        cost=float(result.cost),
        optimality=float(result.optimality),
        function_evaluations=int(result.nfev),
    )


def _scope_frames(targets: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    frames = [("pooled", targets.copy())]
    for dataset, group in targets.groupby("prompt_dataset", observed=True):
        frames.append((f"dataset:{dataset}", group.copy()))
    return frames


def calibrate_candidate_models(
    targets: pd.DataFrame,
    assignments: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Fit every candidate for pooled and prompt-dataset-specific scopes."""

    parameter_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    schemes = ["full_fit", "interpolation", "high_rate_extrapolation"]
    for scope, scope_frame in _scope_frames(targets):
        for scheme in schemes:
            split = attach_split(scope_frame, assignments, scheme)
            training = split.loc[split["partition"].eq("train")]
            for model_name, spec in MODEL_SPECS.items():
                fit = fit_steady_state_model(
                    model_name,
                    training["request_rate"].to_numpy(),
                    training["target_power_kw"].to_numpy(),
                )
                fit_id = f"{scope}|{scheme}|{model_name}"
                prediction = spec.predict(
                    split["request_rate"].to_numpy(), fit.parameters
                )
                residual = prediction - split["target_power_kw"].to_numpy()
                for row_index, (_, row) in enumerate(split.iterrows()):
                    prediction_rows.append(
                        {
                            "fit_id": fit_id,
                            "scope": scope,
                            "split_scheme": scheme,
                            "model_name": model_name,
                            "partition": row["partition"],
                            "profile_id": row["profile_id"],
                            "prompt_dataset": row["prompt_dataset"],
                            "request_rate": row["request_rate"],
                            "observed_power_kw": row["target_power_kw"],
                            "predicted_power_kw": prediction[row_index],
                            "residual_kw": residual[row_index],
                        }
                    )

                raw_parameters = dict(zip(spec.parameter_names, fit.parameters, strict=True))
                derived = derived_parameters(model_name, fit.parameters)
                for name, value in {**raw_parameters, **derived}.items():
                    parameter_rows.append(
                        {
                            "fit_id": fit_id,
                            "scope": scope,
                            "split_scheme": scheme,
                            "model_name": model_name,
                            "parameter": name,
                            "value": float(value),
                            "fit_success": fit.success,
                            "fit_status": fit.status,
                            "fit_message": fit.message,
                            "cost": fit.cost,
                            "optimality": fit.optimality,
                            "function_evaluations": fit.function_evaluations,
                        }
                    )

                partitions = ["train", "all"]
                if scheme != "full_fit":
                    partitions.insert(1, "test")
                for partition in partitions:
                    mask = (
                        np.ones(len(split), dtype=bool)
                        if partition == "all"
                        else split["partition"].eq(partition).to_numpy()
                    )
                    metrics = regression_metrics(
                        split.loc[mask, "target_power_kw"].to_numpy(),
                        prediction[mask],
                        parameter_count=len(spec.parameter_names),
                    )
                    diagnostics = shape_diagnostics(
                        split.loc[mask, "request_rate"].to_numpy(),
                        split.loc[mask, "target_power_kw"].to_numpy(),
                        prediction[mask],
                    )
                    metric_rows.append(
                        {
                            "fit_id": fit_id,
                            "scope": scope,
                            "split_scheme": scheme,
                            "model_name": model_name,
                            "evaluation_partition": partition,
                            "fit_success": fit.success,
                            **metrics,
                            **diagnostics,
                        }
                    )

    parameters = pd.DataFrame(parameter_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)
    holdout = metrics.loc[
        metrics["evaluation_partition"].eq("test") & metrics["fit_success"]
    ].copy()
    best = (
        holdout.sort_values(
            ["scope", "split_scheme", "rmse_kw", "mae_kw", "model_name"],
            kind="stable",
        )
        .groupby(["scope", "split_scheme"], observed=True, as_index=False)
        .first()
    )
    return {
        "parameters": parameters,
        "predictions": predictions,
        "metrics": metrics,
        "best_models": best,
    }


def _save_table(table: pd.DataFrame, output_dir: Path, stem: str) -> dict[str, Path]:
    csv_path = output_dir / f"{stem}.csv"
    parquet_path = output_dir / f"{stem}.parquet"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    table.to_parquet(parquet_path, index=False)
    return {f"{stem}_csv": csv_path, f"{stem}_parquet": parquet_path}


def run_steady_state_calibration(
    feature_path: str | Path,
    audited_manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run target construction, grouped splits, calibration, and evaluation."""

    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    targets = build_online_rate_targets(feature_path, audited_manifest_path)
    assignments = build_request_rate_splits(targets)
    fitted = calibrate_candidate_models(targets, assignments)
    tables = {
        "online_rate_condition_targets": targets,
        "request_rate_split_assignments": assignments,
        "steady_state_model_registry": model_registry_table(),
        "steady_state_parameters": fitted["parameters"],
        "steady_state_predictions": fitted["predictions"],
        "steady_state_metrics": fitted["metrics"],
        "best_model_by_holdout": fitted["best_models"],
    }
    paths: dict[str, Path] = {}
    for stem, table in tables.items():
        paths.update(_save_table(table, directory, stem))
    return {"tables": tables, "paths": paths}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("outputs/feature_analysis/analysis_ready_features.parquet"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("outputs/condition_audit/audited_run_manifest.parquet"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/graybox_steady_state")
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_steady_state_calibration(args.features, args.audit, args.output_dir)
    tables = result["tables"]
    print(f"Condition targets: {len(tables['online_rate_condition_targets']):,}")
    print(f"Calibrated fits: {tables['steady_state_parameters']['fit_id'].nunique():,}")
    print("\nBest held-out models")
    print(
        tables["best_model_by_holdout"][
            ["scope", "split_scheme", "model_name", "rmse_kw", "mape_pct", "r2"]
        ].to_string(index=False)
    )
    print("\nOutputs")
    for name, path in result["paths"].items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

