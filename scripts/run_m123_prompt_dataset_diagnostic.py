"""Prompt-dataset-conditioned diagnostic for the M1/M2/M3 equation families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlr_data.steady_state_models import (  # noqa: E402
    build_condition_table,
    choose_model,
    fit_model,
    read_table,
    split_conditions,
)


MODEL_NAMES = ("M1_linear", "M2_cubic", "M3_logistic")


def evaluate_grouped(
    fitted: dict[tuple[str, str], object],
    evaluation: pd.DataFrame,
    fitting: pd.DataFrame,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    prediction_rows = []
    for model_name in MODEL_NAMES:
        observed_parts = []
        predicted_parts = []
        monotonic_parts = []
        range_parts = []
        for dataset, part in evaluation.groupby("prompt_dataset", observed=True):
            model = fitted[(model_name, str(dataset))]
            prediction = model.predict(part["request_rate"])
            observed = part["power"].to_numpy(float)
            observed_parts.append(observed)
            predicted_parts.append(prediction)
            for index, (_, row) in enumerate(part.iterrows()):
                prediction_rows.append(
                    {
                        "model": model_name,
                        "split": split,
                        "condition_id": row["condition_id"],
                        "prompt_dataset": dataset,
                        "request_rate": row["request_rate"],
                        "observed_power_kw": row["power"],
                        "predicted_power_kw": prediction[index],
                        "residual_kw": prediction[index] - row["power"],
                    }
                )

            group_fit = fitting.loc[fitting["prompt_dataset"].eq(dataset)]
            grid_max = max(model.x_max, float(part["request_rate"].max()))
            grid = np.linspace(model.x_min, grid_max, 500)
            grid_prediction = model.predict(grid)
            scale = max(float(np.ptp(grid_prediction)), float(np.mean(np.abs(observed))), 1e-9)
            monotonic_parts.append(float(np.mean(np.diff(grid_prediction) < -1e-8 * scale)))
            upper = max(float(group_fit["power"].max()) * 1.10, 1e-9)
            range_parts.append(float(np.mean((grid_prediction < 0) | (grid_prediction > upper))))

        y = np.concatenate(observed_parts)
        y_hat = np.concatenate(predicted_parts)
        residual = y_hat - y
        denominator = float(np.sum(np.square(y - np.mean(y))))
        metric_rows.append(
            {
                "model": model_name,
                "split": split,
                "n": len(y),
                "rmse": float(np.sqrt(np.mean(np.square(residual)))),
                "nrmse_mean": float(np.sqrt(np.mean(np.square(residual))) / np.mean(np.abs(y))),
                "mae": float(np.mean(np.abs(residual))),
                "mape_pct": float(np.mean(np.abs(residual) / np.maximum(np.abs(y), 1e-9)) * 100),
                "r2": float(1 - np.sum(np.square(residual)) / denominator) if denominator > 0 else np.nan,
                "monotonic_violation_fraction": float(np.mean(monotonic_parts)),
                "physical_range_violation_fraction": float(np.mean(range_parts)),
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def fit_by_dataset(data: pd.DataFrame) -> dict[tuple[str, str], object]:
    fitted = {}
    for dataset, group in data.groupby("prompt_dataset", observed=True):
        for model_name in MODEL_NAMES:
            fitted[(model_name, str(dataset))] = fit_model(model_name, group)
    return fitted


def parameter_table(fitted: dict[tuple[str, str], object]) -> pd.DataFrame:
    rows = []
    for (model_name, dataset), model in fitted.items():
        for parameter, value in model.parameters.items():
            rows.append(
                {
                    "model": model_name,
                    "prompt_dataset": dataset,
                    "parameter": parameter,
                    "value": value,
                    "fit_rate_min": model.x_min,
                    "fit_rate_max": model.x_max,
                }
            )
    return pd.DataFrame(rows)


def plot_grouped(
    condition: pd.DataFrame,
    fitted: dict[tuple[str, str], object],
    selected: str,
    output: Path,
) -> None:
    datasets = list(condition["prompt_dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), dpi=150, sharey=True)
    axes = np.atleast_1d(axes)
    colors = {"M1_linear": "#111827", "M2_cubic": "#7c3aed", "M3_logistic": "#059669"}
    for ax, dataset in zip(axes, datasets, strict=True):
        part = condition.loc[condition["prompt_dataset"].eq(dataset)]
        ax.scatter(part["request_rate"], part["power"], s=18, color="#4b9cd3", alpha=0.75, label="measured")
        grid = np.linspace(part["request_rate"].min(), part["request_rate"].max(), 600)
        for model_name in MODEL_NAMES:
            model = fitted[(model_name, str(dataset))]
            ax.plot(
                grid,
                model.predict(grid),
                color=colors[model_name],
                lw=3 if model_name == selected else 1.7,
                label=f"{model_name}{' (selected)' if model_name == selected else ''}",
            )
        ax.set_title(str(dataset))
        ax.set_xlabel("Request rate (request/s)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Condition-mean IT power (kW)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("M1/M2/M3 with prompt-dataset-conditioned parameters")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "graybox_steady_state" / "online_rate_condition_targets.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "steady_state_model_comparison",
    )
    args = parser.parse_args()
    condition, _ = build_condition_table(read_table(args.input))
    if "prompt_dataset" not in condition:
        raise ValueError("prompt_dataset is required for this diagnostic")
    train, validation, test, _ = split_conditions(condition)

    train_fitted = fit_by_dataset(train)
    validation_metrics, validation_predictions = evaluate_grouped(
        train_fitted, validation, train, "validation"
    )
    selected, reason = choose_model(validation_metrics)

    development = pd.concat([train, validation], ignore_index=True)
    final_fitted = fit_by_dataset(development)
    test_metrics, test_predictions = evaluate_grouped(final_fitted, test, development, "test")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(exist_ok=True)
    validation_metrics.sort_values("rmse").to_csv(
        args.output_dir / "validation_metrics_prompt_conditioned.csv", index=False
    )
    test_metrics.sort_values("rmse").to_csv(
        args.output_dir / "test_metrics_prompt_conditioned.csv", index=False
    )
    pd.concat([validation_predictions, test_predictions], ignore_index=True).to_csv(
        args.output_dir / "predictions_prompt_conditioned.csv", index=False
    )
    parameter_table(final_fitted).to_csv(
        args.output_dir / "model_parameters_prompt_conditioned.csv", index=False
    )
    summary = {
        "selected_model_family": selected,
        "selection_basis": reason,
        "interpretation": (
            "The equation family is shared, while its parameters are conditioned on prompt_dataset."
        ),
    }
    (args.output_dir / "selected_model_prompt_conditioned.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_grouped(
        condition,
        final_fitted,
        selected,
        args.output_dir / "figures" / "model_comparison_prompt_conditioned.png",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nValidation")
    print(validation_metrics.sort_values("rmse").to_string(index=False))
    print("\nHigh-rate test")
    print(test_metrics.sort_values("rmse").to_string(index=False))


if __name__ == "__main__":
    main()
