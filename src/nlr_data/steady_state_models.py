"""Compare steady-state IT power equations for the NLR online-inference data.

This module deliberately separates the steady-state mapping ``request rate ->
mean IT power`` from any later dynamic lag/noise model.  Conditions, rather
than individual time samples, are split so that repeated measurements from the
same NLR experiment cannot leak across train/validation/test sets.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


COLUMN_ALIASES = {
    "condition_id": (
        "condition_id", "recommended_condition_id", "condition", "condition_key",
        "experiment_condition_id"
    ),
    "request_rate": (
        "request_rate", "request_rate_rps", "requests_per_second", "request_rps",
        "arrival_rate", "rate", "req_rate"
    ),
    "power": (
        "mean_it_power_w", "mean_power_w", "aggregated_it_power_w",
        "aggregate_it_power_w", "it_power_mean_w", "power_mean_w",
        "mean_power", "it_power_w", "power_w", "power"
        , "target_power_kw"
    ),
    "workload_type": (
        "workload_type", "workload_category", "experiment_type", "workload"
    ),
    "prompt_dataset": (
        "prompt_dataset", "prompt_data", "dataset", "dataset_name"
    ),
    "max_output_tokens": (
        "max_output_tokens", "maximum_output_tokens", "max_tokens", "output_tokens"
    ),
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def resolve_columns(frame: pd.DataFrame) -> dict[str, str]:
    normalized = {_norm(c): c for c in frame.columns}
    resolved: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if _norm(alias) in normalized:
                resolved[canonical] = normalized[_norm(alias)]
                break
    # Conservative fallbacks for previously generated feature tables whose
    # column names include units or aggregation suffixes.
    names = list(normalized)
    if "condition_id" not in resolved:
        hit = next((n for n in names if "condition" in n and ("id" in n or "key" in n)), None)
        if hit:
            resolved["condition_id"] = normalized[hit]
    if "request_rate" not in resolved:
        hit = next((n for n in names if "request" in n and "rate" in n), None)
        if hit:
            resolved["request_rate"] = normalized[hit]
    if "power" not in resolved:
        hit = next(
            (
                n for n in names
                if "power" in n
                and "ramp" not in n
                and any(token in n for token in ("mean", "avg", "average", "aggregated", "it"))
            ),
            None,
        )
        if hit:
            resolved["power"] = normalized[hit]
    missing = {"condition_id", "request_rate", "power"} - resolved.keys()
    if missing:
        raise ValueError(
            "Required columns were not found: "
            f"{sorted(missing)}. Available columns: {list(frame.columns)}"
        )
    return resolved


def discover_feature_table(project_root: Path) -> Path:
    """Find a derived feature/condition table without scanning the raw NLR tree."""
    roots = [
        project_root / "outputs",
        project_root / "artifacts",
        project_root / "reports",
        project_root / "data" / "processed",
    ]
    candidates: list[tuple[int, Path]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in list(root.rglob("*.csv")) + list(root.rglob("*.parquet")):
            try:
                head = (
                    pd.read_csv(path, nrows=5)
                    if path.suffix.lower() == ".csv"
                    else pd.read_parquet(path).head(5)
                )
                resolved = resolve_columns(head)
            except Exception:
                continue
            score = 10
            lower = path.name.lower()
            score += 4 * ("condition" in lower)
            score += 3 * ("feature" in lower)
            score += 2 * ("online" in lower)
            score += int("workload_type" in resolved)
            score += int("prompt_dataset" in resolved)
            score += int("max_output_tokens" in resolved)
            candidates.append((score, path))
    if not candidates:
        raise FileNotFoundError(
            "No derived condition/feature table was detected. Pass its path with "
            "--input. The table must contain condition_id, request_rate, and mean "
            "or aggregated IT power columns."
        )
    return sorted(candidates, key=lambda item: (-item[0], str(item[1])))[0][1]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_condition_table(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return one steady-state observation per NLR condition_id."""
    cols = resolve_columns(frame)
    data = frame.copy()

    if "workload_type" in cols:
        text = data[cols["workload_type"]].astype(str).str.lower()
        mask = text.str.contains("online", na=False) & text.str.contains("inference", na=False)
        rate_mask = text.str.contains("rate", na=False)
        if (mask & rate_mask).any():
            data = data.loc[mask & rate_mask].copy()
        elif mask.any():
            data = data.loc[mask].copy()

    for key in ("request_rate", "power"):
        data[cols[key]] = pd.to_numeric(data[cols[key]], errors="coerce")
    data = data.dropna(subset=[cols["condition_id"], cols["request_rate"], cols["power"]])
    data = data.loc[(data[cols["request_rate"]] >= 0) & (data[cols["power"]] >= 0)].copy()

    agg: dict[str, Any] = {
        cols["request_rate"]: "median",
        cols["power"]: "mean",
    }
    for key in ("workload_type", "prompt_dataset", "max_output_tokens"):
        if key in cols:
            agg[cols[key]] = "first"

    condition = data.groupby(cols["condition_id"], as_index=False).agg(agg)
    rename = {
        cols["condition_id"]: "condition_id",
        cols["request_rate"]: "request_rate",
        cols["power"]: "power",
    }
    for key in ("workload_type", "prompt_dataset", "max_output_tokens"):
        if key in cols:
            rename[cols[key]] = key
    condition = condition.rename(columns=rename).sort_values(
        ["request_rate", "condition_id"]
    ).reset_index(drop=True)

    if len(condition) < 18 or condition["request_rate"].nunique() < 6:
        raise ValueError(
            "At least 18 conditions and 6 distinct request-rate levels are required "
            f"for a stable three-way comparison; got {len(condition)} conditions "
            f"and {condition['request_rate'].nunique()} rate levels."
        )
    return condition, cols


def split_conditions(
    condition: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Leakage-safe split by request rate, not by individual profile.

    The highest 20% of request rates form an extrapolation-only test set.  In
    the remaining range every fifth rate is used for interpolation validation.
    Both prompt datasets at one request rate therefore always share a split.
    ``seed`` is retained in the public API for reproducibility metadata; this
    deterministic split intentionally does not use randomness.
    """
    del seed
    rates = np.sort(condition["request_rate"].dropna().unique())
    if len(rates) < 10:
        raise ValueError("At least ten distinct request rates are required")
    test_count = max(1, int(np.ceil(0.20 * len(rates))))
    test_rates = set(rates[-test_count:])
    development_rates = rates[:-test_count]
    validation_rates = {
        rate
        for index, rate in enumerate(development_rates)
        if (index + 1) % 5 == 0 and index not in {0, len(development_rates) - 1}
    }

    assignments = condition[["condition_id", "request_rate"]].copy()
    assignments["split"] = "train"
    assignments.loc[assignments["request_rate"].isin(validation_rates), "split"] = "validation"
    assignments.loc[assignments["request_rate"].isin(test_rates), "split"] = "test"
    assignments["split_design"] = "interpolation-validation + high-rate-extrapolation-test"

    def take(part: str) -> pd.DataFrame:
        selected = set(assignments.loc[assignments["split"] == part, "condition_id"])
        return condition.loc[condition["condition_id"].isin(selected)].copy()

    return take("train"), take("validation"), take("test"), assignments


@dataclass
class FittedModel:
    name: str
    parameters: dict[str, float]
    x_min: float
    x_max: float

    def predict(self, request_rate: Iterable[float]) -> np.ndarray:
        r = np.asarray(request_rate, dtype=float)
        p = self.parameters
        if self.name == "M1_linear":
            return p["p0"] + p["alpha"] * r
        if self.name == "M2_cubic":
            span = max(self.x_max - self.x_min, np.finfo(float).eps)
            u = (r - self.x_min) / span
            return p["b0"] + p["b1"] * u + p["b2"] * u**2 + p["b3"] * u**3
        if self.name == "M3_logistic":
            x = np.log1p(np.maximum(r, 0.0))
            arg = np.clip(-p["k"] * (x - p["x0"]), -700, 700)
            return p["p0"] + p["delta_p"] / (1.0 + np.exp(arg))
        raise KeyError(self.name)


def fit_model(name: str, data: pd.DataFrame) -> FittedModel:
    r = data["request_rate"].to_numpy(dtype=float)
    y = data["power"].to_numpy(dtype=float)
    r_min, r_max = float(r.min()), float(r.max())

    if name == "M1_linear":
        alpha, p0 = np.polyfit(r, y, deg=1)
        params = {"p0": float(p0), "alpha": float(alpha)}

    elif name == "M2_cubic":
        span = max(r_max - r_min, np.finfo(float).eps)
        u = (r - r_min) / span
        b3, b2, b1, b0 = np.polyfit(u, y, deg=3)
        params = {"b0": float(b0), "b1": float(b1), "b2": float(b2), "b3": float(b3)}

    elif name == "M3_logistic":
        x = np.log1p(np.maximum(r, 0.0))
        y_lo, y_hi = float(np.quantile(y, 0.05)), float(np.quantile(y, 0.95))
        delta0 = max(y_hi - y_lo, np.std(y), 1e-6)
        beta0 = np.array([y_lo, math.log(delta0), 0.0, float(np.median(x))])

        def unpack(beta: np.ndarray) -> tuple[float, float, float, float]:
            return float(beta[0]), float(np.exp(beta[1])), float(np.exp(beta[2])), float(beta[3])

        def residual(beta: np.ndarray) -> np.ndarray:
            p0, delta, k, x0 = unpack(beta)
            arg = np.clip(-k * (x - x0), -700, 700)
            return p0 + delta / (1.0 + np.exp(arg)) - y

        x_pad = max(float(np.ptp(x)), 1.0)
        lower = np.array([0.0, math.log(1e-9), math.log(1e-4), float(x.min() - 2 * x_pad)])
        upper = np.array([
            max(float(y.max()) * 1.2, 1.0),
            math.log(max(float(y.max()) * 10.0, 1.0)),
            math.log(1e3),
            float(x.max() + 2 * x_pad),
        ])
        result = least_squares(residual, beta0, bounds=(lower, upper), max_nfev=50_000)
        p0, delta, k, x0 = unpack(result.x)
        params = {"p0": p0, "delta_p": delta, "k": k, "x0": x0}

    else:
        raise KeyError(name)
    return FittedModel(name=name, parameters=params, x_min=r_min, x_max=r_max)


def metric_row(
    model: FittedModel,
    data: pd.DataFrame,
    train_power_max: float,
    split: str,
) -> dict[str, float | str]:
    y = data["power"].to_numpy(dtype=float)
    pred = model.predict(data["request_rate"])
    safe = np.maximum(np.abs(y), max(np.mean(np.abs(y)) * 1e-3, 1e-9))
    grid_max = max(model.x_max, float(data["request_rate"].max()))
    grid = np.linspace(model.x_min, grid_max, 500)
    grid_pred = model.predict(grid)
    scale = max(float(np.ptp(grid_pred)), float(np.mean(np.abs(y))), 1e-9)
    monotonic_violation = float(np.mean(np.diff(grid_pred) < -1e-8 * scale))
    upper = max(train_power_max * 1.10, 1e-9)
    range_violation = float(np.mean((grid_pred < 0.0) | (grid_pred > upper)))
    return {
        "model": model.name,
        "split": split,
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean(np.square(y - pred)))),
        "nrmse_mean": float(np.sqrt(np.mean(np.square(y - pred))) / max(np.mean(np.abs(y)), 1e-9)),
        "mae": float(np.mean(np.abs(y - pred))),
        "mape_pct": float(np.mean(np.abs((y - pred) / safe)) * 100.0),
        "r2": float(1.0 - np.sum(np.square(y - pred)) / np.sum(np.square(y - np.mean(y))))
        if len(y) > 1 and np.sum(np.square(y - np.mean(y))) > np.finfo(float).eps
        else float("nan"),
        "monotonic_violation_fraction": monotonic_violation,
        "physical_range_violation_fraction": range_violation,
    }


def choose_model(validation_metrics: pd.DataFrame) -> tuple[str, str]:
    valid = validation_metrics.loc[
        (validation_metrics["monotonic_violation_fraction"] <= 0.01)
        & (validation_metrics["physical_range_violation_fraction"] <= 0.01)
    ]
    if not valid.empty:
        winner = valid.sort_values(["rmse", "mae", "mape_pct"]).iloc[0]
        reason = "물리 제약(단조 증가·전력 범위)을 통과한 모델 중 validation RMSE 최소"
    else:
        ranked = validation_metrics.assign(
            penalty=validation_metrics["nrmse_mean"]
            + 2.0 * validation_metrics["monotonic_violation_fraction"]
            + 2.0 * validation_metrics["physical_range_violation_fraction"]
        )
        winner = ranked.sort_values(["penalty", "rmse"]).iloc[0]
        reason = "모든 모델이 물리 제약을 위반하여 정규화 RMSE와 위반율의 합성 penalty 최소"
    return str(winner["model"]), reason


def _plot_comparison(
    condition: pd.DataFrame,
    assignments: pd.DataFrame,
    models: dict[str, FittedModel],
    selected: str,
    output: Path,
) -> None:
    merged = condition.merge(assignments[["condition_id", "split"]], on="condition_id")
    colors = {"train": "#9ca3af", "validation": "#f59e0b", "test": "#2563eb"}
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for split, part in merged.groupby("split"):
        ax.scatter(
            part["request_rate"], part["power"], s=34, alpha=0.78,
            color=colors[split], label=f"observed: {split}"
        )
    grid = np.linspace(condition["request_rate"].min(), condition["request_rate"].max(), 600)
    line_colors = {"M1_linear": "#111827", "M2_cubic": "#7c3aed", "M3_logistic": "#059669"}
    for name, model in models.items():
        width = 3.0 if name == selected else 1.8
        ax.plot(grid, model.predict(grid), color=line_colors[name], lw=width,
                label=f"{name}{' (selected)' if name == selected else ''}")
    ax.set_xlabel("Request rate")
    ax.set_ylabel("Condition-mean aggregated IT power")
    ax.set_title("NLR steady-state IT power model comparison")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _subgroup_metrics(
    test: pd.DataFrame, model: FittedModel
) -> pd.DataFrame:
    group_cols = [c for c in ("prompt_dataset", "max_output_tokens") if c in test.columns]
    if not group_cols:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, part in test.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        pred = model.predict(part["request_rate"])
        y = part["power"].to_numpy(dtype=float)
        row = dict(zip(group_cols, keys))
        row.update({
            "n": len(part),
            "rmse": float(np.sqrt(np.mean(np.square(y - pred)))),
            "mae": float(np.mean(np.abs(y - pred))),
            "mean_residual": float(np.mean(y - pred)),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without requiring the tabulate package."""
    values = frame.astype(str)
    header = "| " + " | ".join(values.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(values.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in values.to_numpy()]
    return "\n".join([header, rule, *rows])


def run_comparison(
    input_path: Path | None,
    output_dir: Path,
    project_root: Path,
    seed: int = 42,
) -> dict[str, Any]:
    source = Path(input_path) if input_path else discover_feature_table(project_root)
    raw = read_table(source)
    condition, resolved = build_condition_table(raw)
    train, validation, test, assignments = split_conditions(condition, seed=seed)

    names = ("M1_linear", "M2_cubic", "M3_logistic")
    train_models = {name: fit_model(name, train) for name in names}
    train_power_max = float(train["power"].max())
    val_metrics = pd.DataFrame([
        metric_row(model, validation, train_power_max, "validation")
        for model in train_models.values()
    ])
    selected, reason = choose_model(val_metrics)

    # The test set remains untouched during model-family selection.  Refit all
    # families on train+validation, then report their final test performance.
    development = pd.concat([train, validation], ignore_index=True)
    final_models = {name: fit_model(name, development) for name in names}
    dev_power_max = float(development["power"].max())
    test_metrics = pd.DataFrame([
        metric_row(model, test, dev_power_max, "test")
        for model in final_models.values()
    ])

    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    condition.to_csv(output_dir / "condition_level_dataset.csv", index=False)
    assignments.to_csv(output_dir / "split_assignment.csv", index=False)
    val_metrics.sort_values("rmse").to_csv(output_dir / "validation_metrics.csv", index=False)
    test_metrics.sort_values("rmse").to_csv(output_dir / "test_metrics.csv", index=False)

    parameters = {
        name: {
            "parameters": model.parameters,
            "request_rate_fit_min": model.x_min,
            "request_rate_fit_max": model.x_max,
        }
        for name, model in final_models.items()
    }
    (output_dir / "model_parameters.json").write_text(
        json.dumps(parameters, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    selection = {
        "selected_model": selected,
        "selection_basis": reason,
        "input_path": str(source.resolve()),
        "seed": seed,
        "n_conditions": len(condition),
        "n_train": len(train),
        "n_validation": len(validation),
        "n_test": len(test),
        "resolved_source_columns": resolved,
        "important_note": (
            "The selected equation describes steady-state condition-mean IT power only; "
            "first-order lag and stochastic residuals are intentionally not fitted here."
        ),
    }
    (output_dir / "selected_model.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    subgroup = _subgroup_metrics(test, final_models[selected])
    if not subgroup.empty:
        subgroup.to_csv(output_dir / "selected_model_subgroup_metrics.csv", index=False)

    _plot_comparison(condition, assignments, final_models, selected, figures / "model_comparison.png")

    val_view = _markdown_table(val_metrics.sort_values("rmse").round(5).reset_index(drop=True))
    test_view = _markdown_table(test_metrics.sort_values("rmse").round(5).reset_index(drop=True))
    report = f"""# NLR 정상상태 IT 부하 수식 비교

## 결론

- 선정 모델: **{selected}**
- 선정 기준: {reason}
- 주의: 모델 선정은 validation set으로 수행했으며 test set은 최종 일반화 성능 확인에만 사용함.
- 이 결과는 condition 평균 정상상태 관계만 다루며 1차 지연·burst·확률 residual은 아직 포함하지 않음.

## 데이터 분할

- 입력: `{source.resolve()}`
- 전체 condition: {len(condition)}
- train / validation / test: {len(train)} / {len(validation)} / {len(test)}
- 동일 condition_id의 반복 측정은 서로 다른 split에 포함되지 않음.

## Validation 결과

{val_view}

## Test 결과

{test_view}

## 다음 판단

1. 선정 모델의 test RMSE와 subgroup residual을 확인한다.
2. prompt dataset 또는 maximum output tokens별 편향이 크면 해당 변수를 정상상태 파라미터에 추가한다.
3. 정상상태 수식이 확정된 뒤에만 1차 지연 시정수와 확률 residual을 식별한다.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return {
        **selection,
        "validation_metrics": val_metrics.sort_values("rmse").to_dict("records"),
        "test_metrics": test_metrics.sort_values("rmse").to_dict("records"),
        "output_dir": str(output_dir.resolve()),
    }
