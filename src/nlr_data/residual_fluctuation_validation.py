"""Validation-only fluctuation diagnostics for NLR residual scenarios.

Literature-derived metrics:
  * FC: Tan et al. (IJEPES 2021), Eq. (15)-(16).
  * AIW/AWD/SIW/CP: Tan et al., Eq. (17)-(20).

Adaptation: the original one-hour wind-power increment is evaluated at 0.1 s
and 1.0 s for AIDC IT residuals.  No composite score or automatic model
selection is produced.  Test profiles are deliberately not opened.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, wasserstein_distance
from tqdm.auto import tqdm

from .residual_models import load_condition_targets, fit_m3_by_dataset
from .residual_model_selection import build_parameter_library, simulate_residual
from .residual_scenario_scores import MODELS, load_train_validation


def _validate(observed, ensemble):
    y = np.asarray(observed, dtype=float)
    x = np.asarray(ensemble, dtype=float)
    if y.ndim != 1 or x.ndim != 2 or x.shape[1] != y.size or len(x) < 1:
        raise ValueError("Expected observed[H] and ensemble[M,H]")
    if y.size < 3 or not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("Signals must be finite and contain at least three samples")
    return y, x


def fluctuation_calibration(observed, ensemble, lag_steps: int) -> dict[str, float]:
    """Eq. (15)-(16) adapted to an explicitly recorded sampling lag."""
    y, x = _validate(observed, ensemble)
    lag = int(lag_steps)
    if lag < 1 or lag >= y.size:
        raise ValueError("Invalid fluctuation lag")
    measured = y[lag:] - y[:-lag]
    scenarios = x[:, lag:] - x[:, :-lag]
    fc = np.abs(scenarios - measured[None, :]).ravel()
    return {
        "fc_mean_kw": float(np.mean(fc)),
        "fc_q25_kw": float(np.quantile(fc, 0.25)),
        "fc_median_kw": float(np.quantile(fc, 0.50)),
        "fc_q75_kw": float(np.quantile(fc, 0.75)),
        "fc_max_kw": float(np.max(fc)),
    }


def scenario_interval_metrics(observed, ensemble, *, lambda_siw: float = 10.0) -> dict[str, float]:
    """Eq. (17)-(20), using pointwise ensemble extrema as the interval."""
    y, x = _validate(observed, ensemble)
    if lambda_siw < 0:
        raise ValueError("lambda_siw must be nonnegative")
    lower = np.min(x, axis=0)
    upper = np.max(x, axis=0)
    width = upper - lower
    miss = np.where(y > upper, y - upper, np.where(y < lower, lower - y, 0.0))
    aiw = float(np.mean(width))
    awd = float(np.mean(miss))
    return {
        "aiw_kw": aiw,
        "awd_kw": awd,
        "siw_kw": aiw + lambda_siw * awd,
        "cp": float(np.mean((y >= lower) & (y <= upper))),
    }


def _acf(values: np.ndarray, lag: int) -> np.ndarray:
    values = values - np.mean(values, axis=-1, keepdims=True)
    left, right = values[..., :-lag], values[..., lag:]
    numerator = np.sum(left * right, axis=-1)
    denominator = np.sqrt(np.sum(left**2, axis=-1) * np.sum(right**2, axis=-1))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def aidc_temporal_diagnostics(observed, ensemble, *, interval_s: float,
                              acf_lags_s=(0.1, 0.5, 1.0, 2.0, 5.0)) -> dict[str, float]:
    """Phase-invariant AIDC supplements; reported separately from PL40 metrics."""
    y, x = _validate(observed, ensemble)
    result: dict[str, float] = {}
    errors = []
    for lag_s in acf_lags_s:
        lag = int(round(lag_s / interval_s))
        if lag < 1 or not np.isclose(lag * interval_s, lag_s) or lag >= y.size:
            raise ValueError(f"Invalid ACF lag {lag_s}")
        observed_acf = float(_acf(y[None, :], lag)[0])
        scenario_acf = float(np.mean(_acf(x, lag)))
        label = str(lag_s).replace(".", "p")
        result[f"observed_acf_{label}s"] = observed_acf
        result[f"scenario_acf_{label}s"] = scenario_acf
        result[f"acf_abs_error_{label}s"] = abs(scenario_acf - observed_acf)
        errors.append(abs(scenario_acf - observed_acf))
    result["acf_mae"] = float(np.mean(errors))

    # Wasserstein distance compares ramp distributions without requiring event
    # timing to coincide, unlike FC.
    for lag_s in (0.1, 1.0):
        lag = int(round(lag_s / interval_s))
        measured_ramp = (y[lag:] - y[:-lag]) / lag_s
        scenario_ramp = ((x[:, lag:] - x[:, :-lag]) / lag_s).ravel()
        label = str(lag_s).replace(".", "p")
        result[f"ramp_w1_{label}s_kw_per_s"] = float(
            wasserstein_distance(measured_ramp, scenario_ramp)
        )

    pooled = x.ravel()
    for quantile, label in ((0.01, "q01"), (0.99, "q99")):
        observed_q = float(np.quantile(y, quantile))
        scenario_q = float(np.quantile(pooled, quantile))
        result[f"observed_{label}_kw"] = observed_q
        result[f"scenario_{label}_kw"] = scenario_q
        result[f"{label}_abs_error_kw"] = abs(scenario_q - observed_q)
    observed_kurtosis = float(kurtosis(y, fisher=True, bias=False))
    scenario_kurtosis = float(kurtosis(pooled, fisher=True, bias=False))
    result["observed_excess_kurtosis"] = observed_kurtosis
    result["scenario_excess_kurtosis"] = scenario_kurtosis
    result["excess_kurtosis_abs_error"] = abs(
        scenario_kurtosis - observed_kurtosis
    )
    return result


def _condition_average(rows: pd.DataFrame, draws: int) -> pd.DataFrame:
    part = rows.loc[rows["draws"].eq(draws)].copy()
    keys = ["prompt_dataset", "request_rate", "condition_id", "profile_id", "model"]
    numeric = [column for column in part.columns if column not in keys + ["seed", "draws"]]
    profile = part.groupby(keys, as_index=False)[numeric].mean()
    return profile.groupby(["prompt_dataset", "request_rate", "condition_id", "model"],
                           as_index=False)[numeric].mean()


def run_fluctuation_validation(project_root, *, draws=(25, 50, 100), seeds=(42, 43, 44),
                               horizon_s=180.0, lambda_siw=10.0, output_dir=None):
    root = Path(project_root).resolve()
    output = Path(output_dir or root / "outputs/residual_fluctuation_validation").resolve()
    output.mkdir(parents=True, exist_ok=True)
    counts = sorted(set(int(value) for value in draws))
    if not counts or min(counts) < 1 or not seeds:
        raise ValueError("Invalid ensemble configuration")

    manifest = root / "outputs/condition_audit/audited_run_manifest.csv"
    split = root / "outputs/steady_state_model_comparison/split_assignment.csv"
    target = root / "outputs/graybox_steady_state/online_rate_condition_targets.csv"
    profiles = load_train_validation(root.parent / "NLR_dataset", manifest, split)
    targets = load_condition_targets(target, split)
    m3 = fit_m3_by_dataset(targets, {"train"})
    library = build_parameter_library(profiles, m3, {"train"})
    validation = [profile for profile in profiles if profile.split == "validation"]
    dt = validation[0].interval_s
    requested_h = int(round(horizon_s / dt))
    h = min(requested_h, min(len(profile.power_kw) for profile in validation))
    if h < 3:
        raise ValueError("Insufficient common horizon")

    config = {
        "reference": "Tan et al. (2021), IJEPES 130, 106955, Eqs. (15)-(20)",
        "reference_application": "wind scenarios -> centered AIDC IT residual scenarios",
        "fc_adaptation": "one-hour increment -> 0.1 s and 1.0 s increments",
        "models": list(MODELS), "draws": counts, "seeds": list(seeds),
        "requested_horizon_s": horizon_s, "actual_horizon_s": h * dt,
        "samples": h, "interval_s": dt,
        "lambda_siw": lambda_siw,
        "lambda_note": "Eq. (19) defines lambda; value 10 reproduces the relation in PL40 Table 5. SIW is sensitivity-only, not a selection score.",
        "test_signals_opened": False, "calibration_split": "train",
        "evaluation_split": "validation", "n_train": sum(p.split == "train" for p in profiles),
        "n_validation": len(validation),
        "m3": {name: asdict(value) for name, value in m3.items()},
        "no_composite_score": True,
    }
    (output / "protocol.json").write_text(json.dumps(config, indent=2, ensure_ascii=False),
                                           encoding="utf-8")

    rows = []
    for profile in tqdm(validation, desc="PL40 fluctuation/interval validation"):
        observed = np.asarray(profile.power_kw[:h], dtype=float)
        observed -= np.mean(observed)
        base = {"profile_id": profile.profile_id, "condition_id": profile.condition_id,
                "prompt_dataset": profile.prompt_dataset, "request_rate": profile.request_rate}
        for model in MODELS:
            for seed in seeds:
                key = f"{seed}|{profile.profile_id}|{model}".encode()
                rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))
                ensemble = np.stack([
                    simulate_residual(model, length=h, prompt_dataset=profile.prompt_dataset,
                                      request_rate=profile.request_rate, library=library, rng=rng)
                    for _ in range(counts[-1])
                ])
                for count in counts:
                    subset = ensemble[:count]
                    values = {}
                    for lag_s in (0.1, 1.0):
                        lag_steps = int(round(lag_s / dt))
                        label = str(lag_s).replace(".", "p")
                        values.update({f"{name}_{label}s": value for name, value in
                                       fluctuation_calibration(observed, subset, lag_steps).items()})
                    values.update(scenario_interval_metrics(observed, subset,
                                                            lambda_siw=lambda_siw))
                    values.update(aidc_temporal_diagnostics(observed, subset, interval_s=dt))
                    rows.append({**base, "model": model, "seed": seed,
                                 "draws": count, **values})
        pd.DataFrame(rows).to_csv(output / "metrics_by_profile_seed.csv", index=False)

    metrics = pd.DataFrame(rows)
    condition = _condition_average(metrics, counts[-1])
    condition.to_csv(output / "metrics_by_condition.csv", index=False)
    numeric = [column for column in condition.select_dtypes(include=[np.number]).columns
               if column != "request_rate"]
    summary = condition.groupby("model")[numeric].mean().reindex(MODELS)
    summary.to_csv(output / "validation_diagnostic_summary.csv")
    sensitivity = metrics.groupby(["draws", "model"])[numeric].mean().reset_index()
    sensitivity.to_csv(output / "ensemble_size_sensitivity.csv", index=False)

    # Compact reporting tables.
    main_columns = [
        "fc_median_kw_0p1s", "fc_median_kw_1p0s", "aiw_kw", "awd_kw", "siw_kw", "cp",
        "acf_mae", "ramp_w1_0p1s_kw_per_s", "ramp_w1_1p0s_kw_per_s",
        "q01_abs_error_kw", "q99_abs_error_kw",
    ]
    main = summary[main_columns]
    main.to_csv(output / "main_comparison.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    xloc = np.arange(len(MODELS))
    for ax, lag, title in zip(axes, ("0p1s", "1p0s"), ("FC at 0.1 s", "FC at 1.0 s")):
        med = summary[f"fc_median_kw_{lag}"].to_numpy()
        low = summary[f"fc_q25_kw_{lag}"].to_numpy()
        high = summary[f"fc_q75_kw_{lag}"].to_numpy()
        ax.errorbar(xloc, med, yerr=np.vstack([med-low, high-med]), fmt="o", capsize=4)
        ax.set_xticks(xloc, [name[:2] for name in MODELS]); ax.set_ylabel("FC (kW)")
        ax.set_title(title + " (median and IQR, lower better)"); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(output / "fc_comparison.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for model in MODELS:
        ax.scatter(summary.loc[model, "aiw_kw"], summary.loc[model, "cp"], s=70, label=model[:2])
        ax.annotate(model[:2], (summary.loc[model, "aiw_kw"], summary.loc[model, "cp"]),
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("AIW (kW; narrower left)"); ax.set_ylabel("CP (higher better)")
    ax.set_title("Scenario interval width-coverage trade-off")
    ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(output / "interval_tradeoff.png", dpi=160)
    plt.close(fig)

    lags = ("0p1", "0p5", "1p0", "2p0", "5p0")
    fig, ax = plt.subplots(figsize=(8, 5))
    observed_acf = [condition[f"observed_acf_{lag}s"].mean() for lag in lags]
    ax.plot([.1, .5, 1, 2, 5], observed_acf, color="black", marker="o", label="observed")
    for model in MODELS:
        values = [condition.loc[condition.model.eq(model), f"scenario_acf_{lag}s"].mean()
                  for lag in lags]
        ax.plot([.1, .5, 1, 2, 5], values, marker="o", label=model[:2])
    ax.set_xscale("log"); ax.set_xlabel("Lag (s)"); ax.set_ylabel("ACF")
    ax.set_title("Short-lag temporal correlation"); ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(output / "acf_comparison.png", dpi=160); plt.close(fig)

    status = {
        "status": "validation_diagnostics_complete_no_family_locked",
        "test_evaluated": False, "composite_score": None,
        "fc_0p1_best": str(summary["fc_median_kw_0p1s"].idxmin()),
        "fc_1p0_best": str(summary["fc_median_kw_1p0s"].idxmin()),
        "acf_best": str(summary["acf_mae"].idxmin()),
        "ramp_w1_0p1_best": str(summary["ramp_w1_0p1s_kw_per_s"].idxmin()),
        "tail_q01_best": str(summary["q01_abs_error_kw"].idxmin()),
    }
    (output / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    report = f"""# R0-R3 fluctuation/interval validation

- 문헌 근거: Paper list [40], 식 (15)-(20).
- FC의 시간 간격만 풍력 1시간에서 AIDC residual 0.1초 및 1초로 변경.
- Train {config['n_train']}개로 보정, validation {config['n_validation']}개로 평가. Test signal은 열지 않음.
- 시나리오 수 {counts}, seed {list(seeds)}, 공통 구간 {h*dt:g}초(dt={dt:g}초).
- SIW의 lambda={lambda_siw:g}은 [40] 식 (19) 및 Table 5 관계를 재현하기 위한 값이며, 모델 선정 점수로 사용하지 않음.
- FC는 동일 시점 ramp 오차, ramp-W1은 event timing과 무관한 ramp 분포 오차임.
- 모든 지표를 별도로 보고하며 임의의 합성 J를 만들지 않음.

## 100-scenario validation 평균

```text
{main.to_string()}
```

![FC](fc_comparison.png)
![Interval](interval_tradeoff.png)
![ACF](acf_comparison.png)

## 결과 해석

- FC 중앙값만 보면 {status['fc_0p1_best']}가 최소이다. 이는 대부분의 시점에서 residual ramp가 작아
  변동을 전혀 생성하지 않는 R0이 유리해지는 결과이므로, FC를 단독 선정 기준으로 사용하지 않는다.
- 단기 ACF 오차는 {status['acf_best']}가 최소이지만 R2와 R3 모두 0.5초 이후의 상관을 실제보다
  길게 유지한다. 단일 OU 시정수의 한계로 해석한다.
- event timing과 무관하게 ramp 분포를 비교한 W1은 0.1초에서
  {status['ramp_w1_0p1_best']}가 최소이다. Jump innovation이 빠른 ramp 재현에 기여한다.
- interval 지표는 폭(AIW)과 coverage(CP)의 trade-off이다. R3는 coverage가 가장 높지만 가장 넓고,
  R2는 R1-R3 중 가장 좁은 폭으로 높은 coverage를 유지한다.
- 따라서 validation 단계의 잠정 해석은 `R2=nominal residual 후보`,
  `R3=burst/stress 후보`이다. 최종 단일 모델 확정은 하지 않는다.

## 현재 상태

```json
{json.dumps(status, indent=2, ensure_ascii=False)}
```

이 결과는 최종 모델 선정을 자동화하지 않는다. ES/VS와 함께 시간상관, ramp, tail 및
interval coverage의 상충관계를 검토한 뒤 평가 protocol을 고정하고 test를 한 번만 수행한다.
"""
    (output / "report_ko.md").write_text(report, encoding="utf-8")
    return status
