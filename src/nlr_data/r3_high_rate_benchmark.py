"""Fixed-R3 high-rate benchmark and complete M3+R3 IT profile evaluation.

R3 is fixed before this run.  R0-R2 are reported only as frozen baselines and
cannot replace R3 after inspecting these results.  The historical 81-100
request/s split has appeared in legacy outputs, so it is described as a
high-rate holdout benchmark rather than a pristine untouched test set.

Reference map used in the generated report:
  *  M3 saturation: motivated by NLR Paper list [3]; exact logistic map is this study.
  ** Mean/residual and AR(1): adapted from Paper list [37].
  *** Jump-OU backbone: modified from Paper list [2], Eq. (2).  The signed
      Gaussian-mixture innovation is this study's NLR-specific approximation.
  **** ES/VS/FC/AIW/AWD/SIW/CP: Paper list [40], Eqs. (15)-(20), (24)-(25).
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
from tqdm.auto import tqdm

from .residual_models import load_profiles, load_condition_targets, fit_m3_by_dataset
from .residual_model_selection import (
    build_parameter_library,
    simulate_residual_ensemble,
)
from .residual_scenario_scores import MODELS, block_average_for_scoring, score_prefixes
from .residual_fluctuation_validation import (
    fluctuation_calibration, scenario_interval_metrics, aidc_temporal_diagnostics,
)

FIXED_MODEL = "R3_jump_ou"


def regression_metrics(observed, predicted) -> dict[str, float]:
    y = np.asarray(observed, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or y.size < 2:
        raise ValueError("Expected equal one-dimensional arrays")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Regression inputs must be finite")
    error = p - y
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "mae_kw": float(np.mean(np.abs(error))),
        "rmse_kw": float(np.sqrt(np.mean(error**2))),
        "mape_pct": float(100 * np.mean(np.abs(error) / np.maximum(np.abs(y), 1e-9))),
        "r2": float(1 - np.sum(error**2) / denominator) if denominator > 0 else float("nan"),
        "mean_bias_kw": float(np.mean(error)),
    }


def _condition_average(frame: pd.DataFrame, draws: int) -> pd.DataFrame:
    part = frame.loc[frame["draws"].eq(draws)].copy()
    keys = ["prompt_dataset", "request_rate", "condition_id", "profile_id", "model"]
    numeric = [column for column in part.select_dtypes(include=[np.number]).columns
               if column not in {"request_rate", "seed", "draws"}]
    profile = part.groupby(keys, as_index=False)[numeric].mean()
    return profile.groupby(["prompt_dataset", "request_rate", "condition_id", "model"],
                           as_index=False)[numeric].mean()


def _comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    direction = {
        "residual_es_kw": "lower", "residual_vs_kw": "lower",
        "full_es_kw": "lower", "full_vs_kw": "lower",
        "fc_median_kw_0p1s": "lower", "fc_median_kw_1p0s": "lower",
        "acf_mae": "lower", "ramp_w1_0p1s_kw_per_s": "lower",
        "ramp_w1_1p0s_kw_per_s": "lower", "full_aiw_kw": "context",
        "full_awd_kw": "lower", "full_cp": "higher",
        "q01_abs_error_kw": "lower", "q99_abs_error_kw": "lower",
        "excess_kurtosis_abs_error": "lower",
    }
    rows = []
    for metric, preferred in direction.items():
        selected = float(summary.loc[FIXED_MODEL, metric])
        for baseline in MODELS:
            if baseline == FIXED_MODEL:
                continue
            value = float(summary.loc[baseline, metric])
            rows.append({
                "metric": metric, "preferred": preferred, "fixed_model": FIXED_MODEL,
                "baseline": baseline, "r3_value": selected, "baseline_value": value,
                "r3_minus_baseline": selected - value,
                "relative_change_pct": (100 * (selected - value) / abs(value)) if value != 0 else np.nan,
            })
    return pd.DataFrame(rows)


def run_high_rate_benchmark(project_root, *, draws=(25, 50, 100), seeds=(42, 43, 44),
                            horizon_s=180.0, score_interval_s=1.0, lambda_siw=10.0,
                            output_dir=None):
    root = Path(project_root).resolve()
    output = Path(output_dir or root / "outputs/r3_high_rate_benchmark").resolve()
    output.mkdir(parents=True, exist_ok=True)
    counts = sorted(set(int(value) for value in draws))
    seeds = tuple(int(value) for value in seeds)
    if not counts or min(counts) < 1 or not seeds:
        raise ValueError("Invalid ensemble configuration")

    manifest = root / "outputs/condition_audit/audited_run_manifest.csv"
    split = root / "outputs/steady_state_model_comparison/split_assignment.csv"
    target = root / "outputs/graybox_steady_state/online_rate_condition_targets.csv"
    dataset_root = root.parent / "NLR_dataset"
    profiles = load_profiles(dataset_root=dataset_root, manifest_path=manifest, split_path=split)
    targets = load_condition_targets(target, split)

    calibration_splits = {"train", "validation"}
    m3 = fit_m3_by_dataset(targets, calibration_splits)
    library = build_parameter_library(profiles, m3, calibration_splits)
    if set(library.parameters["split"].unique()) != calibration_splits:
        raise ValueError("R3 calibration did not use exactly train+validation")
    library.parameters.to_csv(output / "train_validation_residual_parameters.csv", index=False)
    pd.DataFrame([
        {"prompt_dataset": dataset, **asdict(parameters)}
        for dataset, parameters in m3.items()
    ]).to_csv(output / "train_validation_m3_parameters.csv", index=False)

    benchmark = [profile for profile in profiles if profile.split == "test"]
    if not benchmark or min(profile.request_rate for profile in benchmark) < 81:
        raise ValueError("Expected the fixed 81-100 request/s high-rate split")
    dt = benchmark[0].interval_s
    if any(not np.isclose(profile.interval_s, dt) for profile in benchmark):
        raise ValueError("Benchmark profiles do not share one time grid")
    requested_h = int(round(horizon_s / dt))
    h = min(requested_h, min(len(profile.power_kw) for profile in benchmark))
    score_factor = int(round(score_interval_s / dt))
    if not np.isclose(score_factor * dt, score_interval_s):
        raise ValueError("Score interval must be an integer multiple of raw interval")

    config = {
        "stage": "fixed-R3 high-rate holdout benchmark",
        "fixed_model_before_benchmark": FIXED_MODEL,
        "baselines_for_reporting_only": [model for model in MODELS if model != FIXED_MODEL],
        "legacy_test_exposure": True,
        "independence_note": "81-100 request/s appeared in earlier legacy outputs; do not call pristine untouched test.",
        "calibration_splits": sorted(calibration_splits), "benchmark_split": "test",
        "request_rate_range": [min(p.request_rate for p in benchmark),
                               max(p.request_rate for p in benchmark)],
        "models": list(MODELS), "draws": counts, "seeds": list(seeds),
        "requested_horizon_s": horizon_s, "actual_horizon_s": h * dt,
        "raw_interval_s": dt, "score_interval_s": score_interval_s,
        "lambda_siw": lambda_siw, "test_conditions": len(benchmark),
        "references": {
            "PL3": "NLR measurement paper: request-rate/prompt-conditioned saturation",
            "PL37": "mean-template plus AR(1) residual construction",
            "PL2": "Eq. (2) OU plus Poisson/log-normal jump backbone; current GMM is modified",
            "PL40": "Eqs. (15)-(20), (24)-(25) scenario metrics",
        },
        "m3": {name: asdict(value) for name, value in m3.items()},
        "selection_locked": True, "reselection_after_benchmark_allowed": False,
        "no_composite_score": True,
    }
    (output / "protocol.json").write_text(json.dumps(config, indent=2, ensure_ascii=False),
                                           encoding="utf-8")

    # Steady-state M3 evaluation uses one target per experimental condition.
    target_test = targets.loc[targets["split"].eq("test")].copy()
    target_test["m3_predicted_kw"] = [
        float(m3[row.prompt_dataset].predict(row.request_rate))
        for row in target_test.itertuples()
    ]
    target_test["m3_error_kw"] = target_test["m3_predicted_kw"] - target_test["target_power_kw"]
    target_test.to_csv(output / "m3_high_rate_predictions.csv", index=False)
    m3_rows = [{"scope": "overall", **regression_metrics(
        target_test["target_power_kw"], target_test["m3_predicted_kw"])}]
    for dataset, group in target_test.groupby("prompt_dataset", observed=True):
        m3_rows.append({"scope": str(dataset), **regression_metrics(
            group["target_power_kw"], group["m3_predicted_kw"])})
    m3_metrics = pd.DataFrame(m3_rows)
    m3_metrics.to_csv(output / "m3_high_rate_metrics.csv", index=False)

    rows = []
    examples = {}
    for profile in tqdm(benchmark, desc="Fixed-R3 high-rate benchmark"):
        measured = np.asarray(profile.power_kw[:h], dtype=float)
        centered = measured - np.mean(measured)
        steady = float(m3[profile.prompt_dataset].predict(profile.request_rate))
        base = {"profile_id": profile.profile_id, "condition_id": profile.condition_id,
                "prompt_dataset": profile.prompt_dataset, "request_rate": profile.request_rate,
                "observed_mean_kw": float(np.mean(measured)), "m3_steady_kw": steady}
        for model in MODELS:
            for seed in seeds:
                key = f"benchmark|{seed}|{profile.profile_id}|{model}".encode()
                rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))
                residual_ensemble = simulate_residual_ensemble(
                    model,
                    draws=counts[-1],
                    length=h,
                    prompt_dataset=profile.prompt_dataset,
                    request_rate=profile.request_rate,
                    library=library,
                    rng=rng,
                )
                full_ensemble = steady + residual_ensemble
                score_y, score_x = block_average_for_scoring(
                    centered, residual_ensemble, raw_interval_s=dt,
                    score_interval_s=score_interval_s)
                residual_score_prefixes = score_prefixes(score_y, score_x, counts)
                full_y, full_x = block_average_for_scoring(
                    measured, full_ensemble, raw_interval_s=dt,
                    score_interval_s=score_interval_s)
                full_score_prefixes = score_prefixes(full_y, full_x, counts)
                max_count_diagnostics = aidc_temporal_diagnostics(
                    centered, residual_ensemble, interval_s=dt)
                for count in counts:
                    residual_subset = residual_ensemble[:count]
                    full_subset = full_ensemble[:count]
                    residual_scores = residual_score_prefixes[count]
                    full_scores = full_score_prefixes[count]
                    values = {
                        "residual_es_kw": residual_scores["es_kw"],
                        "residual_vs_kw": residual_scores["vs_kw"],
                        "full_es_kw": full_scores["es_kw"],
                        "full_vs_kw": full_scores["vs_kw"],
                    }
                    for lag_s in (0.1, 1.0):
                        label = str(lag_s).replace(".", "p")
                        fc = fluctuation_calibration(centered, residual_subset,
                                                     int(round(lag_s / dt)))
                        values.update({f"{name}_{label}s": value for name, value in fc.items()})
                    values.update({f"residual_{name}": value for name, value in
                                   scenario_interval_metrics(centered, residual_subset,
                                                             lambda_siw=lambda_siw).items()})
                    values.update({f"full_{name}": value for name, value in
                                   scenario_interval_metrics(measured, full_subset,
                                                             lambda_siw=lambda_siw).items()})
                    if count == counts[-1]:
                        values.update(max_count_diagnostics)
                    rows.append({**base, "model": model, "seed": seed,
                                 "draws": count, **values})
                if model == FIXED_MODEL and seed == seeds[0] and profile.prompt_dataset not in examples:
                    examples[profile.prompt_dataset] = (profile, measured, steady, residual_ensemble)
        pd.DataFrame(rows).to_csv(output / "benchmark_metrics_by_profile_seed.csv", index=False)

    metrics = pd.DataFrame(rows)
    condition = _condition_average(metrics, counts[-1])
    condition.to_csv(output / "benchmark_metrics_by_condition.csv", index=False)
    numeric = [column for column in condition.select_dtypes(include=[np.number]).columns
               if column != "request_rate"]
    summary = condition.groupby("model")[numeric].mean().reindex(MODELS)
    summary.to_csv(output / "benchmark_summary.csv")
    comparison = _comparison_table(summary)
    comparison.to_csv(output / "r3_vs_frozen_baselines.csv", index=False)
    metrics.groupby(["draws", "model"])[numeric].mean().reset_index().to_csv(
        output / "ensemble_size_sensitivity.csv", index=False)

    # Overview figure uses the preregistered 100-scenario summary.
    labels = [model[:2] for model in MODELS]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, metric, title in (
        (axes[0, 0], "residual_es_kw", "Residual ES* (lower better)"),
        (axes[0, 1], "residual_vs_kw", "Residual VS* (lower better)"),
        (axes[1, 0], "ramp_w1_0p1s_kw_per_s", "0.1 s ramp-W1** (lower better)"),
        (axes[1, 1], "acf_mae", "Short-lag ACF error*** (lower better)"),
    ):
        axis.bar(labels, summary[metric], color=["#9ca3af", "#9ca3af", "#9ca3af", "#dc2626"])
        axis.set_title(title); axis.grid(axis="y", alpha=.2)
    figure.suptitle("81-100 request/s high-rate benchmark; R3 fixed before evaluation")
    figure.tight_layout(); figure.savefig(output / "high_rate_benchmark_overview.png", dpi=160)
    plt.close(figure)

    for dataset, (profile, measured, steady, ensemble) in examples.items():
        time = np.arange(h) * dt
        lower, median, upper = np.quantile(steady + ensemble, [.05, .50, .95], axis=0)
        figure, axis = plt.subplots(figsize=(12, 4.5))
        axis.fill_between(time, lower, upper, alpha=.25, color="#dc2626", label="R3 pointwise 5-95%")
        axis.plot(time, median, color="#dc2626", lw=.8, label="R3 median")
        axis.plot(time, measured, color="black", lw=.7, label="measured")
        axis.axhline(steady, color="#059669", lw=1.2, label="M3 steady")
        axis.set(title=f"M3+R3 | {dataset} | {profile.request_rate:g} request/s",
                 xlabel="Time (s)", ylabel="IT power (kW)")
        axis.legend(ncol=4, fontsize=8); axis.grid(alpha=.2)
        figure.tight_layout(); figure.savefig(output / f"m3_r3_example_{dataset}.png", dpi=160)
        plt.close(figure)

    status = {
        "status": "fixed_r3_high_rate_benchmark_complete",
        "fixed_model": FIXED_MODEL, "model_reselected": False,
        "legacy_test_exposure": True, "test_evaluated_under_new_protocol": True,
        "next_step": "construct and export the final M3+R3 IT profile generator",
    }
    (output / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    selected = summary.loc[FIXED_MODEL]
    r2_baseline = summary.loc["R2_ou"]
    m3_overall = m3_metrics.loc[m3_metrics["scope"].eq("overall")].iloc[0]
    report = rf"""# Fixed-R3 high-rate holdout benchmark

## 1. 고정된 완성형 수식

\[
P_{{IT,k}}=P_{{M3}}(r,d)+e_k \quad *
\]

\[
e_{{k+1}}=\phi(r,d)e_k+\eta_k \quad **
\]

\[
\eta_k\sim(1-g)\mathcal{{N}}(0,\sigma^2)+g\sum_c\pi_c\mathcal{{N}}(\mu_c,\sigma_c^2) \quad ***
\]

- *: NLR [3]의 request-rate 포화특성과 [37]의 mean-template+AR(1) residual 결합을 수정 적용. 정확한 M3 logistic은 본 연구에서 선정.
- **: [37]의 AR(1) residual 및 [2] 식 (2)의 OU+Jump 구조를 이산화.
- ***: [2]의 Poisson/log-normal jump를 NLR의 양·음 event에 맞춰 signed GMM innovation으로 수정한 본 연구 구현.

## 2. 평가 protocol

- R3는 benchmark 전에 최종 후보로 고정함. R0-R2는 비교용이며 결과를 보고 재선택하지 않음.
- M3/R3 파라미터는 Train+Validation에서 재보정함.
- 대상: 81-100 request/s, {len(benchmark)} conditions, {h*dt:g}초, dt={dt:g}초.
- 과거 legacy 결과에 이 구간이 이미 등장하므로 pristine test가 아닌 high-rate holdout benchmark로 명시함.
- ES/VS/FC/AIW/AWD/SIW/CP*: [40] 식 (15)-(20), (24)-(25). FC만 0.1초/1초 AIDC ramp로 수정.
- Ramp-W1, tail quantile 및 excess kurtosis는 일반 확률·분포 보조지표이며 [40]의 직접 수식이 아님.
- 임의 합성 J와 benchmark 이후 모델 재선택은 없음.

## 3. M3 high-rate 평균전력

```text
{m3_metrics.to_string(index=False)}
```

## 4. 100-scenario 평균 결과

```text
{summary[["residual_es_kw","residual_vs_kw","full_es_kw","full_vs_kw","fc_median_kw_0p1s","fc_median_kw_1p0s","ramp_w1_0p1s_kw_per_s","ramp_w1_1p0s_kw_per_s","acf_mae","full_aiw_kw","full_awd_kw","full_cp"]].to_string()}
```

## 5. 고정 R3 결과

- residual ES*: {selected['residual_es_kw']:.6g} kW
- residual VS*: {selected['residual_vs_kw']:.6g} kW
- full-profile ES*: {selected['full_es_kw']:.6g} kW
- 0.1초 ramp-W1: {selected['ramp_w1_0p1s_kw_per_s']:.6g} kW/s
- 1초 ramp-W1: {selected['ramp_w1_1p0s_kw_per_s']:.6g} kW/s
- short-lag ACF MAE: {selected['acf_mae']:.6g}
- residual q01 absolute error: {selected['q01_abs_error_kw']:.6g} kW
- residual q99 absolute error: {selected['q99_abs_error_kw']:.6g} kW
- residual excess-kurtosis absolute error: {selected['excess_kurtosis_abs_error']:.6g}
- full-profile AIW*: {selected['full_aiw_kw']:.6g} kW
- full-profile CP*: {selected['full_cp']:.4%}

![Benchmark](high_rate_benchmark_overview.png)

## 6. 결과 해석과 한계

- M3 평균 운전점은 MAE {m3_overall['mae_kw']:.5f} kW, MAPE {m3_overall['mape_pct']:.3f}%이다.
  다만 포화구간의 실제 평균전력 분산이 매우 작아 R²는 음수가 되므로, 이 구간에서는 MAE/MAPE와
  bias를 중심으로 해석한다.
- 고정 R3는 R2보다 residual ES가
  {100*(selected['residual_es_kw']/r2_baseline['residual_es_kw']-1):.2f}% 낮고,
  0.1초 ramp-W1은
  {100*(1-selected['ramp_w1_0p1s_kw_per_s']/r2_baseline['ramp_w1_0p1s_kw_per_s']):.2f}% 낮다.
  따라서 R3를 선택한 핵심 목적이었던 빠른 ramp 분포 재현은 고율 benchmark에서도 유지되었다.
- 반면 residual VS는 R2보다
  {100*(selected['residual_vs_kw']/r2_baseline['residual_vs_kw']-1):.2f}% 높고,
  short-lag ACF 오차도 사실상 동일하다. 모든 지표에서 R3가 우월하다는 결과는 아니다.
- R3의 q01 오차와 excess-kurtosis 오차는 R2보다 각각
  {100*(1-selected['q01_abs_error_kw']/r2_baseline['q01_abs_error_kw']):.2f}%와
  {100*(1-selected['excess_kurtosis_abs_error']/r2_baseline['excess_kurtosis_abs_error']):.2f}% 낮지만,
  excess-kurtosis 절대오차 자체가 {selected['excess_kurtosis_abs_error']:.2f}로 크다.
  예시 파형에서도 관측된 희소·대규모 dip은 충분히 재현되지 않으므로 극단 event 모델링은 남은 한계이다.
- CP는 R2보다 {100*(selected['full_cp']-r2_baseline['full_cp']):.2f}%p 높지만,
  AIW도 {100*(selected['full_aiw_kw']/r2_baseline['full_aiw_kw']-1):.2f}% 넓다.
  coverage 향상은 더 넓은 예측구간과 함께 해석해야 한다.
- 이 benchmark는 R3를 사후에 다시 선택하기 위한 절차가 아니다. 결과는 고정 R3의 강점과 한계를
  기록하며, 필요하면 향후 별도의 새 실험에서 극단 event 모듈을 고도화한다.

## 7. 다음 단계

R3 family를 다시 선택하지 않고, 재보정된 M3/R3 파라미터와 입력
`(prompt dataset, request rate, seed, horizon, dt, scenario count)`을 연결한 최종 IT profile generator를 export한다.
"""
    (output / "report_ko.md").write_text(report, encoding="utf-8")
    return status
