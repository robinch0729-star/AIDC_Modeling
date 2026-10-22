"""Audit NLR experiment conditions against metadata and the associated paper.

The NLR repository defines workload families through its directory structure and
benchmark protocols, but it does not provide the unified ``condition_id`` used
by this project.  This module checks whether the project's grouping keys preserve
all controlled experiment factors, separates replicate/derived variables, and
emits a corrected condition registry for leakage-safe modelling.
"""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .metadata_join import build_run_manifest


PAPER_REFERENCE = (
    "Vercellino et al. (2026), Measurement of Generative AI Workload Power "
    "Profiles for Whole-Facility Data Center Infrastructure Planning"
)


def _clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed:")].copy()


def _result_relative_path(folder: str, value: Any) -> str:
    filename = PurePosixPath(str(value).replace("\\", "/")).name
    return f"01_aggregated_datasets/{folder}/results/{filename}"


def _stable_value(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _make_condition_id(record: pd.Series, keys: Iterable[str]) -> str:
    return "|".join(f"{key}={_stable_value(record.get(key))}" for key in keys)


def design_registry() -> pd.DataFrame:
    """Document the experiment factors supported by paper and local metadata."""

    rows = [
        {
            "workload_type": "training_or_finetuning",
            "controlled_variables": "model_name;nodes",
            "replicate_variables": "repeat;slurmid",
            "fixed_or_implicit_variables": "MLPerf configuration;Kestrel H100 hardware",
            "derived_or_outcome_variables": "power profile;runtime",
            "current_condition_keys": "model_name;nodes",
            "recommended_condition_keys": "model_name;nodes",
            "paper_evidence": "Sec. 2.2.2 and Sec. 3.2-3.3: model and node count are varied",
            "audit_interpretation": (
                "Current key matches the local release. The paper also reports a "
                "one-node Stable Diffusion case absent from local aggregated metadata."
            ),
        },
        {
            "workload_type": "offline_inference",
            "controlled_variables": "batch_size;max_output_tokens",
            "replicate_variables": "seed;repeat",
            "fixed_or_implicit_variables": "Llama-3 70B;one node;offline mode",
            "derived_or_outcome_variables": "runtime;power profile",
            "current_condition_keys": "batch_size;max_output_tokens",
            "recommended_condition_keys": "batch_size;max_output_tokens",
            "paper_evidence": (
                "Sec. 3.4: number of input prompts and maximum output tokens are varied"
            ),
            "audit_interpretation": (
                "Current key matches the controlled factors; metadata column "
                "batch_size represents the submitted prompt count in this dataset."
            ),
        },
        {
            "workload_type": "online_inference_finite",
            "controlled_variables": "dataset_path;request_rate;max_output_tokens",
            "replicate_variables": "seed;repeat",
            "fixed_or_implicit_variables": "num_prompts=1000;burstiness=100000;Llama-3.1 70B",
            "derived_or_outcome_variables": "throughput;latency;runtime;power profile",
            "current_condition_keys": (
                "request_rate;num_prompts;max_output_tokens;burstiness"
            ),
            "recommended_condition_keys": (
                "dataset_path;request_rate;max_output_tokens"
            ),
            "paper_evidence": (
                "Sec. 3.5: two prompt datasets, request rate, output token length, "
                "and seed are varied; total requests are fixed at 1000"
            ),
            "audit_interpretation": (
                "Current key omits prompt dataset and retains two constants. Strict "
                "experiment-design grouping must include dataset_path."
            ),
        },
        {
            "workload_type": "online_inference_rate",
            "controlled_variables": "dataset_path;request_rate",
            "replicate_variables": "none in local release",
            "fixed_or_implicit_variables": "duration=180 s;burstiness=100000;Llama-3.1 70B",
            "derived_or_outcome_variables": "num_prompts=request_rate*300;power profile",
            "current_condition_keys": "request_rate;num_prompts;burstiness",
            "recommended_condition_keys": "dataset_path;request_rate",
            "paper_evidence": (
                "Sec. 3.5 and Fig. 13: selected request rates are sustained for three "
                "minutes for coding and conversation datasets"
            ),
            "audit_interpretation": (
                "Current key pools the two prompt datasets as if they were repeats. "
                "They are separate controlled cases and no true repeat remains."
            ),
        },
        {
            "workload_type": "inference_whole_facility",
            "controlled_variables": "scale_mw;target_utilization",
            "replicate_variables": "none",
            "fixed_or_implicit_variables": "DIPLOEE inference scenario",
            "derived_or_outcome_variables": "simulated whole-facility power",
            "current_condition_keys": "scale_mw;target_utilization",
            "recommended_condition_keys": "scale_mw;target_utilization",
            "paper_evidence": "Sec. 4.2: 1 MW inference facility at 20/40/60/80% utilization",
            "audit_interpretation": "Scenario registry only; do not use as measured IT training data.",
        },
        {
            "workload_type": "colocation_mixed",
            "controlled_variables": "scale_mw;target_utilization",
            "replicate_variables": "none",
            "fixed_or_implicit_variables": "DIPLOEE colocation scenario",
            "derived_or_outcome_variables": "simulated whole-facility power",
            "current_condition_keys": "scale_mw;target_utilization",
            "recommended_condition_keys": "scale_mw;target_utilization",
            "paper_evidence": "Sec. 4.1: 10 MW colocation facility at 20/40/60/80% utilization",
            "audit_interpretation": "Scenario registry only; do not use as measured IT training data.",
        },
    ]
    table = pd.DataFrame(rows)
    table["source_reference"] = PAPER_REFERENCE
    return table


def _metadata_supplement(dataset_root: Path) -> pd.DataFrame:
    """Load experiment variables omitted from the original project manifest."""

    base = dataset_root / "01_aggregated_datasets"
    frames: list[pd.DataFrame] = []

    training = _clean_columns(pd.read_csv(base / "training" / "metadata.csv"))
    training["relative_path"] = training["path_save"].map(
        lambda value: f"01_aggregated_datasets/{str(value).replace(chr(92), '/')}"
    )
    training["dataset_path"] = np.nan
    training["metadata_repeat"] = training["repeat"]
    training["metadata_seed"] = np.nan
    frames.append(
        training[["relative_path", "dataset_path", "metadata_repeat", "metadata_seed"]]
    )

    folder = "inference_offline_llama3_70b"
    offline = _clean_columns(pd.read_csv(base / folder / "metadata.csv"))
    offline["relative_path"] = offline["path_run"].map(
        lambda value: _result_relative_path(folder, value)
    )
    offline["dataset_path"] = np.nan
    offline["metadata_repeat"] = offline["repeat"]
    offline["metadata_seed"] = offline["seed"]
    frames.append(
        offline[["relative_path", "dataset_path", "metadata_repeat", "metadata_seed"]]
    )

    folder = "inference_online_finite_llama3_70b"
    finite = _clean_columns(pd.read_csv(base / folder / "metadata.csv"))
    finite["relative_path"] = finite["path_run"].map(
        lambda value: _result_relative_path(folder, value)
    )
    finite["dataset_path"] = finite["dataset-path"]
    finite["metadata_repeat"] = finite["_repeat"]
    finite["metadata_seed"] = finite["seed"]
    frames.append(
        finite[["relative_path", "dataset_path", "metadata_repeat", "metadata_seed"]]
    )

    folder = "inference_online_rate_llama3_70b"
    rate = _clean_columns(pd.read_csv(base / folder / "metadata.csv"))
    rate["relative_path"] = rate["path_run"].map(
        lambda value: _result_relative_path(folder, value)
    )
    rate["dataset_path"] = rate["dataset-path"]
    rate["metadata_repeat"] = np.nan
    rate["metadata_seed"] = np.nan
    frames.append(
        rate[["relative_path", "dataset_path", "metadata_repeat", "metadata_seed"]]
    )

    for kind in ("inference", "colocation"):
        metadata = _clean_columns(
            pd.read_csv(dataset_root / "03_whole-facility_profiles" / kind / "metadata.csv")
        )
        metadata["relative_path"] = metadata["name"].map(
            lambda name: f"03_whole-facility_profiles/{kind}/simulated_data/{name}"
        )
        metadata["dataset_path"] = np.nan
        metadata["metadata_repeat"] = np.nan
        metadata["metadata_seed"] = np.nan
        frames.append(
            metadata[["relative_path", "dataset_path", "metadata_repeat", "metadata_seed"]]
        )

    supplement = pd.concat(frames, ignore_index=True)
    if supplement["relative_path"].duplicated().any():
        raise ValueError("Metadata supplement contains duplicate relative paths")
    return supplement


def build_audited_manifest(dataset_root: str | Path) -> pd.DataFrame:
    """Attach experiment factors and a strict design-based condition identifier."""

    root = Path(dataset_root).expanduser().resolve()
    manifest = build_run_manifest(root).rename(
        columns={"condition_id": "current_condition_id"}
    )
    supplement = _metadata_supplement(root)
    audited = manifest.merge(supplement, on="relative_path", how="left", validate="one_to_one")
    if audited["relative_path"].isna().any():
        raise ValueError("Failed to join experiment metadata")

    recommended_keys = {
        "training_or_finetuning": ["model_name", "nodes"],
        "offline_inference": ["batch_size", "max_output_tokens"],
        "online_inference_finite": [
            "dataset_path",
            "request_rate",
            "max_output_tokens",
        ],
        "online_inference_rate": ["dataset_path", "request_rate"],
        "inference_whole_facility": ["scale_mw", "target_utilization"],
        "colocation_mixed": ["scale_mw", "target_utilization"],
    }
    audited["recommended_condition_id"] = audited.apply(
        lambda row: _make_condition_id(row, recommended_keys[row["workload_type"]]),
        axis=1,
    )
    audited["analysis_cohort"] = np.where(
        audited["source_kind"].eq("whole_facility_simulated"),
        "facility_simulated_case",
        "measured_it_workload",
    )
    audited["condition_key_changed"] = audited[
        "current_condition_id"
    ].ne(audited["recommended_condition_id"])
    return audited


def build_validation_summary(audited: pd.DataFrame) -> pd.DataFrame:
    """Compare current and experiment-design-based condition counts."""

    design = design_registry().set_index("workload_type")
    eligible = audited.loc[audited["eligible"].astype(bool)]
    rows: list[dict[str, Any]] = []
    paper_counts = {
        "training_or_finetuning": 9,
        "online_inference_finite": 114,
        "online_inference_rate": 200,
    }
    for workload_type, group in eligible.groupby("workload_type", observed=True):
        current = int(group["current_condition_id"].nunique())
        recommended = int(group["recommended_condition_id"].nunique())
        collision_count = int(
            group.groupby("current_condition_id")["recommended_condition_id"]
            .nunique()
            .gt(1)
            .sum()
        )
        if current == recommended:
            status = "matches_local_experiment_design"
        else:
            status = "revision_required_controlled_factor_omitted"
        rows.append(
            {
                "workload_type": workload_type,
                "eligible_profiles": len(group),
                "current_condition_count": current,
                "recommended_condition_count": recommended,
                "paper_nominal_condition_count_if_explicit": paper_counts.get(
                    workload_type, np.nan
                ),
                "current_conditions_with_collisions": collision_count,
                "audit_status": status,
                "reason": design.loc[workload_type, "audit_interpretation"],
            }
        )
    return pd.DataFrame(rows).sort_values("workload_type").reset_index(drop=True)


def build_collision_report(audited: pd.DataFrame) -> pd.DataFrame:
    """Show current IDs that merge multiple strict experiment conditions."""

    eligible = audited.loc[audited["eligible"].astype(bool)]
    rows: list[dict[str, Any]] = []
    for (workload, current_id), group in eligible.groupby(
        ["workload_type", "current_condition_id"], observed=True
    ):
        strict_ids = sorted(group["recommended_condition_id"].unique())
        if len(strict_ids) <= 1:
            continue
        rows.append(
            {
                "workload_type": workload,
                "current_condition_id": current_id,
                "merged_recommended_condition_count": len(strict_ids),
                "merged_recommended_condition_ids": " || ".join(strict_ids),
                "dataset_paths": " || ".join(
                    sorted(group["dataset_path"].dropna().astype(str).unique())
                ),
                "profile_count": len(group),
            }
        )
    return pd.DataFrame(rows)


def build_condition_registry(audited: pd.DataFrame) -> pd.DataFrame:
    """Create one row per strict condition with replicate structure."""

    eligible = audited.loc[audited["eligible"].astype(bool)]
    rows: list[dict[str, Any]] = []
    condition_columns = [
        "model_name",
        "dataset_path",
        "nodes",
        "scale_mw",
        "target_utilization",
        "request_rate",
        "num_prompts",
        "batch_size",
        "max_output_tokens",
        "burstiness",
    ]
    for (workload, condition_id), group in eligible.groupby(
        ["workload_type", "recommended_condition_id"], observed=True
    ):
        row: dict[str, Any] = {
            "workload_type": workload,
            "recommended_condition_id": condition_id,
            "analysis_cohort": group["analysis_cohort"].iloc[0],
            "profile_count": len(group),
            "current_condition_count": group["current_condition_id"].nunique(),
            "repeat_value_count": group["metadata_repeat"].nunique(dropna=True),
            "seed_value_count": group["metadata_seed"].nunique(dropna=True),
        }
        for column in condition_columns:
            values = group[column].dropna().unique()
            row[column] = values[0] if len(values) == 1 else (
                " || ".join(map(str, sorted(values))) if len(values) else np.nan
            )
            row[f"{column}_unique_count"] = len(values)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["workload_type", "recommended_condition_id"], kind="stable"
    ).reset_index(drop=True)


def build_dependency_report(audited: pd.DataFrame) -> pd.DataFrame:
    """Record fixed, derived, controlled, and replicate-variable evidence."""

    eligible = audited.loc[audited["eligible"].astype(bool)]
    rows: list[dict[str, Any]] = []

    def add(workload: str, variable: str, role: str, evidence: str) -> None:
        subset = eligible.loc[eligible["workload_type"].eq(workload), variable]
        rows.append(
            {
                "workload_type": workload,
                "variable": variable,
                "assigned_role": role,
                "non_null_count": int(subset.notna().sum()),
                "unique_count": int(subset.nunique(dropna=True)),
                "empirical_evidence": evidence,
            }
        )

    add("training_or_finetuning", "model_name", "controlled", "Two model families")
    add("training_or_finetuning", "nodes", "controlled", "Four local node-count levels")
    add("offline_inference", "batch_size", "controlled", "40 prompt-count levels")
    add("offline_inference", "max_output_tokens", "controlled", "512 and 1024 tokens")
    add("offline_inference", "metadata_seed", "replicate", "Five seeds per condition")
    add("offline_inference", "metadata_repeat", "replicate", "Three repeats per seed")
    add("online_inference_finite", "dataset_path", "controlled", "Two prompt datasets")
    add("online_inference_finite", "request_rate", "controlled", "19 request-rate levels")
    add("online_inference_finite", "max_output_tokens", "controlled", "256/512/1024 tokens")
    add("online_inference_finite", "num_prompts", "fixed", "All eligible runs use 1000 prompts")
    add("online_inference_finite", "burstiness", "fixed", "All eligible runs use 100000")
    add("online_inference_finite", "metadata_seed", "replicate", "Three seed levels")
    add("online_inference_finite", "metadata_repeat", "replicate", "Three repeat levels")
    rate = eligible.loc[eligible["workload_type"].eq("online_inference_rate")]
    ratios = rate["num_prompts"] / rate["request_rate"]
    add("online_inference_rate", "dataset_path", "controlled", "Two prompt datasets")
    add("online_inference_rate", "request_rate", "controlled", "100 request-rate levels")
    add(
        "online_inference_rate",
        "num_prompts",
        "derived",
        f"num_prompts/request_rate is constant at {float(ratios.median()):.0f} s",
    )
    add("online_inference_rate", "burstiness", "fixed", "All runs use 100000")
    return pd.DataFrame(rows)


def build_paper_metadata_consistency_report(audited: pd.DataFrame) -> pd.DataFrame:
    """Record differences between the paper narrative and the local release."""

    eligible = audited.loc[audited["eligible"].astype(bool)]
    training = eligible.loc[
        eligible["workload_type"].eq("training_or_finetuning")
    ]
    finite = eligible.loc[
        eligible["workload_type"].eq("online_inference_finite")
    ]
    finite_runs = finite.groupby("recommended_condition_id").size()
    rows = [
        {
            "workload_type": "training_or_finetuning",
            "paper_statement": (
                "Llama-2 uses 2/4/8/16 nodes and Stable Diffusion uses "
                "1/2/4/8/16 nodes"
            ),
            "local_metadata_observation": (
                "Both model families contain only 2/4/8/16 nodes; "
                f"{training['recommended_condition_id'].nunique()} local conditions"
            ),
            "consistency_status": "local_release_subset_or_missing_case",
            "action": (
                "Use eight local conditions for modelling and document that the "
                "paper's one-node Stable Diffusion case is unavailable locally."
            ),
        },
        {
            "workload_type": "offline_inference",
            "paper_statement": (
                "Input-prompt count and maximum output-token length are varied; "
                "curves show replicate percentiles"
            ),
            "local_metadata_observation": (
                "40 prompt-count levels x 2 token levels; five seeds x three "
                "repeats = 15 profiles per condition"
            ),
            "consistency_status": "controlled_factors_consistent_metadata_more_detailed",
            "action": "Retain 80 strict conditions and treat seed/repeat as replicates.",
        },
        {
            "workload_type": "online_inference_finite",
            "paper_statement": (
                "Two datasets, request rate, output-token length and seed are varied; "
                "three repetitions using different seeds"
            ),
            "local_metadata_observation": (
                "2 datasets x 19 rates x 3 token levels = 114 conditions; local "
                "metadata crosses three seed values with three _repeat values, giving "
                f"a median of {float(finite_runs.median()):.0f} profiles per condition"
            ),
            "consistency_status": "condition_grid_consistent_replicate_count_needs_note",
            "action": (
                "Use 114 strict conditions. Treat seed and _repeat as nested replicate "
                "indices in the local release and report the difference from the paper text."
            ),
        },
        {
            "workload_type": "online_inference_rate",
            "paper_statement": (
                "Coding and conversation request streams are sustained for three minutes"
            ),
            "local_metadata_observation": (
                "2 datasets x 100 request rates = 200 profiles; num_prompts equals "
                "request_rate x 300 and there is one profile per strict condition"
            ),
            "consistency_status": "consistent_no_replicates",
            "action": (
                "Use 200 strict conditions; do not interpret the two datasets as repeats "
                "or estimate within-condition uncertainty from this group alone."
            ),
        },
    ]
    return pd.DataFrame(rows)


def build_dataset_effect_report(
    audited: pd.DataFrame, features: pd.DataFrame
) -> pd.DataFrame:
    """Quantify whether prompt dataset can safely be treated as a replicate."""

    metrics = [
        "mean_power_w",
        "peak_to_average_ratio",
        "coefficient_of_variation",
        "p95_abs_ramp_pu_per_s",
        "max_ramp_up_pu_per_s",
        "max_ramp_down_pu_per_s",
    ]
    feature_subset = features[["relative_path", "workload_type", *metrics]].copy()
    merged = audited[["relative_path", "workload_type", "dataset_path", "request_rate", "max_output_tokens", "eligible"]].merge(
        feature_subset,
        on=["relative_path", "workload_type"],
        how="inner",
        validate="one_to_one",
    )
    merged = merged.loc[merged["eligible"].astype(bool)]
    rows: list[dict[str, Any]] = []
    key_map = {
        "online_inference_finite": ["request_rate", "max_output_tokens"],
        "online_inference_rate": ["request_rate"],
    }
    for workload, base_keys in key_map.items():
        subset = merged.loc[merged["workload_type"].eq(workload)]
        for metric in metrics:
            medians = (
                subset.groupby([*base_keys, "dataset_path"], observed=True)[metric]
                .median()
                .unstack("dataset_path")
                .dropna()
            )
            if medians.shape[1] != 2 or medians.empty:
                continue
            left = medians.iloc[:, 0].astype(float)
            right = medians.iloc[:, 1].astype(float)
            scale = (left.abs() + right.abs()) / 2.0
            relative = (left - right).abs() / scale.where(scale > np.finfo(float).eps)
            try:
                p_value = float(wilcoxon(left, right).pvalue)
            except ValueError:
                p_value = np.nan
            rows.append(
                {
                    "workload_type": workload,
                    "metric": metric,
                    "paired_base_condition_count": len(medians),
                    "dataset_a": medians.columns[0],
                    "dataset_b": medians.columns[1],
                    "median_absolute_relative_difference": float(relative.median()),
                    "p95_absolute_relative_difference": float(relative.quantile(0.95)),
                    "wilcoxon_p_value": p_value,
                    "interpretation": (
                        "Dataset is a controlled factor; effect size only informs later "
                        "pooling/ablation and does not redefine it as a replicate."
                    ),
                }
            )
    return pd.DataFrame(rows)


def save_outputs(output_dir: str | Path, tables: dict[str, pd.DataFrame]) -> dict[str, Path]:
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
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


def run_condition_audit(
    dataset_root: str | Path,
    feature_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    audited = build_audited_manifest(dataset_root)
    features = pd.read_parquet(feature_path)
    tables = {
        "experiment_design_registry": design_registry(),
        "audited_run_manifest": audited,
        "condition_validation_summary": build_validation_summary(audited),
        "condition_collision_report": build_collision_report(audited),
        "condition_registry": build_condition_registry(audited),
        "metadata_dependency_report": build_dependency_report(audited),
        "paper_metadata_consistency_report": build_paper_metadata_consistency_report(
            audited
        ),
        "prompt_dataset_effect_report": build_dataset_effect_report(audited, features),
        "modeling_cohort_manifest": audited.loc[
            audited["analysis_cohort"].eq("measured_it_workload")
        ].reset_index(drop=True),
        "facility_case_manifest": audited.loc[
            audited["analysis_cohort"].eq("facility_simulated_case")
        ].reset_index(drop=True),
    }
    paths = save_outputs(output_dir, tables)
    return {"tables": tables, "paths": paths}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("outputs/feature_analysis/analysis_ready_features.parquet"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/condition_audit")
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_condition_audit(args.data_root, args.features, args.output_dir)
    summary = result["tables"]["condition_validation_summary"]
    print(summary.to_string(index=False))
    print("\nOutputs")
    for name, path in result["paths"].items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
