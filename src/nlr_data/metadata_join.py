"""Join NLR run metadata to measured/simulated power profile files.

The output manifest is the authoritative run-level table used by signal
validation and feature extraction. It preserves failed runs with an exclusion
reason instead of silently deleting them.
"""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .schema_registry import infer_path_metadata


MANIFEST_COLUMNS = [
    "profile_id",
    "relative_path",
    "workload_type",
    "model_name",
    "source_kind",
    "measurement_scope",
    "measurement_origin",
    "native_interval_s",
    "analysis_interval_s",
    "resampling_method",
    "run_id",
    "condition_id",
    "run_completed",
    "eligible",
    "exclusion_reason",
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
    "slurmid",
    "duration_metadata_s",
    "mean_power_metadata_w",
    "peak_power_metadata_w",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
]


def _clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed:")].copy()


def _number(record: pd.Series, *columns: str) -> float | None:
    for column in columns:
        if column in record.index:
            value = pd.to_numeric(pd.Series([record[column]]), errors="coerce").iloc[0]
            if pd.notna(value):
                return float(value)
    return None


def _text(record: pd.Series, *columns: str) -> str | None:
    for column in columns:
        if column in record.index and pd.notna(record[column]):
            value = str(record[column]).strip()
            if value:
                return value
    return None


def _boolean(value: Any, default: bool = True) -> bool:
    if pd.isna(value):
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _result_relative_path(folder: str, path_value: Any) -> str:
    """Map original NLR/Linux result paths to the local repository layout."""

    filename = PurePosixPath(str(path_value).replace("\\", "/")).name
    return f"01_aggregated_datasets/{folder}/results/{filename}"


def _condition_id(workload_type: str, values: dict[str, Any]) -> str:
    if workload_type == "training_or_finetuning":
        keys = ["model_name", "nodes"]
    elif workload_type == "offline_inference":
        keys = ["batch_size", "max_output_tokens"]
    elif workload_type == "online_inference_finite":
        keys = ["request_rate", "num_prompts", "max_output_tokens", "burstiness"]
    elif workload_type == "online_inference_rate":
        keys = ["request_rate", "num_prompts", "burstiness"]
    else:
        keys = ["scale_mw", "target_utilization"]
    return "|".join(f"{key}={values.get(key)}" for key in keys)


def _base_record(
    relative_path: str,
    workload_type: str,
    *,
    model_name: str,
    source_kind: str,
    measurement_origin: str,
    native_interval_s: float,
    analysis_interval_s: float,
    resampling_method: str,
) -> dict[str, Any]:
    path_metadata = infer_path_metadata(relative_path)
    run_id = path_metadata["run_id"]
    return {
        "profile_id": f"{workload_type}:{run_id}",
        "relative_path": relative_path,
        "workload_type": workload_type,
        "model_name": model_name,
        "source_kind": source_kind,
        "measurement_scope": (
            "whole_facility_power"
            if source_kind == "whole_facility_simulated"
            else "job_total_IT_power"
        ),
        "measurement_origin": measurement_origin,
        "native_interval_s": native_interval_s,
        "analysis_interval_s": analysis_interval_s,
        "resampling_method": resampling_method,
        "run_id": run_id,
        "run_completed": True,
        "eligible": True,
        "exclusion_reason": "",
        "nodes": path_metadata["node_count"],
        "scale_mw": path_metadata["scale_mw"],
        "target_utilization": path_metadata["target_utilization"],
    }


def _training_records(root: Path) -> Iterable[dict[str, Any]]:
    folder = "training"
    metadata = _clean_columns(
        pd.read_csv(root / "01_aggregated_datasets" / folder / "metadata.csv")
    )
    for _, row in metadata.iterrows():
        relative_path = f"01_aggregated_datasets/{row['path_save']}"
        record = _base_record(
            relative_path,
            "training_or_finetuning",
            model_name=str(row["model"]),
            source_kind="aggregated_measured",
            measurement_origin="measured_resampled",
            native_interval_s=0.2,
            analysis_interval_s=0.2,
            resampling_method="none",
        )
        record.update(
            {
                "nodes": _number(row, "nodes"),
                "repeat": _number(row, "repeat"),
                "slurmid": _text(row, "slurmid"),
            }
        )
        record["condition_id"] = _condition_id(record["workload_type"], record)
        yield record


def _offline_records(root: Path) -> Iterable[dict[str, Any]]:
    folder = "inference_offline_llama3_70b"
    metadata = _clean_columns(
        pd.read_csv(root / "01_aggregated_datasets" / folder / "metadata.csv")
    )
    for _, row in metadata.iterrows():
        relative_path = _result_relative_path(folder, row["path_run"])
        record = _base_record(
            relative_path,
            "offline_inference",
            model_name="Llama3 70B",
            source_kind="aggregated_measured",
            measurement_origin="measured_resampled",
            native_interval_s=0.1,
            analysis_interval_s=0.1,
            resampling_method="none",
        )
        record.update(
            {
                "batch_size": _number(row, "batch_size"),
                "max_output_tokens": _number(row, "max_output_tokens"),
                "repeat": _number(row, "repeat"),
                "seed": _number(row, "seed"),
                "duration_metadata_s": _number(row, "elapsed"),
                "mean_power_metadata_w": _number(row, "mean_power[W]"),
                "peak_power_metadata_w": _number(row, "peak_power[W]"),
            }
        )
        record["condition_id"] = _condition_id(record["workload_type"], record)
        yield record


def _finite_records(root: Path) -> Iterable[dict[str, Any]]:
    folder = "inference_online_finite_llama3_70b"
    metadata = _clean_columns(
        pd.read_csv(root / "01_aggregated_datasets" / folder / "metadata.csv")
    )
    for _, row in metadata.iterrows():
        relative_path = _result_relative_path(folder, row["path_run"])
        record = _base_record(
            relative_path,
            "online_inference_finite",
            model_name="Llama3.1 70B Instruct",
            source_kind="aggregated_measured",
            measurement_origin="measured_resampled",
            native_interval_s=0.1,
            analysis_interval_s=0.1,
            resampling_method="none",
        )
        completed = _boolean(row.get("run_completed", True))
        record.update(
            {
                "run_completed": completed,
                "eligible": completed,
                "exclusion_reason": "" if completed else "run_completed=False",
                "request_rate": _number(row, "request_rate_x", "request_rate_y"),
                "num_prompts": _number(row, "num-prompts", "num_prompts"),
                "max_output_tokens": _number(row, "hf-output-len"),
                "burstiness": _number(row, "burstiness_x", "burstiness_y"),
                "repeat": _number(row, "_repeat"),
                "seed": _number(row, "seed"),
                "duration_metadata_s": _number(row, "execution_time_seconds"),
                "mean_power_metadata_w": _number(row, "mean_power[W]"),
                "peak_power_metadata_w": _number(row, "peak_power[W]"),
                "request_throughput": _number(row, "request_throughput"),
                "output_throughput": _number(row, "output_throughput"),
                "total_token_throughput": _number(row, "total_token_throughput"),
            }
        )
        record["condition_id"] = _condition_id(record["workload_type"], record)
        yield record


def _rate_records(root: Path) -> Iterable[dict[str, Any]]:
    folder = "inference_online_rate_llama3_70b"
    metadata = _clean_columns(
        pd.read_csv(root / "01_aggregated_datasets" / folder / "metadata.csv")
    )
    for _, row in metadata.iterrows():
        relative_path = _result_relative_path(folder, row["path_run"])
        record = _base_record(
            relative_path,
            "online_inference_rate",
            model_name="Llama3.1 70B Instruct",
            source_kind="aggregated_measured",
            measurement_origin="interpolated_from_measured",
            native_interval_s=0.001,
            analysis_interval_s=0.1,
            resampling_method="mean_bin_downsample",
        )
        completed = _boolean(row.get("run_completed", True))
        record.update(
            {
                "run_completed": completed,
                "eligible": completed,
                "exclusion_reason": "" if completed else "run_completed=False",
                "request_rate": _number(row, "request_rate"),
                "num_prompts": _number(row, "num-prompts"),
                "burstiness": _number(row, "burstiness"),
                "duration_metadata_s": _number(row, "execution_time_seconds"),
                "mean_power_metadata_w": _number(row, "mean_power[W]"),
                "peak_power_metadata_w": _number(row, "peak_power[W]"),
            }
        )
        record["condition_id"] = _condition_id(record["workload_type"], record)
        yield record


def _whole_facility_records(root: Path, kind: str) -> Iterable[dict[str, Any]]:
    metadata_path = root / "03_whole-facility_profiles" / kind / "metadata.csv"
    metadata = _clean_columns(pd.read_csv(metadata_path))
    workload_type = "inference_whole_facility" if kind == "inference" else "colocation_mixed"
    for _, row in metadata.iterrows():
        relative_path = (
            f"03_whole-facility_profiles/{kind}/simulated_data/{row['name']}"
        )
        record = _base_record(
            relative_path,
            workload_type,
            model_name="DIPLOEE facility model",
            source_kind="whole_facility_simulated",
            measurement_origin="simulated",
            native_interval_s=60.0,
            analysis_interval_s=60.0,
            resampling_method="none",
        )
        record.update(
            {
                "target_utilization": _number(row, "Average Utilization"),
                "mean_power_metadata_w": (
                    value * 1e6
                    if (value := _number(row, "Power Mean (MW)")) is not None
                    else None
                ),
                "peak_power_metadata_w": (
                    value * 1e6
                    if (value := _number(row, "Power Max (MW)")) is not None
                    else None
                ),
                "request_rate": _number(row, "Avg Incoming Request Rate"),
            }
        )
        record["condition_id"] = _condition_id(record["workload_type"], record)
        yield record


def build_run_manifest(dataset_root: str | Path) -> pd.DataFrame:
    """Build and validate the run-to-profile manifest."""

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    records: list[dict[str, Any]] = []
    for builder in (_training_records, _offline_records, _finite_records, _rate_records):
        records.extend(builder(root))
    records.extend(_whole_facility_records(root, "inference"))
    records.extend(_whole_facility_records(root, "colocation"))

    manifest = pd.DataFrame.from_records(records)
    for column in MANIFEST_COLUMNS:
        if column not in manifest:
            manifest[column] = np.nan
    manifest = manifest[MANIFEST_COLUMNS].copy()

    manifest["path_exists"] = manifest["relative_path"].map(
        lambda value: (root / str(value)).is_file()
    )
    missing_paths = manifest.loc[~manifest["path_exists"], "relative_path"].tolist()
    if missing_paths:
        preview = missing_paths[:10]
        raise FileNotFoundError(f"Manifest references missing profiles: {preview}")
    if manifest["relative_path"].duplicated().any():
        duplicates = manifest.loc[
            manifest["relative_path"].duplicated(keep=False), "relative_path"
        ].tolist()
        raise ValueError(f"Duplicate profile mappings: {duplicates[:10]}")
    if manifest["profile_id"].duplicated().any():
        duplicates = manifest.loc[
            manifest["profile_id"].duplicated(keep=False), "profile_id"
        ].tolist()
        raise ValueError(f"Duplicate profile IDs: {duplicates[:10]}")

    return manifest.sort_values(
        ["workload_type", "relative_path"], kind="stable"
    ).reset_index(drop=True)


def save_run_manifest(
    manifest: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "run_manifest.csv"
    parquet_path = directory / "run_manifest.parquet"
    manifest.to_csv(csv_path, index=False, encoding="utf-8-sig")
    manifest.to_parquet(parquet_path, index=False)
    return {"csv": csv_path, "parquet": parquet_path}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/signal_validation")
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_run_manifest(args.data_root)
    paths = save_run_manifest(manifest, args.output_dir)
    print(
        manifest.groupby(["workload_type", "eligible"], observed=True)
        .size()
        .rename("profiles")
        .to_string()
    )
    print(f"\nManifest CSV: {paths['csv']}")
    print(f"Manifest Parquet: {paths['parquet']}")


if __name__ == "__main__":
    main()

