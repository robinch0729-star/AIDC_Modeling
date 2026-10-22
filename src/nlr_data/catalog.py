"""Build a lightweight, read-only catalog of the NLR GenAI power dataset.

The cataloger intentionally avoids loading complete data files:

* Parquet files: schema/footer metadata and one bounded record batch
* CSV/log files: header and a bounded number of sample rows
* JSON files: bounded line-delimited sample, or a small-file fallback

The resulting table is the data contract for later workload feature extraction.
It identifies time, power, workload, and temperature columns without assigning
physical meaning that has not yet been verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm


DEFAULT_DATA_GROUPS = (
    "00_raw_datasets",
    "01_aggregated_datasets",
    "03_whole-facility_profiles",
)
DEFAULT_EXTENSIONS = (".parquet", ".csv", ".log", ".json", ".txt")
EXCLUDED_DIRECTORIES = {".git", ".venv", "__pycache__", "plots"}

TIME_PATTERN = re.compile(r"timestamp|date|datetime|elapsed|timestep|(^|_)time($|_)", re.I)
POWER_PATTERN = re.compile(r"power|watt|energy|\[w\]|\[kw\]|\[mw\]", re.I)
WORKLOAD_PATTERN = re.compile(
    r"utili[sz]ation|request|rate|token|latency|throughput|prompt|batch|"
    r"sequence|concurren|job|iteration|epoch",
    re.I,
)
TEMPERATURE_PATTERN = re.compile(r"temp|temperature|thermal", re.I)

CATALOG_COLUMNS = [
    "dataset_group",
    "source_kind",
    "workload_type",
    "model_name",
    "measurement_device",
    "experiment_id",
    "scale_mw",
    "node_count",
    "target_utilization",
    "relative_path",
    "extension",
    "file_size_bytes",
    "modified_utc",
    "row_count",
    "column_count",
    "column_names",
    "dtypes",
    "schema_signature",
    "time_columns",
    "power_columns",
    "workload_columns",
    "temperature_columns",
    "primary_time_column",
    "sample_interval_seconds",
    "sample_rows_read",
    "sample_max_missing_fraction",
    "sample_duplicate_time_count",
    "inspection_status",
    "inspection_error",
]


def _json_dump(value: Any) -> str:
    """Serialize catalog list/dict fields consistently."""

    return json.dumps(value, ensure_ascii=False, sort_keys=isinstance(value, dict))


def _matching_columns(columns: Sequence[str], pattern: re.Pattern[str]) -> list[str]:
    return [str(column) for column in columns if pattern.search(str(column))]


def _schema_signature(columns: Sequence[str], dtypes: dict[str, str]) -> str:
    payload = "|".join(f"{column}:{dtypes.get(column, '')}" for column in columns)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _extract_first_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.I)
    return float(match.group(1)) if match else None


def _extract_first_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text, flags=re.I)
    return int(match.group(1)) if match else None


def _classify_path(relative_path: Path) -> dict[str, Any]:
    """Derive non-numerical experiment labels from the repository path."""

    path_text = relative_path.as_posix().lower()
    filename = relative_path.name.lower()
    dataset_group = relative_path.parts[0] if relative_path.parts else "unknown"

    source_kind = {
        "00_raw_datasets": "raw_measured",
        "01_aggregated_datasets": "aggregated_measured",
        "03_whole-facility_profiles": "whole_facility_simulated",
    }.get(dataset_group, "unknown")

    if "colocation" in path_text:
        workload_type = "colocation_mixed"
    elif "inference_online_rate" in path_text or "online_rate" in path_text:
        workload_type = "online_inference_rate"
    elif "inference_online_finite" in path_text or "online_finite" in path_text:
        workload_type = "online_inference_finite"
    elif "inference_offline" in path_text or "offline" in path_text:
        workload_type = "offline_inference"
    elif "training_stable_diffusion" in path_text:
        workload_type = "stable_diffusion_training"
    elif "training" in path_text or "fine" in path_text or "lora" in path_text:
        workload_type = "training_or_finetuning"
    elif "inference" in path_text:
        workload_type = "inference_whole_facility"
    else:
        workload_type = "unknown"

    if "llama3" in path_text:
        model_name = "Llama3"
    elif "llama2" in path_text or "lora" in path_text:
        model_name = "Llama2"
    elif "stable_diffusion" in path_text:
        model_name = "Stable Diffusion"
    else:
        model_name = "unknown"

    if "nvml" in filename:
        measurement_device = "GPU_NVML"
    elif "rapl" in filename:
        measurement_device = "CPU_RAPL"
    elif dataset_group == "03_whole-facility_profiles":
        measurement_device = "whole_facility_model"
    else:
        measurement_device = "unspecified"

    experiment_match = re.search(r"slurmid_(\d+)", filename)
    if experiment_match:
        experiment_id = experiment_match.group(1)
    elif relative_path.stem.isdigit():
        experiment_id = relative_path.stem
    else:
        experiment_id = ""

    return {
        "dataset_group": dataset_group,
        "source_kind": source_kind,
        "workload_type": workload_type,
        "model_name": model_name,
        "measurement_device": measurement_device,
        "experiment_id": experiment_id,
        "scale_mw": _extract_first_float(r"(\d+(?:\.\d+)?)mw", path_text),
        "node_count": _extract_first_int(r"(\d+)nodes", path_text),
        "target_utilization": (
            value / 100.0
            if (value := _extract_first_float(r"_(\d+(?:\.\d+)?)u(?:_|\.)", filename))
            is not None
            else None
        ),
    }


def _infer_sample_interval_seconds(series: pd.Series) -> float | None:
    """Infer a median positive sampling interval from a bounded sample."""

    clean = series.dropna()
    if len(clean) < 2:
        return None

    if pd.api.types.is_numeric_dtype(clean):
        numeric = pd.to_numeric(clean, errors="coerce")
        deltas = numeric.diff().dropna()
    else:
        parsed = pd.to_datetime(clean, errors="coerce", utc=True, format="mixed")
        if parsed.notna().sum() < 2:
            return None
        deltas = parsed.diff().dt.total_seconds().dropna()

    positive = pd.to_numeric(deltas, errors="coerce")
    positive = positive[np.isfinite(positive) & (positive > 0)]
    return float(positive.median()) if not positive.empty else None


def _frame_metadata(frame: pd.DataFrame) -> dict[str, Any]:
    # Pandas restores Parquet index columns (for example ``timestep[s]``) as
    # an index. Materialize named indexes so time detection sees the complete
    # logical schema rather than only the value columns.
    analysis_frame = frame
    index_names = [name for name in frame.index.names if name is not None]
    if index_names and not isinstance(frame.index, pd.RangeIndex):
        analysis_frame = frame.reset_index()

    columns = [str(column) for column in analysis_frame.columns]
    dtypes = {
        str(column): str(dtype) for column, dtype in analysis_frame.dtypes.items()
    }
    time_columns = _matching_columns(columns, TIME_PATTERN)
    power_columns = _matching_columns(columns, POWER_PATTERN)
    workload_columns = _matching_columns(columns, WORKLOAD_PATTERN)
    temperature_columns = _matching_columns(columns, TEMPERATURE_PATTERN)
    primary_time_column = time_columns[0] if time_columns else ""

    if analysis_frame.empty:
        max_missing = None
    else:
        max_missing = float(analysis_frame.isna().mean().max())

    duplicate_time_count: int | None = None
    sample_interval: float | None = None
    if primary_time_column and primary_time_column in analysis_frame:
        duplicate_time_count = int(
            analysis_frame[primary_time_column].duplicated().sum()
        )
        sample_interval = _infer_sample_interval_seconds(
            analysis_frame[primary_time_column]
        )

    return {
        "column_count": len(columns),
        "column_names": _json_dump(columns),
        "dtypes": _json_dump(dtypes),
        "schema_signature": _schema_signature(columns, dtypes),
        "time_columns": _json_dump(time_columns),
        "power_columns": _json_dump(power_columns),
        "workload_columns": _json_dump(workload_columns),
        "temperature_columns": _json_dump(temperature_columns),
        "primary_time_column": primary_time_column,
        "sample_interval_seconds": sample_interval,
        "sample_rows_read": len(analysis_frame),
        "sample_max_missing_fraction": max_missing,
        "sample_duplicate_time_count": duplicate_time_count,
    }


def _inspect_parquet(path: Path, sample_rows: int) -> dict[str, Any]:
    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema_arrow
    columns = [str(name) for name in schema.names]
    dtypes = {str(field.name): str(field.type) for field in schema}

    try:
        batch = next(parquet_file.iter_batches(batch_size=sample_rows))
        frame = batch.to_pandas()
    except StopIteration:
        frame = pd.DataFrame(columns=columns)

    metadata = _frame_metadata(frame)
    metadata.update(
        {
            "row_count": int(parquet_file.metadata.num_rows),
            "column_count": len(columns),
            "column_names": _json_dump(columns),
            "dtypes": _json_dump(dtypes),
            "schema_signature": _schema_signature(columns, dtypes),
        }
    )
    return metadata


def _read_wattameter_log(path: Path, sample_rows: int) -> pd.DataFrame:
    """Read a bounded sample from a WattAMeter whitespace-delimited log."""

    header = ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            candidate = line.strip().lstrip("#").strip()
            if "timestamp" in candidate.lower() and (
                "reading-time" in candidate.lower()
                or "[w]" in candidate.lower()
                or "[mw]" in candidate.lower()
            ):
                header = candidate
                break

    if not header:
        return pd.DataFrame()

    columns = re.split(r"\s+", header)
    return pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=columns,
        nrows=sample_rows,
        engine="python",
    )


def _is_wattameter_log(path: Path) -> bool:
    """Return True only for the structured WattAMeter measurement logs."""

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle):
            header = line.lower()
            if "timestamp" in header and (
                "reading-time" in header or "[w]" in header or "[mw]" in header
            ):
                return True
            if line_number >= 19:
                break
    return False


def _inspect_delimited(path: Path, sample_rows: int) -> dict[str, Any]:
    if path.suffix.lower() == ".log":
        frame = _read_wattameter_log(path, sample_rows)
    else:
        frame = pd.read_csv(path, nrows=sample_rows, sep=None, engine="python")

    metadata = _frame_metadata(frame)
    metadata["row_count"] = None  # Avoid a full-file line count during cataloging.
    return metadata


def _inspect_json(path: Path, sample_rows: int) -> dict[str, Any]:
    if path.stat().st_size <= 5_000_000:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                frame = pd.json_normalize(payload)
            elif isinstance(payload, list):
                frame = pd.json_normalize(payload[:sample_rows])
            else:
                frame = pd.DataFrame({"value": [payload]})
        except json.JSONDecodeError:
            frame = pd.read_json(path, lines=True, nrows=sample_rows)
    else:
        frame = pd.read_json(path, lines=True, nrows=sample_rows)

    metadata = _frame_metadata(frame)
    metadata["row_count"] = len(frame) if path.stat().st_size <= 5_000_000 else None
    return metadata


def _empty_inspection() -> dict[str, Any]:
    return {
        "row_count": None,
        "column_count": None,
        "column_names": _json_dump([]),
        "dtypes": _json_dump({}),
        "schema_signature": "",
        "time_columns": _json_dump([]),
        "power_columns": _json_dump([]),
        "workload_columns": _json_dump([]),
        "temperature_columns": _json_dump([]),
        "primary_time_column": "",
        "sample_interval_seconds": None,
        "sample_rows_read": 0,
        "sample_max_missing_fraction": None,
        "sample_duplicate_time_count": None,
    }


def _iter_data_files(
    dataset_root: Path,
    groups: Sequence[str],
    extensions: set[str],
) -> Iterable[Path]:
    for group in groups:
        group_root = dataset_root / group
        if not group_root.is_dir():
            continue

        for current_root, directories, filenames in os.walk(group_root):
            directories[:] = [
                directory
                for directory in directories
                if directory not in EXCLUDED_DIRECTORIES
            ]
            for filename in filenames:
                path = Path(current_root) / filename
                if path.suffix.lower() in extensions:
                    yield path


def inspect_file(path: Path, dataset_root: Path, sample_rows: int = 2_048) -> dict[str, Any]:
    """Inspect one file and return a single catalog record."""

    relative_path = path.relative_to(dataset_root)
    stat = path.stat()
    record: dict[str, Any] = {
        **_classify_path(relative_path),
        "relative_path": relative_path.as_posix(),
        "extension": path.suffix.lower(),
        "file_size_bytes": int(stat.st_size),
        "modified_utc": pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").isoformat(),
        **_empty_inspection(),
        "inspection_status": "ok",
        "inspection_error": "",
    }

    try:
        if path.suffix.lower() == ".parquet":
            record.update(_inspect_parquet(path, sample_rows))
        elif path.suffix.lower() == ".log" and not _is_wattameter_log(path):
            record["inspection_status"] = "metadata_only"
        elif path.suffix.lower() in {".csv", ".log", ".txt"}:
            record.update(_inspect_delimited(path, sample_rows))
        elif path.suffix.lower() == ".json":
            record.update(_inspect_json(path, sample_rows))
        else:
            record["inspection_status"] = "metadata_only"
    except Exception as exc:  # File-level failures must not abort a full inventory.
        record["inspection_status"] = "error"
        record["inspection_error"] = f"{type(exc).__name__}: {exc}"

    return record


def build_catalog(
    dataset_root: str | Path,
    *,
    sample_rows: int = 2_048,
    groups: Sequence[str] = DEFAULT_DATA_GROUPS,
    extensions: Sequence[str] = DEFAULT_EXTENSIONS,
    max_files: int | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Build a catalog DataFrame without loading complete NLR data files.

    Parameters
    ----------
    dataset_root:
        Root directory containing the NLR repository folders.
    sample_rows:
        Maximum rows read from each file for time-step and quality diagnostics.
    groups:
        Dataset groups to scan. Analysis scripts are excluded by default.
    extensions:
        Data extensions to include.
    max_files:
        Optional smoke-test limit. ``None`` catalogs every matching file.
    show_progress:
        Display a tqdm progress bar.
    """

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"NLR dataset root does not exist: {root}")
    if sample_rows <= 0:
        raise ValueError("sample_rows must be greater than zero")

    normalized_extensions = {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in extensions
    }
    paths = sorted(_iter_data_files(root, groups, normalized_extensions))
    if max_files is not None:
        paths = paths[:max_files]

    records = [
        inspect_file(path, root, sample_rows=sample_rows)
        for path in tqdm(paths, desc="Cataloging NLR files", disable=not show_progress)
    ]
    catalog = pd.DataFrame.from_records(records, columns=CATALOG_COLUMNS)
    if not catalog.empty:
        catalog = catalog.sort_values(
            ["dataset_group", "workload_type", "relative_path"],
            kind="stable",
        ).reset_index(drop=True)
    return catalog


def save_catalog(
    catalog: pd.DataFrame,
    output_dir: str | Path,
    *,
    stem: str = "nlr_file_catalog",
) -> dict[str, Path]:
    """Save the catalog in human-readable CSV and typed Parquet formats."""

    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f"{stem}.csv"
    parquet_path = directory / f"{stem}.parquet"
    catalog.to_csv(csv_path, index=False, encoding="utf-8-sig")
    catalog.to_parquet(parquet_path, index=False)
    return {"csv": csv_path, "parquet": parquet_path}


def summarize_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    """Return a compact group/workload-level coverage and quality summary."""

    if catalog.empty:
        return pd.DataFrame(
            columns=[
                "dataset_group",
                "source_kind",
                "workload_type",
                "file_count",
                "total_size_mb",
                "inspection_success_rate",
                "files_with_time_column",
                "files_with_power_column",
            ]
        )

    working = catalog.copy()
    working["file_size_mb"] = working["file_size_bytes"] / (1024**2)
    working["inspection_ok"] = working["inspection_status"].eq("ok")
    working["has_time"] = working["primary_time_column"].fillna("").ne("")
    working["has_power"] = working["power_columns"].fillna("[]").ne("[]")

    return (
        working.groupby(
            ["dataset_group", "source_kind", "workload_type"],
            dropna=False,
            observed=True,
        )
        .agg(
            file_count=("relative_path", "size"),
            total_size_mb=("file_size_mb", "sum"),
            inspection_success_rate=("inspection_ok", "mean"),
            files_with_time_column=("has_time", "sum"),
            files_with_power_column=("has_power", "sum"),
        )
        .reset_index()
        .sort_values(["dataset_group", "workload_type"], kind="stable")
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="NLR dataset root")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/data_audit"),
        help="Catalog output directory",
    )
    parser.add_argument("--sample-rows", type=int, default=2_048)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    catalog = build_catalog(
        args.data_root,
        sample_rows=args.sample_rows,
        max_files=args.max_files,
        show_progress=not args.no_progress,
    )
    paths = save_catalog(catalog, args.output_dir)
    print(summarize_catalog(catalog).to_string(index=False))
    print(f"\nCSV catalog: {paths['csv']}")
    print(f"Parquet catalog: {paths['parquet']}")


if __name__ == "__main__":
    main()
