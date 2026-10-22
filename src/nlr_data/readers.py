"""Canonical readers for NLR measured and simulated power timeseries."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .schema_registry import (
    SchemaSpec,
    identify_schema,
    infer_path_metadata,
)


DATETIME_FORMAT = "%Y-%m-%d_%H:%M:%S.%f"
SIGNAL_COLUMNS = ["timestamp", "time_s", "power_w", "utilization"]
METADATA_COLUMNS = [
    "schema_name",
    "source_kind",
    "measurement_scope",
    "workload_type",
    "model_name",
    "run_id",
    "relative_path",
]


class UnsupportedDataRoleError(ValueError):
    """Raised when a non-timeseries file is passed to ``read_timeseries``."""


class TimeseriesValidationError(ValueError):
    """Raised when a canonical timeseries fails a required validation check."""

    def __init__(self, report: pd.DataFrame):
        self.report = report
        messages = report.loc[report["severity"].eq("error"), "message"].tolist()
        super().__init__("; ".join(messages))


def _find_wattameter_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            candidate = line.strip().lstrip("#").strip()
            if "timestamp" in candidate.lower() and (
                "reading-time" in candidate.lower()
                or "[w]" in candidate.lower()
                or "[mw]" in candidate.lower()
            ):
                return re.split(r"\s+", candidate)
    raise ValueError(f"WattAMeter column header not found: {path}")


def _read_wattameter(
    path: Path,
    schema: SchemaSpec,
    max_rows: int | None,
) -> pd.DataFrame:
    columns = _find_wattameter_header(path)
    raw = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=columns,
        nrows=max_rows,
        engine="python",
    )

    if schema.time_column not in raw.columns:
        raise ValueError(f"Missing time column {schema.time_column!r}: {path}")
    timestamp = pd.to_datetime(
        raw[schema.time_column],
        format=DATETIME_FORMAT,
        errors="coerce",
    )

    component_pattern = re.compile(schema.power_component_pattern or r"$^")
    components = [column for column in raw.columns if component_pattern.fullmatch(column)]
    if not components:
        raise ValueError(f"No registered power components found: {path}")

    numeric_components = raw[components].apply(pd.to_numeric, errors="coerce")
    power_w = numeric_components.sum(axis=1, min_count=1) * schema.power_scale_to_w

    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "power_w": power_w.astype(float),
            "utilization": np.nan,
        }
    )
    frame["time_s"] = (frame["timestamp"] - frame["timestamp"].iloc[0]).dt.total_seconds()
    return frame[SIGNAL_COLUMNS]


def _read_parquet_sample(path: Path, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None:
        return pd.read_parquet(path)

    parquet_file = pq.ParquetFile(path)
    try:
        batch = next(parquet_file.iter_batches(batch_size=max_rows))
    except StopIteration:
        return pd.DataFrame()
    return batch.to_pandas()


def _materialize_named_index(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame.index, pd.RangeIndex) and any(
        name is not None for name in frame.index.names
    ):
        return frame.reset_index()
    return frame


def _read_aggregated(
    path: Path,
    schema: SchemaSpec,
    max_rows: int | None,
) -> pd.DataFrame:
    raw = _materialize_named_index(_read_parquet_sample(path, max_rows))
    required = [schema.time_column, schema.power_column]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"Missing aggregated columns {missing}: {path}")

    time_s = pd.to_numeric(raw[schema.time_column], errors="coerce")
    if time_s.notna().any():
        time_s = time_s - time_s.dropna().iloc[0]

    frame = pd.DataFrame(
        {
            "timestamp": pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]"),
            "time_s": time_s.astype(float),
            "power_w": (
                pd.to_numeric(raw[schema.power_column], errors="coerce")
                * schema.power_scale_to_w
            ).astype(float),
            "utilization": np.nan,
        }
    )
    return frame[SIGNAL_COLUMNS].reset_index(drop=True)


def _read_whole_facility(
    path: Path,
    schema: SchemaSpec,
    max_rows: int | None,
) -> pd.DataFrame:
    raw = pd.read_csv(path, nrows=max_rows)
    required = [schema.time_column, schema.power_column, schema.utilization_column]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"Missing whole-facility columns {missing}: {path}")

    timestamp = pd.to_datetime(raw[schema.time_column], errors="coerce", format="mixed")
    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "power_w": (
                pd.to_numeric(raw[schema.power_column], errors="coerce")
                * schema.power_scale_to_w
            ).astype(float),
            "utilization": pd.to_numeric(
                raw[schema.utilization_column], errors="coerce"
            ).astype(float),
        }
    )
    frame["time_s"] = (frame["timestamp"] - frame["timestamp"].iloc[0]).dt.total_seconds()
    return frame[SIGNAL_COLUMNS]


def validate_timeseries(frame: pd.DataFrame) -> pd.DataFrame:
    """Return validation checks with error/warning severity."""

    checks: list[dict[str, Any]] = []

    def add(check: str, severity: str, count: int, message: str) -> None:
        checks.append(
            {"check": check, "severity": severity, "count": int(count), "message": message}
        )

    missing_columns = [column for column in SIGNAL_COLUMNS if column not in frame.columns]
    if missing_columns:
        add(
            "required_columns",
            "error",
            len(missing_columns),
            f"Missing canonical columns: {missing_columns}",
        )
        return pd.DataFrame(checks)

    add("nonempty", "error" if frame.empty else "pass", int(frame.empty), "Timeseries is empty")
    if frame.empty:
        return pd.DataFrame(checks)

    invalid_time = int(frame["time_s"].isna().sum())
    add(
        "valid_time",
        "error" if invalid_time else "pass",
        invalid_time,
        "time_s contains missing or unparseable values",
    )

    finite_time = pd.to_numeric(frame["time_s"], errors="coerce")
    non_monotonic = int(not finite_time.is_monotonic_increasing)
    add(
        "monotonic_time",
        "error" if non_monotonic else "pass",
        non_monotonic,
        "time_s is not monotonically increasing",
    )

    duplicate_time = int(finite_time.duplicated().sum())
    add(
        "duplicate_time",
        "warning" if duplicate_time else "pass",
        duplicate_time,
        "time_s contains duplicate samples",
    )

    power = pd.to_numeric(frame["power_w"], errors="coerce")
    invalid_power = int((~np.isfinite(power)).sum())
    add(
        "finite_power",
        "error" if invalid_power == len(frame) else ("warning" if invalid_power else "pass"),
        invalid_power,
        "power_w contains non-finite values",
    )

    negative_power = int((power < 0).sum())
    add(
        "nonnegative_power",
        "warning" if negative_power else "pass",
        negative_power,
        "power_w contains negative values",
    )

    return pd.DataFrame(checks)


def read_timeseries(
    path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    max_rows: int | None = None,
    include_metadata_columns: bool = True,
    validate: bool = True,
) -> pd.DataFrame:
    """Read one registered NLR power timeseries into a common schema.

    The returned signal always contains ``timestamp``, ``time_s``, ``power_w``,
    and ``utilization``. A missing absolute timestamp or utilization is encoded
    as ``NaT``/``NaN`` rather than fabricated.
    """

    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be greater than zero")

    schema = identify_schema(file_path, dataset_root)
    if schema.role != "timeseries":
        raise UnsupportedDataRoleError(
            f"{file_path.name} has role {schema.role!r}, not 'timeseries'"
        )

    if schema.name in {"raw_nvml_power_log", "raw_rapl_power_log"}:
        frame = _read_wattameter(file_path, schema, max_rows)
    elif schema.name == "aggregated_it_power":
        frame = _read_aggregated(file_path, schema, max_rows)
    elif schema.name == "whole_facility_power":
        frame = _read_whole_facility(file_path, schema, max_rows)
    else:
        raise UnsupportedDataRoleError(f"No reader implemented for {schema.name}")

    frame = frame.reset_index(drop=True)
    path_metadata = infer_path_metadata(file_path, dataset_root)
    frame.attrs.update(
        {
            **path_metadata,
            "schema_name": schema.name,
            "source_kind": schema.source_kind,
            "measurement_scope": schema.measurement_scope,
            "input_power_unit": schema.input_power_unit,
            "canonical_power_unit": "W",
            "power_aggregation": schema.power_aggregation,
        }
    )

    if include_metadata_columns:
        metadata_values = {
            "schema_name": schema.name,
            "source_kind": schema.source_kind,
            "measurement_scope": schema.measurement_scope,
            "workload_type": path_metadata["workload_type"],
            "model_name": path_metadata["model_name"],
            "run_id": path_metadata["run_id"],
            "relative_path": path_metadata["relative_path"],
        }
        for column, value in metadata_values.items():
            frame[column] = value

    if validate:
        report = validate_timeseries(frame)
        if report["severity"].eq("error").any():
            raise TimeseriesValidationError(report)
        frame.attrs["validation_report"] = report.to_dict(orient="records")

    ordered_columns = SIGNAL_COLUMNS + (
        METADATA_COLUMNS if include_metadata_columns else []
    )
    return frame[ordered_columns]


def summarize_timeseries(frame: pd.DataFrame) -> dict[str, Any]:
    """Create a compact validation/feature-readiness summary."""

    time_s = pd.to_numeric(frame["time_s"], errors="coerce")
    power = pd.to_numeric(frame["power_w"], errors="coerce")
    deltas = time_s.diff().dropna()
    positive_deltas = deltas[deltas > 0]

    summary: dict[str, Any] = {
        "rows": len(frame),
        "duration_s": float(time_s.max() - time_s.min()) if time_s.notna().any() else None,
        "median_interval_s": (
            float(positive_deltas.median()) if not positive_deltas.empty else None
        ),
        "min_interval_s": float(positive_deltas.min()) if not positive_deltas.empty else None,
        "max_interval_s": float(positive_deltas.max()) if not positive_deltas.empty else None,
        "mean_power_w": float(power.mean()) if power.notna().any() else None,
        "min_power_w": float(power.min()) if power.notna().any() else None,
        "max_power_w": float(power.max()) if power.notna().any() else None,
        "missing_power_count": int(power.isna().sum()),
        "duplicate_time_count": int(time_s.duplicated().sum()),
    }
    for key in [
        "schema_name",
        "source_kind",
        "measurement_scope",
        "workload_type",
        "model_name",
        "run_id",
        "relative_path",
    ]:
        summary[key] = frame.attrs.get(key)
    return summary

