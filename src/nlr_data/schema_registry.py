"""Authoritative schema and unit registry for the NLR GenAI power dataset.

The rules in this module mirror the NLR repository's own ``postprocess.py``
scripts. They keep measured component power, aggregated IT power, and simulated
whole-facility power explicitly separated.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class UnknownSchemaError(ValueError):
    """Raised when a dataset file does not match a registered schema."""


@dataclass(frozen=True, slots=True)
class SchemaSpec:
    """Declarative description of one NLR file schema."""

    name: str
    role: str
    source_kind: str
    path_pattern: str
    time_column: str | None = None
    time_kind: str | None = None
    power_column: str | None = None
    power_component_pattern: str | None = None
    input_power_unit: str | None = None
    power_scale_to_w: float = 1.0
    power_aggregation: str | None = None
    utilization_column: str | None = None
    measurement_scope: str | None = None
    expected_interval_seconds: float | None = None
    description: str = ""

    def matches(self, relative_path: str) -> bool:
        return re.search(self.path_pattern, relative_path, flags=re.I) is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SCHEMA_REGISTRY: tuple[SchemaSpec, ...] = (
    SchemaSpec(
        name="raw_nvml_power_log",
        role="timeseries",
        source_kind="raw_measured",
        path_pattern=r"^00_raw_datasets/.*/nvml[^/]*\.log$",
        time_column="timestamp",
        time_kind="absolute_datetime",
        power_component_pattern=r"^gpu-\d+\[mW\]$",
        input_power_unit="mW",
        power_scale_to_w=1e-3,
        power_aggregation="sum_gpu_components",
        measurement_scope="GPU_power_only",
        expected_interval_seconds=None,
        description=(
            "WattAMeter/NVML per-GPU socket power. GPU mW columns are converted "
            "to W and summed, following the NLR aggregation scripts. Raw "
            "sampling is approximately 0.2 s for training and 0.1 s for "
            "inference in the current release."
        ),
    ),
    SchemaSpec(
        name="raw_rapl_power_log",
        role="timeseries",
        source_kind="raw_measured",
        path_pattern=r"^00_raw_datasets/.*/rapl[^/]*\.log$",
        time_column="timestamp",
        time_kind="absolute_datetime",
        power_component_pattern=r"^cpu-\d+(?:-core)?\[W\]$",
        input_power_unit="W",
        power_scale_to_w=1.0,
        power_aggregation="sum_cpu_package_and_core_components",
        measurement_scope="CPU_power_only",
        expected_interval_seconds=None,
        description=(
            "WattAMeter/RAPL CPU power. The four CPU W columns are summed to "
            "reproduce the NLR postprocessing implementation. Raw sampling is "
            "workload-dependent."
        ),
    ),
    SchemaSpec(
        name="aggregated_it_power",
        role="timeseries",
        source_kind="aggregated_measured",
        path_pattern=r"^01_aggregated_datasets/.*/results/\d+\.parquet$",
        time_column="timestep[s]",
        time_kind="elapsed_seconds",
        power_column="power[W]",
        input_power_unit="W",
        power_scale_to_w=1.0,
        power_aggregation="NLR_preaggregated_GPU_plus_CPU",
        measurement_scope="job_total_IT_power",
        expected_interval_seconds=None,
        description=(
            "NLR-preaggregated measured IT power. The elapsed-time field is "
            "stored as a pandas/Parquet index in most files. Current nominal "
            "steps are 0.2 s (training), 0.1 s (offline/online-finite), and "
            "0.001 s (online-rate)."
        ),
    ),
    SchemaSpec(
        name="whole_facility_power",
        role="timeseries",
        source_kind="whole_facility_simulated",
        path_pattern=(
            r"^03_whole-facility_profiles/(?:inference|colocation)/"
            r"simulated_data/.*_power\.csv$"
        ),
        time_column="timestamp",
        time_kind="absolute_datetime",
        power_column="power_W",
        input_power_unit="W",
        power_scale_to_w=1.0,
        power_aggregation="DIPLOEE_whole_facility_model",
        utilization_column="utilization",
        measurement_scope="whole_facility_power",
        expected_interval_seconds=60.0,
        description=(
            "Synthetic whole-facility profile generated with NLR DIPLOEE; "
            "not a direct PCC measurement."
        ),
    ),
    SchemaSpec(
        name="aggregated_metadata",
        role="metadata",
        source_kind="aggregated_measured",
        path_pattern=r"^01_aggregated_datasets/.*/metadata\.csv$",
        description="Run-level workload, model, timing, and power summary metadata.",
    ),
    SchemaSpec(
        name="whole_facility_metadata",
        role="metadata",
        source_kind="whole_facility_simulated",
        path_pattern=r"^03_whole-facility_profiles/.*/metadata\.csv$",
        description="DIPLOEE profile-level statistics and workload-rate metadata.",
    ),
    SchemaSpec(
        name="raw_configuration",
        role="configuration",
        source_kind="raw_measured",
        path_pattern=r"^00_raw_datasets/.*/config\.json$",
        description="Experiment and Slurm configuration, not a power timeseries.",
    ),
    SchemaSpec(
        name="raw_structured_workload_log",
        role="metadata",
        source_kind="raw_measured",
        path_pattern=r"^00_raw_datasets/.*/(?:log|log_vllm)\.json$",
        description="JSON/JSONL workload execution and performance records.",
    ),
    SchemaSpec(
        name="raw_unstructured_log",
        role="auxiliary_log",
        source_kind="raw_measured",
        path_pattern=r"^00_raw_datasets/.*\.log$",
        description="Non-tabular vLLM/server log retained only as auxiliary metadata.",
    ),
)


def dataset_relative_path(
    path: str | Path,
    dataset_root: str | Path | None = None,
) -> str:
    """Return a normalized path beginning with an NLR dataset group."""

    candidate = Path(path).expanduser()
    if dataset_root is not None:
        root = Path(dataset_root).expanduser().resolve()
        try:
            return candidate.resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Path is outside the NLR dataset root: {candidate}") from exc

    parts = candidate.parts
    for index, part in enumerate(parts):
        if part in {
            "00_raw_datasets",
            "01_aggregated_datasets",
            "03_whole-facility_profiles",
        }:
            return Path(*parts[index:]).as_posix()
    return candidate.as_posix()


def identify_schema(
    path: str | Path,
    dataset_root: str | Path | None = None,
) -> SchemaSpec:
    """Resolve one file to the first matching registered schema."""

    relative_path = dataset_relative_path(path, dataset_root)
    for schema in SCHEMA_REGISTRY:
        if schema.matches(relative_path):
            return schema
    raise UnknownSchemaError(f"No registered NLR schema matches: {relative_path}")


def get_schema(name: str) -> SchemaSpec:
    for schema in SCHEMA_REGISTRY:
        if schema.name == name:
            return schema
    raise UnknownSchemaError(f"Unknown schema name: {name}")


def infer_path_metadata(
    path: str | Path,
    dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    """Infer workload labels and run identifiers without reading file content."""

    relative_path = dataset_relative_path(path, dataset_root)
    text = relative_path.lower()
    filename = Path(relative_path).name.lower()

    if "colocation" in text:
        workload_type = "colocation_mixed"
    elif "inference_online_rate" in text:
        workload_type = "online_inference_rate"
    elif "inference_online_finite" in text:
        workload_type = "online_inference_finite"
    elif "inference_offline" in text:
        workload_type = "offline_inference"
    elif "training_stable_diffusion" in text:
        workload_type = "stable_diffusion_training"
    elif "training" in text or "lora" in text:
        workload_type = "training_or_finetuning"
    elif "inference" in text:
        workload_type = "inference_whole_facility"
    else:
        workload_type = "unknown"

    if "llama3" in text:
        model_name = "Llama3"
    elif "llama2" in text or "lora" in text:
        model_name = "Llama2"
    elif "stable_diffusion" in text:
        model_name = "Stable Diffusion"
    else:
        model_name = "unknown"

    slurm_match = re.search(r"slurmid_(\d+)", filename)
    run_id = (
        slurm_match.group(1)
        if slurm_match
        else Path(filename).stem
    )
    scale_match = re.search(r"(\d+(?:\.\d+)?)mw", filename, flags=re.I)
    nodes_match = re.search(r"(\d+)nodes", filename, flags=re.I)
    utilization_match = re.search(r"_(\d+(?:\.\d+)?)u(?:_|\.)", filename, flags=re.I)

    return {
        "relative_path": relative_path,
        "workload_type": workload_type,
        "model_name": model_name,
        "run_id": run_id,
        "scale_mw": float(scale_match.group(1)) if scale_match else None,
        "node_count": int(nodes_match.group(1)) if nodes_match else None,
        "target_utilization": (
            float(utilization_match.group(1)) / 100.0
            if utilization_match
            else None
        ),
    }


def registry_records() -> list[dict[str, Any]]:
    """Return the registry as serializable dictionaries for documentation."""

    return [schema.to_dict() for schema in SCHEMA_REGISTRY]
