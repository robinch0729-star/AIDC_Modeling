"""Bottom-up aggregation of active and idle inference nodes."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd

from .node_power import M3R3NodePowerGenerator


def _column_name(service: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", service).strip("_").lower()


@dataclass(frozen=True)
class FacilityAggregationResult:
    profile: pd.DataFrame
    operations: pd.DataFrame


@dataclass
class FacilityAggregator:
    total_nodes: int
    node_power: M3R3NodePowerGenerator
    macro_interval_s: float = 60.0
    micro_interval_s: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        if self.total_nodes < 1:
            raise ValueError("total_nodes must be positive")
        if self.macro_interval_s <= 0 or self.micro_interval_s <= 0:
            raise ValueError("time intervals must be positive")
        ratio = self.macro_interval_s / self.micro_interval_s
        if not np.isclose(ratio, round(ratio), atol=1e-9, rtol=0.0):
            raise ValueError("macro_interval_s must be an integer multiple of micro_interval_s")

    def aggregate(self, plan: pd.DataFrame) -> FacilityAggregationResult:
        required = {
            "macro_index",
            "macro_time_s",
            "service",
            "prompt_dataset",
            "active_nodes",
            "request_rate_per_active_instance_rps",
        }
        missing = required.difference(plan.columns)
        if missing:
            raise ValueError(f"plan is missing columns {sorted(missing)}")
        micro_steps = int(round(self.macro_interval_s / self.micro_interval_s))
        rng = np.random.default_rng(self.seed)
        profiles: list[pd.DataFrame] = []
        operation_rows: list[dict[str, float | int]] = []

        for macro_index, group in plan.groupby("macro_index", sort=True, observed=True):
            active_nodes = int(group["active_nodes"].sum())
            idle_nodes = self.total_nodes - active_nodes
            if idle_nodes < 0:
                raise RuntimeError("active nodes exceed facility node count")
            idle_power_kw = idle_nodes * self.node_power.node_spec.idle_power_kw
            service_power: dict[str, np.ndarray] = {}
            active_power_kw = np.zeros(micro_steps, dtype=float)
            for row in group.itertuples(index=False):
                service_column = f"{_column_name(str(row.service))}_power_mw"
                value = self.node_power.aggregate_active_nodes(
                    prompt_dataset=str(row.prompt_dataset),
                    request_rate_rps=float(row.request_rate_per_active_instance_rps),
                    active_nodes=int(row.active_nodes),
                    length=micro_steps,
                    rng=rng,
                )
                service_power[service_column] = value / 1000.0
                active_power_kw += value

            start_s = float(group["macro_time_s"].iloc[0])
            time_s = start_s + np.arange(micro_steps, dtype=float) * self.micro_interval_s
            total_kw = active_power_kw + idle_power_kw
            profiles.append(
                pd.DataFrame(
                    {
                        "time_s": time_s,
                        "macro_index": int(macro_index),
                        "it_power_mw": total_kw / 1000.0,
                        "active_node_power_mw": active_power_kw / 1000.0,
                        "idle_node_power_mw": idle_power_kw / 1000.0,
                        "active_nodes": active_nodes,
                        "idle_nodes": idle_nodes,
                        **service_power,
                    }
                )
            )
            operation_rows.append(
                {
                    "macro_index": int(macro_index),
                    "macro_time_s": start_s,
                    "active_nodes": active_nodes,
                    "idle_nodes": idle_nodes,
                    "utilization": active_nodes / self.total_nodes,
                    "incoming_request_rate_rps": float(
                        group["incoming_request_rate_rps"].sum()
                    ),
                    "processed_request_rate_rps": float(
                        group["processed_request_rate_rps"].sum()
                    ),
                    "incomplete_request_rate_rps": float(
                        group["incomplete_request_rate_rps"].sum()
                    ),
                    "mean_it_power_mw": float(np.mean(total_kw) / 1000.0),
                }
            )
        return FacilityAggregationResult(
            profile=pd.concat(profiles, ignore_index=True),
            operations=pd.DataFrame(operation_rows),
        )
