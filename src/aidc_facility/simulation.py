"""Sizing and orchestration for the NLR-inspired bottom-up IT facility."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .aggregation import FacilityAggregationResult, FacilityAggregator
from .allocation import InstanceAllocator, calibrate_mean_request_rate
from .boundary import FixedPUEBoundary
from .node_power import M3R3NodePowerGenerator, NLRNodeSpec
from .traffic import NLR_INFERENCE_SERVICES, ServiceSpec, TrafficDistribution


@dataclass(frozen=True)
class FacilityCaseConfig:
    target_pcc_mean_mw: float = 20.0
    assumed_pue: float = 1.15
    target_utilization: float = 0.80
    macro_interval_s: float = 60.0
    micro_interval_s: float = 0.1
    seed: int = 42


@dataclass(frozen=True)
class SizingResult:
    total_nodes: int
    target_it_mean_mw: float
    deterministic_it_mean_mw: float
    mean_total_request_rate_rps: float
    achieved_mean_utilization: float
    node_allocation: pd.DataFrame
    traffic: TrafficDistribution
    plan: pd.DataFrame


@dataclass(frozen=True)
class FacilityCaseResult:
    config: FacilityCaseConfig
    boundary: FixedPUEBoundary
    sizing: SizingResult
    aggregation: FacilityAggregationResult


def _deterministic_mean_it_mw(
    plan: pd.DataFrame,
    *,
    total_nodes: int,
    node_power: M3R3NodePowerGenerator,
) -> float:
    values: list[float] = []
    for _, group in plan.groupby("macro_index", sort=True, observed=True):
        active_nodes = int(group["active_nodes"].sum())
        total_kw = (total_nodes - active_nodes) * node_power.node_spec.idle_power_kw
        for row in group.itertuples(index=False):
            total_kw += int(row.active_nodes) * node_power.steady_power_kw(
                str(row.prompt_dataset),
                float(row.request_rate_per_active_instance_rps),
            )
        values.append(total_kw)
    return float(np.mean(values) / 1000.0)


def size_to_mean_it_target(
    normalized_shape: np.ndarray,
    *,
    boundary: FixedPUEBoundary,
    node_power: M3R3NodePowerGenerator,
    target_utilization: float,
    macro_interval_s: float,
    services: tuple[ServiceSpec, ...] = NLR_INFERENCE_SERVICES,
) -> SizingResult:
    """Choose an integer node pool whose bottom-up mean meets the IT target."""
    service_weight = np.array(
        [
            service.request_fraction
            * service.nodes_per_instance
            / service.max_request_rate_rps
            for service in services
        ],
        dtype=float,
    )
    service_weight /= service_weight.sum()
    approximate_active_kw = float(
        np.dot(
            service_weight,
            [
                node_power.steady_power_kw(
                    service.prompt_dataset, service.max_request_rate_rps
                )
                for service in services
            ],
        )
    )
    approximate_node_mean_kw = (
        target_utilization * approximate_active_kw
        + (1.0 - target_utilization) * node_power.node_spec.idle_power_kw
    )
    total_nodes = max(
        1,
        int(round(boundary.target_it_mean_mw * 1000.0 / approximate_node_mean_kw)),
    )

    def evaluate(nodes: int) -> tuple[float, TrafficDistribution, pd.DataFrame, InstanceAllocator]:
        allocator = InstanceAllocator(nodes, services)
        traffic, plan = calibrate_mean_request_rate(
            normalized_shape,
            allocator=allocator,
            target_utilization=target_utilization,
            macro_interval_s=macro_interval_s,
        )
        mean_mw = _deterministic_mean_it_mw(
            plan, total_nodes=nodes, node_power=node_power
        )
        return mean_mw, traffic, plan, allocator

    for _ in range(12):
        mean_mw, _, _, _ = evaluate(total_nodes)
        updated = max(1, int(round(total_nodes * boundary.target_it_mean_mw / mean_mw)))
        if updated == total_nodes:
            break
        total_nodes = updated

    candidates: list[tuple[float, int, float, TrafficDistribution, pd.DataFrame, InstanceAllocator]] = []
    for nodes in range(max(1, total_nodes - 12), total_nodes + 13):
        mean_mw, traffic, plan, allocator = evaluate(nodes)
        candidates.append(
            (
                abs(mean_mw - boundary.target_it_mean_mw),
                nodes,
                mean_mw,
                traffic,
                plan,
                allocator,
            )
        )
    _, nodes, mean_mw, traffic, plan, allocator = min(candidates, key=lambda row: row[0])
    utilization = float(
        plan.groupby("macro_index", observed=True)["active_nodes"].sum().mean()
        / nodes
    )
    return SizingResult(
        total_nodes=nodes,
        target_it_mean_mw=boundary.target_it_mean_mw,
        deterministic_it_mean_mw=mean_mw,
        mean_total_request_rate_rps=float(np.mean(traffic.total_request_rate_rps)),
        achieved_mean_utilization=utilization,
        node_allocation=allocator.node_allocation,
        traffic=traffic,
        plan=plan,
    )


def run_facility_case(
    normalized_shape: np.ndarray,
    *,
    calibrated_generator: Any,
    config: FacilityCaseConfig = FacilityCaseConfig(),
    services: tuple[ServiceSpec, ...] = NLR_INFERENCE_SERVICES,
) -> FacilityCaseResult:
    boundary = FixedPUEBoundary(
        target_pcc_mean_mw=config.target_pcc_mean_mw,
        pue=config.assumed_pue,
    )
    node_power = M3R3NodePowerGenerator(
        calibrated_generator=calibrated_generator,
        node_spec=NLRNodeSpec(),
    )
    sizing = size_to_mean_it_target(
        normalized_shape,
        boundary=boundary,
        node_power=node_power,
        target_utilization=config.target_utilization,
        macro_interval_s=config.macro_interval_s,
        services=services,
    )
    aggregation = FacilityAggregator(
        total_nodes=sizing.total_nodes,
        node_power=node_power,
        macro_interval_s=config.macro_interval_s,
        micro_interval_s=config.micro_interval_s,
        seed=config.seed,
    ).aggregate(sizing.plan)
    return FacilityCaseResult(
        config=config,
        boundary=boundary,
        sizing=sizing,
        aggregation=aggregation,
    )


def config_as_dict(config: FacilityCaseConfig) -> dict[str, Any]:
    return asdict(config)
