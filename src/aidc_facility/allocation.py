"""DIPLOEE-style inference instance and node allocation.

Equations (3)--(7) of Paper list [3] motivate this module:
requests are converted to rates, parallel model instances are activated under
per-instance latency limits, request rate is divided uniformly, and excess
requests are recorded as incomplete.  Integer rounding is an implementation
detail added here because physical instance counts cannot be fractional.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import pandas as pd

from .traffic import ServiceSpec, TrafficDistribution, validate_services


def _largest_remainder(total: int, weights: np.ndarray) -> np.ndarray:
    if total < 1:
        raise ValueError("total must be positive")
    weight = np.asarray(weights, dtype=float)
    if np.any(weight < 0) or float(np.sum(weight)) <= 0:
        raise ValueError("weights must be nonnegative with a positive sum")
    raw = total * weight / float(np.sum(weight))
    allocated = np.floor(raw).astype(int)
    remainder = total - int(np.sum(allocated))
    if remainder:
        order = np.argsort(-(raw - allocated), kind="stable")
        allocated[order[:remainder]] += 1
    return allocated


@dataclass(frozen=True)
class InstanceAllocator:
    """Allocate a fixed node pool and dispatch model instances per macro step."""

    total_nodes: int
    services: tuple[ServiceSpec, ...]

    def __post_init__(self) -> None:
        if self.total_nodes < 1:
            raise ValueError("total_nodes must be positive")
        object.__setattr__(self, "services", validate_services(self.services))

    @property
    def node_allocation(self) -> pd.DataFrame:
        # Paper [3], Eq. (7): N_i is proportional to p_i*n_i/R_i,max.
        weights = np.array(
            [
                service.request_fraction
                * service.nodes_per_instance
                / service.max_request_rate_rps
                for service in self.services
            ],
            dtype=float,
        )
        pools = _largest_remainder(self.total_nodes, weights)
        return pd.DataFrame(
            [
                {
                    "service": service.name,
                    "prompt_dataset": service.prompt_dataset,
                    "request_fraction": service.request_fraction,
                    "max_request_rate_per_instance_rps": service.max_request_rate_rps,
                    "nodes_per_instance": service.nodes_per_instance,
                    "node_pool": int(pool),
                    "max_instances": int(pool // service.nodes_per_instance),
                }
                for service, pool in zip(self.services, pools, strict=True)
            ]
        )

    def plan(self, traffic: TrafficDistribution) -> pd.DataFrame:
        if tuple(traffic.services) != tuple(self.services):
            raise ValueError("traffic services do not match allocator services")
        pool = self.node_allocation.set_index("service")
        result: list[dict[str, float | int | str]] = []
        for row in traffic.by_service().itertuples(index=False):
            service_pool = pool.loc[row.service]
            maximum_instances = int(service_pool["max_instances"])
            incoming = float(row.incoming_request_rate_rps)
            maximum_rate = float(row.max_request_rate_per_instance_rps)
            required = 0 if incoming <= 0 else int(math.ceil(incoming / maximum_rate))
            active_instances = min(required, maximum_instances)
            process_capacity = active_instances * maximum_rate
            processed = min(incoming, process_capacity)
            incomplete = max(incoming - processed, 0.0)
            per_instance = processed / active_instances if active_instances else 0.0
            nodes_per_instance = int(row.nodes_per_instance)
            active_nodes = active_instances * nodes_per_instance
            result.append(
                {
                    **row._asdict(),
                    "node_pool": int(service_pool["node_pool"]),
                    "required_instances": required,
                    "active_instances": active_instances,
                    "active_nodes": active_nodes,
                    "processed_request_rate_rps": processed,
                    "incomplete_request_rate_rps": incomplete,
                    "request_rate_per_active_instance_rps": per_instance,
                }
            )
        plan = pd.DataFrame(result)
        active_by_step = plan.groupby("macro_index", observed=True)["active_nodes"].sum()
        if bool((active_by_step > self.total_nodes).any()):
            raise RuntimeError("active nodes exceed total_nodes")
        return plan

    def mean_utilization(self, traffic: TrafficDistribution) -> float:
        plan = self.plan(traffic)
        active = plan.groupby("macro_index", observed=True)["active_nodes"].sum()
        return float(active.mean() / self.total_nodes)


def calibrate_mean_request_rate(
    normalized_shape: np.ndarray,
    *,
    allocator: InstanceAllocator,
    target_utilization: float,
    macro_interval_s: float,
    tolerance: float = 2.5e-4,
    max_iterations: int = 80,
) -> tuple[TrafficDistribution, pd.DataFrame]:
    """Bisection-calibrate mean requests to a target node utilization.

    Paper [3] uses iterative matching between average daily requests and target
    utilization.  This is the executable short-horizon counterpart.
    """
    if not 0 < target_utilization < 1:
        raise ValueError("target_utilization must lie strictly between zero and one")

    def traffic_for(mean_rate: float) -> TrafficDistribution:
        return TrafficDistribution.from_normalized_shape(
            normalized_shape,
            mean_total_request_rate_rps=mean_rate,
            macro_interval_s=macro_interval_s,
            services=allocator.services,
            source="representative_shape_bisection_to_target_utilization",
        )

    low = 0.0
    high = sum(
        row.max_instances * row.max_request_rate_per_instance_rps
        for row in allocator.node_allocation.itertuples(index=False)
    )
    while allocator.mean_utilization(traffic_for(high)) < target_utilization:
        high *= 2.0
        if high > 1e12:
            raise RuntimeError("could not bracket target utilization")

    best = traffic_for(high)
    best_error = abs(allocator.mean_utilization(best) - target_utilization)
    for _ in range(max_iterations):
        midpoint = 0.5 * (low + high)
        candidate = traffic_for(midpoint)
        utilization = allocator.mean_utilization(candidate)
        error = abs(utilization - target_utilization)
        if error < best_error:
            best = candidate
            best_error = error
        if utilization < target_utilization:
            low = midpoint
        else:
            high = midpoint
        if error <= tolerance:
            break
    return best, allocator.plan(best)
