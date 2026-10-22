"""Node-level M3+R3 replacement for the empirical P(r) block in DIPLOEE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nlr_data.residual_model_selection import simulate_residual_ensemble


@dataclass(frozen=True)
class NLRNodeSpec:
    """NLR Kestrel GPU-node values used by Paper list [3]."""

    rated_compute_power_kw: float = 3.520
    idle_power_kw: float = 0.420

    def __post_init__(self) -> None:
        if self.idle_power_kw <= 0 or self.rated_compute_power_kw <= self.idle_power_kw:
            raise ValueError("node power bounds are invalid")


@dataclass
class M3R3NodePowerGenerator:
    """Aggregate independent node profiles without equivalent MW scaling.

    Paper [3] assigns measured P(request-rate) samples to active nodes.  This
    study replaces only that block with the calibrated M3+R3 generator.  The
    independent residual assumption is explicit because [3] does not document
    cross-node residual correlation.
    """

    calibrated_generator: Any
    node_spec: NLRNodeSpec = NLRNodeSpec()
    residual_model: str = "R3_jump_ou"
    chunk_nodes: int = 256

    def __post_init__(self) -> None:
        if self.chunk_nodes < 1:
            raise ValueError("chunk_nodes must be positive")

    def steady_power_kw(self, prompt_dataset: str, request_rate_rps: float) -> float:
        if prompt_dataset not in self.calibrated_generator.m3_parameters:
            raise KeyError(f"unknown prompt dataset {prompt_dataset!r}")
        value = float(
            self.calibrated_generator.m3_parameters[prompt_dataset].predict(
                request_rate_rps
            )
        )
        return float(
            np.clip(
                value,
                self.node_spec.idle_power_kw,
                self.node_spec.rated_compute_power_kw,
            )
        )

    def aggregate_active_nodes(
        self,
        *,
        prompt_dataset: str,
        request_rate_rps: float,
        active_nodes: int,
        length: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if active_nodes < 0 or length < 1:
            raise ValueError("active_nodes must be nonnegative and length positive")
        if active_nodes == 0:
            return np.zeros(length, dtype=float)
        if request_rate_rps <= 0:
            raise ValueError("active nodes require a positive request rate")
        steady = self.steady_power_kw(prompt_dataset, request_rate_rps)
        aggregate = np.zeros(length, dtype=float)
        remaining = active_nodes
        while remaining:
            draws = min(remaining, self.chunk_nodes)
            residual = simulate_residual_ensemble(
                self.residual_model,
                draws=draws,
                length=length,
                prompt_dataset=prompt_dataset,
                request_rate=float(request_rate_rps),
                library=self.calibrated_generator.residual_library,
                rng=rng,
            )
            node_power = np.clip(
                steady + residual,
                self.node_spec.idle_power_kw,
                self.node_spec.rated_compute_power_kw,
            )
            aggregate += np.sum(node_power, axis=0)
            remaining -= draws
        return aggregate
