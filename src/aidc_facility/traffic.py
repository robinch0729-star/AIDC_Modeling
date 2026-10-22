"""Inference traffic inputs for the bottom-up facility model.

The service split and per-instance rate limits reproduce the inference case in
Paper list [3], Section 4.2.  The representative short-horizon shape is only a
deterministic test driver; it is not claimed to reproduce the proprietary
Azure temporal distribution used by [3].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ServiceSpec:
    """One inference service/prompt class in the DIPLOEE-style dispatcher."""

    name: str
    prompt_dataset: str
    request_fraction: float
    max_request_rate_rps: float
    nodes_per_instance: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("service name must not be empty")
        if not self.prompt_dataset:
            raise ValueError("prompt_dataset must not be empty")
        if not np.isfinite(self.request_fraction) or self.request_fraction <= 0:
            raise ValueError("request_fraction must be finite and positive")
        if (
            not np.isfinite(self.max_request_rate_rps)
            or self.max_request_rate_rps <= 0
        ):
            raise ValueError("max_request_rate_rps must be finite and positive")
        if self.nodes_per_instance < 1:
            raise ValueError("nodes_per_instance must be at least one")


# Paper list [3], Section 4.2: coding/conversation split 38.1/61.9%,
# rate limits 100/50 request/s, and one model instance per node.
NLR_INFERENCE_SERVICES: tuple[ServiceSpec, ...] = (
    ServiceSpec(
        name="coding",
        prompt_dataset="InstructCoder",
        request_fraction=0.381,
        max_request_rate_rps=100.0,
    ),
    ServiceSpec(
        name="conversation",
        prompt_dataset="mlperf-inference-llama2-data",
        request_fraction=0.619,
        max_request_rate_rps=50.0,
    ),
)


def validate_services(services: Sequence[ServiceSpec]) -> tuple[ServiceSpec, ...]:
    result = tuple(services)
    if not result:
        raise ValueError("at least one service is required")
    names = [service.name for service in result]
    if len(set(names)) != len(names):
        raise ValueError("service names must be unique")
    fraction = sum(service.request_fraction for service in result)
    if not np.isclose(fraction, 1.0, atol=1e-9, rtol=0.0):
        raise ValueError(f"request fractions must sum to one, got {fraction:g}")
    return result


@dataclass(frozen=True)
class TrafficDistribution:
    """Facility-wide request-rate time series and its service split."""

    macro_time_s: np.ndarray
    total_request_rate_rps: np.ndarray
    macro_interval_s: float
    services: tuple[ServiceSpec, ...] = NLR_INFERENCE_SERVICES
    source: str = "user_supplied"

    def __post_init__(self) -> None:
        time = np.asarray(self.macro_time_s, dtype=float)
        rate = np.asarray(self.total_request_rate_rps, dtype=float)
        services = validate_services(self.services)
        if time.ndim != 1 or rate.ndim != 1 or len(time) != len(rate):
            raise ValueError("macro_time_s and total_request_rate_rps must be 1-D and aligned")
        if len(time) < 1:
            raise ValueError("traffic series must contain at least one macro step")
        if not np.all(np.isfinite(time)) or not np.all(np.isfinite(rate)):
            raise ValueError("traffic values must be finite")
        if np.any(rate < 0):
            raise ValueError("request rates must be nonnegative")
        if not np.isfinite(self.macro_interval_s) or self.macro_interval_s <= 0:
            raise ValueError("macro_interval_s must be finite and positive")
        object.__setattr__(self, "macro_time_s", time)
        object.__setattr__(self, "total_request_rate_rps", rate)
        object.__setattr__(self, "services", services)

    @classmethod
    def from_normalized_shape(
        cls,
        normalized_shape: np.ndarray,
        *,
        mean_total_request_rate_rps: float,
        macro_interval_s: float,
        services: Sequence[ServiceSpec] = NLR_INFERENCE_SERVICES,
        source: str = "representative_shape",
    ) -> "TrafficDistribution":
        shape = np.asarray(normalized_shape, dtype=float)
        if shape.ndim != 1 or len(shape) < 1:
            raise ValueError("normalized_shape must be a nonempty 1-D array")
        if np.any(~np.isfinite(shape)) or np.any(shape < 0):
            raise ValueError("normalized_shape must be finite and nonnegative")
        if float(np.mean(shape)) <= 0:
            raise ValueError("normalized_shape must have a positive mean")
        if mean_total_request_rate_rps < 0:
            raise ValueError("mean_total_request_rate_rps must be nonnegative")
        shape = shape / float(np.mean(shape))
        return cls(
            macro_time_s=np.arange(len(shape), dtype=float) * macro_interval_s,
            total_request_rate_rps=shape * float(mean_total_request_rate_rps),
            macro_interval_s=float(macro_interval_s),
            services=tuple(services),
            source=source,
        )

    def by_service(self) -> pd.DataFrame:
        rows: list[dict[str, float | int | str]] = []
        for macro_index, (time_s, total_rate) in enumerate(
            zip(self.macro_time_s, self.total_request_rate_rps, strict=True)
        ):
            for service in self.services:
                rows.append(
                    {
                        "macro_index": macro_index,
                        "macro_time_s": float(time_s),
                        "service": service.name,
                        "prompt_dataset": service.prompt_dataset,
                        "request_fraction": service.request_fraction,
                        "requests_in_macro_interval": (
                            float(total_rate)
                            * service.request_fraction
                            * self.macro_interval_s
                        ),
                        "incoming_request_rate_rps": (
                            float(total_rate) * service.request_fraction
                        ),
                        "max_request_rate_per_instance_rps": (
                            service.max_request_rate_rps
                        ),
                        "nodes_per_instance": service.nodes_per_instance,
                    }
                )
        return pd.DataFrame(rows)


def representative_short_horizon_shape(steps: int) -> np.ndarray:
    """Return a smooth, mean-one traffic shape for executable smoke studies.

    This shape is deliberately labelled as a general test input rather than an
    NLR/Azure reproduction.  It can later be replaced without changing the
    DIPLOEE-style allocation and aggregation modules.
    """
    if steps < 2:
        raise ValueError("steps must be at least two")
    phase = np.arange(steps, dtype=float) / steps
    shape = (
        1.0
        + 0.16 * np.sin(2.0 * np.pi * (phase - 0.20))
        + 0.06 * np.sin(4.0 * np.pi * (phase + 0.10))
    )
    shape = np.maximum(shape, 0.05)
    return shape / float(np.mean(shape))
