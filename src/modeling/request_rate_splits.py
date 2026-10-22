"""Leakage-safe train/test assignments grouped by request rate."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_request_rate_splits(
    targets: pd.DataFrame,
    *,
    interpolation_stride: int = 5,
    extrapolation_fraction: float = 0.20,
) -> pd.DataFrame:
    """Build full-fit, interpolation, and high-rate extrapolation scenarios.

    The same request rate always receives the same partition across both prompt
    datasets. This prevents the pooled model from seeing a held-out rate through
    the other prompt dataset.
    """

    if "request_rate" not in targets:
        raise ValueError("Targets must contain request_rate")
    rates = np.sort(pd.to_numeric(targets["request_rate"], errors="coerce").dropna().unique())
    if len(rates) < 10:
        raise ValueError("At least ten unique request rates are required")
    if interpolation_stride < 3:
        raise ValueError("interpolation_stride must be at least three")
    if not 0.0 < extrapolation_fraction < 0.5:
        raise ValueError("extrapolation_fraction must lie between zero and 0.5")

    rows: list[dict[str, object]] = []
    for rate in rates:
        rows.append(
            {"split_scheme": "full_fit", "request_rate": rate, "partition": "train"}
        )

    for index, rate in enumerate(rates):
        is_endpoint = index in {0, len(rates) - 1}
        is_holdout = ((index + 1) % interpolation_stride == 0) and not is_endpoint
        rows.append(
            {
                "split_scheme": "interpolation",
                "request_rate": rate,
                "partition": "test" if is_holdout else "train",
            }
        )

    test_count = max(1, int(np.ceil(len(rates) * extrapolation_fraction)))
    cutoff_index = len(rates) - test_count
    for index, rate in enumerate(rates):
        rows.append(
            {
                "split_scheme": "high_rate_extrapolation",
                "request_rate": rate,
                "partition": "test" if index >= cutoff_index else "train",
            }
        )

    result = pd.DataFrame(rows)
    for scheme, group in result.groupby("split_scheme", observed=True):
        if group["request_rate"].duplicated().any():
            raise RuntimeError(f"Duplicate rate assignment in {scheme}")
        if scheme != "full_fit" and set(group["partition"]) != {"train", "test"}:
            raise RuntimeError(f"Split {scheme} lacks train or test cases")
    return result.sort_values(
        ["split_scheme", "request_rate"], kind="stable"
    ).reset_index(drop=True)


def attach_split(targets: pd.DataFrame, assignments: pd.DataFrame, scheme: str) -> pd.DataFrame:
    """Attach one split scheme to target rows with many-to-one rate validation."""

    selected = assignments.loc[assignments["split_scheme"].eq(scheme)].copy()
    if selected.empty:
        raise ValueError(f"Unknown split scheme: {scheme}")
    merged = targets.merge(
        selected[["request_rate", "partition"]],
        on="request_rate",
        how="left",
        validate="many_to_one",
    )
    if merged["partition"].isna().any():
        raise ValueError(f"Some target rows were not assigned in split {scheme}")
    return merged

