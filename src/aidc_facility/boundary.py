"""Facility-boundary bookkeeping used before a dynamic cooling model exists."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FixedPUEBoundary:
    """Translate a target mean PCC load into a target mean IT load.

    PUE is fundamentally an energy ratio.  At this stage it is used only to set
    the mean IT target; it does not generate an instantaneous cooling profile.
    """

    target_pcc_mean_mw: float = 20.0
    pue: float = 1.15

    def __post_init__(self) -> None:
        if not np.isfinite(self.target_pcc_mean_mw) or self.target_pcc_mean_mw <= 0:
            raise ValueError("target_pcc_mean_mw must be finite and positive")
        if not np.isfinite(self.pue) or self.pue < 1.0:
            raise ValueError("pue must be finite and at least one")

    @property
    def target_it_mean_mw(self) -> float:
        return self.target_pcc_mean_mw / self.pue

    @property
    def implied_non_it_mean_mw(self) -> float:
        return self.target_pcc_mean_mw - self.target_it_mean_mw

    def energy_pue(self, it_power_mw: np.ndarray, interval_s: float) -> float:
        """Return the fixed-PUE proxy, for bookkeeping only."""
        it_energy = float(np.sum(np.asarray(it_power_mw, dtype=float)) * interval_s)
        pcc_energy = self.pue * it_energy
        return pcc_energy / it_energy
