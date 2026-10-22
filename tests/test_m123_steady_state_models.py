"""Regression tests for the M1/M2/M3 steady-state comparison stage."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlr_data.steady_state_models import (  # noqa: E402
    build_condition_table,
    fit_model,
    split_conditions,
)


class M123SteadyStateTests(unittest.TestCase):
    def test_target_aliases_and_condition_aggregation(self) -> None:
        frame = pd.DataFrame(
            {
                "recommended_condition_id": ["a", "a", "b", "b"],
                "request_rate": [1, 1, 2, 2],
                "target_power_kw": [2.0, 2.2, 2.5, 2.7],
                "prompt_dataset": ["x", "x", "x", "x"],
            }
        )
        # The production builder intentionally demands more conditions; this
        # small fixture verifies aliases through a larger repeated construction.
        expanded = pd.concat(
            [frame.assign(
                recommended_condition_id=lambda x, i=i: x["recommended_condition_id"] + str(i),
                request_rate=lambda x, i=i: x["request_rate"] + 2 * i,
            ) for i in range(10)],
            ignore_index=True,
        )
        condition, resolved = build_condition_table(expanded)
        self.assertEqual(resolved["power"], "target_power_kw")
        self.assertAlmostEqual(
            condition.loc[condition["condition_id"].eq("a0"), "power"].item(), 2.1
        )

    def test_request_rate_split_has_no_cross_dataset_leakage(self) -> None:
        rows = []
        for dataset in ("a", "b"):
            for rate in range(1, 101):
                rows.append(
                    {
                        "condition_id": f"{dataset}:{rate}",
                        "prompt_dataset": dataset,
                        "request_rate": rate,
                        "power": 2 + 0.8 * (1 - np.exp(-rate / 20)),
                    }
                )
        condition = pd.DataFrame(rows)
        train, validation, test, assignments = split_conditions(condition)
        for rate, group in assignments.groupby("request_rate"):
            self.assertEqual(group["split"].nunique(), 1, msg=f"leakage at rate {rate}")
        self.assertEqual(set(test["request_rate"].unique()), set(range(81, 101)))
        self.assertTrue(set(train["request_rate"]).isdisjoint(validation["request_rate"]))

    def test_logistic_fit_is_monotone_and_saturating(self) -> None:
        rate = np.arange(1.0, 81.0)
        x = np.log1p(rate)
        power = 2.1 + 0.7 / (1 + np.exp(-4.0 * (x - 3.0)))
        model = fit_model(
            "M3_logistic", pd.DataFrame({"request_rate": rate, "power": power})
        )
        prediction = model.predict(np.linspace(1, 200, 1000))
        self.assertTrue(np.all(np.diff(prediction) >= -1e-10))
        self.assertLess(abs(model.parameters["p0"] - 2.1), 1e-3)
        self.assertLess(abs(model.parameters["delta_p"] - 0.7), 1e-3)


if __name__ == "__main__":
    unittest.main()
