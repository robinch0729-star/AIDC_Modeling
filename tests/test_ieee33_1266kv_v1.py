"""Native-voltage replay checks, not physical connection approval or DOE proof."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandapower as pp
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "ieee33_native_screen", SCRIPTS / "screen_ieee33_1266kv_v1.py"
)
SCREEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCREEN)
CORE = SCREEN.core
OUT = ROOT / "outputs/ieee33_1266kv_screen_v1"


@pytest.fixture(scope="module")
def cfg():
    return SCREEN.read_config()


@pytest.fixture(scope="module")
def result():
    path = OUT / "results.json"
    assert path.is_file(), "Run screen_ieee33_1266kv_v1.py before artifact checks."
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def source(cfg):
    with np.load(ROOT / cfg["aidc"]["profile"]) as data:
        return {key: data[key].copy() for key in data.files}


def test_native_voltage_topology_physical_impedance_and_background_preserved(cfg):
    raw = CORE.original_network()
    net = CORE.build_network(cfg)
    assert len(net.bus) == 33
    assert int(net.line.in_service.sum()) == 32
    assert int((~net.line.in_service).sum()) == 5
    assert np.all(net.bus.vn_kv == 12.66)
    columns = ["from_bus", "to_bus", "length_km", "r_ohm_per_km",
               "x_ohm_per_km", "c_nf_per_km", "parallel", "in_service"]
    pd.testing.assert_frame_equal(net.line[columns], raw.line[columns])
    pd.testing.assert_frame_equal(
        net.load[["bus", "p_mw", "q_mvar", "in_service"]],
        raw.load[["bus", "p_mw", "q_mvar", "in_service"]],
    )
    assert (net.load.scaling == 1).all()
    assert net.load.p_mw.sum() == pytest.approx(3.715)
    assert net.load.q_mvar.sum() == pytest.approx(2.3)
    assert net.trafo.empty and net.gen.empty and net.sgen.empty


def test_new_config_does_not_mutate_previous_33kv_candidate(cfg):
    original_path = ROOT / cfg["native_voltage_test"]["inherits"]
    before = original_path.read_bytes()
    old = json.loads(before)
    assert old["network"]["candidate_voltage_kv"] == 33.0
    assert cfg["network"]["candidate_voltage_kv"] == 12.66
    assert cfg["network"]["voltage_min_pu"] == 0.95
    assert cfg["network"]["voltage_max_pu"] == 1.05
    assert cfg["network"]["uniform_branch_limit_ka"] == 0.6
    assert cfg["network"]["feeder_head_apparent_power_limit_mva"] == 40.0
    cfg["network"]["candidate_voltage_kv"] = 12.66  # No source-file aliasing.
    assert original_path.read_bytes() == before
    assert SCREEN.read_config()["network"]["candidate_voltage_kv"] == 12.66


def test_ess_q_is_zero_and_facility_pf_does_not_apply_to_net_import():
    pnet = np.array([8.0, 8.0, 8.0])
    charge = np.array([2.0, 0.0, 0.0])
    discharge = np.array([0.0, 2.0, 0.0])
    factor = math.tan(math.acos(0.98))
    actual = SCREEN.pcc_q(pnet, charge, discharge, 0.98)
    np.testing.assert_allclose(actual, np.array([6.0, 10.0, 8.0]) * factor)
    assert actual[0] < actual[2] < actual[1]
    np.testing.assert_array_equal(SCREEN.pcc_q(pnet, charge, discharge, 1.0), 0.0)


def test_empty_summary_is_defined_without_fake_extrema():
    summary = SCREEN.summarize([])
    for key in ("intervals", "converged_intervals", "unresolved_intervals",
                "converged_violation_intervals", "feasible_intervals", "duration_h"):
        assert summary[key] == 0
    for key in ("pcc_mean_mw", "pcc_peak_mw", "pcc_min_mw",
                "min_voltage_pu", "max_balance_error_mw", "minimum_pcc_voltage_pu"):
        assert summary[key] is None


def test_solver_failures_are_unresolved_not_certified_constraint_violations():
    rows = [{"converged": False, "feasible": False, "pcc_p_mw": 20.0},
            {"converged": False, "feasible": False, "pcc_p_mw": 21.0}]
    summary = SCREEN.summarize(rows)
    assert summary["unresolved_intervals"] == 2
    assert summary["converged_violation_intervals"] == 0
    assert summary["converged_violation_hours"] == 0
    assert summary["unresolved_hours"] == pytest.approx(2 / 12)
    assert summary["min_voltage_pu"] is None
    assert summary["max_balance_error_mw"] is None
    assert summary["pcc_mean_mw"] == 20.5


def test_mixed_summary_separates_feasible_violating_and_unresolved():
    good = dict(converged=True, feasible=True, pcc_p_mw=20.0,
                min_voltage_pu=0.96, max_line_current_ka=0.3,
                max_line_loading_percent=50.0, head_s_mva=25.0,
                active_power_balance_error_mw=-1e-9, pcc_voltage_pu=0.99,
                under_voltage_buses=[], overloaded_branches=[], head_overloaded=False)
    bad = dict(good, feasible=False, pcc_p_mw=21.0, min_voltage_pu=0.94,
               under_voltage_buses=[18], overloaded_branches=["Branch 1-2"],
               max_line_loading_percent=110.0)
    unresolved = dict(converged=False, feasible=False, pcc_p_mw=22.0)
    summary = SCREEN.summarize([good, bad, unresolved])
    assert summary["intervals"] == 3
    assert summary["converged_intervals"] == 2
    assert summary["feasible_intervals"] == 1
    assert summary["converged_violation_intervals"] == 1
    assert summary["unresolved_intervals"] == 1
    assert summary["under_voltage_intervals"] == summary["line_overload_intervals"] == 1
    assert summary["head_overload_intervals"] == 0
    assert summary["min_voltage_pu"] == 0.94
    assert summary["max_balance_error_mw"] == 1e-9


def test_study_and_source_voltage_criteria_remain_separate(cfg, result):
    study = result["no_aidc_study_bounds"]
    diagnostic = result["no_aidc_source_voltage_bounds_diagnostic"]
    assert study["converged"] and diagnostic["converged"]
    assert study["min_voltage_pu"] == pytest.approx(0.9130904816, abs=1e-8)
    assert diagnostic["min_voltage_pu"] == pytest.approx(study["min_voltage_pu"], abs=1e-10)
    assert not study["feasible"] and study["under_voltage_buses"]
    assert diagnostic["feasible"] and not diagnostic["under_voltage_buses"]
    assert cfg["network"]["voltage_min_pu"] == 0.95


def test_infeasible_background_has_null_not_zero_hosting_boundary(cfg, result):
    for recorded in result["static_limits_not_doe"]:
        assert recorded["baseline_feasible"] is False
        assert recorded["certified_p_mw"] is None
        assert recorded["coarse_flags"][0] is False
    assert {r["bus"] for r in result["static_limits_not_doe"]} == {2, 6, 18}
    fresh = CORE.static_limit(cfg, 2)
    assert fresh["baseline_feasible"] is False
    assert fresh["certified_p_mw"] is None
    assert result["opf_or_doe_implemented"] is False


def test_all_frozen_intervals_and_correct_storage_q_replayed(cfg, result, source):
    prefix = cfg["native_voltage_test"]["frozen_storage_schedule_prefix"]
    p0 = source["no_storage_pcc_mw"]
    pj = source[prefix + "_pcc_mw"]
    q0 = SCREEN.pcc_q(p0, 0, 0, 0.98)
    qj = SCREEN.pcc_q(pj, source[prefix + "_ess_charge_mw"],
                     source[prefix + "_ess_discharge_mw"], 0.98)
    for name, expected_p, expected_q, summary_key in (
        ("no_storage", p0, q0, "no_storage_replay"),
        ("existing_joint", pj, qj, "existing_joint_replay"),
    ):
        rows = json.loads((OUT / (name + "_replay.json")).read_text(encoding="utf-8"))
        assert len(rows) == len(expected_p) == 1438
        assert [r["interval"] for r in rows] == list(range(len(rows)))
        np.testing.assert_allclose([r["pcc_p_mw"] for r in rows], expected_p, rtol=0, atol=0)
        np.testing.assert_allclose([r["pcc_q_mvar"] for r in rows], expected_q, rtol=0, atol=0)
        np.testing.assert_allclose([r["time_h"] for r in rows], source["time_h"], rtol=0, atol=1e-12)
        assert SCREEN.summarize(rows) == result[summary_key]
        assert all(r["converged"] for r in rows)
        assert max(abs(r["active_power_balance_error_mw"]) for r in rows) < 1e-6
        assert result[summary_key]["unresolved_intervals"] == 0


def test_necessary_power_bound_dimensions_and_optimistic_assumptions(cfg, result, source):
    bound = result["necessary_power_bound"]
    n = cfg["network"]
    # kV * kA = MVA. Ignore positive Q and losses for a necessary active bound.
    root_mva = math.sqrt(3) * 12.66 * n["slack_voltage_pu"] * 0.6
    assert bound["root_line_mva_bound"] == pytest.approx(root_mva)
    assert bound["background_p_mw"] == pytest.approx(3.715)
    expected_cap = root_mva - 3.715
    assert bound["optimistic_aidc_active_power_upper_bound_mw"] == pytest.approx(expected_cap)
    storage = json.loads((ROOT / cfg["aidc"]["summary"]).read_text(encoding="utf-8"))
    tes = json.loads((ROOT / "configs/pcc_tes_operating_candidate_v1.json").read_text(encoding="utf-8"))
    # MW + (MWe/MWth)*MWth is a power limit, NOT an energy/duration guarantee.
    reduction = storage["ess_config"]["discharge_limit_mw"] + (
        storage["coefficients"]["reduction_mwe_per_mwth"] * tes["tes"]["discharge_limit_mw_th"]
    )
    assert bound["instantaneous_storage_reduction_upper_bound_mw"] == pytest.approx(reduction)
    optimistic = source["no_storage_pcc_mw"] - reduction
    assert bound["optimistic_after_storage_pcc_min_mw"] == pytest.approx(float(optimistic.min()))
    assert bound["intervals_failing_necessary_bound_even_with_max_discharge"] == int(np.sum(optimistic > expected_cap))
    assert bound["root_current_lower_bound_at_best_pcc_ka"] == pytest.approx(
        (optimistic.min() + 3.715) / (math.sqrt(3) * 12.66)
    )
    assert "ignores SOC and duration" in bound["interpretation"]
    assert "not an achievable schedule" in bound["interpretation"]


def test_independent_nr_bfsw_agreement_at_peak(cfg, result, source):
    check = result["independent_solver_check"]
    assert check["converged"]
    assert check["maximum_voltage_difference_pu"] < 1e-6
    actual, net = CORE.snapshot(cfg, 2, float(source["no_storage_pcc_mw"].max()))
    assert actual["converged"]
    nr_voltages = net.res_bus.vm_pu.to_numpy().copy()
    pp.runpp(net, algorithm="bfsw", init="flat", numba=False, max_iteration=200, tolerance_mva=1e-8)
    assert net.converged
    np.testing.assert_allclose(net.res_bus.vm_pu.to_numpy(), nr_voltages, rtol=0, atol=1e-6)


def test_frozen_sources_and_previous_candidate_files_are_hash_preserved(cfg, result):
    hashes = result["protected_input_hashes"]
    assert hashes and all(result["protected_inputs_unchanged"].values())
    for path, expected in hashes.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    assert any(Path(path).as_posix().endswith("modified_ieee33_candidate_v1.json") for path in hashes)
    assert any(Path(path).name == "schedules_5min.npz" for path in hashes)
    assert hashlib.sha256(SCREEN.CONFIG.read_bytes()).hexdigest() == result["config_sha256"]
    assert json.loads((OUT / "model_contract.json").read_text(encoding="utf-8")) == cfg
    for key in ("storage_reoptimized", "aidc_rescaled", "background_rescaled", "network_reinforced"):
        assert result[key] is False


def test_all_non_slack_locations_screened_without_claiming_collapse(result):
    rows = result["all_non_slack_bus_peak_screen"]
    assert {r["bus"] for r in rows} == set(range(2, 34))
    for row in rows:
        if not row["converged"]:
            assert "not a proof of physical voltage collapse" in row["failure"]
        else:
            assert abs(row["active_power_balance_error_mw"]) < 1e-6

