"""Regression checks for the transparent, study-specific IEEE33 candidate.

These checks validate implementation and reproducibility, not physical cable
ratings, equipment design, grid-code compliance, or multi-period DOE feasibility.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pandapower.networks as pn
import pytest


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/modified_ieee33_candidate_v1"
SPEC = importlib.util.spec_from_file_location(
    "modified_ieee33_screen", ROOT / "scripts/screen_modified_ieee33_v1.py"
)
SCREEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCREEN)


@pytest.fixture(scope="module")
def cfg():
    return SCREEN.read_config()


@pytest.fixture(scope="module")
def profile(cfg):
    with np.load(ROOT / cfg["aidc"]["profile"]) as data:
        return data[cfg["aidc"]["profile_key"]].copy()


@pytest.fixture(scope="module")
def results():
    path = OUT / "results.json"
    assert path.is_file(), "Run screen_modified_ieee33_v1.py before artifact regression checks."
    return json.loads(path.read_text(encoding="utf-8"))


def test_original_topology_background_pq_and_physical_rx_preserved(cfg):
    raw = pn.case33bw()
    net = SCREEN.build_network(cfg)
    line_columns = [
        "from_bus", "to_bus", "length_km", "r_ohm_per_km",
        "x_ohm_per_km", "c_nf_per_km", "parallel", "in_service",
    ]
    pd.testing.assert_frame_equal(net.line[line_columns], raw.line[line_columns])
    pd.testing.assert_frame_equal(
        net.load[["bus", "p_mw", "q_mvar", "in_service"]],
        raw.load[["bus", "p_mw", "q_mvar", "in_service"]],
    )
    assert (net.load.scaling == cfg["network"]["background_load_scale"]).all()
    assert net.bus.index.equals(raw.bus.index)
    assert net.trafo.empty and raw.trafo.empty
    assert net.sgen.empty and net.gen.empty


def test_all_33_buses_connected_and_radial_with_open_ties(cfg):
    net = SCREEN.build_network(cfg)
    active = net.line.loc[net.line.in_service]
    assert len(net.bus) == 33
    assert len(active) == cfg["network"]["closed_branches"] == len(net.bus) - 1
    assert int((~net.line.in_service).sum()) == cfg["network"]["open_ties"]
    adjacent = {int(bus): set() for bus in net.bus.index}
    for line in active.itertuples():
        adjacent[int(line.from_bus)].add(int(line.to_bus))
        adjacent[int(line.to_bus)].add(int(line.from_bus))
    reached, pending = set(), [int(net.ext_grid.bus.iloc[0])]
    while pending:
        bus = pending.pop()
        if bus not in reached:
            reached.add(bus)
            pending.extend(adjacent[bus] - reached)
    assert reached == set(net.bus.index)


def test_33kv_is_a_physical_change_not_base_mva_relabeling(cfg):
    original = SCREEN.build_network(cfg, original_voltage=True)
    candidate = SCREEN.build_network(cfg)
    assert np.all(original.bus.vn_kv == cfg["network"]["original_voltage_kv"])
    assert np.all(candidate.bus.vn_kv == cfg["network"]["candidate_voltage_kv"])
    assert cfg["network"]["candidate_voltage_kv"] == 33.0
    assert original.sn_mva == candidate.sn_mva
    original_result = SCREEN.solve(original)
    candidate_result = SCREEN.solve(candidate)
    assert original_result["converged"] and candidate_result["converged"]
    assert candidate_result["min_voltage_pu"] > original_result["min_voltage_pu"]
    assert candidate_result["network_losses_mw"] < original_result["network_losses_mw"]


def test_calculation_base_10_vs_100_mva_preserves_physical_power_flow(cfg, profile):
    networks, checks = [], []
    for base_mva in (10.0, 100.0):
        net = SCREEN.build_network(cfg, base_mva=base_mva)
        SCREEN.add_aidc(net, cfg["aidc"]["base_case_connection_bus_1based"],
                        float(profile.mean()), cfg["aidc"]["facility_lagging_power_factor"])
        check = SCREEN.solve(net, head_limit_mva=cfg["network"]["feeder_head_apparent_power_limit_mva"])
        assert check["converged"]
        networks.append(net)
        checks.append(check)
    np.testing.assert_allclose(networks[0].res_bus.vm_pu, networks[1].res_bus.vm_pu,
                               rtol=0, atol=1e-8)
    for key in ("head_p_mw", "head_q_mvar", "network_losses_mw", "max_line_current_ka"):
        assert checks[0][key] == pytest.approx(checks[1][key], abs=1e-7)


def test_actual_current_limits_replace_unspecified_case_placeholder(cfg):
    net = SCREEN.build_network(cfg)
    assert cfg["network"]["uniform_branch_limit_ka"] == 0.6
    np.testing.assert_allclose(net.line.max_i_ka, cfg["network"]["uniform_branch_limit_ka"])
    assert (net.line.max_i_ka > 0).all()
    assert (net.line.max_i_ka < 99999).all()
    assert (net.line.max_loading_percent == 100).all()
    assert not cfg["network"]["original_generator_limits_used_as_equipment_ratings"]
    for col in ("min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"):
        if col in net.ext_grid:
            assert net.ext_grid[col].isna().all()


def test_open_ties_excluded_from_loading_criteria(cfg, monkeypatch):
    net = SCREEN.build_network(cfg)
    real_runpp = SCREEN.pp.runpp

    def run_with_unused_branch_sentinels(current, **kwargs):
        real_runpp(current, **kwargs)
        ties = current.line.index[~current.line.in_service]
        current.res_line.loc[ties, "i_ka"] = 99999.0
        current.res_line.loc[ties, "loading_percent"] = 99999.0

    monkeypatch.setattr(SCREEN.pp, "runpp", run_with_unused_branch_sentinels)
    result = SCREEN.solve(net)
    assert result["converged"]
    assert result["max_line_current_ka"] < 99999
    assert result["max_line_loading_percent"] < 99999
    assert not set(result["overloaded_branches"]) & set(net.line.loc[~net.line.in_service, "name"])


def test_head_rating_uses_apparent_not_only_active_power(cfg, profile):
    net = SCREEN.build_network(cfg)
    SCREEN.add_aidc(net, cfg["aidc"]["base_case_connection_bus_1based"],
                    float(profile.mean()), cfg["aidc"]["facility_lagging_power_factor"])
    initial = SCREEN.solve(net, head_limit_mva=float("inf"))
    assert initial["converged"]
    assert initial["head_s_mva"] == pytest.approx(
        math.hypot(initial["head_p_mw"], initial["head_q_mvar"])
    )
    assert initial["head_s_mva"] > initial["head_p_mw"]
    separating_limit = (initial["head_s_mva"] + initial["head_p_mw"]) / 2
    result = SCREEN.solve(net, head_limit_mva=separating_limit)
    assert result["head_p_mw"] < separating_limit < result["head_s_mva"]
    assert result["head_overloaded"] and not result["feasible"]


def test_frozen_pcc_boundary_no_added_site_transformer(cfg, profile, results):
    assert not cfg["aidc"]["profile_rescaled"]
    assert not cfg["network"]["explicit_upstream_transformer"]
    assert cfg["aidc"]["ess_q_mvar"] == 0
    assert results["aidc_mean_pcc_mw"] == pytest.approx(float(profile.mean()), abs=1e-12)
    assert results["aidc_peak_pcc_mw"] == pytest.approx(float(profile.max()), abs=1e-12)
    raw = SCREEN.build_network(cfg)
    n_bus, n_line = len(raw.bus), len(raw.line)
    idx = SCREEN.add_aidc(raw, cfg["aidc"]["base_case_connection_bus_1based"],
                          float(profile.max()), cfg["aidc"]["facility_lagging_power_factor"])
    assert len(raw.bus) == n_bus and len(raw.line) == n_line and raw.trafo.empty
    assert raw.load.at[idx, "p_mw"] == profile.max()
    assert raw.load.at[idx, "q_mvar"] == pytest.approx(
        profile.max() * math.tan(math.acos(cfg["aidc"]["facility_lagging_power_factor"]))
    )


def test_recorded_mean_peak_snapshots_reproduce(cfg, profile, results):
    for label, power in (("mean", float(profile.mean())), ("peak", float(profile.max()))):
        recorded = next(row for row in results["snapshots"]
                        if row["demand"] == label
                        and row["bus"] == cfg["aidc"]["base_case_connection_bus_1based"]
                        and row["voltage_kv"] == cfg["network"]["candidate_voltage_kv"])
        actual, _ = SCREEN.snapshot(cfg, recorded["bus"], power)
        assert actual["converged"] == recorded["converged"]
        assert actual["feasible"] == recorded["feasible"]
        for key in ("min_voltage_pu", "head_s_mva", "max_line_loading_percent"):
            assert actual[key] == pytest.approx(recorded[key], abs=1e-8)
        assert abs(actual["active_power_balance_error_mw"]) < 1e-6


def test_stored_replay_and_snapshot_power_balance(results, profile):
    replay = json.loads((OUT / "replay_5min.json").read_text(encoding="utf-8"))
    assert len(replay) == len(profile) == results["baseline_replay"]["intervals"]
    np.testing.assert_allclose([r["aidc_p_mw"] for r in replay], profile, rtol=0, atol=0)
    converged = [r for r in replay if "active_power_balance_error_mw" in r]
    assert len(converged) == len(replay), "Candidate base replay must supply every bus."
    assert max(abs(r["active_power_balance_error_mw"]) for r in converged) < 1e-6
    for row in results["snapshots"] + results["sensitivities"] + results["all_bus_peak_screen"]:
        if "active_power_balance_error_mw" in row:
            assert abs(row["active_power_balance_error_mw"]) < 1e-6
    assert results["baseline_replay"]["violation_intervals"] == sum(not r["feasible"] for r in replay)


def test_protected_facility_artifacts_and_candidate_contract_unchanged(cfg, results):
    assert results["protected_input_hashes"]
    assert all(results["protected_inputs_unchanged"].values())
    for relative, expected in results["protected_input_hashes"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    assert hashlib.sha256(SCREEN.CONFIG.read_bytes()).hexdigest() == results["config_sha256"]
    assert json.loads((OUT / "model_contract.json").read_text(encoding="utf-8")) == cfg
    assert not results["storage_schedules_changed"]


def test_bus2_is_declared_candidate_not_storage_optimized(cfg, results):
    assert cfg["aidc"]["base_case_connection_bus_1based"] == 2
    assert not results["main_bus_selected_from_peak_or_storage_results"]
    assert not cfg["screening"]["parameter_tuning_after_results"]
    assert not results["network_opf_or_doe_implemented"]
    assert cfg["prior_plan"]["family_previously_selected"]
    assert not cfg["prior_plan"]["numeric_modifications_previously_selected"]
    assert {r["bus"] for r in results["all_bus_peak_screen"]} == set(range(2, 34))
    assert {r["bus"] for r in results["static_limits_not_doe"]} == set(range(2, 34))
    assert "pending_user_review" in cfg["status"]
