"""Adoption, trace provenance, energy alignment and conditional AC checks."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import assess_ieee33_time_varying_background_v1 as s


@pytest.fixture(scope="module")
def config():
    return s.read_config()


@pytest.fixture(scope="module")
def results():
    return json.loads((s.OUT/"results.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def profiles():
    return pd.read_csv(s.OUT/"background_profiles_5min.csv")


def test_confirmed_reference_comparison_and_registry(config):
    cfg,ref=config
    registry=json.loads((ROOT/"configs/grid_testbed_registry.json").read_text(encoding="utf-8"))
    assert ref["reference"]["candidate_id"]==registry["reference_candidate_id"]=="cu400_n3_v1.040"
    assert ref["comparison"]["candidate_id"]==registry["comparison_candidate_id"]=="cu630_n2_v1.040"
    assert "user_confirmed" in ref["status"]
    assert registry["current_reference_config"]==cfg["network_reference"]
    assert ref["fixed"]["nominal_voltage_kv"]==12.66


@pytest.mark.parametrize("role",["reference","comparison"])
def test_network_equivalence_and_only_root_changed(config,role):
    _,ref=config
    n=s.build_network(ref,role); raw=s.previous.core.original_network()
    cols=["from_bus","to_bus","r_ohm_per_km","x_ohm_per_km","length_km","in_service"]
    pd.testing.assert_frame_equal(n.line.loc[1:,cols],raw.line.loc[1:,cols])
    assert n.load.p_mw.sum()==pytest.approx(3.715)
    assert n.load.q_mvar.sum()==pytest.approx(2.3)
    assert len(n.bus)==33 and n.line.in_service.sum()==32
    assert n.trafo.empty and n.sgen.empty


def test_five_minute_conversion_preserves_energy():
    x=np.array([1.,2.,4.,3.])
    y=s.expand_and_trim(x,12,0)
    np.testing.assert_array_equal(y.reshape(-1,3),np.repeat(x[:,None],3,axis=1))
    assert y.sum()/12==pytest.approx(x.sum()/4)
    trimmed=s.expand_and_trim(x,10,1)
    assert trimmed.sum()/12==pytest.approx(x.sum()/4-(x[0]+x[-1])/12)


@pytest.mark.parametrize("x",[[1,np.nan],[1,-1]])
def test_invalid_source_rejected(x):
    with pytest.raises(ValueError): s.expand_and_trim(x,3,0)


def test_source_audit_timezone_anomalies_not_silently_repaired(results):
    a=results["background_profile_audit"]
    assert a["source_package"]=="1.6.3"
    assert a["native_rows"]==35136 and a["annual_duplicate_naive_labels"]==4
    assert a["selected_window_duplicates"]==a["selected_window_non15minute_steps"]==0
    assert a["source_window_intervals"]==480 and a["target_intervals"]==1438
    assert a["original_15min_energy_mwh"]==pytest.approx(a["expanded_full_5min_energy_mwh"])
    assert a["original_15min_energy_mwh"]==pytest.approx(a["trimmed_5min_energy_mwh"]+a["removed_edge_energy_mwh"])
    assert a["aggregate_api_max_error_mw"]<1e-10
    for record in a["source_files"].values():
        assert s.digest(record["path"])==record["sha256"]


def test_scaling_energy_peak_and_phase(profiles):
    mean=profiles.mean_preserved_scale.to_numpy()
    peak=profiles.peak_preserved_scale.to_numpy()
    shift=profiles.mean_shift_6h_scale.to_numpy()
    assert mean.mean()==pytest.approx(1)
    assert peak.max()==pytest.approx(1)
    np.testing.assert_allclose(shift,np.roll(mean,72),atol=1e-12)
    assert shift.sum()==pytest.approx(mean.sum())
    assert profiles.mean_preserved_background_p_mw.mean()==pytest.approx(3.715)
    assert profiles.mean_preserved_background_q_mvar.mean()==pytest.approx(2.3)
    np.testing.assert_allclose(profiles.mean_preserved_background_q_mvar/profiles.mean_preserved_background_p_mw,2.3/3.715)


def test_background_scaling_does_not_scale_aidc(config):
    _,ref=config
    n=s.build_network(ref,"reference"); bg=n.load.index.copy()
    idx=s.previous.core.add_aidc(n,2,23,.98)
    s.apply_operating_point(n,bg,idx,1.3,24,4)
    assert (n.load.loc[bg,"scaling"]==1.3).all()
    assert n.load.at[idx,"scaling"]==1 and n.load.at[idx,"p_mw"]==24
    assert n.load.loc[bg,"p_mw"].sum()==pytest.approx(3.715)


@pytest.mark.parametrize("role",["reference","comparison"])
@pytest.mark.parametrize("case",["mean_preserved","peak_preserved","mean_shift_6h"])
def test_result_accounting_and_solver_checks(results,role,case):
    entry=results["evaluations"][role][case]
    for mode,r in entry["replays"].items():
        assert r["intervals"]==1438
        assert r["feasible_intervals"]+r["converged_violation_intervals"]+r["unresolved_intervals"]==1438
        assert r["unresolved_intervals"]==0
        assert r["max_balance_error_mw"]<1e-6
        assert r["max_reactive_balance_error_mvar"]<1e-6
        assert sum(w["end_interval_exclusive"]-w["start_interval"] for w in r["violation_windows"])==r["converged_violation_intervals"]
        for c in r["solver_crosschecks"]:
            assert c["max_voltage_difference_pu"]<1e-6
            assert c["nr_feasible"]==c["bfsw_feasible"]
    if case=="mean_preserved":
        d=entry["optimistic_discharge_diagnostic"]
        assert d["tested_violating_intervals"]==d["passes_instantaneous_test"]+d["still_violates"]+d["unresolved"]


def test_existing_joint_q_does_not_include_ess_reactive_power():
    data,_=s.previous.inputs(); prefix="constant_25_5_joint"
    rows=json.loads((s.OUT/"reference_mean_preserved_existing_joint.json").read_text(encoding="utf-8"))
    q=(data[prefix+"_pcc_mw"]-data[prefix+"_ess_charge_mw"]+data[prefix+"_ess_discharge_mw"])*math.tan(math.acos(.98))
    np.testing.assert_allclose([r["pcc_q_mvar"] for r in rows],q,atol=1e-10)
    np.testing.assert_array_equal([r["pcc_p_mw"] for r in rows],data[prefix+"_pcc_mw"])


def test_no_source_profile_storage_or_historical_result_mutated(results):
    assert all(results["protected_inputs_unchanged"].values())
    for path,sha in results["protected_input_hashes"].items():
        assert s.digest(ROOT/path)==sha
    assert not results["profile_rescaled"]
    assert not results["storage_reoptimized"]
    assert not results["network_parameters_retuned_after_profile_results"]
    assert not results["doe_implemented"]


def test_fixed_background_new_solver_matches_prior(config):
    _,ref=config
    prior=json.loads((s.previous.OUT/"results.json").read_text(encoding="utf-8"))
    data,_=s.previous.inputs(); k=int(data["no_storage_pcc_mw"].argmax())
    p=float(data["no_storage_pcc_mw"][k])
    n=s.build_network(ref,"reference")
    s.previous.core.add_aidc(n,2,p,.98)
    new=s.solve_fast(n,40)
    old=prior["evaluations"]["allowance_alternative"]["replays"]["no_storage"]
    assert new["min_voltage_pu"]==pytest.approx(old["min_voltage_pu"],abs=1e-7)
    assert new["max_line_loading_percent"]==pytest.approx(old["max_line_loading_percent"],abs=1e-5)
