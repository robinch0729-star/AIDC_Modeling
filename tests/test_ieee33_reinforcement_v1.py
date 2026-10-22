"""Research-equivalent reinforcement checks; no site-design certification."""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandapower as pp
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import screen_ieee33_reinforcement_v1 as s


@pytest.fixture(scope="module")
def cfg():
    return s.read_config()


@pytest.fixture(scope="module")
def data():
    return s.inputs()


@pytest.fixture(scope="module")
def result():
    return json.loads((s.OUT / "results.json").read_text(encoding="utf-8"))


def candidate(cfg, area=630, n=2, v=1.045):
    return next(c for c in s.candidates(cfg) if c["area_mm2"] == area and c["circuits"] == n and c["source_voltage_pu"] == v)


def test_candidate_space_unique_and_ranking_is_explicit(cfg):
    cs = s.candidates(cfg)
    assert len(cs) == 187 == len({c["id"] for c in cs})
    assert len([c for c in cs if c["area_mm2"] == 0]) == 11
    assert max(c["source_voltage_pu"] for c in cs) == 1.05
    assert s.rank(candidate(cfg,630,2)) < s.rank(candidate(cfg,300,3))
    assert s.select([], "nominal") is None


def test_only_root_corridor_changes_and_background_is_preserved(cfg):
    net = s.build_network(cfg,candidate(cfg))
    raw = s.core.original_network()
    cols = ["from_bus","to_bus","length_km","r_ohm_per_km","x_ohm_per_km","c_nf_per_km","parallel","in_service"]
    pd.testing.assert_frame_equal(net.line.loc[1:,cols],raw.line.loc[1:,cols])
    pd.testing.assert_frame_equal(net.load[["bus","p_mw","q_mvar","in_service"]],raw.load[["bus","p_mw","q_mvar","in_service"]])
    assert len(net.bus)==33 and net.line.in_service.sum()==32
    assert (net.bus.vn_kv==12.66).all() and net.f_hz==60
    assert net.load.p_mw.sum()==pytest.approx(3.715)
    assert net.load.q_mvar.sum()==pytest.approx(2.3)
    assert net.trafo.empty and net.gen.empty and net.sgen.empty
    assert (net.line.loc[1:,"max_i_ka"]==.6).all()
    assert cfg["fixed"]["head_limit_mva"]==40


def test_cable_50hz_conversion_and_parallel_values_not_double_divided(cfg):
    line=s.build_network(cfg,candidate(cfg),length=2).line.iloc[0]
    assert line.r_ohm_per_km==.043
    assert line.x_ohm_per_km==pytest.approx(.093*60/50)
    assert line.c_nf_per_km==490
    assert line.length_km==2 and line.parallel==2
    assert line.max_i_ka==.762 and line.df==1


def test_parallel_attribute_matches_two_explicit_circuits(cfg):
    net=s.build_network(cfg,candidate(cfg))
    s.core.add_aidc(net,2,26.55,.98)
    a=s.solve(net,cfg)
    v=net.res_bus.vm_pu.to_numpy().copy()
    line=net.line.iloc[0].copy()
    net.line.at[0,"parallel"]=1
    pp.create_line_from_parameters(net,int(line.from_bus),int(line.to_bus),line.length_km,
        line.r_ohm_per_km,line.x_ohm_per_km,line.c_nf_per_km,line.max_i_ka,
        name="explicit_second_root_circuit")
    b=s.solve(net,cfg)
    np.testing.assert_allclose(net.res_bus.vm_pu,v,rtol=0,atol=1e-8)
    assert b["network_losses_mw"]==pytest.approx(a["network_losses_mw"],abs=1e-8)
    assert net.res_line.at[0,"i_ka"]==pytest.approx(a["root_current_per_circuit_ka"],abs=1e-8)
    assert net.res_line.iloc[-1].i_ka==pytest.approx(a["root_current_per_circuit_ka"],abs=1e-8)


def test_derating_changes_rating_not_voltage(cfg):
    c=candidate(cfg)
    a,na=s.snapshot(cfg,c,26.55,26.55*math.tan(math.acos(.98)))
    b,nb=s.snapshot(cfg,c,26.55,26.55*math.tan(math.acos(.98)),current_factor=.9)
    np.testing.assert_allclose(na.res_bus.vm_pu,nb.res_bus.vm_pu,atol=1e-10)
    assert b["root_loading_percent"]==pytest.approx(a["root_loading_percent"]/.9)
    assert b["root_effective_total_limit_ka"]==pytest.approx(2*.762*.9)


def test_envelope_q_excludes_ess_and_is_not_continuous_schedule(cfg,data):
    d,storage=data
    p=d["no_storage_pcc_mw"]
    points=s.operating_points(cfg,p,storage)
    assert len(points)==9
    ch=next(r for r in points if r["name"]=="peak_plus_rated_charge_envelope")
    assert ch["p"]==pytest.approx(p.max()+1+3*storage["tes_charge_e_mw"]/3)
    assert ch["q"]==pytest.approx((ch["p"]-1)*math.tan(math.acos(.98)))
    assert {r["background"] for r in points if r["p"]==0}=={0,.1,.5,1}


@pytest.mark.parametrize("kwargs",[{"background":-1},{"current_factor":0},{"resistance_factor":0},{"length":0}])
def test_bad_physical_parameters_rejected(cfg,kwargs):
    with pytest.raises(ValueError): s.build_network(cfg,candidate(cfg),**kwargs)


@pytest.mark.parametrize("p,q",[([],[]),([1,2],[1]),([np.nan],[0]),([[1]],[[0]])])
def test_invalid_replays_rejected(cfg,p,q):
    with pytest.raises(ValueError): s.replay(cfg,candidate(cfg),p,q,"invalid_should_not_exist")


def test_selected_candidates_are_finite_space_minima(result):
    rows=json.loads((s.OUT/"candidate_screen.json").read_text(encoding="utf-8"))
    assert len(rows)==result["candidates_count"]==187
    for label,key in (("nominal_minimum","nominal"),("allowance_alternative","allowance")):
        c=result["evaluations"][label]["candidate"]
        assert c==s.select(rows,key)
        assert all(not r[key]["all_feasible"] for r in rows if s.rank(r["candidate"])<s.rank(c))
        assert next(r[key]["all_feasible"] for r in rows if r["candidate"]==c)
    assert all(not r["nominal"]["all_feasible"] for r in rows if not r["candidate"]["area_mm2"])
    assert sum(r["nominal"]["unresolved"]+r["allowance"]["unresolved"] for r in rows)==0


def test_all_full_profiles_preserve_input_and_pass_ac_constraints(result,data):
    d,storage=data
    p=d["no_storage_pcc_mw"]
    prefix="constant_25_5_joint"
    pj=d[prefix+"_pcc_mw"]
    for evaluation in result["evaluations"].values():
        for name,summary in evaluation["replays"].items():
            rows=json.loads((s.OUT/summary["file"]).read_text(encoding="utf-8"))
            assert len(rows)==len(p)==1438
            expected=(pj if name=="existing_joint" else p if name=="no_storage" else p+storage["total_charge_e_mw"])
            np.testing.assert_array_equal([r["pcc_p_mw"] for r in rows],expected)
            if name=="existing_joint":
                expected_q=s.native.pcc_q(pj,d[prefix+"_ess_charge_mw"],d[prefix+"_ess_discharge_mw"],.98)
            else:
                expected_q=(p if name=="no_storage" else p+storage["tes_charge_e_mw"])*math.tan(math.acos(.98))
            np.testing.assert_array_equal([r["pcc_q_mvar"] for r in rows],expected_q)
            assert all(r["converged"] and r["feasible"] for r in rows)
            assert summary["feasible_intervals"]==1438 and summary["unresolved_intervals"]==0
            assert summary["min_voltage_pu"]>=.95-1e-7 and summary["max_voltage_pu"]<=1.05+1e-7
            assert summary["max_root_loading_percent"]<=100+1e-7
            assert summary["max_other_loading_percent"]<=100+1e-7
            assert summary["max_balance_error_mw"]<1e-6
            assert summary["max_q_balance_error_mvar"]<1e-6


def test_original_files_are_hash_preserved_and_config_matches(result):
    assert all(result["protected_inputs_unchanged"].values())
    for path,sha in result["protected_input_hashes"].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha
    assert result["config_sha256"]==hashlib.sha256(s.CONFIG.read_bytes()).hexdigest()
    assert json.loads((s.OUT/"model_contract.json").read_text(encoding="utf-8"))==s.read_config()
    for key in ("aidc_rescaled","storage_resized","storage_reoptimized","doe_implemented","source_capacity_increased","source_transformer_explicitly_modeled"):
        assert result[key] is False
    assert "not an executable SOC-constrained schedule" in result["full_charge_envelope_note"]


def test_independent_solver_and_static_boundary_checks(result):
    for e in result["evaluations"].values():
        assert e["solver_crosscheck"]["max_voltage_difference_pu"]<1e-6
        assert e["solver_crosscheck"]["max_line_current_difference_ka"]<1e-6
        boundary=e["static_limit_not_doe"]
        assert boundary["last_feasible"]["feasible"]
        assert not boundary["first_infeasible"]["feasible"]
        assert 0<boundary["first_infeasible_mw"]-boundary["capacity_mw"]<=.001
        outage=next(r for r in e["sensitivity"] if r["kind"]=="one_parallel_circuit_out_of_service")
        assert outage["converged"] and not outage["feasible"]
