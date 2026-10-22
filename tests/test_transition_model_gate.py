import pytest
from aidc_facility.transition_model_gate import decide_transition_model


def example(complete=True, class_a=False):
    return {"all_files_eof_complete": complete, "families": {"online_rate": {"gate": {
        "class_A_verified_rate_change_and_arrival_time": class_a,
        "class_B_start_stop_and_power_candidate": True,
        "class_C_fixed_rate_operating_points": True}}}}


def test_no_tau_from_start_stop_or_steady():
    d = decide_transition_model(example())
    assert d["status"] == "no_identifiable_operating_transition_in_audited_material"
    assert d["new_time_constant_s"] is None
    assert d["new_transient_model_selected"] is False


def test_incomplete_not_absence_proof():
    assert decide_transition_model(example(False))["status"] == "audit_incomplete_no_model_selection"


def test_candidate_data_not_automatic_fit():
    d = decide_transition_model(example(class_a=True))
    assert d["status"] == "candidate_transition_data_require_further_validation"
    assert d["new_transient_model_selected"] is False


def test_missing_audit_rejected():
    with pytest.raises(ValueError):
        decide_transition_model({})
