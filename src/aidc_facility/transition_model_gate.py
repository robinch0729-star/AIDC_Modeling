"""Evidence gate for a new mean-power transient model.

No time constant is inferred from steady operating points, R3 correlation,
startup boundaries, or untimestamped access-log lines. A candidate structure
is not a fitted or selected AIDC response model.
"""
from __future__ import annotations


def decide_transition_model(audit: dict) -> dict:
    if not isinstance(audit.get("families"), dict) or not audit["families"]:
        raise ValueError("An explicit nonempty per-family audit is required")
    complete = audit.get("all_files_eof_complete") is True
    supported = [name for name, item in audit["families"].items()
                 if item.get("gate", {}).get("class_A_verified_rate_change_and_arrival_time") is True]
    if not complete:
        status = "audit_incomplete_no_model_selection"
        reason = "Some required raw files were not scanned to EOF; absence of evidence cannot be generalized."
    elif not supported:
        status = "no_identifiable_operating_transition_in_audited_material"
        reason = "No verified operating-rate transition with corresponding arrival timing was identified by the audit."
    else:
        status = "candidate_transition_data_require_further_validation"
        reason = "Candidate data require clock, confound, repeatability and held-out checks before any fitting or selection."
    return {
        "status": status, "reason": reason, "audit_complete": complete,
        "families_with_candidate_class_A_evidence": supported,
        "new_transient_model_selected": False, "new_time_constant_s": None,
        "new_delay_s": None, "M3_R3_parameters_changed": False,
        "baseline": {"model": "existing quasi-steady M3 + continuous R3 residual",
                     "role": "reference only, not validated fast power response"},
        "candidate_structures_not_fitted": [
            {"name": "pure delay", "requires": "clock-aligned repeatable delay distinguishable from timestamp offset"},
            {"name": "unity-DC-gain first-order mean response", "requires": "resolved gradual transitions and repeatability across runs"},
            {"name": "first order plus delay", "requires": "both identifiable delay and gradual response; justified gain over simpler candidates"},
            {"name": "serving-state response", "requires": "aligned arrival, token, running/queue/latency and power evidence; input remapping must be recalibrated"},
        ],
        "prohibited_shortcuts": ["using R3 OU correlation time as mean-response time constant",
            "concatenating different fixed-rate runs as measured transitions",
            "using startup/drain as ordinary hot operating-rate response without validation",
            "choosing tau to make a visually smooth curve", "filtering the entire M3+R3 trace"],
        "scope": "Evidence sufficiency for the audited local files, not a claim that no suitable data exist elsewhere.",
        "next_required_data": ["same warm serving instance with documented within-run rate changes",
            "request send/start/completion timestamps and input/output token counts",
            "synchronized raw GPU/CPU power and explicit meter integration interval",
            "running/queued requests and resource clocks/power-cap/settings",
            "repeated up/down transitions and separate runs for held-out validation"],
    }
