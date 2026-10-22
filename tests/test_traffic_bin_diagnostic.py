import numpy as np
import pytest
from aidc_facility.traffic_bin_diagnostic import aggregate_arrivals, bin_edges, describe_bins, project_bins, ramp_metrics, to_dispatch


@pytest.mark.parametrize("width", [.1, 1, 5, 10])
@pytest.mark.parametrize("phase", [0, .5])
def test_count_and_duration_conservation(width, phase):
    t = np.array([0, .05, .1, 1, 4.999, 10, 179.999])
    frame = aggregate_arrivals(t, 180, width, phase)
    assert frame.source_count.sum() == len(t)
    assert frame.bin_s.sum() == pytest.approx(180)
    assert np.sum(frame.source_rate_rps * frame.bin_s) == pytest.approx(len(t))


def test_phase_partial_edges():
    assert bin_edges(20, 10, .5) == pytest.approx([0, 5, 15, 20])
    a = aggregate_arrivals([0, 4.9, 5, 15], 20, 10, .5)
    assert a.source_count.tolist() == [2, 1, 1]
    assert a.source_rate_rps.tolist() == pytest.approx([.4, .1, .2])


def test_end_exclusive():
    with pytest.raises(ValueError):
        aggregate_arrivals([180], 180, 10)


def test_projection_preserves_integral():
    f = aggregate_arrivals([0, 5, 11], 20, 10, .5)
    v, b = project_bins(f, "source_rate_rps")
    assert len(v) == 200
    assert v.sum() * .1 == pytest.approx(3)
    assert np.flatnonzero(b).tolist() == [50, 150]


def test_half_point_one_phase_not_grid_aligned():
    with pytest.raises(ValueError):
        project_bins(aggregate_arrivals([0], 1, .1, .5), "source_rate_rps")


def test_ramp_boundary_metric():
    v = np.r_[np.ones(20), np.ones(20) * 2]
    b = np.zeros(40, bool)
    b[20] = True
    m = ramp_metrics(v, b)
    assert m["max_abs_ramp_0p1s_mw_per_s"] == pytest.approx(10)
    assert m["within_bin_max_abs_ramp_0p1s_mw_per_s"] == 0
    assert m["boundary_abs_change_share_1s"] == pytest.approx(1)


def test_no_silent_clipping():
    import pandas as pd
    f = pd.DataFrame({"support_status": ["unsupported_extrapolation"]})
    with pytest.raises(ValueError):
        to_dispatch(f)


def test_weighted_statistics_on_partial_bins():
    import pandas as pd
    f = pd.DataFrame({"bin_s": [1., 3.], "source_rate_rps": [0., 4.],
                      "source_count": [0, 12], "node_rate_rps": [0., 8.],
                      "offered_rps": [0., 16.],
                      "support_status": ["unsupported_extrapolation", "calibration"]})
    d = describe_bins(f)
    assert d["source_mean_rps"] == 3
    assert d["source_std_rps_time_weighted"] == pytest.approx(np.sqrt(3))
    assert d["zero_bin_fraction"] == .5
    assert d["zero_time_fraction"] == .25
    assert d["node_p95_rps_time_weighted"] == 8
    assert d["scenario_offered_requests"] == 48


def test_no_within_bin_one_second_window_is_nan():
    b = np.zeros(40, bool)
    b[[10, 20, 30]] = True
    m = ramp_metrics(np.arange(40, dtype=float), b)
    assert np.isnan(m["within_bin_max_abs_ramp_1s_mw_per_s"])


def test_energy_unit_conversion():
    m = ramp_metrics(np.ones(1800), np.zeros(1800, bool))
    assert m["energy_kwh"] == pytest.approx(50)
