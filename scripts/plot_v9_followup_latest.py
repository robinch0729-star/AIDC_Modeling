"""Plot the saved eta90 joint dispatch; never modify or rerun the model."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ess_tes_pcc_scheduling_v3_eta90"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    summary = read(OUT / "summary.json")
    solution = read(OUT / "constant_25_5_joint_solution.json")
    cfg = read(ROOT / "configs/ess_tes_pcc_scheduling_v3_eta90.json")
    candidate = read(ROOT / cfg["candidate_config"])
    assert solution["optimal"] and solution["feasible"]
    assert cfg["ess"]["charge_efficiency"] == 0.9
    assert cfg["ess"]["discharge_efficiency"] == 0.9
    assert candidate["tes"]["charge_efficiency"] == 0.9
    assert candidate["tes"]["discharge_efficiency"] == 0.9

    s = {k: np.asarray(v, dtype=float) for k, v in solution["schedule"].items()}
    dt = cfg["step_s"] / 3600
    n = len(s["pcc_mw"])
    hours = np.arange(n) * dt
    edges = np.arange(n + 1) * dt
    ess_delta = s["ess_discharge_mw"] - s["ess_charge_mw"]
    coefficients = summary["coefficients"]
    tes_delta = (
        coefficients["reduction_mwe_per_mwth"] * s["tes_discharge_mw_th"]
        - coefficients["increase_mwe_per_mwth"] * s["tes_charge_mw_th"]
    )
    np.testing.assert_allclose(
        s["no_storage_pcc_mw"] - s["pcc_mw"],
        ess_delta + tes_delta,
        atol=1e-9,
    )
    m = solution["metrics"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#7d8f98", "axes.labelcolor": "#183c4a",
        "text.color": "#183c4a", "xtick.color": "#425a65",
        "ytick.color": "#425a65", "svg.fonttype": "none",
    })
    blue, teal, orange, grey = "#175c95", "#087b75", "#cb782d", "#b3bdc3"
    fig, axes = plt.subplots(3, 1, figsize=(13.2, 9.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1, 1.05]})
    fig.subplots_adjust(left=.087, right=.985, bottom=.15, top=.85, hspace=.28)
    fig.suptitle("ESS-TES joint operation under a prescribed PCC limit",
                 x=.087, y=.974, ha="left", fontsize=20, fontweight="bold")
    fig.text(.087, .93,
             "ESS: 1 MW / 4 MWh  |  TES: 3 MWth / 10 MWhth  |  Charge/discharge efficiency: 0.90 / 0.90",
             fontsize=11.3)

    a = axes[0]
    a.plot(hours, s["no_storage_pcc_mw"], color=grey, lw=1.7,
           label="No storage", zorder=2)
    a.plot(hours, s["pcc_mw"], color=teal, lw=1.25,
           label="Joint ESS-TES operation", zorder=3)
    a.step(hours, s["pcc_upper_mw"], where="post", color="#b54143", lw=1.6,
           ls="--", label="25.5 MW limit", zorder=4)
    a.set_ylabel("PCC power (MW)")
    a.set_ylim(20.4, 27.6)
    a.set_title("(a) Five-minute average PCC demand", loc="left", fontsize=12,
                fontweight="bold", pad=9)
    a.legend(loc="upper left", frameon=False, ncols=3, fontsize=10)

    a = axes[1]
    a.stairs(ess_delta, edges, color=blue, lw=1.1, label="ESS electrical contribution")
    a.stairs(tes_delta, edges, color=orange, lw=1.3, label="TES contribution at PCC")
    a.axhline(0, color="#87969c", lw=.65)
    a.set_ylim(-1.15, 1.2)
    a.set_ylabel("PCC reduction (MW)")
    a.set_title("(b) Storage contributions: positive = lower import; negative = charging",
                loc="left", fontsize=12, fontweight="bold", pad=9)
    a.legend(loc="upper left", frameon=False, fontsize=10, ncols=1)

    a = axes[2]
    a.plot(edges, s["ess_energy_mwh"] / cfg["ess"]["energy_capacity_mwh"] * 100,
           color=blue, lw=1.5, label="ESS SOC")
    a.plot(edges, s["tes_energy_mwh_th"] / candidate["tes"]["energy_capacity_mwh_th"] * 100,
           color=orange, lw=1.5, label="TES SOC")
    a.axhline(50, ls=":", lw=1, color=grey, label="Initial / terminal SOC = 50%")
    a.set_ylim(0, 90)
    a.set_ylabel("Storage SOC (%)")
    a.set_xlabel("Elapsed simulation time (h)")
    a.set_title("(c) Storage states and terminal restoration", loc="left",
                fontsize=12, fontweight="bold", pad=9)
    a.legend(loc="upper right", frameon=False, fontsize=10, ncols=3)
    for a in axes:
        a.set_xlim(0, 120)
        a.set_xticks(np.arange(0, 121, 24))
        a.grid(axis="both", color="#dce4e8", alpha=.65, lw=.6)
        a.set_axisbelow(True)

    fig.text(.087, .075,
             f"Peak PCC: {m['no_storage_pcc_peak_mw']:.3f} -> {m['pcc_peak_mw']:.3f} MW"
             f"   |   Added energy vs. no storage: {m['pcc_energy_increment_mwh']:.3f} MWh",
             fontsize=11, fontweight="bold")
    fig.text(.087, .041,
             "Prescribed test limit, not a DSO-derived envelope. Perfect-foresight, energy-minimizing scheduling.\n"
             "119.83 h (1,438 five-minute intervals); compliance concerns five-minute averages, not instantaneous demand.",
             fontsize=9.5, color="#5a6f79", linespacing=1.5)
    for suffix in ["png", "svg"]:
        fig.savefig(OUT / f"joint_dispatch_eta90.{suffix}", dpi=200,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({
        "figure": str(OUT / "joint_dispatch_eta90.png"),
        "duration_h": edges[-1],
        "pcc_peak_no_storage_mw": m["no_storage_pcc_peak_mw"],
        "pcc_peak_joint_mw": m["pcc_peak_mw"],
        "extra_energy_mwh": m["pcc_energy_increment_mwh"],
        "ess_discharge_mwh": m["ess_discharge_energy_mwh"],
        "tes_discharge_mwh_th": m["tes_discharge_energy_mwh_th"],
        "tes_peak_pcc_reduction_mw": float(np.max(tes_delta)),
        "max_power_decomposition_error_mw": float(np.max(np.abs(
            s["no_storage_pcc_mw"] - s["pcc_mw"] - ess_delta - tes_delta))),
    }, indent=2))


if __name__ == "__main__":
    main()
