"""Run the NLR [3]-inspired bottom-up IT facility aggregation case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aidc_facility.reporting import save_facility_case  # noqa: E402
from aidc_facility.simulation import FacilityCaseConfig, run_facility_case  # noqa: E402
from aidc_facility.traffic import representative_short_horizon_shape  # noqa: E402
from nlr_data.final_it_profile_generator import FinalITProfileGenerator  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=ROOT / "outputs/r3_high_rate_benchmark",
    )
    parser.add_argument("--target-pcc-mean-mw", type=float, default=20.0)
    parser.add_argument("--pue", type=float, default=1.15)
    parser.add_argument("--target-utilization", type=float, default=0.80)
    parser.add_argument("--duration-min", type=float, default=15.0)
    parser.add_argument("--macro-interval-s", type=float, default=60.0)
    parser.add_argument("--micro-interval-s", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/nlr_bottom_up_facility_17mw_it",
    )
    args = parser.parse_args()
    macro_steps = int(round(args.duration_min * 60.0 / args.macro_interval_s))
    if macro_steps < 2:
        raise ValueError("duration must contain at least two macro steps")

    calibrated = FinalITProfileGenerator.from_benchmark_artifact(args.artifact_dir)
    config = FacilityCaseConfig(
        target_pcc_mean_mw=args.target_pcc_mean_mw,
        assumed_pue=args.pue,
        target_utilization=args.target_utilization,
        macro_interval_s=args.macro_interval_s,
        micro_interval_s=args.micro_interval_s,
        seed=args.seed,
    )
    result = run_facility_case(
        representative_short_horizon_shape(macro_steps),
        calibrated_generator=calibrated,
        config=config,
    )
    output = save_facility_case(result, args.output_dir)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps({"output_dir": str(output), **summary}, indent=2, ensure_ascii=False))
