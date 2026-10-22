"""Generate final locked M3+R3 IT-load scenarios for CSV/MATLAB use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlr_data.final_it_profile_generator import (  # noqa: E402
    FinalITProfileGenerator,
    GenerationRequest,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=ROOT / "outputs/r3_high_rate_benchmark",
    )
    parser.add_argument("--prompt-dataset", required=True)
    parser.add_argument("--request-rate", type=float, required=True)
    parser.add_argument("--duration-s", type=float, default=180.0)
    parser.add_argument("--interval-s", type=float, default=0.1)
    parser.add_argument("--scenario-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-it-mean-mw", type=float)
    parser.add_argument("--allow-unsupported-extrapolation", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    generator = FinalITProfileGenerator.from_benchmark_artifact(args.artifact_dir)
    request = GenerationRequest(
        prompt_dataset=args.prompt_dataset,
        request_rate=args.request_rate,
        duration_s=args.duration_s,
        interval_s=args.interval_s,
        scenario_count=args.scenario_count,
        seed=args.seed,
        target_it_mean_mw=args.target_it_mean_mw,
        allow_unsupported_extrapolation=args.allow_unsupported_extrapolation,
    )
    profiles = generator.generate(request)
    output = generator.save_generation(profiles, request, args.output_dir)
    result = {
        "status": "complete",
        "output_dir": str(output),
        "rows": len(profiles),
        "scenario_count": request.scenario_count,
        "samples_per_scenario": int(len(profiles) / request.scenario_count),
        "prompt_dataset": request.prompt_dataset,
        "request_rate": request.request_rate,
        "request_rate_status": str(profiles["request_rate_status"].iloc[0]),
        "target_it_mean_mw": request.target_it_mean_mw,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
