"""Build the portable M3+R3 generator and run facility aggregation cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlr_data.workload_generator import (  # noqa: E402
    build_generator_from_nlr,
    run_aggregation_case_study,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT.parent / "NLR_dataset")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "outputs" / "condition_audit" / "audited_run_manifest.csv")
    parser.add_argument("--condition-targets", type=Path, default=PROJECT_ROOT / "outputs" / "graybox_steady_state" / "online_rate_condition_targets.csv")
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "outputs" / "steady_state_model_comparison" / "split_assignment.csv")
    parser.add_argument("--selection", type=Path, default=PROJECT_ROOT / "outputs" / "residual_model_comparison" / "selected_residual_model.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "workload_aggregation")
    parser.add_argument("--target-it-mw", type=float, default=20.0)
    parser.add_argument("--draws", type=int, default=10)
    args = parser.parse_args()

    generator = build_generator_from_nlr(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        condition_target_path=args.condition_targets,
        split_path=args.split,
        selection_path=args.selection,
    )
    result = run_aggregation_case_study(
        generator=generator,
        output_dir=args.output_dir,
        target_it_mean_mw=args.target_it_mw,
        draws=args.draws,
    )
    print(json.dumps(result["decision"], indent=2, ensure_ascii=False))
    print(f"Outputs: {result['output_dir']}")


if __name__ == "__main__":
    main()
