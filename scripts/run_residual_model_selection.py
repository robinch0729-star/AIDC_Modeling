"""Run MAD-free R1/R2/R3 validation and high-rate test selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlr_data.residual_model_selection import run_residual_model_selection  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT.parent / "NLR_dataset")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "condition_audit" / "audited_run_manifest.csv",
    )
    parser.add_argument(
        "--condition-targets",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "graybox_steady_state" / "online_rate_condition_targets.csv",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "steady_state_model_comparison" / "split_assignment.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "residual_model_selection_v2",
    )
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_residual_model_selection(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        condition_target_path=args.condition_targets,
        split_path=args.split,
        output_dir=args.output_dir,
        draws=args.draws,
        seed=args.seed,
    )
    print(json.dumps(result["decision"], indent=2, ensure_ascii=False))
    print(f"Outputs: {result['output_dir']}")


if __name__ == "__main__":
    main()

