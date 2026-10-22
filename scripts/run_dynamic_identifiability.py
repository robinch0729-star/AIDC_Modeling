"""Run NLR M3 dynamic-identifiability and residual audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlr_data.dynamic_identifiability import run_dynamic_identifiability  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT.parent / "NLR_dataset",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "condition_audit" / "audited_run_manifest.csv",
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "steady_state_model_comparison"
            / "model_parameters_prompt_conditioned.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "dynamic_identifiability",
    )
    args = parser.parse_args()
    result = run_dynamic_identifiability(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        parameter_path=args.parameters,
        output_dir=args.output_dir,
    )
    summary = result["tables"]["dynamic_summary"]
    print(summary.to_string(index=False))
    print("\nDecision")
    print(json.dumps(result["decision"], indent=2, ensure_ascii=False))
    print(f"\nOutputs: {result['output_dir']}")


if __name__ == "__main__":
    main()

