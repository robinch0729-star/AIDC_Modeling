"""CLI entry point for M1/M2/M3 NLR steady-state equation comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nlr_data.steady_state_models import run_comparison  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Derived NLR condition/feature CSV or Parquet. Omit for auto-discovery.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "steady_state_model_comparison",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_comparison(
        input_path=args.input,
        output_dir=args.output_dir,
        project_root=PROJECT_ROOT,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

