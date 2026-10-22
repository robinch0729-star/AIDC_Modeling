"""Run the fixed-R3 high-rate holdout benchmark."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlr_data.r3_high_rate_benchmark import run_high_rate_benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, nargs="+", default=[25, 50, 100])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--horizon-s", type=float, default=180.0)
    parser.add_argument("--score-interval-s", type=float, default=1.0)
    parser.add_argument("--lambda-siw", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = run_high_rate_benchmark(
        ROOT, draws=args.draws, seeds=args.seeds, horizon_s=args.horizon_s,
        score_interval_s=args.score_interval_s, lambda_siw=args.lambda_siw,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2))

