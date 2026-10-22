"""Validation only: keep high-rate test sealed until protocol review."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from nlr_data.residual_scenario_scores import run_validation

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--draws",type=int,nargs="+",default=[25,50,100])
    parser.add_argument("--seeds",type=int,nargs="+",default=[42,43,44])
    parser.add_argument("--horizon-s",type=float,default=180.)
    parser.add_argument("--score-interval-s",type=float,default=1.)
    parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args()
    print(json.dumps(run_validation(ROOT,draws=args.draws,seeds=args.seeds,
                     horizon_s=args.horizon_s,score_interval_s=args.score_interval_s,
                     output_dir=args.output_dir),indent=2))
