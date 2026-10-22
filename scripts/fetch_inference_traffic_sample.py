"""Fetch a bounded first-common-window sample of official Azure arrival traces."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aidc_facility.arrival_trace import prepare_sample

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/raw/azure_inference_sample")
    parser.add_argument("--duration-s", type=float, default=180)
    parser.add_argument("--bin-s", type=float, default=10)
    parser.add_argument("--max-mib-per-service", type=int, default=8)
    args = parser.parse_args()
    print(prepare_sample(args.output_dir, args.duration_s, args.bin_s, args.max_mib_per_service), flush=True)
