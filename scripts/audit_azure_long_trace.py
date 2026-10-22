"""Audit the full official Azure trace download; no model/scaling changes."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aidc_facility.azure_trace_audit import run_audit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/azure_inference_2024_full")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/azure_long_trace_audit_v1")
    parser.add_argument("--chunksize", type=int, default=250_000)
    args = parser.parse_args()
    run_audit(args.raw_dir, args.output_dir, args.chunksize)
