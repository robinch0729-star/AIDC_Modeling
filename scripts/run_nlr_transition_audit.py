"""Audit all locally supplied NLR inference raw logs without fitting dynamics."""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aidc_facility.nlr_transition_audit import audit_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path(r"D:\DC_modeling\NLR_dataset\00_raw_datasets"))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "nlr_transition_audit_v1")
    args = parser.parse_args()
    started = time.perf_counter()
    summary = audit_dataset(args.raw_root, args.output)
    print(json.dumps({"files_scanned": summary["files_scanned"],
                      "bytes_scanned": summary["bytes_scanned"],
                      "all_files_eof_complete": summary["all_files_eof_complete"],
                      "families": summary["families"],
                      "seconds": time.perf_counter()-started}, indent=2))


if __name__ == "__main__":
    main()
