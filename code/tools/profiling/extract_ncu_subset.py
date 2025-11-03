#!/usr/bin/env python3
"""Collate Nsight Compute CSV metrics referenced in the manuscript tables."""

import csv
import glob
import pathlib
import sys
from typing import Dict, Iterable, List, Tuple


WANT: Dict[str, str] = {
    "Achieved Occupancy": "achieved_occupancy",
    "Warp Execution Efficiency": "warp_execution_efficiency",
    "SM % Peak": "sm_pct_of_peak",
    "GPU DRAM Throughput": "gpu_dram_throughput",
    "L2 Hit Rate": "l2_hit_rate",
    "Shared Load Bytes": "shared_ld_bytes",
    "Shared Store Bytes": "shared_st_bytes",
    "FP32 Instructions": "fp32_inst_count",
    "FP16 Instructions": "fp16_inst_count",
    "Branch Uniformity": "branch_uniformity",
    "Eligible Warps Per Cycle": "eligible_warps_per_cycle",
}


def harvest_csv(path: pathlib.Path) -> List[Tuple[str, str, str, str]]:
    rows: List[Tuple[str, str, str, str]] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for entry in reader:
            metric_name = entry.get("Name") or entry.get("Metric Name") or ""
            metric_value = entry.get("Metric Value") or entry.get("Value") or ""
            kernel_name = entry.get("Kernel Name") or entry.get("ID") or "kernel"
            section = entry.get("Section") or ""
            if metric_name in WANT and metric_value:
                rows.append((kernel_name, section, metric_name, metric_value))
    return rows


def collect(pattern: str) -> List[Dict[str, str]]:
    out_rows: List[Dict[str, str]] = []
    for match in glob.glob(pattern):
        tag = pathlib.Path(match).stem
        for kernel, section, metric_name, metric_value in harvest_csv(pathlib.Path(match)):
            out_rows.append(
                {
                    "tag": tag,
                    "kernel": kernel,
                    "section": section,
                    "metric": WANT[metric_name],
                    "value": metric_value,
                }
            )
    return out_rows


def main(args: Iterable[str]) -> int:
    try:
        pattern = next(iter(args))
    except StopIteration:
        print("usage: python tools/extract_ncu_subset.py 'output/reports/*.csv'", file=sys.stderr)
        return 2

    rows = collect(pattern)
    if not rows:
        print("No metrics found—did you pass the Nsight Compute CSV files?", file=sys.stderr)
        return 1

    output_dir = pathlib.Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "metrics_summary.csv"
    fieldnames = ["tag", "kernel", "section", "metric", "value"]
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
