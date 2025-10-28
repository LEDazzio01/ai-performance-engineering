"""Torch profiler helper focused on CUDA memory diagnostics."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Memory profiler wrapper")
    parser.add_argument("script", help="Python script to execute under the profiler")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments forwarded to the script")
    parser.add_argument("--trace", type=Path, help="Optional path to write Chrome trace JSON")
    parser.add_argument("--sort", default="self_cuda_memory_usage", help="Metric to sort by in the summary table")
    parser.add_argument("--row-limit", type=int, default=25, help="Rows to display in summary table")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_path = Path(args.script).resolve()
    if not script_path.exists():
        raise SystemExit(f"Script not found: {script_path}")

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    print(f"Profiling memory usage for {script_path} ...")
    sys.argv = [str(script_path)] + args.script_args
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        runpy.run_path(str(script_path), run_name="__main__")

    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(args.trace))
        print(f"Chrome trace written to {args.trace}")

    table = prof.key_averages().table(sort_by=args.sort, row_limit=args.row_limit)
    print("\n=== Memory Profiling Summary ===")
    print(table)


if __name__ == "__main__":
    main()
