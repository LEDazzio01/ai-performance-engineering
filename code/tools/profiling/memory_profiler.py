"""Torch profiler helper focused on CUDA memory diagnostics."""

from __future__ import annotations

import argparse
import os
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
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment variable assignments applied before running the script",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_path = Path(args.script).resolve()
    if not script_path.exists():
        raise SystemExit(f"Script not found: {script_path}")

    # Ensure script-relative imports (e.g., arch_config) resolve when running from repo root
    script_parent = script_path.parent
    repo_root = script_parent.parent
    for path in (script_parent, repo_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    # Apply environment overrides
    for assignment in args.env:
        if "=" not in assignment:
            raise SystemExit(f"Invalid --env assignment '{assignment}' (expected KEY=VALUE)")
        key, value = assignment.split("=", 1)
        os.environ[key] = value

    # Add script directory and repo root to sys.path for imports
    script_dir = script_path.parent
    repo_root = Path.cwd()  # Assuming memory_profiler is run from repo root
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

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
