#!/usr/bin/env python3
"""
Lightweight helper to regenerate Nsight Compute CSV metrics for chapter examples.

Usage examples:
  python tools/extract_ncu_metrics.py --example ch6_add_parallel \
      --metrics gpu__time_duration.avg,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed

  python tools/extract_ncu_metrics.py --example ch6_add_parallel \
      --metrics default --output profiles/metrics/ch6_add_parallel.csv

By default we collect a set of latency/throughput counters that are useful when
contrasting vectorised vs scalar kernels. The script reuses the profiling harness'
example registry, so any registered example name can be passed via --example.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from profile_harness import (  # type: ignore  # pylint: disable=wrong-import-position
    CUDA_BIN_DIRS,
    CUDA_LIB_DIRS,
    REPO_ROOT as HARNESS_ROOT,
    base_env,
    example_run_command,
    EXAMPLE_BY_NAME,
)

DEFAULT_METRICS = [
    "gpu__time_duration.avg",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
]


def resolve_ncu() -> str:
    """Return the Nsight Compute binary path, searching CUDA toolchain locations."""
    for directory in CUDA_BIN_DIRS:
        candidate = Path(directory) / "ncu"
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return "ncu"


def ensure_dirs(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Nsight Compute metrics to CSV.")
    parser.add_argument(
        "--example",
        required=True,
        help="Example name registered in example_registry (e.g. ch6_add_parallel).",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma separated list of metric identifiers, or 'default'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV output path. Defaults to profiles/metrics/<example>_<timestamp>.csv",
    )
    parser.add_argument(
        "--replay-mode",
        default="kernel",
        choices=["kernel", "application", "singlepass"],
        help="Nsight Compute replay mode (default: kernel).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    example = EXAMPLE_BY_NAME.get(args.example)
    if example is None:
        raise SystemExit(f"Unknown example '{args.example}'. Run scripts/profile_harness.py --list for options.")

    metrics: List[str]
    if args.metrics.strip().lower() == "default":
        metrics = DEFAULT_METRICS
    else:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
        if not metrics:
            raise SystemExit("Metric list may not be empty.")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_path = args.output.resolve()
    else:
        output_dir = REPO_ROOT / "profiles" / "metrics"
        output_path = output_dir / f"{example.name}_{timestamp}.csv"
    ensure_dirs(output_path)

    ncu_path = resolve_ncu()
    base_command = [
        ncu_path,
        "--set",
        "full",
        "--metrics",
        ",".join(metrics),
        "--replay-mode",
        args.replay_mode,
        "--csv",
    ]

    command = base_command + example_run_command(example, HARNESS_ROOT)

    env = base_env(example)
    # Guarantee Nsight can resolve libraries when CUDA toolkit is not on the default loader path.
    for var, candidates in (("PATH", CUDA_BIN_DIRS), ("LD_LIBRARY_PATH", CUDA_LIB_DIRS)):
        segments = [p for p in env.get(var, "").split(os.pathsep) if p]
        for candidate in candidates:
            if os.path.isdir(candidate) and candidate not in segments:
                segments.insert(0, candidate)
        env[var] = os.pathsep.join(segments)

    print(f"[extract-ncu] Running: {' '.join(command)}")
    completed = subprocess.run(  # noqa: S603
        command,
        cwd=str(example.resolved_workdir(HARNESS_ROOT)),
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    raw_output = completed.stdout
    stderr_path = output_path.with_suffix(".stderr.log")
    stderr_path.write_text(completed.stderr)

    # Filter narration from CSV output (retain only comma-delimited rows)
    csv_lines: List[str] = []
    for line in raw_output.splitlines():
        if "," in line and not line.startswith("=="):
            csv_lines.append(line)
    if csv_lines:
        Path(output_path).write_text("\n".join(csv_lines) + "\n")
    else:
        Path(output_path).write_text(raw_output)

    if completed.returncode != 0:
        print(f"[extract-ncu] Nsight Compute exited with {completed.returncode}. See {stderr_path}", file=sys.stderr)
        raise SystemExit(completed.returncode)

    print(f"[extract-ncu] Metrics written to {output_path}")

    # Optional summary for Chapter 16 RadixAttention example
    if args.example == "ch16_radix_attention":
        summary = {}
        naive_match = re.search(r"Naive approach took:\s*([0-9.]+) seconds", raw_output)
        cached_match = re.search(r"RadixAttention approach took:\s*([0-9.]+) seconds", raw_output)
        speedup_match = re.search(r"Speedup:\s*([0-9.]+)x", raw_output)
        reuse_matches = re.findall(r"Cached prefix length for prompt of (\d+) tokens:\s*(\d+)", raw_output)

        if naive_match:
            summary["naive_seconds"] = float(naive_match.group(1))
        if cached_match:
            summary["cached_seconds"] = float(cached_match.group(1))
        if speedup_match:
            summary["speedup"] = float(speedup_match.group(1))
        if reuse_matches:
            summary["cached_prefix_lengths"] = {
                int(total): int(reused) for total, reused in reuse_matches
            }

        if summary:
            base = output_path.parent / output_path.stem
            json_path = base.with_suffix(".summary.json")
            md_path = base.with_suffix(".summary.md")
            json_path.write_text(json.dumps(summary, indent=2))

            lines = ["# RadixAttention Profiling Summary\n"]
            if "speedup" in summary:
                lines.append(f"* Speedup: {summary['speedup']:.2f}x")
            if "naive_seconds" in summary and "cached_seconds" in summary:
                lines.append(
                    f"* Naive vs Cached wall time: {summary['naive_seconds']:.3f}s → {summary['cached_seconds']:.3f}s"
                )
            if "cached_prefix_lengths" in summary:
                reuse_lines = ", ".join(
                    f"{total}→{reused}" for total, reused in summary["cached_prefix_lengths"].items()
                )
                lines.append(f"* Cached prefix lengths: {reuse_lines}")
            md_path.write_text("\n".join(lines) + "\n")
            print(f"[extract-ncu] Summary written to {json_path} and {md_path}")


if __name__ == "__main__":
    main()
