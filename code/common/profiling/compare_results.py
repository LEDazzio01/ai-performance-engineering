#!/usr/bin/env python3
"""compare_results.py - Compare baseline vs optimized profiling results."""

import json
import sys
from pathlib import Path
from typing import Dict, Any


def compare_benchmark_results(baseline_file: Path, optimized_file: Path) -> None:
    """Compare two benchmark JSON files."""
    try:
        with open(baseline_file) as f:
            baseline = json.load(f)
        with open(optimized_file) as f:
            optimized = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("BENCHMARK COMPARISON")
    print("="*70)
    
    print(f"\nBaseline:  {baseline_file.name}")
    print(f"Optimized: {optimized_file.name}")
    
    # Compare metrics
    metrics_to_compare = ["mean_ms", "throughput", "bandwidth_gbs", "tflops"]
    
    print("\n" + "-"*70)
    print(f"{'Metric':<20} {'Baseline':>15} {'Optimized':>15} {'Speedup':>12}")
    print("-"*70)
    
    for metric in metrics_to_compare:
        if metric in baseline and metric in optimized:
            base_val = baseline[metric]
            opt_val = optimized[metric]
            
            # For timing metrics, speedup is baseline/optimized
            # For throughput metrics, speedup is optimized/baseline
            if "ms" in metric or "time" in metric:
                speedup = base_val / opt_val if opt_val > 0 else float('inf')
            else:
                speedup = opt_val / base_val if base_val > 0 else float('inf')
            
            print(f"{metric:<20} {base_val:>15.3f} {opt_val:>15.3f} {speedup:>11.2f}x")
    
    print("-"*70)
    print()


def print_usage():
    print("Usage: python compare_results.py <baseline.json> <optimized.json>")
    print("\nCompares two benchmark result files and shows speedup.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print_usage()
        sys.exit(1)
    
    baseline_file = Path(sys.argv[1])
    optimized_file = Path(sys.argv[2])
    
    compare_benchmark_results(baseline_file, optimized_file)

