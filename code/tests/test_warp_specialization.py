#!/usr/bin/env python3
"""Simple test script to compare warp specialization benchmarks.

Usage:
    python tests/test_warp_specialization.py [--iterations N] [--warmup N] [--test-mode triton|cuda|both]
"""

import os
import sys
import time
import argparse
import torch
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

IS_CI = os.environ.get("CI", "").lower() in {"1", "true", "yes"}
DEV_ITER_CAP = int(os.environ.get("WARP_SPEC_DEV_ITER_CAP", "20"))
DEV_WARMUP_CAP = int(os.environ.get("WARP_SPEC_DEV_WARMUP_CAP", "3"))

# Global CLI arguments (set by main())
_cli_iterations = None
_cli_warmup = None


def _resolve_loop_counts(benchmark, default_iterations: int, default_warmup: int):
    """Resolve iteration/warmup counts based on CLI flags, BenchmarkConfig, and defaults."""
    default_iterations = max(1, int(default_iterations))
    default_warmup = max(0, int(default_warmup))
    config = None

    if hasattr(benchmark, "get_config"):
        try:
            config = benchmark.get_config()
        except Exception as exc:
            print(f"  Warning: failed to load BenchmarkConfig ({exc})")

    # CLI flags take precedence
    if _cli_iterations is not None:
        iterations = _cli_iterations
    elif IS_CI:
        # CI mode: use config or default (no cap)
        iterations = config.iterations if config else default_iterations
    else:
        # Dev mode: use config or default with cap
        iterations = config.iterations if config else default_iterations
        iterations = min(iterations, DEV_ITER_CAP)

    if _cli_warmup is not None:
        warmup = _cli_warmup
    elif IS_CI:
        # CI mode: use config or default (no cap)
        warmup = config.warmup if config else default_warmup
    else:
        # Dev mode: use config or default with cap
        warmup = config.warmup if config else default_warmup
        warmup = min(warmup, DEV_WARMUP_CAP)

    return max(1, iterations), max(0, warmup)


def run_benchmark(benchmark_class, iterations=50, warmup=5):
    """Run a benchmark and return average time in ms."""
    benchmark = benchmark_class()
    iterations, warmup = _resolve_loop_counts(benchmark, iterations, warmup)
    
    # Setup
    benchmark.setup()
    torch.cuda.synchronize()
    
    # Warmup
    for _ in range(warmup):
        benchmark.benchmark_fn()
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        benchmark.benchmark_fn()
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    # Teardown
    benchmark.teardown()
    
    avg_time = sum(times) / len(times)
    median_time = sorted(times)[len(times) // 2]
    return avg_time, median_time, times

def main():
    parser = argparse.ArgumentParser(
        description="Compare warp specialization benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of benchmark iterations (overrides defaults and config)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Number of warmup iterations (overrides defaults and config)"
    )
    parser.add_argument(
        "--test-mode",
        type=str,
        choices=["triton", "cuda", "both"],
        default=os.environ.get("WARP_SPEC_TEST_MODE", "triton").strip().lower(),
        help="Test mode: triton, cuda, or both (default: triton)"
    )
    
    args = parser.parse_args()
    
    # Validate test mode
    if args.test_mode not in {"triton", "cuda", "both"}:
        print(f"Warning: unknown test-mode='{args.test_mode}'. Defaulting to 'triton'.")
        args.test_mode = "triton"
    
    # Set global CLI arguments for _resolve_loop_counts
    global _cli_iterations, _cli_warmup
    _cli_iterations = args.iterations
    _cli_warmup = args.warmup
    
    WARP_SPEC_TEST_MODE = args.test_mode
    
    print("=" * 80)
    print("Warp Specialization Benchmark Comparison")
    print("=" * 80)
    if _cli_iterations is not None or _cli_warmup is not None:
        print(f"CLI overrides: iterations={_cli_iterations}, warmup={_cli_warmup}")
    print(f"Test mode: {WARP_SPEC_TEST_MODE}")
    print()
    
    results = {}
    
    # Chapter 1
    print("\n--- Chapter 1 ---")
    try:
        from ch01.baseline_warp_specialization import BaselineWarpSpecializationBenchmark
        from ch01.optimized_warp_specialization import OptimizedWarpSpecializationBenchmark
        
        print("Running baseline...")
        baseline_avg, baseline_median, baseline_times = run_benchmark(BaselineWarpSpecializationBenchmark, iterations=10, warmup=5)
        print(f"  Baseline: {baseline_avg:.3f} ms (median: {baseline_median:.3f} ms)")
        
        print("Running optimized...")
        optimized_avg, optimized_median, optimized_times = run_benchmark(OptimizedWarpSpecializationBenchmark, iterations=10, warmup=5)
        print(f"  Optimized: {optimized_avg:.3f} ms (median: {optimized_median:.3f} ms)")
        
        speedup = baseline_avg / optimized_avg if optimized_avg > 0 else 0
        print(f"  Speedup: {speedup:.2f}x")
        results['ch01'] = (baseline_avg, optimized_avg, speedup)
    except Exception as e:
        print(f"  Error: {e}")
    
    # Chapter 9
    print("\n--- Chapter 9 ---")
    try:
        from ch09.baseline_warp_specialization_producer_consumer import BaselineWarpSpecializationProducerConsumerBenchmark
        print("Running baseline...")
        baseline_avg, baseline_median, baseline_times = run_benchmark(BaselineWarpSpecializationProducerConsumerBenchmark)
        print(f"  Baseline: {baseline_avg:.3f} ms (median: {baseline_median:.3f} ms)")

        def run_variant(label: str, loader):
            try:
                bench_cls = loader()
            except Exception as exc:
                print(f"  {label} benchmark unavailable ({exc})")
                return None
            print(f"Running optimized ({label})...")
            opt_avg, opt_median, _ = run_benchmark(bench_cls)
            print(f"  Optimized ({label}): {opt_avg:.3f} ms (median: {opt_median:.3f} ms)")
            speedup = baseline_avg / opt_avg if opt_avg > 0 else 0
            print(f"  Speedup ({label}): {speedup:.2f}x")
            results[f'ch09_{label}'] = (baseline_avg, opt_avg, speedup)
            return True

        def load_triton():
            from ch09.optimized_warp_specialization_producer_consumer_triton import (
                OptimizedWarpSpecializationProducerConsumerBenchmark as _Bench
            )
            return _Bench

        def load_cuda():
            from ch09.optimized_warp_specialization_producer_consumer_cuda import (
                OptimizedWarpSpecializationProducerConsumerCUDABenchmark as _Bench
            )
            return _Bench

        if args.test_mode in {"triton", "both"}:
            run_variant("triton", load_triton)
        if args.test_mode in {"cuda", "both"}:
            run_variant("cuda", load_cuda)
    except Exception as e:
        print(f"  Error: {e}")
    
    # Chapter 13
    print("\n--- Chapter 13 ---")
    try:
        from ch13.baseline_warp_specialization_training import BaselineWarpSpecializationTrainingBenchmark
        from ch13.optimized_warp_specialization_training import OptimizedWarpSpecializationTrainingBenchmark
        
        print("Running baseline...")
        baseline_avg, baseline_median, baseline_times = run_benchmark(BaselineWarpSpecializationTrainingBenchmark, iterations=10, warmup=5)
        print(f"  Baseline: {baseline_avg:.3f} ms (median: {baseline_median:.3f} ms)")
        
        print("Running optimized...")
        optimized_avg, optimized_median, optimized_times = run_benchmark(OptimizedWarpSpecializationTrainingBenchmark, iterations=10, warmup=5)
        print(f"  Optimized: {optimized_avg:.3f} ms (median: {optimized_median:.3f} ms)")
        
        speedup = baseline_avg / optimized_avg if optimized_avg > 0 else 0
        print(f"  Speedup: {speedup:.2f}x")
        results['ch13'] = (baseline_avg, optimized_avg, speedup)
    except Exception as e:
        print(f"  Error: {e}")
    
    # Chapter 18
    print("\n--- Chapter 18 ---")
    try:
        from ch18.baseline_warp_specialization_attention import BaselineWarpSpecializationAttentionBenchmark
        from ch18.optimized_warp_specialization_attention import OptimizedWarpSpecializationAttentionBenchmark
        
        print("Running baseline...")
        baseline_avg, baseline_median, baseline_times = run_benchmark(BaselineWarpSpecializationAttentionBenchmark)
        print(f"  Baseline: {baseline_avg:.3f} ms (median: {baseline_median:.3f} ms)")
        
        print("Running optimized...")
        optimized_avg, optimized_median, optimized_times = run_benchmark(OptimizedWarpSpecializationAttentionBenchmark)
        print(f"  Optimized: {optimized_avg:.3f} ms (median: {optimized_median:.3f} ms)")
        
        speedup = baseline_avg / optimized_avg if optimized_avg > 0 else 0
        print(f"  Speedup: {speedup:.2f}x")
        results['ch18'] = (baseline_avg, optimized_avg, speedup)
    except Exception as e:
        print(f"  Error: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for chapter, (baseline, optimized, speedup) in results.items():
        better = "✓ Optimized better" if speedup > 1.0 else "✗ Optimized worse" if speedup < 1.0 else "≈ Same"
        print(f"{chapter.upper()}: Baseline={baseline:.3f}ms, Optimized={optimized:.3f}ms, Speedup={speedup:.2f}x {better}")

if __name__ == "__main__":
    main()

