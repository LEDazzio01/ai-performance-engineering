"""Multi-GPU wrapper for optimized continuous batching; skips when GPUs < 2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from ch04.optimized_continuous_batching import OptimizedContinuousBatchingBenchmark


def get_benchmark() -> OptimizedContinuousBatchingBenchmark:
    if torch.cuda.device_count() < 2:
        raise RuntimeError("SKIPPED: optimized_continuous_batching_multigpu requires >=2 GPUs")
    return OptimizedContinuousBatchingBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode, BenchmarkConfig

    parser = argparse.ArgumentParser(description="Run optimized continuous batching on multiple GPUs.")
    parser.add_argument("--iterations", type=int, default=None, help="Measurement iterations (overrides harness default).")
    parser.add_argument("--warmup", type=int, default=None, help="Warmup iterations (overrides harness default).")
    args = parser.parse_args()

    bench = get_benchmark()
    config = BenchmarkConfig()
    if args.iterations is not None:
        config.iterations = args.iterations
    if args.warmup is not None:
        config.warmup = args.warmup

    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    result = harness.benchmark(bench)
    print(f"Continuous batching optimized (multi-GPU): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
