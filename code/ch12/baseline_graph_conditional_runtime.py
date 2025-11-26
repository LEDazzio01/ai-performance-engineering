"""Baseline: Fresh kernel launches without CUDA graph.

This baseline demonstrates the overhead of launching kernels individually
without CUDA graph replay. Each operation requires a separate kernel launch
with associated driver overhead.

The optimized version captures all operations in a CUDA graph and replays
them with a single submission, eliminating the per-kernel launch overhead.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


class BaselineGraphBenchmark(BaseBenchmark):
    """Baseline: Fresh kernel launches each iteration.
    
    This shows the overhead of launching kernels individually:
    - Each operation triggers a kernel launch
    - Driver overhead per launch (~5-10us)
    - No batching of operations
    
    For small/fast kernels, launch overhead can dominate.
    """
    
    def __init__(self):
        super().__init__()
        self.batch_size = 32
        self.seq_len = 512
        self.hidden_dim = 2048
        
        self.data: Optional[torch.Tensor] = None
        
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def _compute_ops(self) -> None:
        """Same operations as optimized, but fresh launches each time."""
        # Each of these triggers a kernel launch
        self.data.mul_(0.99)
        for _ in range(16):  # 16 loops × 2 ops = 32 kernel launches
            self.data.add_(0.001)
            self.data.mul_(1.0001)
        self.data.relu_()
        self.data.mul_(1.001)
    
    def setup(self) -> None:
        """Setup data tensor."""
        torch.manual_seed(42)
        
        dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        
        self.data = torch.randn(
            self.batch_size, self.seq_len, self.hidden_dim,
            device=self.device, dtype=dtype
        )
        
        # Warmup
        for _ in range(5):
            self._compute_ops()
        
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark fresh kernel launches."""
        with self._nvtx_range("fresh_kernel_launches"):
            self._compute_ops()
        self._synchronize()
    
    def teardown(self) -> None:
        """Clean up."""
        self.data = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=100,
            warmup=20,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return baseline graph metrics using standard helpers."""
        from common.python.benchmark_metrics import compute_graph_metrics
        
        # Baseline has full launch overhead per iteration
        baseline_launch_us = 8.0 * 35  # ~35 ops × 8us each
        
        metrics = compute_graph_metrics(
            baseline_launch_overhead_us=baseline_launch_us,
            graph_launch_overhead_us=baseline_launch_us,  # Same as baseline (no graph)
            num_nodes=35,
            num_iterations=100,
        )
        metrics["graph.uses_cuda_graph"] = 0.0
        return metrics


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselineGraphBenchmark()


def main() -> None:
    """Standalone execution."""
    print("=" * 70)
    print("Baseline: Fresh Kernel Launches (No CUDA Graph)")
    print("=" * 70)
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    print()
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=100, warmup=20)
    )
    
    benchmark = get_benchmark()
    result = harness.benchmark(benchmark)
    
    print(f"Results:")
    print(f"  Average: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"  Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print()
    print("Overhead sources:")
    print("  - ~35 kernel launches per iteration")
    print("  - Driver overhead per launch (~5-10us)")
    print("  - No operation batching")
    print()
    print("See optimized version for CUDA graph improvement.")


if __name__ == "__main__":
    main()
