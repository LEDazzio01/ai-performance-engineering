"""baseline_kernel_launches.py - Many small kernel launches (baseline).

Demonstrates performance issue: many sequential kernel launches with overhead.
Implements BaseBenchmark for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from typing import Optional

from common.python.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


def many_small_ops_regular(x: torch.Tensor, iterations: int = 100) -> torch.Tensor:
    """Run many small operations WITHOUT CUDA graphs (many kernel launches).
    
    Each operation launches a separate kernel:
    - x + 1.0 → kernel launch 1
    - x * 0.99 → kernel launch 2
    - torch.relu(x) → kernel launch 3
    
    Total: 3 * iterations kernel launches = high overhead!
    """
    for _ in range(iterations):
        x = x + 1.0
        x = x * 0.99
        x = torch.relu(x)
    return x


class BaselineKernelLaunchesBenchmark(BaseBenchmark):
    """Benchmark implementation following BaseBenchmark."""
    
    def __init__(self):
        super().__init__()
        self.x = None
        self.output = None
        self.size = (1024, 1024)
        self.iterations = 1000
        tokens = self.size[0] * self.size[1]
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: initialize tensor."""
        # Use bfloat16 for GPU performance
        dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.x = torch.randn(*self.size, device=self.device, dtype=dtype)
    
    def benchmark_fn(self) -> None:
        """Function to benchmark."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("kernel_launches", enable=enable_nvtx):
            with torch.no_grad():
                self.output = many_small_ops_regular(self.x.clone(), self.iterations)
            self._synchronize()

    
    def teardown(self) -> None:
        """Cleanup."""
        self.x = None
        self.output = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=30,
            warmup=5,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_graph_metrics
        return compute_graph_metrics(
            baseline_launch_overhead_us=getattr(self, '_baseline_launch_us', 10.0),
            graph_launch_overhead_us=getattr(self, '_graph_launch_us', 1.0),
            num_nodes=getattr(self, 'num_nodes', 10),
            num_iterations=getattr(self, 'num_iterations', 100),
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.x is None:
            return "Input tensor not initialized"
        if self.output is None:
            return "Output tensor not initialized"
        if not torch.isfinite(self.output).all():
            return "Output contains non-finite values"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselineKernelLaunchesBenchmark()


def main() -> None:
    """Standalone execution with timing."""
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=30, warmup=5)
    )
    benchmark = BaselineKernelLaunchesBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print("Baseline: Many Small Kernel Launches")
    print("=" * 70)
    print(f"Tensor size: {benchmark.size}")
    print(f"Operations per iteration: 3 (add, multiply, relu)")
    print(f"Total kernel launches: {benchmark.iterations * 3} per run")
    print("Problem: Each operation = separate kernel launch overhead\n")
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")
    print(f"Total kernel launches: {benchmark.iterations * 3}")
    print(f"Overhead: High (300 launches per run)")
    print("\n Tip: Use CUDA Graphs to reduce launch overhead")


if __name__ == "__main__":
    main()
