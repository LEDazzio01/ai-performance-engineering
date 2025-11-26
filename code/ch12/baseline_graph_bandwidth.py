"""baseline_graph_bandwidth.py - Separate kernel launches for bandwidth measurement (baseline)."""

from __future__ import annotations

import sys
from pathlib import Path

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

# Import CUDA extension
from ch12.cuda_extensions import load_graph_bandwidth_extension


class BaselineGraphBandwidthBenchmark(BaseBenchmark):
    """Separate kernel launches - measures bandwidth without graphs (uses CUDA extension)."""
    
    def __init__(self):
        super().__init__()
        self.src = None
        self.dst = None
        self.N = 50_000_000
        self.iterations = 10
        self._extension = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.N * self.iterations),
        )
    
    def setup(self) -> None:
        """Setup: Initialize tensors and load CUDA extension."""
        # Load CUDA extension (will compile on first call)
        self._extension = load_graph_bandwidth_extension()
        
        torch.manual_seed(42)
        self.src = torch.randn(self.N, dtype=torch.float32, device=self.device)
        self.dst = torch.empty_like(self.src)
        torch.cuda.synchronize(self.device)
        # Dry run to amortize first-use overhead (extension launch/cuda events)
        self._extension.separate_kernel_launches(self.dst, self.src, 1)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Separate kernel launches (memory copy)."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("graph_bandwidth", enable=enable_nvtx):
            for _ in range(self.iterations):
                self._extension.separate_kernel_launches(self.dst, self.src, 1)
                self._synchronize()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.src = None
        self.dst = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=5,
            warmup=1,
            enable_memory_tracking=False,
            enable_profiling=False,
            setup_timeout_seconds=120,  # CUDA extension compilation can take time
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
        if self.dst is None:
            return "Destination tensor not initialized"
        if self.src is None:
            return "Source tensor not initialized"
        if self.dst.shape[0] != self.N:
            return f"Destination size mismatch: expected {self.N}, got {self.dst.shape[0]}"
        if not torch.isfinite(self.dst).all():
            return "Destination tensor contains non-finite values"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineGraphBandwidthBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Graph Bandwidth (CUDA Extension): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
