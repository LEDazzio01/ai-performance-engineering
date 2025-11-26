"""baseline_cuda_graphs.py - Separate kernel launches (baseline)."""

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

from ch12.cuda_extensions import load_cuda_graphs_extension


class BaselineCudaGraphsBenchmark(BaseBenchmark):
    """Separate kernel launches - multiple launches without graph optimization (uses CUDA extension)."""
    
    def __init__(self):
        super().__init__()
        self.data = None
        self.N = 1 << 18  # Smaller problem amplifies launch overhead.
        self.iterations = 200
        self._extension = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.N),
        )
    
    def setup(self) -> None:
        """Setup: Initialize tensors and load CUDA extension."""
        self._extension = load_cuda_graphs_extension()
        
        torch.manual_seed(42)
        self.data = torch.linspace(0.0, 1.0, self.N, dtype=torch.float32, device=self.device)
        torch.cuda.synchronize(self.device)
        # Warm up kernel launches so compilation/init costs are excluded.
        self._extension.separate_kernel_launches(self.data, 1)
        torch.cuda.synchronize()
        torch.manual_seed(42)
        self.data = torch.linspace(0.0, 1.0, self.N, dtype=torch.float32, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Separate kernel launches."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("cuda_graphs", enable=enable_nvtx):
            self._extension.separate_kernel_launches(self.data, self.iterations)
            self._synchronize()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=5,
            warmup=1,
            enable_memory_tracking=False,
            enable_profiling=False,
            setup_timeout_seconds=120,  # CUDA extension compilation can take time
            measurement_timeout_seconds=120,
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
        if self.data is None:
            return "Data tensor not initialized"
        if self.data.shape[0] != self.N:
            return f"Data size mismatch: expected {self.N}"
        if not torch.isfinite(self.data).all():
            return "Data contains non-finite values"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineCudaGraphsBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline CUDA Graphs (Separate Launches): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
