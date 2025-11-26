"""baseline_work_queue.py - Static work distribution (baseline)."""

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
from ch12.cuda_extensions import load_work_queue_extension


class BaselineWorkQueueBenchmark(BaseBenchmark):
    """Static work distribution - each thread processes fixed indices (uses CUDA extension)."""
    
    def __init__(self):
        super().__init__()
        self.input_data = None
        self.output_data = None
        self.N = 1 << 20  # 1M elements
        self.iterations = 5
        self._extension = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.N * self.iterations),
        )
    
    def setup(self) -> None:
        """Setup: Initialize tensors and load CUDA extension."""
        # Load CUDA extension (will compile on first call)
        self._extension = load_work_queue_extension()
        
        torch.manual_seed(42)
        self.input_data = torch.linspace(0.0, 1.0, self.N, dtype=torch.float32, device=self.device)
        self.output_data = torch.zeros(self.N, dtype=torch.float32, device=self.device)
        torch.cuda.synchronize(self.device)
        self._extension.static_work_distribution(self.input_data, self.output_data, 1)
        torch.cuda.synchronize()
        torch.manual_seed(42)
        self.input_data = torch.linspace(0.0, 1.0, self.N, dtype=torch.float32, device=self.device)
        self.output_data = torch.zeros(self.N, dtype=torch.float32, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Static work distribution."""

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("work_queue", enable=enable_nvtx):
            # Call CUDA extension with static work distribution
            self._extension.static_work_distribution(self.input_data, self.output_data, self.iterations)
        self._synchronize()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input_data = None
        self.output_data = None
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
        if self.input_data is None or self.output_data is None:
            return "Data tensors not initialized"
        if self.input_data.shape[0] != self.N or self.output_data.shape[0] != self.N:
            return f"Data size mismatch: expected {self.N}"
        if not torch.isfinite(self.output_data).all():
            return "Output contains non-finite values"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineWorkQueueBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Work Queue (Static Distribution): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
