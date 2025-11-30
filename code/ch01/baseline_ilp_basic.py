"""baseline ilp basic - Baseline with low instruction-level parallelism. Implements BaseBenchmark for harness integration."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

try:
    import ch01.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

from typing import Optional

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.benchmark.utils import warn_benchmark_scaling


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch01")
    return torch.device("cuda")


class BaselineIlpBasicBenchmark(BaseBenchmark):
    """Baseline: Sequential operations with low ILP.
    
    ILP: This baseline has low instruction-level parallelism.
    Operations are sequential and dependent, limiting parallel execution.
    """
    
    def __init__(self):
        super().__init__()
        self.device = resolve_device()
        self.input = None
        self.output = None
        # Target workload size for optimal ILP demonstration
        original_N = 100_000_000  # 100M elements (~400 MB FP32)
        
        # Scale workload based on available GPU memory (match optimized scale)
        # Scale down for smaller GPUs to ensure it fits
        if torch.cuda.is_available():
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if total_memory_gb >= 16:  # Large GPU (B200/B300, GB200, etc.)
                self.N = 100_000_000  # 100M elements
            elif total_memory_gb >= 8:  # Medium GPU (RTX 3090, etc.)
                self.N = 50_000_000  # 50M elements
            elif total_memory_gb >= 4:  # Small GPU (RTX 3060, etc.)
                self.N = 25_000_000  # 25M elements
            else:  # Very small GPU
                self.N = 10_000_000  # 10M elements
        else:
            self.N = 100_000_000  # Fallback (shouldn't happen - CUDA required)
        
        # Warn if workload was reduced
        warn_benchmark_scaling(
            scaling_type="ILP workload size",
            original_values={"N": original_N},
            scaled_values={"N": self.N},
            impact_description="Smaller workloads may not fully demonstrate ILP benefits; speedup ratios may be lower than production-scale",
            recommendation="For accurate production benchmarks, use GPUs with >=16GB memory"
        )
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        torch.manual_seed(42)
        # Baseline: Sequential operations (low ILP)
        # Each operation depends on the previous one
        # Low instruction-level parallelism
        
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Sequential operations with low ILP."""
        # Use conditional NVTX ranges - only enabled when profiling

        from core.profiling.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_ilp_basic", enable=enable_nvtx):
            # Baseline: Sequential operations - low ILP
            # Each operation depends on previous one
            # Cannot execute operations in parallel
            val = self.input
            val = val * 2.0      # Op 1
            val = val + 1.0     # Op 2 (depends on Op 1)
            val = val * 3.0     # Op 3 (depends on Op 2)
            val = val - 5.0     # Op 4 (depends on Op 3)
            self.output = val
            
            # Baseline: Low ILP issues
            # Sequential dependencies prevent parallel execution
            # Cannot hide instruction latency
        torch.cuda.synchronize(self.device)

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.output = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_environment_metrics
        return compute_environment_metrics(
            gpu_count=getattr(self, 'gpu_count', 1),
            gpu_memory_gb=getattr(self, 'gpu_memory_gb', 80.0),
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.input is None or self.output is None:
            return "Tensors not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineIlpBasicBenchmark()


if __name__ == '__main__':
    from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    timing = result.timing
    if timing:
        print(f"\nBaseline ILP Basic: {timing.mean_ms:.3f} ms")
    else:
        print("\nBaseline ILP Basic: No timing data available")
