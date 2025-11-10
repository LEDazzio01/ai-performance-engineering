"""optimized_autotuning.py - Optimized with autotuning.

Demonstrates autotuning to find optimal kernel parameters.
Autotuning searches parameter space to maximize performance.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

# Import arch_config to apply Triton patch for sm_12x support
# The patch removes 'a' suffix from sm_121a -> sm_121 for ptxas compatibility
try:
    import arch_config  # noqa: F401
except ImportError:
    pass  # Continue if arch_config not available

from typing import Optional, List, Tuple
import torch.nn.functional as F

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch6")
    return torch.device("cuda")

class OptimizedAutotuningBenchmark(Benchmark):
    """Optimized: Uses autotuning to find optimal parameters."""
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.N = 4_000_000
        self.candidates = [1024, 2048, 4096, 8192]
        self.optimal_chunk = None
        self.timer_results: List[Tuple[int, float]] = []
    
    def setup(self) -> None:
        """Setup: Initialize tensors and perform autotuning."""
        
        torch.manual_seed(42)
        # Optimization: Autotune to find optimal parameters
        # Autotuning searches parameter space (block size, tile size, etc.)
        # to find configuration that maximizes performance
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        self.optimal_chunk = self._autotune_chunk_size()
        torch.cuda.synchronize()

    def _transform(self, tensor: torch.Tensor) -> torch.Tensor:
        out = tensor.mul(1.75)
        out = out.add(0.1)
        return F.silu(out)

    def _autotune_chunk_size(self) -> int:
        """Benchmark several staging chunk sizes using CUDA events."""
        best = None
        best_time = float("inf")
        for chunk in self.candidates:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for offset in range(0, self.N, chunk):
                span = min(chunk, self.N - offset)
                window = self.input[offset:offset + span]
                transformed = self._transform(window)
                self.output[offset:offset + span].copy_(transformed)
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)
            self.timer_results.append((chunk, elapsed))
            if elapsed < best_time:
                best_time = elapsed
                best = chunk
        assert best is not None
        return best
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations with autotuned parameters."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_autotuning", enable=enable_nvtx):
            chunk = self.optimal_chunk
            for offset in range(0, self.N, chunk):
                span = min(chunk, self.N - offset)
                window = self.input[offset:offset + span]
                transformed = self._transform(window)
                self.output[offset:offset + span].copy_(transformed, non_blocking=True)
            torch.cuda.synchronize()
            
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.output = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.output is None:
            return "Output tensor not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedAutotuningBenchmark()

if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    
    print(f"\nOptimized Autotuning: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(" Tip: Autotuning finds optimal kernel parameters automatically for best performance")
