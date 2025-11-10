"""optimized_adaptive.py - Optimized with adaptive runtime optimization.

Demonstrates adaptive optimization that adjusts parameters at runtime.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn.functional as F
from typing import Optional
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch6")
    return torch.device("cuda")

class OptimizedAdaptiveBenchmark(Benchmark):
    """Optimized: Adaptive runtime optimization."""
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.N = 4_000_000
        self.adaptive_chunk = None
        self.prefetch_stream: Optional[torch.cuda.Stream] = None
        self.stage_buffers: list[torch.Tensor] = []
    
    def setup(self) -> None:
        """Setup: Initialize with adaptive configuration."""
        
        torch.manual_seed(42)
        # Optimization: Adaptive runtime optimization
        # Adjusts parameters at runtime based on workload characteristics
        # Adapts to changing input sizes, data patterns, etc.
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        props = torch.cuda.get_device_properties(self.device)
        sm_count = props.multi_processor_count
        warp_allocation = props.warp_size * sm_count
        # Choose a chunk that keeps at least two CTAs per SM resident but still fits L2.
        self.adaptive_chunk = min(self.N, warp_allocation * 256)
        self.prefetch_stream = torch.cuda.Stream()
        self.stage_buffers = [
            torch.empty(self.adaptive_chunk, device=self.device, dtype=torch.float32)
            for _ in range(2)
        ]
        torch.cuda.synchronize()

    def _transform(self, tensor: torch.Tensor) -> torch.Tensor:
        out = tensor.mul(1.75)
        out = out.add(0.1)
        return F.silu(out)
    
    def benchmark_fn(self) -> None:
        """Benchmark: Adaptive optimization operations."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_adaptive", enable=enable_nvtx):
            assert self.prefetch_stream is not None
            chunk_plan = []
            start = 0
            while start < self.N:
                span = min(self.adaptive_chunk, self.N - start)
                chunk_plan.append((start, span))
                start += span
            for idx, (start, span) in enumerate(chunk_plan):
                buf = self.stage_buffers[idx % len(self.stage_buffers)]
                slice_buf = buf[:span]
                next_slice = self.input[start:start + span]
                with torch.cuda.stream(self.prefetch_stream):
                    slice_buf.copy_(next_slice, non_blocking=True)
                torch.cuda.current_stream().wait_stream(self.prefetch_stream)
                transformed = self._transform(slice_buf)
                self.output[start:start + span].copy_(transformed, non_blocking=True)
            torch.cuda.synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.output = None
        self.stage_buffers = []
        self.prefetch_stream = None
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
    return OptimizedAdaptiveBenchmark()

if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Adaptive: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print("  Tip: Adaptive optimization adjusts parameters at runtime for optimal performance")
