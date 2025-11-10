"""optimized_roofline.py - Optimized with roofline analysis in memory access/GEMM context.

Demonstrates roofline analysis for performance optimization.
Roofline: Uses roofline analysis to identify compute/memory bottlenecks.
Guides optimization strategy based on arithmetic intensity.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch7")
    return torch.device("cuda")

class OptimizedRooflineBenchmark(Benchmark):
    """Optimized: Roofline analysis for performance optimization.
    
    Roofline: Uses roofline analysis to identify compute/memory bottlenecks.
    Guides optimization strategy based on arithmetic intensity.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        # Optimization: Compile model for kernel fusion and optimization

        # Optimization: Compile model for kernel fusion and optimization

        self.input = None
        self.roofline_data = None
    
    def setup(self) -> None:
        """Setup: Initialize model with roofline analysis."""
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        torch.manual_seed(42)
        self.model = nn.Sequential(
            nn.Linear(1024, 2048, bias=False),
            nn.GELU(),
            nn.Linear(2048, 1024, bias=False),
        ).to(self.device).eval()

        self.batch = 512
        host_activation = torch.randn(self.batch, 1024)
        self.staging = host_activation.pin_memory()
        self.host_pinned = self.staging
        self.device_input = host_activation.to(self.device, non_blocking=True)
        self.input = self.device_input

        try:
            self.compiled = torch.compile(self.model, mode="max-autotune", fullgraph=True)
        except Exception:
            self.compiled = self.model

        self.roofline_data = {
            "compute_bound": False,
            "memory_bound": True,
            "arithmetic_intensity": 0.0,
        }
        self._ridge_point = 4.0  # FLOP/byte crossover for this workload
        self._prefetch_stream = torch.cuda.Stream()
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations with roofline analysis."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_roofline", enable=enable_nvtx):
            with torch.no_grad():
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                output = self.compiled(self.device_input)
                end_event.record()
                torch.cuda.synchronize()

                elapsed_ms = start_event.elapsed_time(end_event)
                bytes_accessed = self.device_input.numel() * self.device_input.element_size()
                bytes_accessed += output.numel() * output.element_size()
                flops = 2.0 * self.device_input.size(0) * self.device_input.size(1) * 2048 * 2
                arithmetic_intensity = flops / bytes_accessed if bytes_accessed else 0.0
                is_memory_bound = arithmetic_intensity < self._ridge_point

                self.roofline_data["compute_bound"] = not is_memory_bound
                self.roofline_data["memory_bound"] = is_memory_bound
                self.roofline_data["arithmetic_intensity"] = arithmetic_intensity
                self.roofline_data["elapsed_ms"] = elapsed_ms

                if is_memory_bound:
                    fused = output
                    for _ in range(2):
                        fused = self.compiled(fused)
                    torch.cuda.synchronize()
                    self.device_input.copy_(fused)
                else:
                    with torch.cuda.stream(self._prefetch_stream):
                        self.device_input.copy_(self.host_pinned, non_blocking=True)
                    torch.cuda.current_stream().wait_stream(self._prefetch_stream)

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
        self.roofline_data = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.input is None:
            return "Input not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedRooflineBenchmark()

def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedRooflineBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: Roofline")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")

if __name__ == "__main__":
    main()
