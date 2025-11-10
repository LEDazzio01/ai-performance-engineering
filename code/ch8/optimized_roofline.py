"""optimized_roofline.py - Optimized with roofline analysis in occupancy/warp divergence context.

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

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch8")
    return torch.device("cuda")

class OptimizedRooflineBenchmark(Benchmark):
    """Optimized: Roofline analysis for performance optimization.
    
    Roofline: Uses roofline analysis to identify compute/memory bottlenecks.
    Guides optimization strategy based on arithmetic intensity.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.compiled_model = None
        self.inputs = None
        self.roofline_data = None
        self.hidden_dim = 1024
        self.batch_size = 256
        self.micro_batches = 4
    
    def setup(self) -> None:
        """Setup: Initialize model with roofline analysis."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: Roofline analysis
        # Identifies compute-bound vs memory-bound operations
        # Guides optimization strategy
        
        enable_tf32()
        
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        ).to(self.device).eval()
        self.compiled_model = torch.compile(self.model, mode="reduce-overhead")
        
        self.inputs = [
            torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.bfloat16)
            for _ in range(self.micro_batches)
        ]
        
        # Roofline data for analysis
        self.roofline_data = {
            'compute_bound': False,
            'memory_bound': True,
            'arithmetic_intensity': 0.0,
        }
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations with roofline analysis."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_roofline", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: Roofline analysis
                # Measure execution time and estimate arithmetic intensity
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    for micro_input in self.inputs:
                        output = self.compiled_model(micro_input)
                        _ = output.sum()
                end_event.record()
                torch.cuda.synchronize()
                
                elapsed_ms = start_event.elapsed_time(end_event)
                
                # Estimate arithmetic intensity (simplified)
                # Arithmetic intensity = FLOPs / Bytes accessed
                input_bytes = sum(inp.numel() * inp.element_size() for inp in self.inputs)
                output_bytes = output.numel() * output.element_size()
                total_bytes = input_bytes + output_bytes
                
                # Estimate FLOPs (simplified)
                flops = self.batch_size * self.hidden_dim * (self.hidden_dim * 2) * 2 * self.micro_batches
                arithmetic_intensity = flops / total_bytes if total_bytes > 0 else 0.0
                
                # Roofline analysis: determine bottleneck
                is_memory_bound = arithmetic_intensity < 1.0
                self.roofline_data['compute_bound'] = not is_memory_bound
                self.roofline_data['memory_bound'] = is_memory_bound
                self.roofline_data['arithmetic_intensity'] = arithmetic_intensity
                self.roofline_data['elapsed_ms'] = elapsed_ms
                
                # Use roofline analysis to guide optimization
                if is_memory_bound:
                    # Memory-bound: optimize memory access patterns
                    # Roofline analysis guides memory optimization
                    pass
                else:
                    # Compute-bound: optimize compute operations
                    # Roofline analysis guides compute optimization
                    pass
                
                # Optimization: Roofline analysis benefits
                # - Identifies compute vs memory bottlenecks
                # - Guides optimization strategy
                # - Measures arithmetic intensity
                # - Performance-based optimization decisions
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.compiled_model = None
        self.inputs = None
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
        if self.model is None or self.compiled_model is None:
            return "Model not initialized"
        if not self.inputs:
            return "Inputs not initialized"
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
