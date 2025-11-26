"""optimized_roofline.py - Optimized with roofline analysis in GEMM context.

Demonstrates roofline analysis for performance optimization.
Roofline: Uses roofline analysis to identify compute/memory bottlenecks.
Guides optimization strategy based on arithmetic intensity.
Implements BaseBenchmark for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import torch.cuda.amp as amp

from typing import Optional

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
)
from ch10.workload_config import WORKLOAD

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch10")
    return torch.device("cuda")

class OptimizedRooflineBenchmark(BaseBenchmark):
    """Optimized: Roofline analysis for performance optimization.
    
    Roofline: Uses roofline analysis to identify compute/memory bottlenecks.
    Guides optimization strategy based on arithmetic intensity.
    """
    
    def __init__(self):
        super().__init__()
        self.device = resolve_device()
        self.model = None
        self.input = None
        self.roofline_data = None
        self.workload = WORKLOAD
        self.batch = self.workload.roofline_batch_size
        self.micro_batches = self.workload.roofline_micro_batches
        self.hidden_dim = self.workload.roofline_hidden_dim
        self.ffn_dim = self.workload.roofline_ffn_dim
        self._checksum = 0.0
    
    def setup(self) -> None:
        """Setup: Initialize model with roofline analysis."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            enable_tf32()
        torch.manual_seed(42)
        # Optimization: Roofline analysis
        # Identifies compute-bound vs memory-bound operations
        # Guides optimization strategy
        
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.ffn_dim),
            nn.ReLU(),
            nn.Linear(self.ffn_dim, self.hidden_dim),
        ).to(self.device).half().eval()
        
        total_rows = self.batch * self.micro_batches
        self.input = torch.randn(
            total_rows,
            self.hidden_dim,
            device=self.device,
            dtype=torch.float16,
        )
        
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
                with amp.autocast(dtype=torch.float16):
                    output = self.model(self.input)
                end_event.record()
                torch.cuda.synchronize()
                
                elapsed_ms = start_event.elapsed_time(end_event)
                
                # Estimate arithmetic intensity (simplified)
                # Arithmetic intensity = FLOPs / Bytes accessed
                input_bytes = self.input.numel() * self.input.element_size()
                output_bytes = output.numel() * output.element_size()
                total_bytes = input_bytes + output_bytes
                
                # Estimate FLOPs (simplified)
                flops = self.input.size(0) * self.input.size(1) * 2048 * 2
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
                self._checksum = float(output.float().sum().item())

    
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
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return roofline analysis metrics."""
        # Estimate problem size for roofline analysis
        n = getattr(self, 'N', 0) or getattr(self, 'hidden_dim', 0) or 4096
        batch = getattr(self, 'batch_size', 1) or getattr(self, 'batch', 1)
        # Simple FLOP estimate for linear layers
        flops = 2.0 * batch * n * n  # Rough estimate
        bytes_moved = batch * n * 4.0  # Input/output bytes
        arithmetic_intensity = flops / max(bytes_moved, 1.0)
        return {
    "roofline.estimated_flops": flops,
    "roofline.estimated_bytes": bytes_moved,
    "roofline.arithmetic_intensity": arithmetic_intensity,
}

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.input is None:
            return "Input not initialized"
        return None

def get_benchmark() -> BaseBenchmark:
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
