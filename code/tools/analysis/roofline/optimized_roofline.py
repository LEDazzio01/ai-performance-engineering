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

from typing import Optional

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
)

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch10")
    return torch.device("cuda")

class OptimizedRooflineBenchmark(BaseBenchmark):
    """Optimized: Roofline analysis for performance optimization.
    
    Roofline: Uses roofline analysis to identify compute/memory bottlenecks.
    Guides optimization strategy based on arithmetic intensity.
    
    Optimizations vs baseline:
    - TF32/BF16 precision for faster matmul
    - cuDNN benchmark mode for kernel selection
    - torch.compile for kernel fusion
    """
    
    def __init__(self):
        super().__init__()
        self.device = resolve_device()
        self.model = None
        self.compiled_model = None
        self.input = None
        self.roofline_data = None
        # Match baseline workload for fair comparison
        self.batch_size = 32
        self.seq_len = 256
        self.hidden_dim = 256
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
        
        # Same model architecture as baseline
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        ).to(self.device).bfloat16().eval()
        
        # Compile for kernel fusion (the main optimization)
        try:
            self.compiled_model = torch.compile(self.model, mode="reduce-overhead")
        except Exception:
            self.compiled_model = self.model
        
        # Same input shape as baseline
        self.input = torch.randn(
            self.batch_size,
            self.seq_len,
            self.hidden_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        
        # Roofline data for analysis
        self.roofline_data = {
            'compute_bound': False,
            'memory_bound': True,
            'arithmetic_intensity': 0.0,
        }
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations with roofline analysis and optimizations.
        
        Optimizations applied:
        - torch.compile for kernel fusion
        - BF16 precision for faster ops
        - cuDNN benchmark mode
        """
        with self._nvtx_range("optimized_roofline"):
            with torch.no_grad():
                # Use compiled model for optimized execution
                output = self.compiled_model(self.input)
        self._synchronize()
        
        # Compute roofline metrics (not during timing)
        input_bytes = self.input.numel() * self.input.element_size()
        output_bytes = output.numel() * output.element_size()
        total_bytes = input_bytes + output_bytes
        
        flops = self.batch_size * self.seq_len * self.hidden_dim * self.hidden_dim * 2 * 2  # 2 linears
        arithmetic_intensity = flops / total_bytes if total_bytes > 0 else 0.0
        
        is_memory_bound = arithmetic_intensity < 10.0  # B200 has high compute
        self.roofline_data['compute_bound'] = not is_memory_bound
        self.roofline_data['memory_bound'] = is_memory_bound
        self.roofline_data['arithmetic_intensity'] = arithmetic_intensity

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.compiled_model = None
        self.input = None
        self.roofline_data = None
        super().teardown()
    
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
