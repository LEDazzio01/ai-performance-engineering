"""optimized_attention.py - Optimized attention with tensor core acceleration in GEMM context.

Demonstrates attention optimized for tensor cores.
Attention: Uses tensor core-optimized attention kernels.
Leverages tensor cores for efficient attention computation.
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
        raise RuntimeError("CUDA required for ch10")
    return torch.device("cuda")

class OptimizedAttentionBenchmark(Benchmark):
    """Optimized: Tensor core-optimized attention.
    
    Attention: Uses tensor core-optimized attention kernels.
    Leverages tensor cores for efficient attention computation.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        # Optimization: Compile model for kernel fusion and optimization
        try:

        # Optimization: Compile model for kernel fusion and optimization
        try:

        self.input = None
    
    def setup(self) -> None:
        """Setup: Initialize tensor core-optimized attention model."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(42)
        # Optimization: Tensor core-optimized attention
        # Uses FP16/BF16 precision to leverage tensor cores
        
        hidden_dim = 256
        num_heads = 8
        
        self.model = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        ).to(self.device).to(torch.float16).eval()  # Convert model to FP16 to match inputs
        
        # Input sequence (FP16 for tensor core acceleration)
        batch_size = 4
        seq_len = 128
        self.input = torch.randn(batch_size, seq_len, hidden_dim, device=self.device, dtype=torch.float16)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Tensor core-optimized attention computation."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_attention", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: Tensor core-optimized attention
                # Uses FP16/BF16 precision to leverage tensor cores
                # Attention mechanism: optimized for tensor core acceleration
                output, _ = self.model(self.input, self.input, self.input)
                
                # Optimization: Tensor core-optimized attention benefits
                # - Leverages tensor cores for matrix operations
                # - FP16/BF16 precision for tensor core acceleration
                # - Better performance through tensor core utilization

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
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
    return OptimizedAttentionBenchmark()

def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedAttentionBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: Attention")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")

if __name__ == "__main__":
    main()
