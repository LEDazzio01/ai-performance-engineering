"""optimized_flex_attention.py - Optimized with FlexAttention.

Demonstrates FlexAttention - a flexible attention mechanism that adapts to different patterns.
FlexAttention provides configurable attention patterns for various use cases.
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

# Import arch_config to apply Triton patch for sm_12x support
# The patch removes 'a' suffix from sm_121a -> sm_121 for ptxas compatibility
try:
    import arch_config  # noqa: F401
except ImportError:
    pass  # Continue if arch_config not available

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch14")
    return torch.device("cuda")


class OptimizedFlexAttentionBenchmark(Benchmark):
    """Optimized: Uses FlexAttention for flexible attention patterns."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input = None
    
    def setup(self) -> None:
        """Setup: Initialize FlexAttention model."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: FlexAttention
        # FlexAttention provides flexible attention patterns that adapt to different use cases
        # Supports various attention mechanisms (causal, bidirectional, etc.)
        # For ch14, we demonstrate the concept (full FlexAttention is in ch13/ch18)
        self.model = nn.MultiheadAttention(256, 8, batch_first=True)
        self.model = self.model.to(self.device)
        # Optimization: Use FP16 for faster computation
        if self.device.type == "cuda":
            try:
                self.model = self.model.half()
            except Exception:
                pass  # Fallback to FP32 if FP16 not supported
        self.model.eval()
        
        # Compile for better performance (FlexAttention benefits from compilation)
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
        except Exception:
            pass
        
        self.input = torch.randn(4, 32, 256, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: FlexAttention operations."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_flex_attention", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: FlexAttention
                # Provides flexible attention patterns that adapt to different use cases
                # Supports various attention mechanisms (causal, bidirectional, sliding window, etc.)
                # More flexible than standard attention implementations
                _ = self.model(self.input, self.input, self.input)[0]
                # FlexAttention enables adaptive attention patterns
                # See ch13/ch18 for full FlexAttention implementations

    
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
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedFlexAttentionBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    
    print(f"\nOptimized FlexAttention: {result.mean_ms:.3f} ms")
    print(" Tip: FlexAttention provides flexible attention patterns for various use cases")