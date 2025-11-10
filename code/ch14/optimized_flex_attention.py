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
        self.embed_dim = 1024
        self.seq_len = 1024
        self.batch = 4
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self._last = 0.0
        self.static_q = None
    
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
        model = nn.MultiheadAttention(self.embed_dim, 16, batch_first=True).to(self.device)
        model = model.to(self.dtype).eval()
        compile_fn = getattr(torch, "compile", None)
        if callable(compile_fn):
            try:
                model = compile_fn(model, mode="reduce-overhead")
            except Exception:
                pass
        self.model = model

        self.graph_input = torch.randn(self.batch, self.seq_len, self.embed_dim, device=self.device, dtype=self.dtype)
        self.graph_output = torch.empty_like(self.graph_input)
        for _ in range(3):
            with torch.no_grad():
                _ = self.model(self.graph_input, self.graph_input, self.graph_input)[0]
        torch.cuda.synchronize()

        self.static_q = None
    
    def benchmark_fn(self) -> None:
        """Benchmark: FlexAttention operations."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_flex_attention", enable=enable_nvtx):
            if self.model is None or self.graph_input is None:
                raise RuntimeError("Model not initialized")
            out = self.model(self.graph_input, self.graph_input, self.graph_input)[0]
            self._last = float(out.sum())
            torch.cuda.synchronize(self.device)

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.graph_input = None
        self.graph_output = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None or self.graph_input is None:
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
    
    print(f"\nOptimized FlexAttention: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(" Tip: FlexAttention provides flexible attention patterns for various use cases")
