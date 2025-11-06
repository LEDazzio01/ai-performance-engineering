"""optimized_disaggregated.py - Optimized disaggregated inference in FlexAttention/KV cache context.

Demonstrates disaggregated inference separating prefill and decode.
Disaggregated: Separates prefill (parallel) and decode (autoregressive) stages.
Improves utilization by allowing parallel execution.
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
        raise RuntimeError("CUDA required for ch18")
    return torch.device("cuda")

class OptimizedDisaggregatedBenchmark(Benchmark):
    """Optimized: Disaggregated inference separating prefill and decode.
    
    Disaggregated: Separates prefill (parallel) and decode (autoregressive) stages.
    Improves utilization by allowing parallel execution.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.prefill_model = None
        # Optimization: Compile model for kernel fusion and optimization
        try:

        self.decode_model = None
        self.prefill_input = None
        self.decode_input = None
        self.prefill_stream = None
        self.decode_stream = None
    
    def setup(self) -> None:
        """Setup: Initialize disaggregated models."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: Disaggregated inference
        # Separates prefill (parallel) and decode (autoregressive) stages
        # Disaggregated: allows parallel execution
        
        hidden_dim = 256
        
        # Prefill model (optimized for parallel processing)
        self.prefill_model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=2
        ).to(self.device).eval()
        
        # Decode model (optimized for autoregressive processing)
        self.decode_model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=2
        ).to(self.device).eval()
        
        # Separate streams for disaggregated execution
        self.prefill_stream = torch.cuda.Stream()
        self.decode_stream = torch.cuda.Stream()
        
        self.prefill_input = torch.randn(4, 32, hidden_dim, device=self.device)
        self.decode_input = torch.randn(4, 1, hidden_dim, device=self.device)
        # Create dummy memory tensor for TransformerDecoder (encoder output)
        self.memory = torch.randn(4, 32, hidden_dim, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Disaggregated inference."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_disaggregated", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: Disaggregated inference
                # Prefill and decode execute in parallel on separate streams
                # Disaggregated: allows parallel execution
                
                # Prefill phase (disaggregated: parallel execution)
                # TransformerDecoder requires both tgt and memory arguments
                with torch.cuda.stream(self.prefill_stream):
                    _ = self.prefill_model(self.prefill_input, self.memory)
                
                # Decode phase (disaggregated: parallel with prefill)
                # TransformerDecoder requires both tgt and memory arguments
                with torch.cuda.stream(self.decode_stream):
                    _ = self.decode_model(self.decode_input, self.memory)
                
                # Synchronize streams (disaggregated: ensure completion)
                self.prefill_stream.synchronize()
                self.decode_stream.synchronize()
                
                # Optimization: Disaggregated inference benefits
                # - Separates prefill and decode stages
                # - Parallel execution improves utilization
                # - Better resource allocation
                # - Improved throughput through disaggregation

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.prefill_model = None
        self.decode_model = None
        self.prefill_input = None
        self.decode_input = None
        self.prefill_stream = None
        self.decode_stream = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.prefill_model is None or self.decode_model is None:
            return "Models not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedDisaggregatedBenchmark()

def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedDisaggregatedBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: Disaggregated")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")

if __name__ == "__main__":
    main()
