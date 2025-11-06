"""baseline_speculative_decoding.py - Baseline decoding without speculative execution.

Demonstrates standard autoregressive decoding without speculative decoding optimization.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

try:
    import ch1.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass
import torch.nn as nn

from common.python.compile_utils import compile_model

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch1")
    return torch.device("cuda")


class BaselineSpeculativeDecodingBenchmark(Benchmark):
    """Baseline: Standard autoregressive decoding (no speculative execution)."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.embedding = None
        self.input_ids = None
        self.max_length = 20
    
    def setup(self) -> None:
        """Setup: Initialize model and input."""
        # Simple decoder model with embedding layer
        # TransformerDecoder expects embedded inputs, not raw token IDs
        vocab_size = 1000
        d_model = 256
        self.embedding = nn.Embedding(vocab_size, d_model).to(self.device)
        self.model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=8, batch_first=True),
            num_layers=2
        )
        self.model = self.model.to(self.device).eval()
        
        # Baseline: Standard decoding - generate tokens one at a time
        # No speculative decoding - cannot generate multiple tokens in parallel
        batch_size = 4
        seq_len = 10
        self.input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Standard autoregressive decoding."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_speculative_decoding", enable=enable_nvtx):
            with torch.no_grad():
                # Baseline: Standard autoregressive decoding
                # Generate tokens sequentially, one at a time
                # No speculative execution - cannot parallelize token generation
                # Embed input tokens (TransformerDecoder expects embedded inputs)
                embedded_input = self.embedding(self.input_ids)
                # Create memory tensor (simulating encoder output in decoder-only scenario)
                memory = torch.randn(self.input_ids.size(0), self.input_ids.size(1), 256, device=self.device)
                output = self.model(embedded_input, memory)
                # Baseline: sequential processing without speculation
                _ = output.sum()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.embedding = None
        self.input_ids = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineSpeculativeDecodingBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Speculative Decoding (Standard): {result.mean_ms:.3f} ms")
    print("NOTE: Standard autoregressive decoding - tokens generated sequentially, no speculation")

