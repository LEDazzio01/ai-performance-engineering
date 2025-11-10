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
from ch1.workload_config import WORKLOAD


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
        self.decode_tokens = None
        self.memory = None
        self.workload = WORKLOAD
        self.batch_size = self.workload.batch_size
        self.tokens_per_step = self.workload.tokens_per_step
        self.decode_steps = self.workload.decode_steps
        self.vocab_size = 1000
    
    def setup(self) -> None:
        """Setup: Initialize model and input."""
        # Simple decoder model with embedding layer
        # TransformerDecoder expects embedded inputs, not raw token IDs
        d_model = 256
        self.embedding = nn.Embedding(self.vocab_size, d_model).to(self.device)
        self.model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=8, batch_first=True),
            num_layers=2
        )
        self.model = self.model.to(self.device).eval()
        
        # Baseline: Standard decoding - generate tokens one at a time
        # No speculative decoding - cannot generate multiple tokens in parallel
        # Pre-generate the decode stream so every iteration replays the same workload.
        torch.manual_seed(42)
        decode = torch.randint(
            0,
            self.vocab_size,
            (self.decode_steps, self.batch_size, self.tokens_per_step),
            device=self.device,
        )
        self.decode_tokens = decode
        self.memory = torch.randn(
            self.batch_size, self.tokens_per_step, d_model, device=self.device
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Standard autoregressive decoding."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_speculative_decoding", enable=enable_nvtx):
            with torch.no_grad():
                for step in range(self.decode_steps):
                    tokens = self.decode_tokens[step]
                    embedded = self.embedding(tokens)
                    memory = self.memory.expand(self.batch_size, self.tokens_per_step, -1)
                    output = self.model(embedded, memory)
                    _ = output.sum()
            if self.device.type == "cuda":
                torch.cuda.synchronize()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.embedding = None
        self.input_ids = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=5,
            warmup=2,
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
    timing = result.timing
    if timing:
        print(f"\nBaseline Speculative Decoding (Standard): {timing.mean_ms:.3f} ms")
    else:
        print("\nBaseline Speculative Decoding (Standard): No timing data available")
    print("NOTE: Standard autoregressive decoding - tokens generated sequentially, no speculation")
