"""baseline guided decoding - Baseline standard decoding without guidance. Implements Benchmark protocol for harness integration."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

try:
    import ch1.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

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


class BaselineGuidedDecodingBenchmark(Benchmark):
    """Baseline: Standard decoding without guidance (free-form generation).
    
    Guided decoding: This baseline does not use guided decoding.
    Generates tokens freely without schema or structure constraints.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input_ids = None
        self.max_length = 20
    
    def setup(self) -> None:
        """Setup: Initialize model and input."""
        torch.manual_seed(42)
        # Baseline: Standard decoding without guidance
        # Guided decoding uses schema/constraints to guide generation
        # This baseline generates tokens freely without constraints
        
        vocab_size = 1000
        hidden_dim = 256
        
        self.model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=2
        ).to(self.device).eval()
        
        batch_size = 4
        seq_len = 10
        self.input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Standard decoding without guidance."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_guided_decoding", enable=enable_nvtx):
            with torch.no_grad():
                # Baseline: Standard decoding
                # Generates tokens freely without schema/constraints
                # No guidance - model generates any valid token
                embedded_input = torch.randn(self.input_ids.size(0), self.input_ids.size(1), 256, device=self.device)
                memory = torch.randn(self.input_ids.size(0), self.input_ids.size(1), 256, device=self.device)
                output = self.model(embedded_input, memory)
                
                # Baseline: No guidance - free-form generation
                # Cannot enforce structure or schema constraints
                _ = output.sum()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
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
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineGuidedDecodingBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Guided Decoding (Standard): {result.mean_ms:.3f} ms")
