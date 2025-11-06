"""baseline_disaggregated.py - Baseline monolithic inference in FlexAttention/KV cache context.

Demonstrates monolithic inference without disaggregation.
Disaggregated: This baseline does not use disaggregated inference.
Combines prefill and decode in single service.
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


class BaselineDisaggregatedBenchmark(Benchmark):
    """Baseline: Monolithic inference (no disaggregation).
    
    Disaggregated: This baseline does not use disaggregated inference.
    Combines prefill and decode in single service, blocking each other.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.prefill_input = None
        self.decode_input = None
    
    def setup(self) -> None:
        """Setup: Initialize model."""
        torch.manual_seed(42)
        # Baseline: Monolithic inference - single service
        # Disaggregated inference separates prefill (parallel) and decode (autoregressive)
        # This baseline does not use disaggregated inference
        
        hidden_dim = 256
        self.model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=2
        ).to(self.device).eval()
        
        # Prefill and decode inputs (monolithic)
        self.prefill_input = torch.randn(4, 32, hidden_dim, device=self.device)
        self.decode_input = torch.randn(4, 1, hidden_dim, device=self.device)
        # Create dummy memory tensor for TransformerDecoder (encoder output)
        self.memory = torch.randn(4, 32, hidden_dim, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Monolithic inference."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_disaggregated", enable=enable_nvtx):
            with torch.no_grad():
                # Baseline: Monolithic inference
                # Prefill and decode compete for same resources
                # Disaggregated inference would separate them
                
                # Prefill phase (monolithic: blocks decode)
                # TransformerDecoder requires both tgt and memory arguments
                _ = self.model(self.prefill_input, self.memory)
                torch.cuda.synchronize()
                
                # Decode phase (monolithic: blocked by prefill)
                _ = self.model(self.decode_input, self.memory)
                
                # Baseline: No disaggregated inference
                # Prefill and decode block each other

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.prefill_input = None
        self.decode_input = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineDisaggregatedBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineDisaggregatedBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: Disaggregated")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
