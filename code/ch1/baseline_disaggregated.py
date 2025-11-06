"""baseline disaggregated - Baseline monolithic inference. Implements Benchmark protocol for harness integration."""

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


class BaselineDisaggregatedBenchmark(Benchmark):
    """Baseline: Monolithic inference (prefill and decode share resources).
    
    Disaggregated inference: This baseline does not separate prefill and decode phases.
    Both phases compete for same GPU resources, causing interference and poor utilization.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.prefill_input = None
        self.decode_input = None
    
    def setup(self) -> None:
        """Setup: Initialize model and inputs."""
        torch.manual_seed(42)
        # Baseline: Monolithic inference - prefill and decode share same resources
        # Disaggregated inference separates prefill (parallel) and decode (autoregressive)
        # This baseline does not separate prefill and decode phases
        self.model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        ).to(self.device).eval()
        
        # Simulate prefill (long context) and decode (single token) inputs
        # Match optimized scale for fair comparison
        self.prefill_input = torch.randn(8, 4096, 256, device=self.device)  # Match optimized scale
        self.decode_input = torch.randn(8, 1, 256, device=self.device)  # Single token
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
                # Prefill and decode phases share same resources
                # This causes interference - prefill blocks decode and vice versa
                # Disaggregated inference separates these phases for better efficiency
                
                # Process prefill (long context) - competes with decode for resources
                prefill_output = self.model(self.prefill_input)
                
                # Process decode (autoregressive) - competes with prefill for resources
                decode_output = self.model(self.decode_input)
                
                # Baseline: No separation - both phases interfere with each other
                # This leads to poor GPU utilization and latency spikes
                _ = prefill_output.sum() + decode_output.sum()

    
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
        if self.prefill_input is None or self.decode_input is None:
            return "Inputs not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineDisaggregatedBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Disaggregated (Monolithic): {result.mean_ms:.3f} ms")
