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
from ch1.workload_config import WORKLOAD


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
        self.prefill_inputs = None
        self.decode_inputs = None
        self.workload = WORKLOAD
        self.batch_size = self.workload.batch_size
        self.tokens_per_step = self.workload.tokens_per_step
        self.decode_steps = max(1, self.workload.decode_steps // 2)
        self.prefill_chunks = max(2, self.workload.prefill_chunks // 2)
    
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
        
        # Simulate prefill (long context) and decode inputs using a shared
        # workload so every iteration replays the same compute footprint.
        token_dim = 256
        samples_per_batch = self.batch_size * self.tokens_per_step
        torch.manual_seed(42)
        self.prefill_inputs = torch.randn(
            self.prefill_chunks, samples_per_batch, token_dim, device=self.device
        )
        self.decode_inputs = torch.randn(
            self.decode_steps, samples_per_batch, token_dim, device=self.device
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Monolithic inference."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_disaggregated", enable=enable_nvtx):
            with torch.no_grad():
                for step in range(self.decode_steps):
                    prefill_batch = self.prefill_inputs[step % self.prefill_chunks]
                    decode_batch = self.decode_inputs[step]
                    prefill_output = self.model(prefill_batch)
                    decode_output = self.model(decode_batch)
                    _ = prefill_output.sum() + decode_output.sum()
            if self.device.type == "cuda":
                torch.cuda.synchronize()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.prefill_inputs = None
        self.decode_inputs = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=5,
            warmup=1,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.prefill_inputs is None or self.decode_inputs is None:
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
    print(f"\nBaseline Disaggregated (Monolithic): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
