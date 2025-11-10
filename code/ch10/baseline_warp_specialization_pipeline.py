"""baseline_warp_specialization.py - Baseline without warp specialization.

Demonstrates sequential processing without warp specialization.
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
    BenchmarkHarness,
    BenchmarkMode,
)
from ch10.workload_config import WORKLOAD, is_smoke_test


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch10")
    return torch.device("cuda")


class BaselineWarpSpecializationPipelineBenchmark(Benchmark):
    """Baseline: Sequential processing without warp specialization or pipelining."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.inputs_host = None
        self.workload = WORKLOAD
        self.smoke_test = is_smoke_test()
        self.micro_batches = self.workload.pipeline_micro_batches_for_mode(self.smoke_test)
        self.chunk_tokens = self.workload.pipeline_chunk_tokens_for_mode(self.smoke_test)
        self.hidden_dim = self.workload.pipeline_hidden_dim_for_mode(self.smoke_test)
        self._checksum = 0.0
    
    def setup(self) -> None:
        """Setup: Initialize model without warp specialization."""
        torch.manual_seed(42)
        
        # Baseline: Sequential model - no warp specialization
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        ).to(self.device).eval()
        
        # Match optimized workload size
        self.inputs_host = torch.randn(
            self.micro_batches,
            self.chunk_tokens,
            self.hidden_dim,
            pin_memory=True,
        )
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Sequential forward pass."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("baseline_warp_specialization_pipeline", enable=enable_nvtx):
            with torch.no_grad():
                # Baseline: Sequential processing
                # All warps do the same work - no specialization
                total = 0.0
                for idx in range(self.micro_batches):
                    host_chunk = self.inputs_host[idx]
                    device_chunk = host_chunk.to(self.device, non_blocking=False)
                    output = self.model(device_chunk)
                    total += float(output.sum().item())
                    torch.cuda.synchronize()
                self._checksum = total
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.inputs_host = None
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
        if self.inputs_host is None:
            return "Inputs not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineWarpSpecializationPipelineBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    config = benchmark.get_config()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Warp Specialization (Pipeline): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
