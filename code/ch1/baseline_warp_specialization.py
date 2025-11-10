"""baseline warp specialization - Baseline without warp specialization. Implements Benchmark protocol for harness integration."""

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


class BaselineWarpSpecializationBenchmark(Benchmark):
    """Baseline: No warp specialization - all warps do same work.
    
    Warp specialization: This baseline does not use warp specialization.
    All warps execute same operations, not leveraging specialized warp roles.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.inputs = None
        self.microsteps = WORKLOAD.performance_microbatches // 2
    
    def setup(self) -> None:
        """Setup: Initialize model without warp specialization."""
        torch.manual_seed(42)
        # Baseline: No warp specialization
        # Warp specialization assigns different roles to warps (producer/consumer)
        # This baseline does not use warp specialization
        
        self.model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        ).to(self.device).eval()
        
        self.inputs = [
            torch.randn(64, 1024, device=self.device) for _ in range(self.microsteps)
        ]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations without warp specialization."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_warp_specialization", enable=enable_nvtx):
            total = 0.0
            for microbatch in self.inputs:
                fragments = microbatch.chunk(8, dim=0)
                stage_outputs = []
                for frag in fragments:
                    hidden = torch.nn.functional.linear(frag, self.model[0].weight, self.model[0].bias)
                    hidden = torch.relu(hidden)
                    stage_outputs.append(torch.nn.functional.linear(hidden, self.model[2].weight, self.model[2].bias))
                total += torch.cat(stage_outputs, dim=0).sum()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self._checksum = total

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.inputs = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
            use_subprocess=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.inputs is None:
            return "Input not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineWarpSpecializationBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    config = benchmark.get_config()
    config.use_subprocess = False
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=config
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Warp Specialization: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
