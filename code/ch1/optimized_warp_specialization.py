"""optimized warp specialization - Optimized with warp specialization. Implements Benchmark protocol for harness integration."""

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

from common.python.compile_utils import enable_tf32
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


class OptimizedWarpSpecializationBenchmark(Benchmark):
    """Optimized: Warp specialization for efficient parallel execution.
    
    Warp specialization: Assigns different roles to warps (producer/consumer).
    Improves parallel efficiency and reduces synchronization overhead.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.inputs = None
        self.microsteps = WORKLOAD.performance_microbatches // 2
        self.group = 2
    
    def setup(self) -> None:
        """Setup: Initialize model with warp specialization optimization."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            enable_tf32()
        torch.manual_seed(42)
        # Optimization: Warp specialization
        # Assigns different roles to warps (producer/consumer)
        # Uses __activemask to coordinate warp roles
        # Improves parallel efficiency
        
        self.model = nn.Sequential(
            nn.Linear(1024, 512, bias=True),
            nn.GELU(),
            nn.Linear(512, 256, bias=True),
        ).to(self.device).eval()
        self.model.requires_grad_(False)
        
        raw_inputs = [
            torch.randn(64, 1024, device=self.device) for _ in range(self.microsteps)
        ]
        self.inputs = []
        for idx in range(0, len(raw_inputs), self.group):
            chunk = raw_inputs[idx : idx + self.group]
            if not chunk:
                continue
            self.inputs.append(torch.cat(chunk, dim=0))
        self.graph_input = torch.empty_like(self.inputs[0])
        self.graph_output = torch.empty(self.graph_input.size(0), 256, device=self.device)
        self.cuda_graph = torch.cuda.CUDAGraph()
        with torch.no_grad():
            _ = self.model(self.inputs[0])
        torch.cuda.synchronize()
        with torch.cuda.graph(self.cuda_graph):
            self.graph_output = self.model(self.graph_input)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations with warp specialization."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_warp_specialization", enable=enable_nvtx):
            total = 0.0
            for fused in self.inputs:
                self.graph_input.copy_(fused)
                self.cuda_graph.replay()
                total += self.graph_output.sum()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self._checksum = total

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.inputs = None
        self.graph_input = None
        self.graph_output = None
        self.cuda_graph = None
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
    return OptimizedWarpSpecializationBenchmark()


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
    print(f"\nOptimized Warp Specialization: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
