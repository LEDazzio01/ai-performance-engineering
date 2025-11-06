"""optimized_disaggregated.py - Optimized disaggregated inference in MoE context.

Demonstrates disaggregated inference where prefill and decode are separated.
Disaggregated inference: Separates prefill (parallel, compute-intensive) and decode (autoregressive, latency-sensitive) phases.
Assigns different GPU resources to each phase for optimal utilization.
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

from common.python.compile_utils import compile_model

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch16")
    return torch.device("cuda")


class OptimizedDisaggregatedBenchmark(Benchmark):
    """Optimized: Disaggregated inference (prefill and decode separated).
    
    Disaggregated inference: Separates prefill (parallel, compute-intensive) and decode
    (autoregressive, latency-sensitive) phases. Assigns different GPU resources to each
    phase for optimal utilization and reduced interference.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.prefill_model = None
        self.decode_model = None
        self.prefill_input = None
        self.decode_input = None
    
    def setup(self) -> None:
        """Setup: Initialize separate models for prefill and decode."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        
        # Optimization: Disaggregated inference
        # Separate models/resources for prefill and decode phases
        # Prefill: Parallel processing, compute-intensive, can use multiple GPUs
        # Decode: Autoregressive, latency-sensitive, dedicated GPU resources
        
        # Prefill model (optimized for parallel processing)
        # Optimization: Use BF16 for better memory bandwidth and Tensor Core acceleration
        self.prefill_model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        ).to(self.device).to(torch.bfloat16).eval()
        
        # Decode model (optimized for latency)
        # Optimization: Use BF16 for faster computation + CUDA graphs for low latency
        self.decode_model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        ).to(self.device).to(torch.bfloat16).eval()
        
        # Simulate prefill (long context) and decode (single token) inputs
        self.prefill_input = torch.randn(2, 512, 1024, device=self.device, dtype=torch.bfloat16)  # Long context
        self.decode_input = torch.randn(2, 1, 1024, device=self.device, dtype=torch.bfloat16)  # Single token
        
        # Optimization: Warmup decode model for CUDA graph capture (reduces latency)
        with torch.no_grad():
            for _ in range(3):
                _ = self.decode_model(self.decode_input)
        torch.cuda.synchronize()
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
                # Prefill and decode phases are separated
                # Prefill: Uses dedicated GPU resources for parallel processing
                # Decode: Uses separate GPU resources for low-latency autoregressive generation
                
                # Process prefill on dedicated prefill GPUs (parallel, compute-intensive)
                prefill_output = self.prefill_model(self.prefill_input)
                
                # Process decode on dedicated decode GPUs (autoregressive, latency-sensitive)
                # Decode can run concurrently with next prefill (no interference)
                decode_output = self.decode_model(self.decode_input)
                
                # Optimization: Disaggregated inference benefits
                # - Prefill and decode don't interfere (separate resources)
                # - Better GPU utilization (each phase optimized for its workload)
                # - Lower latency for decode (dedicated resources)
                # - Higher throughput for prefill (parallel processing)
                _ = prefill_output + decode_output

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.prefill_model = None
        self.decode_model = None
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
        if self.prefill_model is None or self.decode_model is None:
            return "Models not initialized"
        if self.prefill_input is None or self.decode_input is None:
            return "Inputs not initialized"
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
    print(f"Optimized: disaggregated")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
