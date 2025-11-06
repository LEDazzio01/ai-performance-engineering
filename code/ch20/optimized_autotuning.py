"""optimized_autotuning.py - Optimized with autotuning in AI optimization context.

Demonstrates operations with autotuning to find optimal parameters.
Autotuning: Automatically finds optimal kernel parameters through search/optimization.
Adapts kernel configuration to workload for maximum performance.
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
        raise RuntimeError("CUDA required for ch20")
    return torch.device("cuda")


class OptimizedAutotuningBenchmark(Benchmark):
    """Optimized: Autotuning to find optimal parameters.
    
    Autotuning: Automatically finds optimal kernel parameters through search/optimization.
    Adapts kernel configuration (tile sizes, block sizes, etc.) to workload for maximum performance.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input = None
    
    def setup(self) -> None:
        """Setup: Initialize model with autotuning."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: Autotuning to find optimal parameters
        # Automatically searches for best kernel configuration
        
        self.model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        
        # Optimization: Autotuning enabled
        # torch.compile with max-autotune mode explores optimal configurations
        # Searches tile sizes, block sizes, loop unrolling, etc.
        self.model = self.model.to(self.device)
        # Optimization: Use FP16 for faster computation
        if self.device.type == "cuda":
            try:
                self.model = self.model.half()
            except Exception:
                pass  # Fallback to FP32 if FP16 not supported
        self.model.eval()
        # Enable autotuning via torch.compile with optimal mode
        # This triggers automatic kernel parameter search
        # Select optimal compile mode based on GPU SM count
        from common.python.compile_utils import get_optimal_compile_mode
        compile_mode = get_optimal_compile_mode("max-autotune")
        try:
            self.model = torch.compile(self.model, mode=compile_mode, backend="inductor")
        except Exception:
            # Fallback if torch.compile not available
            self.model = compile_model(self.model, mode=compile_mode)
        
        self.input = torch.randn(4, 32, 256, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations with autotuned parameters."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_autotuning", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: Autotuned kernel configuration
                # Parameters (tile sizes, block sizes, etc.) are optimized for this workload
                # Autotuning explores search space to find best configuration
                output = self.model(self.input)

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
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
        if self.input is None:
            return "Input not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedAutotuningBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
    mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedAutotuningBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: autotuning")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
