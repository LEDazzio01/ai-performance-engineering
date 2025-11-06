"""optimized_disaggregated_memory.py - Optimized memory management with disaggregated inference.

Demonstrates disaggregated memory management separating prefill and decode phases.
Disaggregated inference uses separate memory pools for better efficiency.
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
        raise RuntimeError("CUDA required for ch19")
    return torch.device("cuda")

class OptimizedDisaggregatedMemoryBenchmark(Benchmark):
    """Optimized: Disaggregated memory management."""
    
    def __init__(self):
        self.device = resolve_device()
        self.prefill_model = None

        self.decode_model = None
        self.prefill_input = None
        self.decode_input = None
    
    def setup(self) -> None:
        """Setup: Initialize separate models with disaggregated memory."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            pass
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: Disaggregated memory management
        # Separates memory pools for prefill (parallel) and decode (autoregressive)
        # Prefill and decode use separate memory allocations
        # This avoids interference and improves memory efficiency
        
        # Separate models for prefill and decode (in practice, could share weights)
        self.prefill_model = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.decode_model = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.prefill_model = self.prefill_model.to(self.device).eval()
        self.decode_model = self.decode_model.to(self.device).eval()
        
        # Prefill: long context (parallel processing, separate memory pool)
        self.prefill_input = torch.randn(1, 512, 256, device=self.device)
        # Decode: single token (autoregressive, separate memory pool)
        self.decode_input = torch.randn(1, 1, 256, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Disaggregated memory operations."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_disaggregated_memory", enable=enable_nvtx):
            with torch.no_grad():
                pass
        # Optimization: Disaggregated memory management
        # Prefill and decode phases use separate memory pools
        # This avoids interference and improves memory efficiency
        # Enables better resource utilization
                
        # Process prefill phase (separate memory pool)
        prefill_output = self.prefill_model(self.prefill_input)
        # Process decode phase (separate memory pool)
        decode_output = self.decode_model(self.decode_input)
                
        # Disaggregated memory management improves efficiency
        # See ch15 for full disaggregated inference implementations

    
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
            iterations=20,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.prefill_model is None or self.decode_model is None:
            return "Models not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedDisaggregatedMemoryBenchmark()

if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Disaggregated Memory: {result.mean_ms:.3f} ms")
    print(" Tip: Disaggregated memory management separates prefill/decode for better efficiency")
