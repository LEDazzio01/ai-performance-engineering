"""optimized disaggregated - Optimized disaggregated inference. Implements Benchmark protocol for harness integration."""

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
        self._prefill_stream = None
        self._decode_stream = None
    
    def setup(self) -> None:
        """Setup: Initialize separate models for prefill and decode."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(42)
        
        # Optimization: Disaggregated inference
        # Separate models/resources for prefill and decode phases
        # Prefill: Parallel processing, compute-intensive, can use multiple GPUs
        # Decode: Autoregressive, latency-sensitive, dedicated GPU resources
        
        # Prefill model (optimized for parallel processing)
        prefill_model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        ).to(self.device).eval()
        
        # Decode model (optimized for latency)
        decode_model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        ).to(self.device).eval()
        
        # Optimization: Use FP16 for faster computation - this is the key optimization
        # Concurrent streams provide additional benefit, but FP16 is the main speedup
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        try:
            prefill_model = prefill_model.half()
            decode_model = decode_model.half()
        except Exception:
            dtype = torch.float32
        
        self.prefill_model = prefill_model
        self.decode_model = decode_model
        
        # Scale inputs appropriately - concurrent execution benefits require sufficient work
        # But not so large that it dominates the benchmark time
        self.prefill_input = torch.randn(8, 4096, 256, device=self.device, dtype=dtype)  # Large context
        self.decode_input = torch.randn(8, 1, 256, device=self.device, dtype=dtype)  # Single token
        
        # Create streams once in setup to reduce overhead in benchmark loop
        self._prefill_stream = torch.cuda.Stream()
        self._decode_stream = torch.cuda.Stream()
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Disaggregated inference."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_disaggregated", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: Disaggregated inference with optimized execution
                # Key insight: Prefill and decode can run concurrently on different streams
                # This eliminates interference and improves GPU utilization
                
                # Optimization: Process prefill and decode concurrently on separate streams
                # Key insight: With large enough workloads, concurrent execution reduces total time
                # Total time ≈ max(prefill_time, decode_time) instead of sum
                # Streams created in setup to reduce overhead
                
                # Process concurrently - prefill is much larger, decode runs in parallel
                # The concurrent execution is the key optimization
                with torch.cuda.stream(self._prefill_stream):
                    prefill_output = self.prefill_model(self.prefill_input)
                
                with torch.cuda.stream(self._decode_stream):
                    decode_output = self.decode_model(self.decode_input)
                
                # Single synchronization point for both operations
                # This is faster than sequential execution where total_time = prefill_time + decode_time
                torch.cuda.synchronize()
                
                # Optimization: Disaggregated inference benefits
                # - Concurrent execution eliminates interference
                # - Better GPU utilization (both phases run simultaneously)
                # - Lower latency (decode doesn't wait for prefill)
                # - Single synchronization point reduces overhead
                _ = prefill_output.sum() + decode_output.sum()

    
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
            iterations=100,
            warmup=20,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.prefill_model is None or self.decode_model is None:
            return "Models not initialized"
        if self.prefill_input is None or self.decode_input is None:
            return "Inputs not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedDisaggregatedBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Disaggregated: {result.mean_ms:.3f} ms")
