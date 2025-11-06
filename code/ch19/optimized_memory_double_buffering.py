"""optimized_memory_double_buffering.py - Optimized memory management with double buffering.

Demonstrates double buffering (ping-pong) for overlapping memory operations.
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

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch19")
    return torch.device("cuda")

class OptimizedMemoryDoubleBufferingBenchmark(Benchmark):
    """Optimized: double buffering for overlapping operations."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None

        self.buffer_a = None
        self.buffer_b = None
        self.stream1 = None
        self.stream2 = None
        self.batch_size = 4
        self.seq_len = 1024
        self.hidden_dim = 1024
    
    def setup(self) -> None:
        """Setup: Initialize model and double buffers."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )
        self.model = self.model.to(self.device).to(dtype=torch.bfloat16)
        # Optimization: Use FP16 for faster computation
        if self.device.type == "cuda":
            try:
                self.model = self.model.half()
            except Exception:
                pass  # Fallback to FP32 if FP16 not supported
        self.model.eval()
        
        # Optimization: Double buffering (ping-pong buffers)
        # Two buffers allow overlapping copy and compute operations
        self.buffer_a = torch.randn(
            self.batch_size, self.seq_len, self.hidden_dim,
            device=self.device, dtype=torch.bfloat16
        )
        self.buffer_b = torch.empty_like(self.buffer_a)
        
        # Create streams for overlapping operations
        self.stream1 = torch.cuda.Stream()
        self.stream2 = torch.cuda.Stream()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Double buffering with overlapping operations."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_memory_double_buffering", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: Double buffering (ping-pong)
                # While processing buffer_a on stream1, copy buffer_b on stream2
                # This overlaps memory transfers with compute
                
                # Process buffer_a on stream1
                with torch.cuda.stream(self.stream1):
                    output_a = self.model(self.buffer_a)
                
                # Copy buffer_b on stream2 (overlaps with compute)
                with torch.cuda.stream(self.stream2):
                    self.buffer_b.copy_(self.buffer_a, non_blocking=True)
                
                # Synchronize streams
                self.stream1.synchronize()
                self.stream2.synchronize()
                
        # Swap buffers for next iteration
        self.buffer_a, self.buffer_b = self.buffer_b, self.buffer_a

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.buffer_a, self.buffer_b
        if self.stream1:
            self.stream1 = None
        if self.stream2:
            self.stream2 = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.buffer_a is None or self.buffer_b is None:
            return "Buffers not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedMemoryDoubleBufferingBenchmark()

def main() -> None:
    """Standalone execution."""
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=20, warmup=5)
    )
    benchmark = OptimizedMemoryDoubleBufferingBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print("Optimized: Memory Double Buffering (Ping-Pong)")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(" Tip: Double buffering overlaps memory transfers with compute operations")

if __name__ == "__main__":
    main()
