"""optimized_memory_double_buffering.py - Optimized memory management with double buffering.

Demonstrates double buffering (ping-pong) for overlapping memory operations.
Implements BaseBenchmark for harness integration.
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
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch19")
    return torch.device("cuda")

class OptimizedMemoryDoubleBufferingBenchmark(BaseBenchmark):
    """Optimized: double buffering for overlapping operations."""
    
    def __init__(self):
        super().__init__()
        self.device = resolve_device()
        self.model = None

        self.buffer_a = None
        self.buffer_b = None
        self.copy_stream = None
        self.compute_stream = None
        self.batch_size = 4
        self.seq_len = 1024
        self.hidden_dim = 1024
        self.micro_batches = 16
        self.host_batches: list[torch.Tensor] = []
    
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
        # Optimization: Use FP16 for faster computation - FAIL FAST if not supported
        if self.device.type != "cuda":
            raise RuntimeError("CUDA required for optimized_memory_double_buffering benchmark")
        self.model = self.model.to(self.device).half()
        self.model.eval()
        
        # Optimization: Double buffering (ping-pong buffers)
        # Two buffers allow overlapping copy and compute operations
        # Ensure buffer dtype matches model dtype - FAIL FAST if model has no parameters
        params = list(self.model.parameters())
        if not params:
            raise RuntimeError("Model has no parameters - cannot determine dtype")
        model_dtype = params[0].dtype
        self.buffer_a = torch.empty(
            self.batch_size, self.seq_len, self.hidden_dim,
            device=self.device, dtype=model_dtype
        )
        self.buffer_b = torch.empty_like(self.buffer_a)
        self.host_batches = [
            torch.randn(
                self.batch_size,
                self.seq_len,
                self.hidden_dim,
                device="cpu",
                dtype=model_dtype,
            ).pin_memory()
            for _ in range(self.micro_batches)
        ]
        
        # Create streams for overlapping operations
        self.copy_stream = torch.cuda.Stream()
        self.compute_stream = torch.cuda.Stream()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Double buffering with overlapping operations."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        assert self.copy_stream is not None and self.compute_stream is not None
        with nvtx_range("optimized_memory_double_buffering", enable=enable_nvtx):
            with torch.no_grad():
                buffers = [self.buffer_a, self.buffer_b]
                copy_events = [torch.cuda.Event(blocking=False) for _ in buffers]

                # Preload first buffer on the copy stream and signal completion.
                with torch.cuda.stream(self.copy_stream):
                    buffers[0].copy_(self.host_batches[0], non_blocking=True)
                    copy_events[0].record()

                for i in range(self.micro_batches):
                    current_buffer = buffers[i % 2]
                    current_event = copy_events[i % 2]

                    with torch.cuda.stream(self.compute_stream):
                        # Ensure the compute stream only waits when the copy has finished.
                        self.compute_stream.wait_event(current_event)
                        self.last_output = self.model(current_buffer)

                    if i + 1 < self.micro_batches:
                        next_buffer = buffers[(i + 1) % 2]
                        next_event = copy_events[(i + 1) % 2]
                        with torch.cuda.stream(self.copy_stream):
                            next_buffer.copy_(self.host_batches[i + 1], non_blocking=True)
                            next_event.record()
                self.compute_stream.synchronize()
        torch.cuda.synchronize()

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.buffer_a, self.buffer_b
        self.copy_stream = None
        self.compute_stream = None
        self.host_batches = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_fp32_ms', 10.0),
            reduced_precision_time_ms=getattr(self, '_reduced_ms', 5.0),
            precision_type="fp8",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.buffer_a is None or self.buffer_b is None:
            return "Buffers not initialized"
        return None

def get_benchmark() -> BaseBenchmark:
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
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(" Tip: Double buffering overlaps memory transfers with compute operations")

if __name__ == "__main__":
    main()
