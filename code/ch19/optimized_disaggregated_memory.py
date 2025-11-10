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
        self.output = None
        self.prefill_stream = None
        self.decode_stream = None
        self.transfer_event = None
        self.kv_cache_host = None
        self.decode_buffer = None
        self.hidden_dim = 256
        self.prefill_tokens = 2048
        self.decode_tokens = 128
    
    def setup(self) -> None:
        """Setup: Initialize separate models with disaggregated memory."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: Disaggregated memory management
        # Separates memory pools for prefill (parallel) and decode (autoregressive)
        # Prefill and decode use separate memory allocations
        # This avoids interference and improves memory efficiency
        
        # Separate models for prefill and decode (in practice, could share weights)
        self.prefill_model = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        self.decode_model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.prefill_model = self.prefill_model.to(self.device).half().eval()
        self.decode_model = self.decode_model.to(self.device).half().eval()
        
        # Prefill: long context (parallel processing, separate memory pool)
        self.prefill_input = torch.randn(
            self.prefill_tokens, self.hidden_dim, device=self.device, dtype=torch.float16
        )
        # Decode: single token (autoregressive, separate memory pool)
        self.decode_input = torch.randn(
            self.decode_tokens, self.hidden_dim, device=self.device, dtype=torch.float16
        )
        self.output = torch.empty_like(self.decode_input)
        self.prefill_stream = torch.cuda.Stream()
        self.decode_stream = torch.cuda.Stream()
        self.transfer_event = torch.cuda.Event(blocking=False)
        self.kv_cache_host = torch.empty(
            self.decode_tokens, self.hidden_dim, pin_memory=True, dtype=torch.float16
        )
        self.decode_buffer = torch.empty_like(self.decode_input)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Disaggregated memory operations."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("disaggregated_memory", enable=enable_nvtx):
            with torch.no_grad():
                # Prefill and decode run on dedicated streams so transfers overlap compute
                with torch.cuda.stream(self.prefill_stream):
                    prefill_output = self.prefill_model(self.prefill_input)
                    newest = prefill_output[-self.decode_tokens:].contiguous()
                    self.kv_cache_host.copy_(newest, non_blocking=True)
                    self.transfer_event.record(self.prefill_stream)

                with torch.cuda.stream(self.decode_stream):
                    self.decode_stream.wait_event(self.transfer_event)
                    self.decode_buffer.copy_(self.kv_cache_host, non_blocking=True)
                    decode_tokens = self.decode_buffer + self.decode_input
                    self.output = self.decode_model(decode_tokens)

            torch.cuda.synchronize()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.prefill_model = None
        self.decode_model = None
        self.prefill_input = None
        self.decode_input = None
        self.output = None
        self.prefill_stream = None
        self.decode_stream = None
        self.transfer_event = None
        self.kv_cache_host = None
        self.decode_buffer = None
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
        if self.output is None:
            return "Decode output missing"
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
    print(f"\nOptimized Disaggregated Memory: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(" Tip: Disaggregated memory management separates prefill/decode for better efficiency")
