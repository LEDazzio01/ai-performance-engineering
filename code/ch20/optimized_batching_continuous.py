"""optimized_batching_continuous.py - Continuous batching optimization (optimized).

Continuous batching processes requests as they arrive, without waiting for full batch.
Efficiently handles variable-length sequences with better GPU utilization.

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
        raise RuntimeError("CUDA required for ch20")
    return torch.device("cuda")

class SimpleModel(nn.Module):
    """Simple model for batching demonstration."""
    
    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class OptimizedBatchingContinuousBenchmark(Benchmark):
    """Continuous batching optimization - processes requests immediately."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        # Optimization: Compile model for kernel fusion and optimization
        try:

        # Optimization: Compile model for kernel fusion and optimization
        try:

        self.request_queue = None
        self.hidden_dim = 1024
    
    def setup(self) -> None:
        """Setup: Initialize model and prepare requests."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        
        self.model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device).half().eval()
        
        # Simulate incoming requests with different sequence lengths
        # Continuous batching: process requests as they arrive, group by length
        self.request_queue = []
        for i in range(8):
            seq_len = 64 + (i * 8)  # Different lengths: 64, 72, 80, ...
            x = torch.randn(1, seq_len, self.hidden_dim, device=self.device, dtype=torch.float16)
            self.request_queue.append(x)
        
        # Warmup
        with torch.no_grad():
            # Process requests grouped by similar length (continuous batching strategy)
            for req in self.request_queue[:3]:
                _ = self.model(req)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - continuous batching processes immediately."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_batching_continuous", enable=enable_nvtx):
            # Continuous batching: group requests by similar length and process immediately
            # No waiting for full batch - better GPU utilization
            
            # Group requests by sequence length for efficient batching
            requests_by_length = {}
            for req in self.request_queue:
                seq_len = req.size(1)
                if seq_len not in requests_by_length:
                    requests_by_length[seq_len] = []
                requests_by_length[seq_len].append(req)
            
            # Process each group (continuous batching)
            with torch.no_grad():
                for seq_len, reqs in requests_by_length.items():
                    if len(reqs) > 1:
                        # Batch multiple requests of same length
                        batch = torch.cat(reqs, dim=0)
                        _ = self.model(batch)
                    else:
                        # Process single request immediately (no waiting)
                        _ = self.model(reqs[0])

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.request_queue
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedBatchingContinuousBenchmark()

if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Continuous Batching: {result.mean_ms:.3f} ms")

