"""baseline_batching_static.py - Static batching baseline (baseline).

Static batching waits for full batch before processing, leading to low GPU utilization
and high latency for individual requests.

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


class BaselineBatchingStaticBenchmark(Benchmark):
    """Static batching baseline - waits for full batch."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.request_queue = None
        self.batch_size = 8  # Must wait for full batch
        self.hidden_dim = 1024
    
    def setup(self) -> None:
        """Setup: Initialize model and prepare requests."""
        torch.manual_seed(42)
        
        self.model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device).half().eval()
        
        # Simulate incoming requests with different sequence lengths
        # Static batching: must wait until batch_size requests arrive
        self.request_queue = []
        for i in range(self.batch_size):
            seq_len = 64 + (i * 8)  # Different lengths: 64, 72, 80, ...
            x = torch.randn(1, seq_len, self.hidden_dim, device=self.device, dtype=torch.float16)
            self.request_queue.append(x)
        
        # Warmup
        with torch.no_grad():
            # Process one full batch (with padding)
            max_seq_len = max(req.size(1) for req in self.request_queue)
            padded_batch = []
            for req in self.request_queue:
                seq_len = req.size(1)
                if seq_len < max_seq_len:
                    padding = torch.zeros(1, max_seq_len - seq_len, self.hidden_dim, 
                                        device=self.device, dtype=torch.float16)
                    padded_req = torch.cat([req, padding], dim=1)
                else:
                    padded_req = req
                padded_batch.append(padded_req)
            batch = torch.cat(padded_batch, dim=0)
            _ = self.model(batch)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - static batching waits for full batch."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_batching_static", enable=enable_nvtx):
            # Static batching: must wait for full batch before processing
            # Simulate waiting by padding shorter sequences to match longest
            max_seq_len = max(req.size(1) for req in self.request_queue)
            
            # Pad all sequences to max length (wasteful but required for static batching)
            padded_batch = []
            for req in self.request_queue:
                seq_len = req.size(1)
                if seq_len < max_seq_len:
                    padding = torch.zeros(1, max_seq_len - seq_len, self.hidden_dim, 
                                        device=self.device, dtype=torch.float16)
                    padded_req = torch.cat([req, padding], dim=1)
                else:
                    padded_req = req
                padded_batch.append(padded_req)
            
            # Process full batch (static batching)
            # All sequences are now padded to max_seq_len
            batch = torch.cat(padded_batch, dim=0)  # [batch_size, max_seq_len, hidden_dim]
            with torch.no_grad():
                _ = self.model(batch)

    
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
    return BaselineBatchingStaticBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Static Batching: {result.mean_ms:.3f} ms")

