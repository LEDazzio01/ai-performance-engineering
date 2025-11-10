"""optimized_data_parallelism.py - Optimized data parallelism for multi-GPU inference.

Demonstrates data parallelism by replicating model across multiple GPUs.
Data parallelism: Replicates model across GPUs for parallel processing of different batches.
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


class OptimizedDataParallelismBenchmark(Benchmark):
    """Optimized: Data parallelism with model replication across GPUs."""
    
    def __init__(self):
        self.device = resolve_device()
        self.models = None
        self.requests_per_gpu = None
        self.workload = WORKLOAD
        self.num_requests = self.workload.total_requests
        self.parallel_width = max(1, self.workload.data_parallel_chunk)
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.hidden_dim = 256
    
    def setup(self) -> None:
        """Setup: Initialize model replicas on multiple GPUs."""
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        
        torch.manual_seed(42)
        self.models = []
        for gpu_id in range(self.num_gpus):
            model = nn.Sequential(
                nn.Linear(self.hidden_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 10),
            ).to(torch.device(f"cuda:{gpu_id}")).eval()
            self.models.append(model)
        base_requests = torch.randn(self.num_requests, self.hidden_dim, device=self.device)
        # Slice the shared request tensor so each GPU gets a shard.
        self.requests_per_gpu = []
        for gpu_id in range(self.num_gpus):
            device = torch.device(f"cuda:{gpu_id}")
            shard = base_requests[gpu_id :: self.num_gpus]
            self.requests_per_gpu.append(shard.to(device).contiguous())
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Data parallelism processing across multiple GPUs."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_data_parallelism", enable=enable_nvtx):
            with torch.no_grad():
                if self.num_gpus > 1:
                    streams = [torch.cuda.Stream(device=gpu_id) for gpu_id in range(self.num_gpus)]
                    for gpu_id, (model, shard, stream) in enumerate(zip(self.models, self.requests_per_gpu, streams)):
                        with torch.cuda.stream(stream):
                            for start in range(0, shard.size(0), self.parallel_width):
                                batch = shard[start : start + self.parallel_width]
                                _ = model(batch)
                    for stream in streams:
                        stream.synchronize()
                else:
                    shard = self.requests_per_gpu[0]
                    model = self.models[0]
                    for start in range(0, shard.size(0), self.parallel_width):
                        batch = shard[start : start + self.parallel_width]
                        _ = model(batch)
                    torch.cuda.synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.models = None
        self.requests = None
        if torch.cuda.is_available():
            for gpu_id in range(self.num_gpus):
                torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.models is None or len(self.models) == 0:
            return "Models not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedDataParallelismBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)
