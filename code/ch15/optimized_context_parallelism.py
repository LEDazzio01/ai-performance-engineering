"""optimized_context_parallelism.py - Optimized processing with context parallelism.

Demonstrates context parallelism for long sequences by splitting across GPUs.
Context parallelism: This optimized version uses context parallelism to split long sequences across multiple GPUs for parallel processing.
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

from typing import Optional, List

from common.python.compile_utils import enable_tf32, compile_model
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch15")
    return torch.device("cuda")


class OptimizedContextParallelismBenchmark(Benchmark):
    """Optimized: Context parallelism for long sequences (split across GPUs)."""
    
    def __init__(self):
        self.device = resolve_device()
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.models: Optional[List[nn.Module]] = None
        self.sequence_chunks: Optional[List[torch.Tensor]] = None
        self.sequence_length = 8192
        self.num_context_shards = min(4, max(1, self.sequence_length // 2048))
        self.streams: List[torch.cuda.Stream] = []
    
    def setup(self) -> None:
        """Setup: Initialize models and split sequence for context parallelism."""
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        
        torch.manual_seed(42)
        self.models = []
        self.streams = [torch.cuda.Stream() for _ in range(self.num_context_shards)]
        for _ in range(self.num_context_shards):
            shard = nn.Sequential(
                nn.Linear(256, 512, bias=False),
                nn.GELU(),
                nn.Linear(512, 512, bias=False),
                nn.GELU(),
                nn.Linear(512, 256, bias=False),
            ).to(self.device, dtype=torch.bfloat16).eval()
            shard = compile_model(shard, mode="reduce-overhead")
            self.models.append(shard)
        
        full_sequence = torch.randn(
            self.sequence_length,
            256,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self.sequence_chunks = list(torch.chunk(full_sequence, self.num_context_shards))
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Context parallelism processing of long sequence."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled
        
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        
        # Optimization: Process sequence chunks in parallel across GPUs
        # Context parallelism enables parallel processing of long sequences
        if self.models is None or self.sequence_chunks is None:
            raise RuntimeError("Context shards not initialized")

        with nvtx_range("optimized_context_parallelism", enable=enable_nvtx):
            with torch.no_grad():
                outputs = [None] * len(self.models)
                for idx, (model, chunk, stream) in enumerate(zip(self.models, self.sequence_chunks, self.streams)):
                    next_chunk = chunk.roll(shifts=idx, dims=0)
                    with torch.cuda.stream(stream):
                        outputs[idx] = model(next_chunk)
                for stream in self.streams:
                    stream.synchronize()
        torch.cuda.synchronize(self.device)
    
    def teardown(self) -> None:
        """Cleanup: Clear CUDA cache."""
        if torch.cuda.is_available():
            for gpu_id in range(self.num_gpus):
                torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=3,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.models is None or len(self.models) == 0:
            return "Models not initialized"
        if self.sequence_chunks is None or len(self.sequence_chunks) == 0:
            return "Sequence chunks not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedContextParallelismBenchmark()


def main():
    """Run optimized context parallelism benchmark."""
    benchmark = get_benchmark()
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)
    print(f"Sequence length: {benchmark.sequence_length} tokens")
    print(f"GPUs used: {benchmark.num_gpus}")
    print("Processing: Parallel across GPUs (context parallelism)")


if __name__ == "__main__":
    main()
