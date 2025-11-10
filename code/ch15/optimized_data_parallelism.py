"""optimized_data_parallelism.py - Optimized inference with data parallelism.

Demonstrates data parallelism for inference by replicating model across GPUs.
Data parallelism: This optimized version uses data parallelism to process multiple requests in parallel.
In inference, data parallelism replicates the entire model on multiple GPUs for higher throughput.
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

from ch15.baseline_data_parallelism import _build_mlp
from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch15")
    return torch.device("cuda")


def _infer_device_index(device: torch.device) -> int:
    """Return the concrete CUDA device index."""
    if device.index is not None:
        return device.index
    return torch.cuda.current_device()


class OptimizedDataParallelismBenchmark(Benchmark):
    """Optimized: Virtual data parallelism with batched execution and multi-stream replay.

    When only one physical GPU is available we still want to demonstrate the
    effect of data parallelism by keeping the device busy with large batches
    of independent requests. We emulate multiple data-parallel replicas
    ("virtual shards") that process chunks on dedicated CUDA streams and use
    torch.compile to fuse the per-request work.
    """

    def __init__(self):
        self.device = resolve_device()
        self.hidden_dim = 256
        self.num_requests = 128
        self.virtual_shards = 1
        self.micro_batch_size = 16
        self.window_size = None
        self.model: Optional[nn.Module] = None
        self.requests: Optional[torch.Tensor] = None
        self.streams: List[torch.cuda.Stream] = []
        self.compiled = False

    def setup(self) -> None:
        """Setup: Initialize model, compile it, and pre-batch requests."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()

        torch.manual_seed(42)
        self.requests = torch.randn(
            self.num_requests,
            self.hidden_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )

        model = _build_mlp(self.hidden_dim).to(self.device, dtype=torch.bfloat16).eval()
        try:
            model = torch.compile(model, mode="reduce-overhead")
            self.compiled = True
        except Exception:
            self.compiled = False
        self.model = model

        device_index = _infer_device_index(self.device)
        sm_count = torch.cuda.get_device_properties(device_index).multi_processor_count
        # Derive a small virtual data-parallel world size that fits on a single GPU.
        self.virtual_shards = max(1, min(4, sm_count // 8))
        self.micro_batch_size = max(8, self.num_requests // (self.virtual_shards * 4))
        self.window_size = self.virtual_shards * self.micro_batch_size
        self.streams = [torch.cuda.Stream(priority=-1) for _ in range(self.virtual_shards)]
        torch.cuda.synchronize()

    def _execute_virtual_data_parallel(self, micro_batch: torch.Tensor) -> None:
        """Shard a micro-batch across the virtual replicas and launch on streams."""
        if self.model is None:
            raise RuntimeError("Model not compiled")
        shard_inputs = torch.chunk(micro_batch, self.virtual_shards, dim=0)
        current_stream = torch.cuda.current_stream()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for stream, shard in zip(self.streams, shard_inputs):
                if shard.numel() == 0:
                    continue
                with torch.cuda.stream(stream):
                    _ = self.model(shard)
            for stream in self.streams:
                current_stream.wait_stream(stream)

    def benchmark_fn(self) -> None:
        """Benchmark: Batched inference with virtual data parallel replicas."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        if self.model is None or self.requests is None:
            raise RuntimeError("Model/requests not initialized")

        with nvtx_range("optimized_data_parallelism", enable=enable_nvtx):
            # Process the queue in windows so every virtual shard stays busy.
            for start_idx in range(0, self.requests.shape[0], self.window_size):
                window = self.requests[start_idx : start_idx + self.window_size]
                if window.numel() == 0:
                    continue
                for micro in window.split(self.micro_batch_size, dim=0):
                    if micro.numel() == 0:
                        continue
                    self._execute_virtual_data_parallel(micro)
        torch.cuda.synchronize()

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.requests = None
        self.streams = []
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=12,
            warmup=2,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.requests is None:
            return "Requests not initialized"
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
