"""optimized_tensor_parallelism.py - Optimized tensor parallelism across GPUs.

Demonstrates tensor parallelism by splitting tensors across multiple GPUs.
Tensor parallelism: Splits tensors across GPUs for parallel computation.
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

from typing import Callable, List, Optional

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch17")
    return torch.device("cuda")


class OptimizedTensorParallelismBenchmark(Benchmark):
    """Optimized: Tensor parallelism with tensors split across GPUs."""

    def __init__(self):
        self.device = resolve_device()
        self.hidden_size = 2048
        self.batch_size = 256
        self.available_gpus = max(1, torch.cuda.device_count())
        self.tensor_parallel_world = min(4, max(2, self.available_gpus))
        self.shard_sizes: List[int] = []
        self.shard_modules: List[nn.Module] = []
        self.streams: List[torch.cuda.Stream] = []
        self.input_data = None
        self.static_shard_inputs: List[torch.Tensor] = []
        self.graphed_modules: List[Optional[Callable[[torch.Tensor], torch.Tensor]]] = []

    def setup(self) -> None:
        """Setup: Initialize model shards across multiple GPUs."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()

        torch.manual_seed(42)
        base = self.hidden_size // self.tensor_parallel_world
        remainder = self.hidden_size % self.tensor_parallel_world
        self.shard_sizes = [base + (1 if idx < remainder else 0) for idx in range(self.tensor_parallel_world)]

        self.shard_modules = []
        self.static_shard_inputs = []
        self.graphed_modules = []
        for shard_idx, shard_dim in enumerate(self.shard_sizes):
            module = nn.Sequential(
                nn.Linear(shard_dim, shard_dim * 2),
                nn.GELU(),
                nn.Linear(shard_dim * 2, shard_dim),
            ).to(torch.device(f"cuda:{shard_idx % self.available_gpus}"), dtype=torch.bfloat16).eval()
            self.shard_modules.append(module)
            static_input = torch.zeros(
                self.batch_size,
                shard_dim,
                device=torch.device(f"cuda:{shard_idx % self.available_gpus}"),
                dtype=torch.bfloat16,
            )
            if hasattr(torch.cuda, "make_graphed_callables"):
                try:
                    graph_callable = torch.cuda.make_graphed_callables(module, (static_input,))
                except Exception:
                    graph_callable = None
            else:
                graph_callable = None
            self.static_shard_inputs.append(static_input)
            self.graphed_modules.append(graph_callable)

        self.streams = [torch.cuda.Stream(priority=-1) for _ in self.shard_modules]
        self.input_data = torch.randn(self.batch_size, self.hidden_size, device=self.device, dtype=torch.bfloat16)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Benchmark: Tensor parallelism processing across multiple GPUs."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        if self.input_data is None or not self.shard_modules:
            raise RuntimeError("Tensor parallel benchmark not initialized")

        chunks = torch.split(self.input_data, self.shard_sizes, dim=1)
        outputs: List[Optional[torch.Tensor]] = [None] * len(self.shard_modules)

        with nvtx_range("optimized_tensor_parallelism", enable=enable_nvtx):
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                for idx, (module, shard_input, stream) in enumerate(zip(self.shard_modules, chunks, self.streams)):
                    target_device = next(module.parameters()).device
                    with torch.cuda.stream(stream):
                        local_input = shard_input.to(target_device)
                        static_input = self.static_shard_inputs[idx]
                        static_input.copy_(local_input, non_blocking=True)
                        graph_callable = self.graphed_modules[idx]
                        if graph_callable is not None:
                            outputs[idx] = graph_callable(static_input)
                        else:
                            outputs[idx] = module(static_input)
                current_stream = torch.cuda.current_stream()
                for stream in self.streams:
                    current_stream.wait_stream(stream)

        combined = torch.cat([out.to(self.device) for out in outputs if out is not None], dim=1)
        _ = combined
        torch.cuda.synchronize()

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.shard_modules = []
        self.input_data = None
        self.static_shard_inputs = []
        self.graphed_modules = []
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=12,
            warmup=2,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if not self.shard_modules:
            return "Shard modules not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedTensorParallelismBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)
