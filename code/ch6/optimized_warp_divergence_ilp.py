"""optimized_warp_divergence_ilp.py - Optimized ILP avoiding warp divergence.

Demonstrates ILP optimization by avoiding warp divergence (predication).
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Callable, Tuple

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)
from ch6.workload_config import WORKLOAD, is_smoke_test

try:
    TORCH_COMPILE_AVAILABLE = hasattr(torch, "compile")
except Exception:  # pragma: no cover - defensive
    TORCH_COMPILE_AVAILABLE = False


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch6")
    return torch.device("cuda")


def _branchless_kernel(
    result: torch.Tensor,
    mask_source: torch.Tensor,
    iterations: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shared branchless transform used for eager + compiled paths."""
    for iteration in range(iterations):
        activations = torch.sigmoid(mask_source)
        mask = torch.gt(activations, 0.5).to(result.dtype)
        inv_mask = 1.0 - mask

        positive = torch.tanh(result * 1.11 + 0.25)
        positive = positive * 1.003 + 0.0005 * positive * positive

        negative = torch.sin(result * 0.77 - 0.35)
        negative = negative * 0.997 - 0.0004 * negative * negative

        result = mask * positive + inv_mask * negative
        mask_source = 0.92 * mask_source + 0.08 * torch.roll(result, shifts=iteration + 1, dims=0)
    return result, mask_source


class OptimizedWarpDivergenceILPBenchmark(Benchmark):
    """Optimized: High ILP by avoiding warp divergence."""

    def __init__(self):
        self.device = resolve_device()
        self.workload = WORKLOAD
        self.smoke_test = is_smoke_test()
        self.N = self.workload.warp_elements_for_mode(self.smoke_test)
        self.branch_iterations = self.workload.warp_branch_iterations
        self.input: Optional[torch.Tensor] = None
        self.routing_logits: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._checksum = 0.0
        self.streams: list[torch.cuda.Stream] = []
        self._compiled_step: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None
        self._branchless_fn: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None

    def setup(self) -> None:
        torch.manual_seed(42)
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.routing_logits = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty_like(self.input)
        props = torch.cuda.get_device_properties(self.device.index or 0)
        stream_count = min(4, max(1, props.multi_processor_count // 8))
        self.streams = [torch.cuda.Stream(priority=-1) for _ in range(stream_count)]

        def branchless_fn(chunk: torch.Tensor, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            return _branchless_kernel(chunk, logits, self.branch_iterations)

        self._branchless_fn = branchless_fn
        self._compiled_step = self._maybe_compile(branchless_fn)
        torch.cuda.synchronize()

    def _maybe_compile(self, fn: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]):
        if not TORCH_COMPILE_AVAILABLE:
            return None
        try:
            return torch.compile(fn, fullgraph=True)
        except Exception:
            return None

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_warp_divergence_ilp", enable=enable_nvtx):
            assert self.input is not None and self.routing_logits is not None
            chunked_inputs = torch.chunk(self.input, len(self.streams))
            chunked_logits = torch.chunk(self.routing_logits, len(self.streams))
            updated_chunks: list[torch.Tensor] = [torch.empty(0, device=self.device)] * len(self.streams)
            updated_logits: list[torch.Tensor] = [torch.empty(0, device=self.device)] * len(self.streams)
            step_fn = self._compiled_step or self._branchless_fn
            assert step_fn is not None

            for idx, (stream, chunk, logits) in enumerate(zip(self.streams, chunked_inputs, chunked_logits)):
                with torch.cuda.stream(stream):
                    chunk_contig = chunk.contiguous()
                    logits_contig = logits.contiguous()
                    out_chunk, out_logits = step_fn(chunk_contig, logits_contig)
                    updated_chunks[idx] = out_chunk
                    updated_logits[idx] = out_logits

            torch.cuda.synchronize()
            self.output = torch.cat(updated_chunks, dim=0)
            self.routing_logits = torch.cat(updated_logits, dim=0)
            self._checksum = float(self.output.sum().item())

    def teardown(self) -> None:
        self.input = None
        self.output = None
        self.routing_logits = None
        self.streams = []
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=self.workload.ilp_iterations,
            warmup=self.workload.ilp_warmup,
        )

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output tensor not initialized"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedWarpDivergenceILPBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Warp Divergence ILP: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print("  Tip: Avoiding warp divergence maximizes instruction-level parallelism")
