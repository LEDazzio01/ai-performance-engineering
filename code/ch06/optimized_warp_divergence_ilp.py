"""optimized_warp_divergence_ilp.py - Optimized ILP avoiding warp divergence.

Chapter 6: Occupancy and Instruction-Level Parallelism

Demonstrates how to avoid warp divergence using branchless operations.
The baseline (baseline_warp_divergence_ilp.py) uses conditional indexing
which causes warp divergence. This optimized version uses torch.compile
to convert branches into predicated/branchless operations.

FORWARD REFERENCE: This file uses torch.compile (TorchInductor), which is
covered in depth in Chapter 14. Here we use it to demonstrate the ILP
benefits of eliminating warp divergence through compiler optimization.
See ch14/*compile*.py for detailed torch.compile analysis.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

from core.utils.compile_utils import compile_callable
from core.optimization.inductor_guard import (
    InductorCudagraphState,
    disable_inductor_cudagraph_features,
    restore_inductor_cudagraph_features,
)
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch06.workload_config import WORKLOAD

_MARK_CUDAGRAPH_STEP = getattr(torch.compiler, "cudagraph_mark_step_begin", None)


def _mark_cudagraph_step() -> None:
    if callable(_MARK_CUDAGRAPH_STEP):
        _MARK_CUDAGRAPH_STEP()


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


class OptimizedWarpDivergenceILPBenchmark(BaseBenchmark):
    """Optimized: High ILP by avoiding warp divergence."""

    def __init__(self):
        super().__init__()
        self.skip_output_check = True
        self.skip_input_check = True
        self.workload = WORKLOAD
        self.N = self.workload.warp_elements
        self.branch_iterations = self.workload.warp_branch_iterations
        self.input: Optional[torch.Tensor] = None
        self.routing_logits: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._checksum = 0.0
        self.streams: list[torch.cuda.Stream] = []
        self._compiled_step: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None
        self._branchless_fn: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None
        self._inductor_state: Optional[InductorCudagraphState] = None
        token_count = self.N * self.branch_iterations
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.branch_iterations),
            tokens_per_iteration=float(token_count),
        )

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
        if self._inductor_state is None:
            self._inductor_state = disable_inductor_cudagraph_features()
        self._compiled_step = compile_callable(
            branchless_fn,
            fullgraph=True,
            mode="reduce-overhead",
        )
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert self.input is not None and self.routing_logits is not None
        with self._nvtx_range("optimized_warp_divergence_ilp"):
            chunked_inputs = torch.chunk(self.input, len(self.streams))
            chunked_logits = torch.chunk(self.routing_logits, len(self.streams))
            updated_chunks: list[torch.Tensor] = [torch.empty(0, device=self.device)] * len(self.streams)
            updated_logits: list[torch.Tensor] = [torch.empty(0, device=self.device)] * len(self.streams)
            step_fn = self._compiled_step
            assert step_fn is not None

            for idx, (stream, chunk, logits) in enumerate(zip(self.streams, chunked_inputs, chunked_logits)):
                with torch.cuda.stream(stream):
                    chunk_contig = chunk.contiguous()
                    logits_contig = logits.contiguous()
                    _mark_cudagraph_step()
                    out_chunk, out_logits = step_fn(chunk_contig, logits_contig)
                    out_chunk = out_chunk.clone()
                    out_logits = out_logits.clone()
                    updated_chunks[idx] = out_chunk
                    updated_logits[idx] = out_logits

            self._synchronize()
            self.output = torch.cat(updated_chunks, dim=0)
            self.routing_logits = torch.cat(updated_logits, dim=0)
            self._checksum = float(self.output.sum().item())

    def teardown(self) -> None:
        self.input = None
        self.output = None
        self.routing_logits = None
        self.streams = []
        restore_inductor_cudagraph_features(self._inductor_state)
        self._inductor_state = None
        torch.cuda.empty_cache()

    def skip_output_verification(self) -> bool:
        return True

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=self.workload.ilp_iterations,
            warmup=self.workload.ilp_warmup,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_kernel_fundamentals_metrics
        return compute_kernel_fundamentals_metrics(
            num_elements=getattr(self, 'N', getattr(self, 'num_elements', 1024)),
            num_iterations=1,
        )

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedWarpDivergenceILPBenchmark()
