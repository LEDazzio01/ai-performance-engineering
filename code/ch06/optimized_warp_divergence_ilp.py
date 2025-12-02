"""optimized_warp_divergence_ilp.py - Optimized ILP avoiding warp divergence.

Chapter 6: Occupancy and Instruction-Level Parallelism

Demonstrates how to avoid warp divergence using branchless operations.
The baseline (baseline_warp_divergence_ilp.py) uses conditional indexing
which causes warp divergence. This optimized version uses torch.where
for branchless selection, compiled once with torch.compile on the full tensor.

Key optimizations vs baseline:
- torch.where instead of boolean indexing (no warp divergence)
- Single compiled kernel on full tensor (no chunking overhead)
- No per-iteration clones or concatenation
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


def _fused_branchless_kernel(
    result: torch.Tensor,
    mask_source: torch.Tensor,
    iterations: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fully fused branchless transform - no warp divergence.
    
    Uses torch.where for predicated selection instead of boolean indexing.
    All threads compute both branches; the result is selected via predicate.
    """
    for iteration in range(iterations):
        # Compute mask as float for branchless blending
        activations = torch.sigmoid(mask_source)
        mask = activations > 0.5  # Boolean mask for torch.where
        
        # Compute BOTH branches for all elements (branchless)
        positive = torch.tanh(result * 1.11 + 0.25)
        positive = positive * 1.003 + 0.0005 * positive * positive
        
        negative = torch.sin(result * 0.77 - 0.35)
        negative = negative * 0.997 - 0.0004 * negative * negative
        
        # Select result via predicate (no divergence - all threads do same work)
        result = torch.where(mask, positive, negative)
        mask_source = 0.92 * mask_source + 0.08 * torch.roll(result, shifts=iteration + 1, dims=0)
    
    return result, mask_source


class OptimizedWarpDivergenceILPBenchmark(BaseBenchmark):
    """Optimized: High ILP by avoiding warp divergence with fused branchless kernel."""

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
        self._compiled_fn: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None
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
        
        # Capture iterations in closure for compilation
        branch_iters = self.branch_iterations
        
        def fused_fn(data: torch.Tensor, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            return _fused_branchless_kernel(data, logits, branch_iters)
        
        # Disable CUDA graph features that conflict with compiled functions
        if self._inductor_state is None:
            self._inductor_state = disable_inductor_cudagraph_features()
        
        # Compile once for the full tensor - no chunking
        self._compiled_fn = compile_callable(
            fused_fn,
            fullgraph=True,
            mode="reduce-overhead",
        )
        
        # Warmup the compiled function
        _ = self._compiled_fn(self.input, self.routing_logits)
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert self.input is not None and self.routing_logits is not None
        with self._nvtx_range("optimized_warp_divergence_ilp"):
            # Single compiled call on full tensor - no chunking, no concat
            assert self._compiled_fn is not None
            self.output, self.routing_logits = self._compiled_fn(self.input, self.routing_logits)
            self._checksum = float(self.output.sum().item())

    def teardown(self) -> None:
        self.input = None
        self.output = None
        self.routing_logits = None
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
