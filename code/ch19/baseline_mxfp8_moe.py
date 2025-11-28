"""Baseline MXFP8 MoE microbenchmark with naive Python-side block quantization."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.python.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402
from ch19.mxfp8_moe_common import (  # noqa: E402
    MX_BLOCK_SIZE,
    balanced_assignments,
    block_dequantize_mxfp8,
    block_quantize_mxfp8,
    bucket_by_expert,
    require_blackwell,
    restore_bucketed,
)


class _NaiveMXFP8Matmul:
    """Reference MXFP8 GEMM that quantizes on the host side each iteration."""

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        weights: torch.Tensor,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.weights = weights  # [E, N, K]

    def __call__(self, expert_input: torch.Tensor, expert_idx: int) -> torch.Tensor:
        weight = self.weights[expert_idx]  # [N, K]
        q_act, act_scale = block_quantize_mxfp8(expert_input, block_size=MX_BLOCK_SIZE)
        q_wt, wt_scale = block_quantize_mxfp8(weight, block_size=MX_BLOCK_SIZE)
        act = block_dequantize_mxfp8(q_act, act_scale, block_size=MX_BLOCK_SIZE, dtype=torch.float16)
        wt = block_dequantize_mxfp8(q_wt, wt_scale, block_size=MX_BLOCK_SIZE, dtype=torch.float16)
        return act @ wt.transpose(0, 1)


class BaselineMXFP8MoEBenchmark(BaseBenchmark):
    """MXFP8 MoE forward path with unfused quantization and per-expert GEMMs."""

    def __init__(self) -> None:
        super().__init__()
        self.num_tokens = 4096
        self.hidden_dim = 4096
        self.ffn_dim = 14336
        self.num_experts = 8
        self.inputs: Optional[torch.Tensor] = None
        self.assignments: Optional[torch.Tensor] = None
        self.bucketed_inputs: Optional[torch.Tensor] = None
        self.bucket_indices: Optional[torch.Tensor] = None
        self.expert_order: Optional[torch.Tensor] = None
        self.m_splits: List[int] = []
        self.weights: Optional[torch.Tensor] = None
        self.matmul_ref: Optional[_NaiveMXFP8Matmul] = None

    def setup(self) -> None:
        require_blackwell("ch19 baseline_mxfp8_moe")
        torch.manual_seed(7)
        self.inputs = torch.randn(
            self.num_tokens, self.hidden_dim, device=self.device, dtype=torch.bfloat16
        )
        # Store weights as [E, N, K] to quantize along K (reduction) blocks.
        self.weights = torch.randn(
            self.num_experts, self.ffn_dim, self.hidden_dim, device=self.device, dtype=torch.bfloat16
        )
        self.assignments = balanced_assignments(
            num_tokens=self.num_tokens, num_experts=self.num_experts, device=self.device
        )
        bucketed, m_splits, bucket_indices, expert_order, _ = bucket_by_expert(
            self.inputs, self.assignments, num_experts=self.num_experts
        )
        self.bucketed_inputs = bucketed
        self.bucket_indices = bucket_indices
        self.expert_order = expert_order
        self.m_splits = m_splits
        self.matmul_ref = _NaiveMXFP8Matmul(
            hidden_dim=self.hidden_dim,
            ffn_dim=self.ffn_dim,
            weights=self.weights,
        )
        tokens_per_iteration = float(self.num_tokens)
        self.register_workload_metadata(tokens_per_iteration=tokens_per_iteration)
        torch.cuda.synchronize(self.device)

    def _run_naive(self) -> torch.Tensor:
        assert (
            self.bucketed_inputs is not None
            and self.bucket_indices is not None
            and self.expert_order is not None
            and self.weights is not None
            and self.matmul_ref is not None
        )
        outputs: List[torch.Tensor] = []
        offset = 0
        for expert_idx, m in zip(self.expert_order.tolist(), self.m_splits):
            expert_slice = self.bucketed_inputs.narrow(0, offset, m)
            outputs.append(self.matmul_ref(expert_slice, expert_idx))
            offset += m
        bucketed_out = torch.cat(outputs, dim=0)
        return restore_bucketed(
            bucketed_out, self.bucket_indices, num_tokens=self.num_tokens
        )

    def benchmark_fn(self) -> None:
        enable_nvtx = get_nvtx_enabled(self.get_config())
        with nvtx_range("mxfp8_moe_baseline", enable=enable_nvtx):
            _ = self._run_naive()
        self._synchronize()

    def teardown(self) -> None:
        self.inputs = None
        self.weights = None
        self.assignments = None
        self.bucketed_inputs = None
        self.bucket_indices = None
        self.expert_order = None
        self.m_splits = []
        self.matmul_ref = None
        torch.cuda.empty_cache()

    def validate_result(self) -> Optional[str]:
        if self.inputs is None or self.weights is None:
            return "Inputs not initialized"
        if any(m == 0 for m in self.m_splits):
            return "Empty expert bucket detected"
        return None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=8, warmup=5, deterministic=False, enable_nvtx=True)

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_fp32_ms', 10.0),
            reduced_precision_time_ms=getattr(self, '_reduced_ms', 5.0),
            precision_type="fp8",
        )


def get_benchmark() -> BaseBenchmark:
    return BaselineMXFP8MoEBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    mean_ms = result.timing.mean_ms if result.timing else 0.0
    print(f"Baseline MXFP8 MoE: {mean_ms:.3f} ms")
