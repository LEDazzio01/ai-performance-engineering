"""optimized_regional_triton.py - Regional compile with Triton-fused MLP.

Shows steady-state gains by fusing the MLP (RMSNorm → GELU → Linear) in Triton
while keeping the rest of the block eager. Targets Blackwell/GB10; falls back
to standard ops if Triton is unavailable.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python import compile_utils as _compile_utils_patch  # noqa: F401
from common.python.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def rmsnorm_gelu_linear_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        y_ptr,
        B,
        D,
        H,
        stride_xm,
        stride_xd,
        stride_wd,
        stride_wh,
        stride_ym,
        stride_yh,
        eps,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        mask_m = offs_m < B
        mask_n = offs_n < H

        sum_sq = tl.zeros([BLOCK_M], dtype=tl.float32)
        k = 0
        while k < D:
            k_ids = k + offs_k
            mask_k = k_ids < D
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_ids[None, :] * stride_xd
            x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            sum_sq += tl.sum(x * x, axis=1)
            k += BLOCK_K

        rms = tl.sqrt(sum_sq / D + eps)

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        k = 0
        while k < D:
            k_ids = k + offs_k
            mask_k = k_ids < D
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_ids[None, :] * stride_xd
            x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            x = x / rms[:, None]

            w_ptrs = w_ptr + k_ids[:, None] * stride_wd + offs_n[None, :] * stride_wh
            w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

            acc += tl.dot(x, w)
            k += BLOCK_K

        b_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc = acc + b_vals[None, :]

        sig = 1.0 / (1.0 + tl.exp(-1.702 * acc))
        gelu = acc * sig

        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yh
        tl.store(y_ptrs, gelu.to(tl.float16), mask=mask_m[:, None] & mask_n[None, :])


    def rmsnorm_gelu_linear_triton(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        B, D = x.shape
        _, H = w.shape
        y = torch.empty((B, H), device=x.device, dtype=x.dtype)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid = (triton.cdiv(B, BLOCK_M), triton.cdiv(H, BLOCK_N))

        rmsnorm_gelu_linear_kernel[grid](
            x,
            w,
            b,
            y,
            B,
            D,
            H,
            x.stride(0),
            x.stride(1),
            w.stride(0),
            w.stride(1),
            y.stride(0),
            y.stride(1),
            eps,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=2,
        )
        return y
else:
    def rmsnorm_gelu_linear_triton(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(var + eps)
        x_norm = x / rms
        return F.gelu(F.linear(x_norm, w.t(), b))


class TritonMLP(nn.Module):
    """MLP hot-region fused in Triton and regionally compiled."""

    def __init__(self, hidden: int, mlp_hidden: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden, mlp_hidden, dtype=torch.float16))
        self.bias = nn.Parameter(torch.zeros(mlp_hidden, dtype=torch.float16))
        self.proj_out = nn.Linear(mlp_hidden, hidden)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(hidden)
        nn.init.uniform_(self.bias, -bound, bound)
        self.eps = eps
        # Regionally compile just this fused path
        self._compiled = torch.compile(
            self._forward_impl,
            backend="aot_eager",
            fullgraph=False,
            dynamic=True,
            mode="default",
        )

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        y = rmsnorm_gelu_linear_triton(x, self.weight, self.bias, eps=self.eps)
        return self.proj_out(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._compiled(x)


class TinyTransformerBlock(nn.Module):
    def __init__(self, hidden: int = 1024, num_heads: int = 8, mlp_hidden: int = 4096):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden)
        self.mlp = TritonMLP(hidden, mlp_hidden)
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln1(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = residual + attn_out

        residual = x
        x = self.ln2(x)
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)
        mlp_out = self.mlp(x_flat)
        mlp_out = mlp_out.reshape(B, T, D)
        x = residual + mlp_out
        return x


class OptimizedRegionalTritonBenchmark(BaseBenchmark):
    """Optimized: Triton-fused MLP as the compiled region plus eager rest."""

    def __init__(self):
        super().__init__()
        self.hidden = 1024
        self.num_heads = 8
        self.mlp_hidden = 4096
        self.batch_size = 8
        self.sequence_schedule: List[int] = [128, 256, 384, 512]
        self._step = 0
        self.model: Optional[nn.Module] = None
        self.inputs: Dict[int, torch.Tensor] = {}
        max_tokens = self.batch_size * max(self.sequence_schedule) * self.hidden
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(max_tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(0)
        self.model = TinyTransformerBlock(
            hidden=self.hidden,
            num_heads=self.num_heads,
            mlp_hidden=self.mlp_hidden,
        ).to(self.device, dtype=torch.bfloat16).eval()

        for seq in self.sequence_schedule:
            self.inputs[seq] = torch.randn(
                self.batch_size,
                seq,
                self.hidden,
                device=self.device,
                dtype=torch.bfloat16,
            )

        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )
        self._synchronize()

    def _next_sequence_length(self) -> int:
        seq = self.sequence_schedule[self._step % len(self.sequence_schedule)]
        self._step += 1
        return seq

    def benchmark_fn(self) -> None:
        if self.model is None:
            raise RuntimeError("Model not initialized")
        seq_len = self._next_sequence_length()
        x = self.inputs[seq_len]
        with torch.no_grad(), self._nvtx_range("optimized_regional_triton"):
            _ = self.model(x)
        self._synchronize()

    def teardown(self) -> None:
        self.model = None
        self.inputs.clear()
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=8,
            warmup=10,  # Required for Triton JIT - excludes compile overhead from timing
            enable_memory_tracking=False,
            enable_profiling=False,
            setup_timeout_seconds=300,
            measurement_timeout_seconds=300,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_triton_metrics
        return compute_triton_metrics(
            num_elements=getattr(self, 'N', getattr(self, 'num_elements', 1024)),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            block_size=getattr(self, 'BLOCK_SIZE', 1024),
            num_warps=getattr(self, 'num_warps', 4),
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedRegionalTritonBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(
        f"\nOptimized regional (Triton MLP): {result.timing.mean_ms if result.timing else 0.0:.3f} ms"
    )
