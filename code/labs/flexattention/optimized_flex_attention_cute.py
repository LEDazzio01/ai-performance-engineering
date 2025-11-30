"""Optimized FlexAttention using flash-attn CuTe DSL backend.

The CuTe DSL (flash_attn.cute.interface._flash_attn_fwd) IS the optimized path.
It's a highly-tuned CUDA kernel using NVIDIA's CuTe template library for tensor
cores. This benchmark directly invokes the CuTe kernel - no torch.compile needed
since the kernel is already maximally optimized at the CUDA level.

NOTE: On exit, you may see a harmless "TypeError: 'NoneType' object is not callable"
from CudaDialectJitModule.__del__. This is a known issue in nvidia_cutlass_dsl where
cuda.cuModuleUnload becomes None during Python interpreter shutdown. The benchmark
results are not affected.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.utils.compile_utils import enable_tf32
from labs.flexattention.flexattention_common import build_qkv_inputs

try:
    from flash_attn.cute.interface import _flash_attn_fwd
except Exception as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "SKIPPED: flash-attn with CuTe DSL support is required (pip install flash-attn)"
    ) from exc


class OptimizedFlexAttentionCuteBenchmark(BaseBenchmark):
    """CuTe DSL FlexAttention - the CuTe kernel IS the optimization.
    
    The CuTe DSL backend in flash-attn uses NVIDIA's CuTe template library
    to generate highly-optimized tensor core kernels. This is NOT wrapped
    in torch.compile because:
    1. The CuTe kernel is already maximally optimized at the CUDA level
    2. torch.compile cannot introspect non-Python CUDA functions
    3. Adding torch.compile would only add overhead, not optimization
    """

    def __init__(self) -> None:
        super().__init__()
        self.dtype = torch.bfloat16
        self.seq_len = 1024
        self.batch = 2
        self.heads = 8
        self.head_dim = 64
        self.block_size = 128
        self.doc_span = 256
        self.q: Optional[torch.Tensor] = None
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None
        tokens = self.batch * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        enable_tf32()

        self.q, self.k, self.v = build_qkv_inputs(
            batch=self.batch,
            heads=self.heads,
            seq_len=self.seq_len,
            head_dim=self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )

        # Warmup the CuTe kernel to ensure CUDA context and JIT are ready
        with torch.inference_mode():
            _ = _flash_attn_fwd(self.q, self.k, self.v)
        self._synchronize()

    def benchmark_fn(self) -> None:
        if any(t is None for t in (self.q, self.k, self.v)):
            raise RuntimeError("CuTe FlexAttention inputs not initialized")

        with self._nvtx_range("flexattention_cute_dsl"):
            with torch.inference_mode():
                # Direct CuTe kernel call - this IS the optimized path
                _ = _flash_attn_fwd(self.q, self.k, self.v)
            self._synchronize()

    def teardown(self) -> None:
        # Clear tensors first
        self.q = None
        self.k = None
        self.v = None
        # Force garbage collection to clean up CUTLASS JIT objects
        # This prevents destructor errors during interpreter shutdown
        gc.collect()
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=12, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return attention-specific roofline metrics."""
        # FlexAttention FLOP estimate: 4 * B * H * S^2 * D (Q@K + softmax + attn@V)
        B, H, S, D = self.batch, self.heads, self.seq_len, self.head_dim
        flops = 4.0 * B * H * S * S * D
        # Memory: Q, K, V, O all of size [B, H, S, D]
        bytes_moved = 4.0 * B * H * S * D * 2  # bfloat16 = 2 bytes
        arithmetic_intensity = flops / max(bytes_moved, 1.0)
        return {
            "flex_attention_cute.estimated_flops": flops,
            "flex_attention_cute.estimated_bytes": bytes_moved,
            "flex_attention_cute.arithmetic_intensity": arithmetic_intensity,
        }

    def validate_result(self) -> Optional[str]:
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedFlexAttentionCuteBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode

    bench = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=bench.get_config())
    result = harness.benchmark(bench)
    mean_ms = result.timing.mean_ms if result and result.timing else 0.0
    print(f"[flexattention cute optimized] mean iteration {mean_ms:.3f} ms")
