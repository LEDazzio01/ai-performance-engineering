"""Persistent decode in Triton with a device-side work queue."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import triton
import triton.language as tl

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.persistent_decode.persistent_decode_common import (
    build_inputs,
    get_decode_options,
    get_decode_profile,
    resolve_device,
    resolve_shapes,
    tokens_per_iteration,
)


@triton.jit
def persistent_decode_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    OUT_ptr,
    work_seq_ids_ptr,
    work_steps_ptr,
    num_items,
    head_dim: tl.constexpr,
    max_steps: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)  # one program per sequence
    if pid >= num_items:
        return

    seq_id = tl.load(work_seq_ids_ptr + pid)
    num_step = tl.load(work_steps_ptr + pid)

    for t in range(0, max_steps):
        active = num_step > t
        base = (seq_id * max_steps + t) * head_dim
        offs = tl.arange(0, BLOCK_K)
        dot = tl.zeros((), dtype=tl.float32)

        for k0 in range(0, head_dim, BLOCK_K):
            k_idx = k0 + offs
            mask = (k_idx < head_dim) & active
            q = tl.load(Q_ptr + base + k_idx, mask=mask, other=0.0)
            k = tl.load(K_ptr + base + k_idx, mask=mask, other=0.0)
            dot += tl.sum(q * k, axis=0)

        # Write vector output = V * dot
        for k0 in range(0, head_dim, BLOCK_K):
            k_idx = k0 + offs
            mask = (k_idx < head_dim) & active
            v = tl.load(V_ptr + base + k_idx, mask=mask, other=0.0)
            tl.store(OUT_ptr + base + k_idx, v * dot, mask=mask)


class OptimizedPersistentDecodeTritonBenchmark(BaseBenchmark):
    """Persistent decode using Triton."""

    def __init__(self) -> None:
        super().__init__()
        self.device = resolve_device()
        self.inputs = None
        self.options = get_decode_options()
        self.profile = get_decode_profile()
        self.batch, self.seq_len, self.head_dim = resolve_shapes()
        self.block_k = self.profile.block_k
        self.num_programs = self.profile.num_programs
        self.register_workload_metadata(tokens_per_iteration=tokens_per_iteration())

    def setup(self) -> None:
        self.inputs = build_inputs(self.device)
        self._synchronize()

        # Precompile the Triton kernel to keep measurement under benchmark timeouts.
        grid = (max(1, min(self.batch, self.num_programs)),)
        BLOCK_K = self.block_k
        persistent_decode_kernel[grid](
            self.inputs.q,
            self.inputs.k,
            self.inputs.v,
            self.inputs.out,
            self.inputs.work_seq_ids,
            self.inputs.work_steps,
            self.batch,
            head_dim=self.head_dim,
            max_steps=self.seq_len,
            BLOCK_K=BLOCK_K,
            num_warps=2,
            num_stages=1,
        )
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.inputs is None:
            raise RuntimeError("Inputs not initialized")

        num_items = min(self.batch, self.num_programs)
        grid = (max(1, num_items),)
        BLOCK_K = self.block_k
        with self._nvtx_range("persistent_decode_triton"):
            persistent_decode_kernel[grid](
                self.inputs.q,
                self.inputs.k,
                self.inputs.v,
                self.inputs.out,
                self.inputs.work_seq_ids,
                self.inputs.work_steps,
                num_items,
                head_dim=self.head_dim,
                max_steps=self.seq_len,
                BLOCK_K=BLOCK_K,
                num_warps=2,
                num_stages=1,
            )
            self._synchronize()

    def teardown(self) -> None:
        torch.cuda.empty_cache()
        self.inputs = None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=3,
            warmup=5,
            enable_profiling=False,
            enable_ncu=False,
            enable_nsys=False,
            measurement_timeout_seconds=90,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return inference metrics."""
        return {
            "persistent_decode_tr.batch_size": float(getattr(self, 'batch_size', 0)),
            "persistent_decode_tr.seq_len": float(getattr(self, 'seq_len', 0)),
            "persistent_decode_tr.hidden_dim": float(getattr(self, 'hidden_dim', 0)),
        }

    def validate_result(self) -> str | None:
        if self.inputs is None:
            return "Inputs not initialized"
        if not torch.isfinite(self.inputs.out).all():
            return "Non-finite output detected"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedPersistentDecodeTritonBenchmark()

if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    bench = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=bench.get_config())
    result = harness.benchmark(bench)
    mean_ms = result.timing.mean_ms if result and result.timing else 0.0
    print(f"[{bench.__class__.__name__}] mean iteration {mean_ms:.3f} ms")
