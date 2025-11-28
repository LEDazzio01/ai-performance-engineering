"""Baseline prefill vs. decode microbench without TMA shaping.

Emits two phases for Nsight Systems:
- Prefill: sequential "TMA-like" bulk copies + compute on the default stream.
- Decode: per-token work (host-driven loop).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig
from common.python.blackwell_requirements import ensure_blackwell_tma_supported
from labs.persistent_decode.persistent_decode_common import (
    build_inputs,
    resolve_device,
    resolve_shapes,
    tokens_per_iteration,
)


class BaselineTmaPrefillDecodeBenchmark(BaseBenchmark):
    """Sequential copy/compute prefill followed by host-driven decode."""

    def __init__(self) -> None:
        super().__init__()
        self.device = resolve_device()
        self.inputs = None
        self.batch, self.seq_len, self.head_dim = resolve_shapes()
        self.prefill_chunks = 8
        self.prefill_chunk_elems = 128 * 128
        self.register_workload_metadata(tokens_per_iteration=tokens_per_iteration())

    def setup(self) -> None:
        ensure_blackwell_tma_supported("baseline_tma_prefill_decode")
        self.inputs = build_inputs(self.device)
        self.prefill_src = torch.randn(
            self.prefill_chunks, self.prefill_chunk_elems, device=self.device
        )
        self.prefill_dst = torch.zeros_like(self.prefill_src)
        self._synchronize()

    def _prefill_sequential(self) -> None:
        """Sequential copy + compute without any pipelining."""
        for idx in range(self.prefill_chunks):
            # Real copy operation - no artificial delays
            self.prefill_dst[idx].copy_(self.prefill_src[idx])
            # Add computation to simulate processing
            self.prefill_dst[idx].add_(self.prefill_src[idx])

    def _decode_host_loop(self) -> None:
        assert self.inputs is not None
        for t in range(self.seq_len):
            q_t = self.inputs.q[:, t, :]
            k_t = self.inputs.k[:, t, :]
            v_t = self.inputs.v[:, t, :]
            dot = (q_t * k_t).sum(dim=-1, keepdim=True)
            self.inputs.out[:, t, :] = v_t * dot

    def benchmark_fn(self) -> None:
        if self.inputs is None:
            raise RuntimeError("Inputs not initialized")

        with self._nvtx_range("prefill_baseline"):
            self._prefill_sequential()
        with self._nvtx_range("decode_baseline"):
            self._decode_host_loop()
        self._synchronize()

    def teardown(self) -> None:
        torch.cuda.empty_cache()
        self.inputs = None

    def get_config(self) -> BenchmarkConfig:
        # Keep short; this is primarily for profiling with --profile / nsys
        return BenchmarkConfig(
            iterations=8,
            warmup=5,
            use_subprocess=False,
            measurement_timeout_seconds=120,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return inference metrics."""
        return {
            "tma_prefill_decode.batch_size": float(getattr(self, 'batch_size', 0)),
            "tma_prefill_decode.seq_len": float(getattr(self, 'seq_len', 0)),
            "tma_prefill_decode.hidden_dim": float(getattr(self, 'hidden_dim', 0)),
        }

    def validate_result(self) -> str | None:
        if self.inputs is None:
            return "Inputs not initialized"
        if not torch.isfinite(self.inputs.out).all():
            return "Non-finite output detected"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineTmaPrefillDecodeBenchmark()

if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    bench = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=bench.get_config())
    result = harness.benchmark(bench)
    mean_ms = result.timing.mean_ms if result and result.timing else 0.0
    print(f"[{bench.__class__.__name__}] mean iteration {mean_ms:.3f} ms")
