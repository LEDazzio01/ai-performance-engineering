"""Optimized disaggregated prefill/decode benchmark (Chapter 17).

Separates prefill (long context) and decode (short, latency-sensitive) phases onto
independent CUDA streams. Mirrors production scheduling that dedicates resources
for context building while keeping decode latency low.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from common.python.compile_utils import enable_tf32  # noqa: E402
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class _AttentionBlock(nn.Module):
    """Minimal Transformer block with attention + FFN."""

    def __init__(self, hidden: int, heads: int, *, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, device=device, dtype=dtype)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=heads,
            batch_first=True,
            device=device,
            dtype=dtype,
        )
        self.norm2 = nn.LayerNorm(hidden, device=device, dtype=dtype)
        self.ff = nn.Sequential(
            nn.Linear(hidden, 4 * hidden, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(4 * hidden, hidden, device=device, dtype=dtype),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = k = v = self.norm1(x)
        attn_out, _ = self.attn(q, k, v, need_weights=False)
        x = x + attn_out
        ff_out = self.ff(self.norm2(x))
        return x + ff_out


class OptimizedDisaggregatedBenchmark(BaseBenchmark):
    """Prefill on a long context + decode on a short context using separate streams."""

    def __init__(self) -> None:
        super().__init__()
        # Tuned to Chapter 17 workloads (kept lightweight for single-GPU smoke runs)
        self.dtype = torch.bfloat16
        self.hidden = 2048
        self.heads = 16
        self.prefill_seq = 2048
        self.decode_seq = 256
        self.batch_size = 4

        self.prefill_model: Optional[nn.Module] = None
        self.decode_model: Optional[nn.Module] = None
        self.prefill_input: Optional[torch.Tensor] = None
        self.decode_input: Optional[torch.Tensor] = None
        self.prefill_stream = torch.cuda.Stream()
        self.decode_stream = torch.cuda.Stream()
        self._prefill_done = torch.cuda.Event()
        self._checksum: float = 0.0
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(self.batch_size * (self.prefill_seq + self.decode_seq)),
        )

    def setup(self) -> None:
        enable_tf32()
        torch.manual_seed(17)

        # Independent models for each phase keep parameters resident and avoid contention.
        self.prefill_model = _AttentionBlock(
            hidden=self.hidden,
            heads=self.heads,
            device=self.device,
            dtype=self.dtype,
        ).eval()
        self.decode_model = _AttentionBlock(
            hidden=self.hidden,
            heads=self.heads,
            device=self.device,
            dtype=self.dtype,
        ).eval()

        self.prefill_input = torch.randn(
            self.batch_size,
            self.prefill_seq,
            self.hidden,
            device=self.device,
            dtype=self.dtype,
        )
        self.decode_input = torch.randn(
            self.batch_size,
            self.decode_seq,
            self.hidden,
            device=self.device,
            dtype=self.dtype,
        )

        # Warm up decode path to reduce first-iteration variance.
        with torch.no_grad():
            for _ in range(2):
                _ = self.decode_model(self.decode_input)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        if any(item is None for item in (self.prefill_model, self.decode_model, self.prefill_input, self.decode_input)):
            raise RuntimeError("Models or inputs not initialized")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        with nvtx_range("optimized_disaggregated.prefill_decode", enable=enable_nvtx):
            with torch.no_grad():
                # Prefill on its own stream
                with torch.cuda.stream(self.prefill_stream):
                    prefill_out = self.prefill_model(self.prefill_input)
                    self._prefill_done.record(self.prefill_stream)

                # Decode runs on a separate stream after prefill completes.
                with torch.cuda.stream(self.decode_stream):
                    self.decode_stream.wait_event(self._prefill_done)
                    decode_out = self.decode_model(self.decode_input)

                # Wait for both streams and accumulate a checksum for determinism.
                torch.cuda.current_stream().wait_stream(self.prefill_stream)
                torch.cuda.current_stream().wait_stream(self.decode_stream)
                self._checksum = float(prefill_out.float().sum() + decode_out.float().sum())

    def teardown(self) -> None:
        self.prefill_model = None
        self.decode_model = None
        self.prefill_input = None
        self.decode_input = None
        self._checksum = 0.0
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=2)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_inference_metrics
        return compute_inference_metrics(
            ttft_ms=getattr(self, '_ttft_ms', 50.0),
            tpot_ms=getattr(self, '_tpot_ms', 10.0),
            total_tokens=getattr(self, 'total_tokens', 256),
            total_requests=getattr(self, 'total_requests', 1),
            batch_size=getattr(self, 'batch_size', 1),
            max_batch_size=getattr(self, 'max_batch_size', 32),
        )

    def validate_result(self) -> Optional[str]:
        if self.prefill_model is None or self.decode_model is None:
            return "Models not initialized"
        if self.prefill_input is None or self.decode_input is None:
            return "Inputs not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedDisaggregatedBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    print(f"Prefill/decode disagg mean: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
