"""Baseline benchmark for the dynamic quantized cache helpers."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from ch19 import dynamic_quantized_cache as dq  # noqa: E402


class _DynamicQuantizedCacheBenchmark(BaseBenchmark):
    """Shared harness that compresses synthetic KV caches."""

    def __init__(self, *, schedule_bits: List[int]):
        super().__init__()
        self.schedule_bits = schedule_bits
        self.tensor: Optional[torch.Tensor] = None
        self._history: Dict[str, List[float]] = {"latency_ms": [], "error": []}
        total_tokens = len(schedule_bits) * 64
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(total_tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(7)
        self.tensor = torch.randn(8, 32, 128, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize(self.device)

    def _quantize(self, bits: int) -> float:
        if self.tensor is None:
            raise RuntimeError("Tensor not initialized")
        qmax = (1 << (bits - 1)) - 1
        scale = self.tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / qmax
        quant = torch.clamp((self.tensor / scale).round(), -qmax, qmax)
        approx = quant * scale
        return float((self.tensor - approx).abs().max().item())

    def benchmark_fn(self) -> Dict[str, List[float]]:
        if self.tensor is None:
            raise RuntimeError("Tensor not initialized")

        errors: List[float] = []
        torch.cuda.synchronize(self.device)
        start = self._record_start()

        for bits in self.schedule_bits:
            errors.append(self._quantize(bits))

        torch.cuda.synchronize(self.device)
        latency_ms = self._record_stop(start)
        self._history["latency_ms"].append(latency_ms)
        self._history["error"].extend(errors)
        return {"errors": errors}

    def teardown(self) -> None:
        self.tensor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        if not self._history["latency_ms"]:
            return None
        avg_ms = statistics.mean(self._history["latency_ms"])
        avg_err = statistics.mean(self._history["error"])
        payload_bits = sum(self.schedule_bits) * self.tensor.numel() if self.tensor is not None else 0
        throughput_gbps = 0.0
        if avg_ms > 0 and payload_bits:
            throughput_gbps = (payload_bits / avg_ms) / 1e6
        return {
            "kv_cache.mean_latency_ms": float(avg_ms),
            "kv_cache.mean_error": float(avg_err),
            "kv_cache.throughput_gbps": float(throughput_gbps),
        }


class BaselineDynamicQuantizedCacheBenchmark(_DynamicQuantizedCacheBenchmark):
    """Static INT8 quantization without fallback."""

    def __init__(self) -> None:
        schedule = [8] * 32
        super().__init__(schedule_bits=schedule)


def get_benchmark():
    return BaselineDynamicQuantizedCacheBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    print(result)
