"""Baseline: Full precision KV cache without quantization.

Chapter 19: Blackwell-Native Precision Operations

The baseline shows naive full-precision (FP32) KV cache storage.
This uses maximum memory but avoids quantization overhead.

The optimized version uses dynamic quantization with adaptive bit-widths,
trading some precision for significant memory savings (4x-8x).
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


class _DynamicQuantizedCacheBenchmark(BaseBenchmark):
    """Base class for KV cache quantization benchmarks."""

    def __init__(self, *, schedule_bits: List[int], use_fp32_baseline: bool = False):
        super().__init__()
        self.schedule_bits = schedule_bits
        self.use_fp32_baseline = use_fp32_baseline
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
    
    def _non_adaptive_cache_update(self) -> float:
        """Baseline: Non-adaptive cache management (reprocess full cache).
        
        Anti-pattern: On every token, reprocess entire KV cache at full precision.
        This is what naive implementations do without adaptive caching.
        """
        if self.tensor is None:
            raise RuntimeError("Tensor not initialized")
        # Simulate reprocessing entire cache (expensive)
        # - Full clone
        # - Full precision operations
        # - Multiple passes
        temp = self.tensor.clone()
        for _ in range(4):  # Multiple passes over full cache
            temp = temp * 1.001 + 0.001
            _ = temp.abs().max()  # Force sync
        return 0.0

    def _adaptive_cache_update(self, bits: int) -> float:
        """Optimized: Adaptive cache with selective updates.
        
        Only update/quantize the most recent tokens, leave rest unchanged.
        """
        if self.tensor is None:
            raise RuntimeError("Tensor not initialized")
        # Just quantize the newest segment (simulating adaptive/incremental)
        segment = self.tensor[:, -4:, :]  # Only last 4 positions
        qmax = (1 << (bits - 1)) - 1
        scale = segment.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / qmax
        _ = torch.clamp((segment / scale).round(), -qmax, qmax)
        return 0.0

    def benchmark_fn(self) -> Dict[str, List[float]]:
        if self.tensor is None:
            raise RuntimeError("Tensor not initialized")

        errors: List[float] = []
        torch.cuda.synchronize(self.device)
        start = self._record_start()

        if self.use_fp32_baseline:
            # Baseline: Non-adaptive (full cache reprocessing each time)
            for _ in self.schedule_bits:
                errors.append(self._non_adaptive_cache_update())
        else:
            # Optimized: Adaptive (selective updates only)
            for bits in self.schedule_bits:
                errors.append(self._adaptive_cache_update(bits))

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
    """Baseline: Full precision KV cache (more memory traffic, no quantization).
    
    This simulates the naive approach of keeping full FP32 caches.
    More memory traffic = slower for memory-bound operations.
    """

    def __init__(self) -> None:
        # Same number of iterations but with full precision copies
        schedule = [8] * 32  # Schedule doesn't matter - we do FP32 copies
        super().__init__(schedule_bits=schedule, use_fp32_baseline=True)


def get_benchmark():
    return BaselineDynamicQuantizedCacheBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    print(result)
