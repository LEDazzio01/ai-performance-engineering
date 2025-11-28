"""Baseline harness for Chapter 17 dynamic routing."""

from __future__ import annotations

import random
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
from ch17.dynamic_routing import DisaggregatedRouter, Priority, Request, WorkerMetrics  # noqa: E402


class _DynamicRoutingBenchmark(BaseBenchmark):
    """Shared logic for baseline/optimized routing harnesses."""

    def __init__(self, *, batch_size: int, vectorized: bool):
        super().__init__()
        self.batch_size = batch_size
        self.vectorized = vectorized
        self.router = DisaggregatedRouter()
        self._history: Dict[str, List[float]] = {"lat_ms": []}
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(batch_size),
            tokens_per_iteration=float(batch_size * 128),
        )
        # Pre-generated requests (created once in setup, reused in benchmark)
        self._cached_requests: List[Request] = []
        # Pre-allocated tensors for vectorized path (reused each iteration)
        self._prompt_lengths: Optional[torch.Tensor] = None
        self._cached_lengths: Optional[torch.Tensor] = None
        self._queue_lengths: Optional[torch.Tensor] = None
        self._priorities: Optional[torch.Tensor] = None

    def setup(self) -> None:
        now = time.time()
        for idx in range(4):
            self.router.prefill_workers[f"prefill-{idx}"] = self._make_metrics(queue=idx, now=now)
            self.router.decode_workers[f"decode-{idx}"] = self._make_metrics(queue=idx // 2, now=now)
        
        # Pre-generate requests once (not part of benchmark timing)
        self._cached_requests = self._generate_requests()
        
        # Pre-allocate tensors for vectorized routing (avoids allocation in hot path)
        if self.vectorized:
            self._prompt_lengths = torch.tensor(
                [len(r.prompt_tokens) for r in self._cached_requests], dtype=torch.int32
            )
            self._cached_lengths = torch.tensor(
                [r.prefix_cached_length for r in self._cached_requests], dtype=torch.int32
            )
            self._queue_lengths = torch.zeros(self.batch_size, dtype=torch.int32)
            self._priorities = torch.tensor(
                [0 if r.priority is Priority.LOW else (2 if r.priority is Priority.HIGH else 1) 
                 for r in self._cached_requests], dtype=torch.int32
            )

    def _make_metrics(self, queue: int, now: float):
        return WorkerMetrics(
            queue_length=queue,
            gpu_utilization=random.uniform(0.4, 0.8),
            memory_usage=random.uniform(30.0, 70.0),
            kv_cache_usage=random.uniform(10.0, 50.0),
            active_requests=random.randint(1, 4),
            last_updated=now,
        )

    def _generate_requests(self) -> List[Request]:
        reqs: List[Request] = []
        for idx in range(self.batch_size):
            prompt_len = random.randint(64, 2048)
            cached = random.randint(0, min(prompt_len // 2, 512))
            reqs.append(
                Request(
                    id=f"req-{idx}",
                    prompt_tokens=list(range(prompt_len)),
                    priority=random.choice(list(Priority)),
                    timestamp=time.time(),
                    prefix_cached_length=cached,
                    expected_output_length=random.randint(16, 128),
                )
            )
        return reqs

    def benchmark_fn(self) -> Dict[str, float]:
        # Use pre-generated requests (generation time excluded from benchmark)
        requests = self._cached_requests
        rejects = 0
        offloaded = 0
        start = self._record_start()

        if self.vectorized and self._prompt_lengths is not None:
            # Vectorized routing decisions using pre-allocated tensors
            # Only randomize queue lengths each iteration (simulates changing load)
            self._queue_lengths.random_(0, 10)

            # Vectorized boolean operations (single pass over data)
            long_prefill = (self._prompt_lengths - self._cached_lengths) > self.router.PREFILL_LENGTH_THRESHOLD
            capacity = self._queue_lengths < self.router.PREFILL_QUEUE_MAX
            offload_mask = long_prefill & capacity

            load_estimate = self._queue_lengths.float() * self.router.avg_prefill_time_per_req
            slo_mask = load_estimate <= self.router.TTFT_SLO_MAX
            admit_mask = torch.logical_or(slo_mask, self._priorities == 2)

            rejects = int((~admit_mask).sum().item())
            offloaded = int(torch.logical_and(admit_mask, offload_mask).sum().item())
        else:
            # Python loop-based routing (sequential, one-at-a-time)
            for req in requests:
                if not self.router.admit_request(req):
                    rejects += 1
                    continue
                queue_depth = random.randint(0, self.router.PREFILL_QUEUE_MAX + 5)
                if self.router.should_offload_prefill(len(req.prompt_tokens), req.prefix_cached_length, queue_depth):
                    offloaded += 1

        elapsed_ms = self._record_stop(start)
        self._history["lat_ms"].append(elapsed_ms)
        served = len(requests) - rejects

        return {
            "requests": float(len(requests)),
            "served": float(served),
            "rejected": float(rejects),
            "offloaded": float(offloaded),
        }

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=8, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        if not self._history["lat_ms"]:
            return None
        return {
            "routing.latency_ms": float(statistics.mean(self._history["lat_ms"])),
        }


class BaselineDynamicRoutingBenchmark(_DynamicRoutingBenchmark):
    """Python loop-based routing decisions."""
    def __init__(self) -> None:
        # Match optimized batch size for fair comparison
        super().__init__(batch_size=1024, vectorized=False)


def get_benchmark():
    return BaselineDynamicRoutingBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    print(result)
