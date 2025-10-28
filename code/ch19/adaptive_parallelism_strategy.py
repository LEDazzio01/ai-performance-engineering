"""Adaptive parallelism routing demo (Chapter 19)."""

from __future__ import annotations
import arch_config  # noqa: F401 - Configure Blackwell optimizations

import logging
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ParallelismStrategy(Enum):
    TENSOR = "tensor_parallel"
    PIPELINE = "pipeline_parallel"
    HYBRID = "hybrid"
    DATA = "data_parallel"


@dataclass
class WorkloadMetrics:
    seq_len: int
    batch_size: int
    gpu_mem_util: float
    concurrent_reqs: int
    avg_latency_ms: float
    throughput_tokens: float


@dataclass
class StrategyConfig:
    strategy: ParallelismStrategy
    tp: int
    pp: int
    dp: int
    est_latency_ms: float
    est_throughput: float
    mem_efficiency: float


class DynamicParallelismRouter:
    def __init__(self, gpus: int = 8) -> None:
        self.gpus = gpus
        self.current = ParallelismStrategy.TENSOR
        self.last_switch = 0.0
        self.cooldown = 5.0
        self.profiles: Dict[ParallelismStrategy, StrategyConfig] = {
            ParallelismStrategy.TENSOR: StrategyConfig(ParallelismStrategy.TENSOR, gpus, 1, 1, 50.0, 900.0, 0.7),
            ParallelismStrategy.PIPELINE: StrategyConfig(ParallelismStrategy.PIPELINE, 1, gpus, 1, 85.0, 700.0, 0.9),
            ParallelismStrategy.HYBRID: StrategyConfig(ParallelismStrategy.HYBRID, gpus // 2, 2, 1, 65.0, 800.0, 0.85),
            ParallelismStrategy.DATA: StrategyConfig(ParallelismStrategy.DATA, 1, 1, gpus, 120.0, 1400.0, 0.6),
        }

    def choose(self, metrics: WorkloadMetrics) -> ParallelismStrategy:
        if metrics.concurrent_reqs > 32 and metrics.seq_len <= 256:
            return ParallelismStrategy.DATA
        if metrics.seq_len > 1024 or metrics.gpu_mem_util > 0.8:
            return ParallelismStrategy.PIPELINE if metrics.seq_len > 2048 else ParallelismStrategy.HYBRID
        return ParallelismStrategy.TENSOR

    def score(self, metrics: WorkloadMetrics, strategy: ParallelismStrategy) -> float:
        cfg = self.profiles[strategy]
        latency = max(0.0, 1.0 - metrics.avg_latency_ms / 200.0)
        throughput = min(1.0, metrics.throughput_tokens / 2000.0)
        memory = 1.0 - metrics.gpu_mem_util
        if strategy == ParallelismStrategy.TENSOR:
            return 0.6 * latency + 0.3 * throughput + 0.1 * memory
        if strategy == ParallelismStrategy.PIPELINE:
            return 0.2 * latency + 0.3 * throughput + 0.5 * memory
        if strategy == ParallelismStrategy.HYBRID:
            return 0.4 * latency + 0.4 * throughput + 0.2 * memory
        return 0.1 * latency + 0.7 * throughput + 0.2 * memory

    def maybe_switch(self, metrics: WorkloadMetrics) -> Optional[StrategyConfig]:
        now = time.time()
        if now - self.last_switch < self.cooldown:
            return None
        proposed = self.choose(metrics)
        if proposed == self.current:
            return None
        current_score = self.score(metrics, self.current)
        proposed_score = self.score(metrics, proposed)
        if proposed_score > current_score * 1.1:
            logger.info("Switching %s -> %s (score %.2f -> %.2f)", self.current.value, proposed.value, current_score, proposed_score)
            self.current = proposed
            self.last_switch = now
            return self.profiles[proposed]
        return None


def collect_metrics() -> WorkloadMetrics:
    seq_len = random.choice([128, 256, 512, 1024, 2048, 4096])
    load = random.uniform(0.3, 0.9)
    return WorkloadMetrics(
        seq_len=seq_len,
        batch_size=random.randint(4, 32),
        gpu_mem_util=0.4 + load * 0.5,
        concurrent_reqs=int(load * 50),
        avg_latency_ms=60 + load * 90,
        throughput_tokens=1100 * (2.0 - load),
    )


def simulate(router: DynamicParallelismRouter, seconds: float = 10.0) -> None:
    start = time.time()
    switches = 0
    processed = 0
    latency_total = 0.0
    while time.time() - start < seconds:
        metrics = collect_metrics()
        cfg = router.maybe_switch(metrics)
        if cfg:
            switches += 1
        cfg = router.profiles[router.current]
        simulated_latency = cfg.est_latency_ms * (metrics.seq_len / 1024.0) * (1.0 + metrics.gpu_mem_util * 0.3)
        processing_time = simulated_latency / 1000.0
        time.sleep(processing_time)
        processed += 1
        latency_total += simulated_latency
        if processed % 50 == 0:
            avg = latency_total / processed
            logger.info("Processed %d requests, avg latency %.1f ms, strategy %s, switches %d",
                        processed, avg, router.current.value, switches)

    print(f"Processed {processed} requests in {seconds:.1f}s, switches={switches}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    router = DynamicParallelismRouter(gpus=8)
    print("Chapter 19: Adaptive parallelism demo")
    simulate(router)


if __name__ == "__main__":
    main()
