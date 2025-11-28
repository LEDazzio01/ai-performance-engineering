"""Benchmark/utility that records GPU↔NUMA topology to artifacts/topology/."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.dynamic_router.topology import detect_topology, write_topology


class TopologyProbeBenchmark(BaseBenchmark):
    """Capture a snapshot of GPU↔NUMA mapping for downstream routing demos."""

    def __init__(self) -> None:
        super().__init__()
        self.snapshot = None
        self.output_path: Optional[Path] = None

    def setup(self) -> None:
        # Nothing to initialize besides ensuring artifacts dir exists (handled by write_topology).
        return

    def benchmark_fn(self) -> None:
        topo = detect_topology()
        self.output_path = write_topology(topo)
        self.snapshot = topo

    def get_config(self) -> Optional[BenchmarkConfig]:
        # Single-shot capture
        return BenchmarkConfig(iterations=1, warmup=5)

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        if self.snapshot is None:
            return None
        gpu_numa = {f"gpu{idx}_numa": float(node) if node is not None else -1.0 for idx, node in self.snapshot.gpu_numa.items()}
        gpu_numa["num_gpus_detected"] = float(len(self.snapshot.gpu_numa))
        gpu_numa["numa_nodes_known"] = float(len(self.snapshot.distance))
        return gpu_numa


def get_benchmark() -> BaseBenchmark:
    return TopologyProbeBenchmark()


if __name__ == "__main__":
    bench = get_benchmark()
    bench.benchmark_fn()
    print(bench.get_custom_metrics())
