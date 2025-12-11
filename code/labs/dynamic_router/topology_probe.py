"""Benchmark/utility that records GPU↔NUMA topology to artifacts/topology/."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.dynamic_router.topology import detect_topology, write_topology


class TopologyProbeBenchmark(BaseBenchmark):
    """Capture a snapshot of GPU↔NUMA mapping for downstream routing demos."""

    def __init__(self) -> None:
        super().__init__()
        self.snapshot = None
        self.output_path: Optional[Path] = None
        self.metrics: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None

    def setup(self) -> None:
        # Nothing to initialize besides ensuring artifacts dir exists (handled by write_topology).
        return

    def benchmark_fn(self) -> None:
        topo = detect_topology()
        self.output_path = write_topology(topo)
        self.snapshot = topo
        metrics_dict = self.get_custom_metrics() or {}
        metric_values = list(metrics_dict.values()) or [0.0]
        expected_shape = (1, len(metric_values))
        if self.metrics is None or tuple(self.metrics.shape) != expected_shape:
            self.metrics = torch.randn(expected_shape, dtype=torch.float32)
        summary_tensor = torch.tensor([metric_values], dtype=torch.float32)
        self.output = (summary_tensor + self.metrics).detach()

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

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        if self.output is None:
            raise RuntimeError("benchmark_fn() must be called before verification")
        return self.output

    def get_input_signature(self) -> dict:
        shape = tuple(self.metrics.shape) if self.metrics is not None else (1, max(1, len((self.get_custom_metrics() or {})) or 1))
        return {"type": "topology_probe", "shapes": {"metrics": shape}}

    def get_output_tolerance(self) -> tuple:
        return (0.1, 1.0)

    def teardown(self) -> None:
        self.metrics = None
        self.output = None
        self.snapshot = None
        self.output_path = None
        super().teardown()



def get_benchmark() -> BaseBenchmark:
    return TopologyProbeBenchmark()


if __name__ == "__main__":
    bench = get_benchmark()
    bench.benchmark_fn()
    print(bench.get_custom_metrics())
