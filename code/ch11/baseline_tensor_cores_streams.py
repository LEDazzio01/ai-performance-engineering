"""Baseline tensor-core stream workload without overlap."""

from __future__ import annotations

from ch11.stream_overlap_base import StridedStreamBaseline


class BaselineTensorCoresStreamsBenchmark(StridedStreamBaseline):
    def __init__(self) -> None:
        super().__init__("baseline_tensor_cores_streams")


def get_benchmark() -> StridedStreamBaseline:
    return BaselineTensorCoresStreamsBenchmark()
