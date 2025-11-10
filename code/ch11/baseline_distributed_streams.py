"""Baseline single-GPU stream workload (no distributed overlap)."""

from __future__ import annotations

from ch11.stream_overlap_base import StridedStreamBaseline


class BaselineDistributedStreamsBenchmark(StridedStreamBaseline):
    def __init__(self) -> None:
        super().__init__("baseline_distributed_streams")


def get_benchmark() -> StridedStreamBaseline:
    return BaselineDistributedStreamsBenchmark()
