"""Baseline stream workload without adaptive scheduling."""

from __future__ import annotations

from ch11.stream_overlap_base import StridedStreamBaseline


class BaselineAdaptiveStreamsBenchmark(StridedStreamBaseline):
    def __init__(self) -> None:
        super().__init__("baseline_adaptive_streams")


def get_benchmark() -> StridedStreamBaseline:
    return BaselineAdaptiveStreamsBenchmark()
