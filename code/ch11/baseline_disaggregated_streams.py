"""Baseline disaggregated stream workload without overlap."""

from __future__ import annotations

from ch11.stream_overlap_base import StridedStreamBaseline


class BaselineDisaggregatedStreamsBenchmark(StridedStreamBaseline):
    def __init__(self) -> None:
        super().__init__("baseline_disaggregated_streams")


def get_benchmark() -> StridedStreamBaseline:
    return BaselineDisaggregatedStreamsBenchmark()
