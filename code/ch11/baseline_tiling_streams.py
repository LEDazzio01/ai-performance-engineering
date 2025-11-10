"""Baseline stream workload without tiling/overlap."""

from __future__ import annotations

from ch11.stream_overlap_base import StridedStreamBaseline


class BaselineTilingStreamsBenchmark(StridedStreamBaseline):
    def __init__(self) -> None:
        super().__init__("baseline_tiling_streams")


def get_benchmark() -> StridedStreamBaseline:
    return BaselineTilingStreamsBenchmark()
