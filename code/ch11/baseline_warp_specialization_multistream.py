"""Baseline warp-specialization stream workload (no overlap)."""

from __future__ import annotations

from ch11.stream_overlap_base import StridedStreamBaseline


class BaselineWarpSpecializationStreamsBenchmark(StridedStreamBaseline):
    def __init__(self) -> None:
        super().__init__("baseline_warp_specialization_multistream")


def get_benchmark() -> StridedStreamBaseline:
    return BaselineWarpSpecializationStreamsBenchmark()
