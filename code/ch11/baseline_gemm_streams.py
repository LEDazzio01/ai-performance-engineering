"""Baseline GEMM stream workload without overlap."""

from __future__ import annotations

from ch11.stream_overlap_base import StridedStreamBaseline


class BaselineGemmStreamsBenchmark(StridedStreamBaseline):
    def __init__(self) -> None:
        super().__init__("baseline_gemm_streams")


def get_benchmark() -> StridedStreamBaseline:
    return BaselineGemmStreamsBenchmark()
