"""Optimized NVSHMEM IBGDA microbenchmark (GPU-initiated path)."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch04.nvshmem_ibgda_microbench import NvshmemIbgdaMicrobench

_DEFAULT_KWARGS = dict(
    mode="p",
    bytes_per_message=1024,
    ctas=32,
    threads=256,
    iters=500,
)


class OptimizedNvshmemIbgdaMicrobench(NvshmemIbgdaMicrobench):
    def __init__(self) -> None:
        super().__init__(enable_ibgda=True, **_DEFAULT_KWARGS)


def get_benchmark() -> NvshmemIbgdaMicrobench:
    return OptimizedNvshmemIbgdaMicrobench()
