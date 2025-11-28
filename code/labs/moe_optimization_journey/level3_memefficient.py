#!/usr/bin/env python3
"""Level 3: Memory Efficient Execution.

ADDS: Reuse pre-allocated buffers instead of creating new tensors.

Benefits:
- Reduces memory allocation overhead
- Less garbage collection pressure
- Better memory locality

Cumulative: batched + fused + mem_efficient
"""
import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level3MemEfficient(MoEJourneyBenchmark):
    """Level 3: + Memory efficient (buffer reuse)."""
    LEVEL = 3


def get_benchmark() -> Level3MemEfficient:
    return Level3MemEfficient()


if __name__ == "__main__":
    run_level(3)

