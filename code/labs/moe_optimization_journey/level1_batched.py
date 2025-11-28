#!/usr/bin/env python3
"""Level 1: Batched Expert Execution.

ADDS: Batched matmul for parallel expert computation.
- Eliminates Python loops
- All experts computed via single batched einsum
- ~2-3x speedup over Level 0

Cumulative: Level 0 + batched execution
"""
import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level1Batched(MoEJourneyBenchmark):
    """Level 1: + Batched expert execution."""
    LEVEL = 1


def get_benchmark() -> Level1Batched:
    return Level1Batched()


if __name__ == "__main__":
    run_level(1)

