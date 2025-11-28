#!/usr/bin/env python3
"""Level 0: Naive MoE Baseline.

NO OPTIMIZATIONS - Sequential expert execution with Python loops.
This is our starting point for measuring compound improvements.

Expected: ~25ms baseline
"""
import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level0Naive(MoEJourneyBenchmark):
    """Level 0: Naive baseline."""
    LEVEL = 0


def get_benchmark() -> Level0Naive:
    return Level0Naive()


if __name__ == "__main__":
    run_level(0)
