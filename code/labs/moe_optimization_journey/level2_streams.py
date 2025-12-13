#!/usr/bin/env python3
"""Level 2: Multi-Stream Expert Parallelism.

ADDS: Run top-K experts on parallel CUDA streams.
- Overlaps expert computations
- Based on ch15/expert_parallelism.py pattern
- Reduces total execution time by hiding latency

Cumulative: batched + multi-stream
"""
import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent.parent
import torch

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level2Streams(MoEJourneyBenchmark):
    """Level 2: + Multi-stream expert parallelism."""
    LEVEL = 2

def get_benchmark() -> Level2Streams:
    return Level2Streams()


if __name__ == "__main__":
    run_level(2)
