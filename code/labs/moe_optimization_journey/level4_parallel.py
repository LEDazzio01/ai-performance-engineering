#!/usr/bin/env python3
"""Level 4: Multi-Stream Expert Parallelism.

ADDS: Run expert groups on different CUDA streams.
Based on ch15/expert_parallelism.py and 
ch15/optimized_moe_shared_expert_overlap.py patterns.

- Overlaps expert computation across streams
- Better GPU utilization
- Real expert parallelism pattern

Cumulative: batched + permuted + grouped + parallel
"""
import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent.parent
import torch

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level4Parallel(MoEJourneyBenchmark):
    """Level 4: + Multi-stream expert parallelism."""
    LEVEL = 4

def get_benchmark() -> Level4Parallel:
    return Level4Parallel()


if __name__ == "__main__":
    run_level(4)


