#!/usr/bin/env python3
"""Level 2: Triton Fused SiLU*Up.

ADDS: Custom Triton kernel that fuses SiLU activation with elementwise multiply.

Before: gate → memory → SiLU → memory → multiply → memory
After:  gate → memory → fused_silu_mul → memory

This eliminates one kernel launch and one memory round-trip.

Cumulative: batched + fused
"""
import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level2Fused(MoEJourneyBenchmark):
    """Level 2: + Triton fused SiLU*up."""
    LEVEL = 2


def get_benchmark() -> Level2Fused:
    return Level2Fused()


if __name__ == "__main__":
    run_level(2)

