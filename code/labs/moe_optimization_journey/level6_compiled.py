#!/usr/bin/env python3
"""Level 6: torch.compile - The Grand Finale!

ADDS: torch.compile with mode='max-autotune'.

torch.compile does ALL of the previous optimizations automatically:
- Kernel fusion (like our Triton fused SiLU*up)
- Memory planning (like our buffer reuse)
- Operator reordering
- Triton code generation
- CUDA graph capture (with reduce-overhead mode)

The key insight: torch.compile achieves similar performance to
5 levels of manual optimization with ONE LINE OF CODE!

Cumulative: ALL previous optimizations + torch.compile magic
"""
import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level6Compiled(MoEJourneyBenchmark):
    """Level 6: + torch.compile (does ALL of the above!)."""
    LEVEL = 6


def get_benchmark() -> Level6Compiled:
    return Level6Compiled()


if __name__ == "__main__":
    run_level(6)

