#!/usr/bin/env python3
"""Level 4: Grouped GEMM (Production MoE Pattern).

ADDS: Sort tokens by expert + run contiguous per-expert GEMM.

This is how production MoE systems work (vLLM, SGLang):
1. bucket_by_expert() from ch19/mxfp8_moe_common.py
2. Run GEMM per expert on contiguous memory
3. Restore original token order

Benefits:
- Contiguous memory access per expert
- Better cache utilization
- Enables CUTLASS grouped GEMM

Cumulative: batched + fused + mem_efficient + grouped
"""
import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level4Grouped(MoEJourneyBenchmark):
    """Level 4: + Grouped GEMM (sort + per-expert)."""
    LEVEL = 4


def get_benchmark() -> Level4Grouped:
    return Level4Grouped()


if __name__ == "__main__":
    run_level(4)

