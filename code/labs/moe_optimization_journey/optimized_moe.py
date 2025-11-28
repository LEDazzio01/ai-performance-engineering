#!/usr/bin/env python3
"""Optimized MoE: Level 6 (torch.compile - Full Optimization)."""
import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from labs.moe_optimization_journey.level6_compiled import Level6Compiled, get_benchmark
__all__ = ["Level6Compiled", "get_benchmark"]
