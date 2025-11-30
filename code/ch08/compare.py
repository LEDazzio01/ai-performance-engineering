"""Chapter 8: Compare baseline vs optimized implementations using formal harness.

Uses the BaseBenchmark - benchmarks provide get_benchmark() function,
harness measures directly (no subprocess, no output parsing).
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import torch

# Import arch_config early to set up torch inductor cache directory
# This prevents C++ compilation errors when torch.compile is used
try:
    from ch08 import arch_config  # noqa: F401 - triggers cache setup
except ImportError:
    pass  # If arch_config not available, continue without it

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
)
from core.utils.chapter_compare_template import (
    profile_template,
)


def profile() -> Dict[str, Any]:
    """Compare all baseline/optimized pairs using formal harness."""
    chapter_dir = Path(__file__).parent
    
    return profile_template(
        chapter='ch08',
        chapter_dir=chapter_dir,
        harness_config=BenchmarkConfig(iterations=20, warmup=5),
    )


if __name__ == '__main__':
    result = profile()
    print("\nMetrics:", result)
