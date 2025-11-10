"""Chapter 15: Compare baseline vs optimized implementations using formal harness.

Uses the Benchmark protocol - benchmarks provide get_benchmark() function,
harness measures directly (no subprocess, no output parsing).
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import (
    BenchmarkConfig,
)
from common.python.chapter_compare_template import (
    profile_template,
)


def profile() -> Dict[str, Any]:
    """Compare all baseline/optimized pairs using formal harness."""
    chapter_dir = Path(__file__).parent
    
    return profile_template(
        chapter='ch15',
        chapter_dir=chapter_dir,
        harness_config=BenchmarkConfig(iterations=20, warmup=5),
    )


if __name__ == '__main__':
    result = profile()
    print("\nMetrics:", result)
