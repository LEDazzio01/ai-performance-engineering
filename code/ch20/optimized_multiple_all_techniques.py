"""optimized_multiple_all_techniques.py - Alias to the combined techniques benchmark.

This file exists for backwards compatibility with older references and
integration tests. The canonical implementation lives in
`optimized_multiple_unoptimized.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import ch20.arch_config  # noqa: F401 - Apply chapter defaults

from ch20.optimized_multiple_unoptimized import OptimizedAllTechniquesBenchmark
from core.harness.benchmark_harness import BaseBenchmark


def get_benchmark() -> BaseBenchmark:
    return OptimizedAllTechniquesBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)

