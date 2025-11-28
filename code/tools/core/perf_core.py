"""
Shared PerformanceCore facade used by CLI, MCP, and libraries.

This wraps the dashboard's PerformanceCore without starting an HTTP server,
so other layers can reuse the business logic without subclassing an HTTP
handler or duplicating GPU/system collection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from tools.dashboard.server import PerformanceCore
from tools.analysis.performance_analyzer import (
    PerformanceAnalyzer,
    load_benchmark_data as load_benchmark_results,
)


class PerfCore(PerformanceCore):
    """
    Standalone PerformanceCore that skips HTTP handler initialization.

    Only sets up the analyzer and data_file; callers can directly invoke the
    methods defined on PerformanceCore (GPU/system info, analysis, etc.).
    """

    def __init__(self, data_file: Optional[Path] = None):
        # Do NOT call the HTTP handler constructor.
        self.data_file = data_file
        self._analyzer: Optional[PerformanceAnalyzer] = PerformanceAnalyzer(
            lambda: load_benchmark_results(self.data_file)
        )


_CORE_SINGLETON: Optional[PerfCore] = None


def get_core(data_file: Optional[Path] = None, refresh: bool = False) -> PerfCore:
    """
    Get a singleton PerfCore instance.

    Args:
        data_file: Optional path to benchmark results.
        refresh: If True, force a new instance (e.g., after changing data_file).
    """
    global _CORE_SINGLETON
    if refresh or _CORE_SINGLETON is None:
        _CORE_SINGLETON = PerfCore(data_file=data_file)
    return _CORE_SINGLETON

