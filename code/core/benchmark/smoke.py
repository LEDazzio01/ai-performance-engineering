"""Smoke test mode detection.

When running with --smoke-test, benchmarks should use reduced workloads
for quick validation. In full mode, benchmarks use realistic workloads.
"""

import os


def is_smoke_mode() -> bool:
    """Check if running in smoke test mode.
    
    Returns True if BENCHMARK_SMOKE_TEST env var is set or if running
    with reduced resources. Full benchmarks should use this to scale
    down workloads appropriately.
    """
    # Check environment variable
    if os.environ.get("BENCHMARK_SMOKE_TEST", "").lower() in ("1", "true", "yes"):
        return True
    
    # Check for --smoke-test argument in sys.argv
    import sys
    if "--smoke-test" in sys.argv or "--smoke" in sys.argv:
        return True
    
    return False






