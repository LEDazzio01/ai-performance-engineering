"""Helpers for documenting allocator tuning (jemalloc / tcmalloc) in Chapter 3.

These utilities do **not** force a different allocator at runtime. Instead they
surface the recommended environment variables, detect whether the current
process is already running with those knobs, and print actionable guidance so
the benchmark scripts can record the state in their console output.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional


# Recommended knobs mirrored from ch03/docker_gpu_optimized.dockerfile
_RECOMMENDED_ENV: Dict[str, str] = {
    "MALLOC_CONF": "narenas:8,dirty_decay_ms:10000,muzzy_decay_ms:10000,background_thread:true",
    "TCMALLOC_MAX_TOTAL_THREAD_CACHE_BYTES": "536870912",
    "TCMALLOC_RELEASE_RATE": "16",
}

# Candidate library paths for jemalloc / tcmalloc on Ubuntu based systems.
_JEMALLOC_CANDIDATES = (
    Path("/usr/lib/x86_64-linux-gnu/libjemalloc.so.2"),
    Path("/usr/lib/aarch64-linux-gnu/libjemalloc.so.2"),
    Path("/usr/lib64/libjemalloc.so.2"),
)
_TCMALLOC_CANDIDATES = (
    Path("/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"),
    Path("/usr/lib/aarch64-linux-gnu/libtcmalloc_minimal.so.4"),
    Path("/usr/lib64/libtcmalloc_minimal.so.4"),
)


def _find_first_existing(paths: tuple[Path, ...]) -> Optional[str]:
    for candidate in paths:
        if candidate.exists():
            return str(candidate)
    return None


def recommended_allocator_env() -> Dict[str, str]:
    """Return a copy of the recommended allocator environment variables."""
    return dict(_RECOMMENDED_ENV)


def detect_jemalloc() -> Optional[str]:
    """Return the first jemalloc shared library path that exists, if any."""
    return _find_first_existing(_JEMALLOC_CANDIDATES)


def detect_tcmalloc() -> Optional[str]:
    """Return the first tcmalloc-minimal shared library path that exists, if any."""
    return _find_first_existing(_TCMALLOC_CANDIDATES)


def is_allocator_env_tuned() -> bool:
    """Check whether the current process already uses the recommended settings."""
    for key, expected in _RECOMMENDED_ENV.items():
        if os.environ.get(key) != expected:
            return False
    ld_preload = os.environ.get("LD_PRELOAD", "")
    jemalloc_path = detect_jemalloc()
    return bool(jemalloc_path and jemalloc_path in ld_preload)


def format_recommended_command() -> str:
    """Return a shell snippet that enables the recommended allocator knobs."""
    parts = []
    jemalloc_path = detect_jemalloc()
    if jemalloc_path:
        parts.append(f'LD_PRELOAD="{jemalloc_path} ${{LD_PRELOAD:-}}"')
    for key, value in _RECOMMENDED_ENV.items():
        parts.append(f'{key}="{value}"')
    if not parts:
        return "python"
    return "env " + " ".join(parts) + " python"


def describe_allocator_state() -> Dict[str, Optional[str]]:
    """Summarise the allocator related environment for logging purposes."""
    state: Dict[str, Optional[str]] = {
        key: os.environ.get(key) for key in _RECOMMENDED_ENV.keys()
    }
    state["LD_PRELOAD"] = os.environ.get("LD_PRELOAD")
    state["jemalloc_available"] = detect_jemalloc()
    state["tcmalloc_available"] = detect_tcmalloc()
    return state


def log_allocator_guidance(
    example_name: str,
    optimized: bool,
    logger=None,
) -> None:
    """Emit guidance for the chapter benchmarks about allocator tuning."""

    def _emit(message: str) -> None:
        if logger is None:
            print(message, file=sys.stderr)
        else:
            logger(message)

    tuned = is_allocator_env_tuned()
    status = "enabled" if tuned else "disabled"
    _emit(f"[allocator] {example_name}: recommended allocator tuning is {status}.")

    if optimized and not tuned:
        _emit(
            "[allocator] To enable jemalloc tuning, re-run with:\n"
            f"  {format_recommended_command()} {example_name}.py"
        )
        jemalloc_path = detect_jemalloc()
        if jemalloc_path is None:
            _emit(
                "[allocator] jemalloc shared library not found. "
                "Install package 'libjemalloc2' to enable LD_PRELOAD."
            )
    elif not optimized and tuned:
        _emit(
            "[allocator] Baseline is running with tuned allocator settings. "
            "For a fair comparison, consider disabling LD_PRELOAD/MALLOC_CONF."
        )


