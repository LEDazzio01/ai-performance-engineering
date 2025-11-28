"""
Thin wrappers over the shared PerfCore GPU helpers.

This keeps GPU/system collection centralized in one implementation
(`tools/core/perf_core.py` -> `PerformanceCore`) instead of duplicating
`nvidia-smi` parsing in multiple places.
"""

from __future__ import annotations

from typing import Any, Dict

from tools.core.perf_core import get_core


def get_gpu_info() -> Dict[str, Any]:
    """Return detailed GPU info from the shared core."""
    return get_core().get_gpu_info()


def get_all_gpus() -> Dict[str, Any]:
    """Return multi-GPU details when available."""
    core = get_core()
    info = core.get_gpu_info() or {}
    if "gpus" in info:
        return {"gpus": info.get("gpus", [])}
    try:
        topo = core.get_gpu_topology()
        return topo
    except Exception:
        return {"gpus": []}


def get_gpu_count() -> int:
    """Return the number of GPUs detected."""
    info = get_gpu_info() or {}
    if "gpus" in info and isinstance(info["gpus"], list):
        return len(info["gpus"])
    # Fallback to single entry if the API is summarized
    return int(info.get("gpu_count", 1) or 1)


def get_topology() -> Dict[str, Any]:
    """Return GPU topology matrix and NVLink/PCIe relationships."""
    return get_core().get_gpu_topology()


def get_nvlink_status() -> Dict[str, Any]:
    """Return NVLink status and bandwidth."""
    return get_core().get_nvlink_status()

