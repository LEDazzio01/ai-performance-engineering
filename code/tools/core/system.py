"""
Thin wrappers over PerfCore for system/software/dependency introspection.

Centralizes system collection so CLI/MCP/AI layers do not drift.
"""

from __future__ import annotations

from typing import Any, Dict

from tools.core.perf_core import get_core


def get_software_info() -> Dict[str, Any]:
    return get_core().get_software_info()


def get_dependency_health() -> Dict[str, Any]:
    return get_core().get_dependency_health()


def check_dependency_updates() -> Dict[str, Any]:
    return get_core().check_dependency_updates()


def get_full_system_context() -> Dict[str, Any]:
    return get_core().get_full_system_context()


def list_available_benchmarks() -> Dict[str, Any]:
    return get_core().get_available_benchmarks()


