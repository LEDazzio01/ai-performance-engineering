"""
Lightweight capability registry for aisp plugins/extensions.

Core code can query registered capabilities to shape help output or behavior
without hard dependencies on optional extensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Capability:
    name: str
    description: str = ""
    provider: Optional[str] = None  # e.g., plugin package name


_CAPABILITIES: Dict[str, Capability] = {}


def register_capability(name: str, description: str = "", provider: Optional[str] = None) -> None:
    """Register a capability by name (idempotent)."""
    if not name:
        return
    if name in _CAPABILITIES:
        return
    _CAPABILITIES[name] = Capability(name=name, description=description, provider=provider)


def list_capabilities() -> List[Capability]:
    """Return all registered capabilities."""
    return list(_CAPABILITIES.values())


def has_capability(name: str) -> bool:
    """Check if a capability is registered."""
    return name in _CAPABILITIES
