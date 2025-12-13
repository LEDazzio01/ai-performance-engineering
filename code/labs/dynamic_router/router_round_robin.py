"""
Round-robin router policy (tool module).

This version intentionally does *not* use TTFT/TPOT feedback. It emulates a
single undifferentiated pool and round-robins requests. Use this as the
control variant against the telemetry-driven router policy.
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


@dataclass
class Request:
    """Minimal request metadata used by the simulator."""

    req_id: str
    prompt_tokens: int
    expected_new_tokens: int
    priority: int = 0


class BaselineRouter:
    """
    Round-robin router with a single pool and no feedback.

    This mirrors a naive admission strategy:
      - No prefill/decode separation
      - No TTFT/TPOT awareness
      - No KV locality or migration
    """

    def __init__(self, gpu_ids: Iterable[str]) -> None:
        gpu_list: List[str] = list(gpu_ids)
        if not gpu_list:
            raise ValueError("BaselineRouter requires at least one GPU id")
        self._gpu_ids = gpu_list
        self._rr = itertools.cycle(self._gpu_ids)
        self._inflight: Dict[str, str] = {}  # req_id -> gpu_id

    def route(self, req: Request) -> str:
        """
        Pick the next GPU in round-robin order.

        Returns the chosen gpu_id so the caller can record placement.
        """
        gpu = next(self._rr)
        self._inflight[req.req_id] = gpu
        return gpu

    def complete(self, req_id: str) -> Optional[str]:
        """Mark a request complete; returns the GPU it was on."""
        return self._inflight.pop(req_id, None)

    def inflight(self) -> Dict[str, str]:
        """Expose current placements for debugging/metrics."""
        return dict(self._inflight)
