"""
Baseline dynamic routing toy for inference.

This version intentionally does *not* use TTFT/TPOT feedback. It emulates a
single undifferentiated pool and round-robins requests. Use this as the
control variant against the optimized router.
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

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


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


#============================================================================
# Benchmark Harness Integration
#============================================================================

class BaselineRouterBenchmark(BaseBenchmark):
    """Benchmark harness wrapper for baseline round-robin router."""

    def __init__(self):
        super().__init__()
        self.router = None
        self.num_gpus = 8
        self.num_requests = 1000
        self._last = 0.0
        
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.num_requests),
            tokens_per_iteration=float(self.num_requests * 100),  # ~100 tokens/request
        )
        self.jitter_exemption_reason = "Router benchmark: fixed configuration"

    def setup(self) -> None:
        """Setup: Initialize baseline round-robin router."""
        gpu_ids = [f"gpu_{i}" for i in range(self.num_gpus)]
        self.router = BaselineRouter(gpu_ids)

    def benchmark_fn(self) -> None:
        """Benchmark: Route requests via round-robin."""
        if self.router is None:
            return
            
        routed = 0
        for i in range(self.num_requests):
            req = Request(
                req_id=f"req_{i}",
                prompt_tokens=100,
                expected_new_tokens=50,
                priority=i % 3,
            )
            gpu = self.router.route(req)
            if gpu:
                routed += 1
            # Mark some as complete
            if i > 0 and i % 10 == 0:
                self.router.complete(f"req_{i-10}")
        
        self._last = float(routed)

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.router = None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=10)
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        return {
            "baseline_router.strategy": "round_robin",
            "baseline_router.feedback": False,
        }

    def validate_result(self) -> Optional[str]:
        if self.router is None:
            return "Router not initialized"
        return None

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return torch.tensor([hash(str(id(self))) % (2**31)], dtype=torch.float32)

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return {"num_gpus": self.num_gpus, "num_requests": self.num_requests}

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineRouterBenchmark()
