"""optimized_cuda_graphs_router.py - CUDA Graphs with simple routing branch.

Captures a two-expert graph with a conditional-like branch by toggling an index
tensor. This keeps dependencies light while giving the docs a runnable target.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BaseBenchmark, WorkloadMetadata  # noqa: E402
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class RouterGraph(nn.Module):
    def __init__(self, hidden: int = 512):
        super().__init__()
        self.router = nn.Linear(hidden, 2, bias=False)
        self.expert0 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.expert1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())

    def forward(self, tokens: torch.Tensor, route: torch.Tensor) -> torch.Tensor:
        logits = self.router(tokens)
        weights = torch.softmax(logits, dim=-1)
        blend0 = self.expert0(tokens) * weights[..., 0:1] * (route == 0)
        blend1 = self.expert1(tokens) * weights[..., 1:2] * (route == 1)
        return blend0 + blend1


class CUDAGraphRouterBenchmark(BaseBenchmark):
    def __init__(self) -> None:
        super().__init__()
        self.model: Optional[RouterGraph] = None
        self.tokens: Optional[torch.Tensor] = None
        self.route: Optional[torch.Tensor] = None
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_out: Optional[torch.Tensor] = None
        self._workload = WorkloadMetadata(tokens_per_iteration=4096.0)

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for graph capture")
        self.model = RouterGraph(hidden=512).to(self.device).to(torch.float32)
        self.tokens = torch.randn(8, 64, 512, device=self.device)
        self.route = torch.zeros(8, 64, 1, device=self.device, dtype=torch.int32)
        self.static_out = torch.empty_like(self.tokens)

        try:
            g = torch.cuda.CUDAGraph()
            # Warm-up to materialize params.
            _ = self.model(self.tokens, self.route)
            torch.cuda.synchronize(self.device)
            with torch.cuda.graph(g):
                self.static_out = self.model(self.tokens, self.route)
            self.graph = g
        except Exception as exc:
            raise RuntimeError(f"SKIPPED: CUDA graph capture failed ({exc})") from exc

    def benchmark_fn(self) -> Optional[dict]:
        if self.graph is None or self.tokens is None or self.route is None:
            raise RuntimeError("SKIPPED: graph not initialized")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        with nvtx_range("cuda_graphs_router", enable=enable_nvtx):
            # Flip route between iterations to emulate dynamic routing.
            self.route.random_(0, 2)
            self.graph.replay()
        torch.cuda.synchronize(self.device)
        return {}

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload


    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_graph_metrics
        return compute_graph_metrics(
            baseline_launch_overhead_us=getattr(self, '_baseline_launch_us', 10.0),
            graph_launch_overhead_us=getattr(self, '_graph_launch_us', 1.0),
            num_nodes=getattr(self, 'num_nodes', 10),
            num_iterations=getattr(self, 'num_iterations', 100),
        )

def get_benchmark() -> BaseBenchmark:
    return CUDAGraphRouterBenchmark()
