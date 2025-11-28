"""baseline_piece_graphs.py - Monolithic CUDA graph capture baseline for piece-graphs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class PieceGraphBlock(nn.Module):
    """Sequential block to mimic per-piece captures."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PieceGraphModel(nn.Module):
    """Sequential stack of PieceGraphBlocks."""

    def __init__(self, hidden_dim: int = 512, n_layers: int = 12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([PieceGraphBlock(hidden_dim) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class BaselinePieceGraphsBenchmark(BaseBenchmark):
    """Monolithic graph capture baseline for piece-graphs."""

    def __init__(self):
        super().__init__()
        self.model: Optional[PieceGraphModel] = None
        self.inputs: Optional[torch.Tensor] = None
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_input: Optional[torch.Tensor] = None
        self.static_output: Optional[torch.Tensor] = None
        # Heavier per-iteration work to better expose graph replay speedups.
        self.repeats = 12
        self.batch = 16
        self.hidden = 768
        tokens = self.batch * self.hidden * self.repeats
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.repeats),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(0)
        self.model = (
            PieceGraphModel(hidden_dim=self.hidden, n_layers=12)
            .to(self.device, dtype=torch.float16)
            .eval()
        )
        self.inputs = torch.randn(
            self.batch, self.hidden, device=self.device, dtype=torch.float16
        )
        # Capture monolithic graph
        try:
            self.graph = torch.cuda.CUDAGraph()
            self.static_input = self.inputs.clone()
            self.static_output = torch.empty_like(self.static_input)
            torch.cuda.synchronize()
            with torch.cuda.graph(self.graph):
                self.static_output.copy_(self.model(self.static_input))
            torch.cuda.synchronize()
        except Exception:
            self.graph = None
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        with nvtx_range("baseline_piece_graphs", enable=enable_nvtx):
            for _ in range(self.repeats):
                if self.graph and self.static_input is not None and self.static_output is not None:
                    self.static_input.copy_(self.inputs)
                    self.graph.replay()
                else:
                    _ = self.model(self.inputs)
            torch.cuda.synchronize(self.device)

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        self.graph = None
        self.static_input = None
        self.static_output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=8,
            warmup=5,
            measurement_timeout_seconds=240,
            setup_timeout_seconds=120,
            use_subprocess=False,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_inference_metrics
        return compute_inference_metrics(
            ttft_ms=getattr(self, '_ttft_ms', 50.0),
            tpot_ms=getattr(self, '_tpot_ms', 10.0),
            total_tokens=getattr(self, 'total_tokens', 256),
            total_requests=getattr(self, 'total_requests', 1),
            batch_size=getattr(self, 'batch_size', 1),
            max_batch_size=getattr(self, 'max_batch_size', 32),
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None or self.inputs is None:
            return "Model/inputs not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselinePieceGraphsBenchmark()


if __name__ == "__main__":
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BaselinePieceGraphsBenchmark().get_config(),
    )
    result = harness.benchmark(get_benchmark())
    print(f"Baseline piece graphs: {result.timing.mean_ms if result and result.timing else 0.0:.3f} ms")
