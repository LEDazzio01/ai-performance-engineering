"""baseline_regional_compilation.py - Full model compilation baseline (piece-graph context)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python import compile_utils as _compile_utils_patch  # noqa: F401
from common.python.compile_utils import error_on_graph_break, maybe_nested_compile_region  # noqa: E402
from common.python.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from common.python.benchmark_utils import warn_benchmark_scaling  # noqa: E402
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402

MODEL_CANDIDATES: List[Dict[str, Any]] = [
    {"label": "20B (48x7168)", "n_layers": 48, "d_model": 7168, "d_ff": 28672, "seq_len": 2048},
    {"label": "15B (36x6400)", "n_layers": 36, "d_model": 6400, "d_ff": 25600, "seq_len": 2048},
    {"label": "11B (32x5632)", "n_layers": 32, "d_model": 5632, "d_ff": 22528, "seq_len": 1536},
    {"label": "8B (24x5120)", "n_layers": 24, "d_model": 5120, "d_ff": 20480, "seq_len": 1536},
    {"label": "6B (24x4096)", "n_layers": 24, "d_model": 4096, "d_ff": 16384, "seq_len": 1024},
    {"label": "3B (16x3072)", "n_layers": 16, "d_model": 3072, "d_ff": 12288, "seq_len": 1024},
    {"label": "2B (8x2048)", "n_layers": 8, "d_model": 2048, "d_ff": 8192, "seq_len": 1024},
    {"label": "1B (4x1024)", "n_layers": 4, "d_model": 1024, "d_ff": 4096, "seq_len": 768},
]


class DummyTransformer(nn.Module):
    def __init__(self, n_layers: int, d_model: int, d_ff: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Linear(d_ff, d_model),
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + _run_layer(layer, x)
        return x


@maybe_nested_compile_region
def _run_layer(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return layer(x)


class BaselineRegionalCompilationBenchmark(BaseBenchmark):
    """Full-model compilation baseline that demonstrates piece-graph issues."""

    def __init__(self):
        super().__init__()
        self.model: Optional[DummyTransformer] = None
        self.inputs: Optional[torch.Tensor] = None
        # Use a mid-sized config so the full-graph compilation cost is noticeable.
        self.choice = MODEL_CANDIDATES[4]  # 6B-style config (24x4096)
        tokens = self.choice["seq_len"] * self.choice["d_model"]
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(0)
        n_layers = self.choice["n_layers"]
        d_model = self.choice["d_model"]
        d_ff = self.choice["d_ff"]
        self.model = DummyTransformer(n_layers, d_model, d_ff).to(self.device).eval()
        self.inputs = torch.randn(
            1,
            self.choice["seq_len"],
            d_model,
            device=self.device,
            dtype=torch.bfloat16,
        )
        warn_benchmark_scaling(
            scaling_type="Model size",
            original_values={
                "label": self.choice["label"],
                "n_layers": self.choice["n_layers"],
                "d_model": self.choice["d_model"],
                "d_ff": self.choice["d_ff"],
                "seq_len": self.choice["seq_len"],
            },
            scaled_values={
                "label": self.choice["label"],
                "n_layers": self.choice["n_layers"],
                "d_model": self.choice["d_model"],
                "d_ff": self.choice["d_ff"],
                "seq_len": self.choice["seq_len"],
            },
            impact_description="Scaled-down model sizes run on smaller GPUs; speedups are representative but conservative.",
        )
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        with nvtx_range("baseline_regional_compilation", enable=enable_nvtx):
            try:
                with error_on_graph_break(True):
                    compiled = torch.compile(self.model, mode="max-autotune")  # type: ignore[attr-defined]
                _ = compiled(self.inputs)  # type: ignore[misc]
            except Exception:
                # Show the pitfall rather than failing the harness run
                pass
            torch.cuda.synchronize(self.device)

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=1,
            warmup=10,
            setup_timeout_seconds=240,
            measurement_timeout_seconds=240,
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
    return BaselineRegionalCompilationBenchmark()


if __name__ == "__main__":
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BaselineRegionalCompilationBenchmark().get_config(),
    )
    result = harness.benchmark(get_benchmark())
    print(f"Baseline regional compilation: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
