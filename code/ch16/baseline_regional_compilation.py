"""baseline_regional_compilation.py - Baseline: Full model compilation (hangs hard).

Demonstrates the problem: torch.compile on entire large model (>40B params) hangs indefinitely.
This baseline shows what NOT to do for large models.

Regional Compilation: This baseline compiles the ENTIRE model at once, which causes hangs
on models >40B parameters due to graph explosion and memory exhaustion. There is no eager
fallback here—if compilation fails, the harness marks the benchmark as skipped so we can
surface the limitation explicitly.
"""

from __future__ import annotations

from common.python import compile_utils as _compile_utils_patch  # noqa: F401
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)
from common.python.benchmark_utils import warn_benchmark_scaling

MODEL_CANDIDATES: List[Dict[str, Any]] = [
    {
        "label": "20B (48x7168)",
        "n_layers": 48,
        "d_model": 7168,
        "d_ff": 28672,
        "seq_len": 2048,
    },
    {
        "label": "15B (36x6400)",
        "n_layers": 36,
        "d_model": 6400,
        "d_ff": 25600,
        "seq_len": 2048,
    },
    {
        "label": "11B (32x5632)",
        "n_layers": 32,
        "d_model": 5632,
        "d_ff": 22528,
        "seq_len": 1536,
    },
    {
        "label": "8B (24x5120)",
        "n_layers": 24,
        "d_model": 5120,
        "d_ff": 20480,
        "seq_len": 1536,
    },
    {
        "label": "6B (16x4096)",
        "n_layers": 16,
        "d_model": 4096,
        "d_ff": 16384,
        "seq_len": 1536,
    },
]


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch16")
    return torch.device("cuda")


class LargeTransformerBlock(nn.Module):
    """A large transformer block that's computationally expensive."""
    
    def __init__(self, d_model: int = 8192, d_ff: int = 32768):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=64, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln1(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + attn_out
        residual = x
        x = self.ln2(x)
        x = residual + self.mlp(x)
        return x


class LargeTransformerModel(nn.Module):
    """A large transformer model (~40B+ parameters) that causes compilation hangs."""
    
    def __init__(self, n_layers: int = 48, d_model: int = 8192, d_ff: int = 32768):
        super().__init__()
        self.embed = nn.Embedding(50304, d_model)
        self.blocks = nn.ModuleList([
            LargeTransformerBlock(d_model, d_ff) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, 50304, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x


class BaselineRegionalCompilationBenchmark(Benchmark):
    """Baseline: Full model execution with no regional compilation optimizations."""

    def __init__(self):
        self.device = resolve_device()
        self.model: Optional[nn.Module] = None
        self.sequence_schedule = [512, 768, 1024, 1280, 1536, 1792, 2048]
        self._iteration = 0

    def setup(self) -> None:
        """Instantiate the transformer model in eager mode."""
        candidate = MODEL_CANDIDATES[-1]
        self.model = LargeTransformerModel(
            n_layers=candidate["n_layers"],
            d_model=candidate["d_model"],
            d_ff=candidate["d_ff"],
        ).to(self.device, dtype=torch.float32).eval()

    def benchmark_fn(self) -> None:
        """Function to benchmark - runs inference."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        if self.model is None:
            raise RuntimeError("Model not initialized")

        seq_len = self.sequence_schedule[self._iteration % len(self.sequence_schedule)]
        self._iteration += 1
        input_data = torch.randint(0, 50304, (1, seq_len), device=self.device, dtype=torch.long)
        with nvtx_range("regional_compilation", enable=enable_nvtx):
            with torch.no_grad():
                _ = self.model(input_data)
        torch.cuda.synchronize()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=1,
            warmup=0,
            setup_timeout_seconds=240,
            measurement_timeout_seconds=240,
            use_subprocess=False,
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None

    def teardown(self) -> None:
        self.model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()



def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineRegionalCompilationBenchmark()


def main():
    """Run the baseline benchmark."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = BaselineRegionalCompilationBenchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\n[OK] Baseline completed: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print("This benchmark purposefully compiles the full 20B model to show how long")
    print("end-to-end compilation takes before regional compilation is applied.")


if __name__ == "__main__":
    main()
