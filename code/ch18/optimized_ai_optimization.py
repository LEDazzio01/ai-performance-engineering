"""Optimized attention path that uses learned heuristics to batch work efficiently."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from common.python.benchmark_harness import Benchmark, BenchmarkConfig
from ch18.workload_config import WORKLOAD, is_smoke_test

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch18")
    return torch.device("cuda")


class FusedAttentionModel(nn.Module):
    """Attention block that leverages learned heuristics for chunk sizing."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.optimizer_model = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, seq, hidden = tokens.shape
        features = torch.tensor(
            [[seq, batch, hidden]],
            dtype=tokens.dtype,
            device=tokens.device,
        )
        ratio = self.optimizer_model(features).item()
        chunk = max(256, min(seq, int(seq * (0.75 + 0.25 * ratio))))

        outputs: list[torch.Tensor] = []
        with torch.cuda.amp.autocast(dtype=torch.float16):
            for chunk_tokens in torch.split(tokens, chunk, dim=1):
                fused, _ = self.attn(chunk_tokens, chunk_tokens, chunk_tokens, need_weights=False)
                outputs.append(fused)
        return torch.cat(outputs, dim=1)


class OptimizedAiOptimizationBenchmark(Benchmark):
    """AI-assisted attention implementation that batches work adaptively."""

    def __init__(self):
        self.device = resolve_device()
        self.workload = WORKLOAD
        self.smoke_test = is_smoke_test()
        self.hidden_dim = self.workload.attention_hidden_dim
        self.num_heads = self.workload.attention_num_heads
        self.batch_size = self.workload.attention_batch_size
        self.sequence_length = self.workload.seq_len(self.smoke_test)
        self.micro_batches = self.workload.micro_batches_for_mode(self.smoke_test)

        self.model: Optional[FusedAttentionModel] = None
        self.token_cache: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.manual_seed(42)
        self.model = FusedAttentionModel(self.hidden_dim, self.num_heads).to(self.device).half().eval()
        self.token_cache = torch.randn(
            self.batch_size,
            self.sequence_length,
            self.hidden_dim,
            dtype=torch.float16,
            device=self.device,
        )
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        assert self.model is not None
        assert self.token_cache is not None

        with nvtx_range("optimized_ai_optimization", enable=enable_nvtx):
            with torch.no_grad():
                for micro in range(self.micro_batches):
                    tokens = torch.roll(self.token_cache, shifts=micro * 32, dims=1).contiguous()
                    _ = self.model(tokens)

    def teardown(self) -> None:
        self.model = None
        self.token_cache = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=2,
            warmup=1,
            enable_memory_tracking=False,
            measurement_timeout_seconds=120,
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Attention model not initialized"
        if self.token_cache is None:
            return "Input cache not initialized"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedAiOptimizationBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=BenchmarkConfig(iterations=3, warmup=1))
    result = harness.benchmark(OptimizedAiOptimizationBenchmark())
    print(f"Optimized AI optimization mean: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
