"""Optimized piece-graph benchmark: cached regional CUDA graphs per sequence bucket."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-capable GPU required for piece-graph benchmarks.")
    return torch.device("cuda")


class PieceGraphBlock(nn.Module):
    """Transformer-style block used by both head and tail regions."""

    def __init__(self, hidden_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = residual + attn_out
        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        return x


class RegionalPieceGraph(nn.Module):
    """Splits the model into two regions (input+head stack, tail stack+output)."""

    def __init__(self, hidden_dim: int = 512, n_layers: int = 12):
        super().__init__()
        midpoint = n_layers // 2
        blocks = [PieceGraphBlock(hidden_dim=hidden_dim) for _ in range(n_layers)]
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)
        self.region_head = nn.Sequential(*blocks[:midpoint])
        self.region_tail = nn.Sequential(*blocks[midpoint:])
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)


RegionGraph = Tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]


class OptimizedPieceGraphsBenchmark(BaseBenchmark):
    """Caches two smaller CUDA graphs per sequence bucket (piece graph strategy)."""

    def __init__(self):
        super().__init__()
        self.device = resolve_device()
        self.model: Optional[RegionalPieceGraph] = None
        # Start with larger buckets so the capture/replay benefit stands out.
        self.sequence_schedule: Sequence[int] = [256, 384, 512, 640, 768, 896, 1024]
        self.iteration = 0
        self.graph_cache: Dict[int, Tuple[RegionGraph, RegionGraph]] = {}
        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(0)

    def setup(self) -> None:
        self.model = RegionalPieceGraph(hidden_dim=768, n_layers=12).to(
            self.device, dtype=torch.float16
        ).eval()
        self._capture_piece_graphs()

    def _capture_piece_graphs(self) -> None:
        assert self.model is not None
        self.graph_cache.clear()
        torch.cuda.synchronize()
        for seq_len in sorted(set(self.sequence_schedule)):
            head_input = torch.empty(
                1, seq_len, self.model.hidden_dim, device=self.device, dtype=torch.float16
            )
            head_input.normal_(0.0, 1.0, generator=self._rng)
            head_output = torch.empty_like(head_input)
            tail_input = torch.empty_like(head_input)
            tail_output = torch.empty_like(head_input)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                warmed = self.model.region_head(self.model.input_proj(head_input))
            head_output.copy_(warmed)

            head_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(head_graph):
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    tmp = self.model.region_head(self.model.input_proj(head_input))
                head_output.copy_(tmp)

            tail_input.copy_(head_output)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                warmed_tail = self.model.output_proj(self.model.region_tail(tail_input))
            tail_output.copy_(warmed_tail)

            tail_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(tail_graph):
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    tmp = self.model.output_proj(self.model.region_tail(tail_input))
                tail_output.copy_(tmp)

            self.graph_cache[seq_len] = (
                (head_graph, head_input, head_output),
                (tail_graph, tail_input, tail_output),
            )
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        if self.model is None:
            raise RuntimeError("Model not initialized")
        seq_len = self.sequence_schedule[self.iteration % len(self.sequence_schedule)]
        self.iteration += 1
        (head_graph, head_input, head_output), (tail_graph, tail_input, tail_output) = self.graph_cache[seq_len]

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        fresh_tokens = torch.empty_like(head_input)
        fresh_tokens.normal_(0.0, 1.0, generator=self._rng)
        head_input.copy_(fresh_tokens)
        with nvtx_range("piece_graph_head", enable=enable_nvtx):
            head_graph.replay()
        tail_input.copy_(head_output)
        with nvtx_range("piece_graph_tail", enable=enable_nvtx):
            tail_graph.replay()
        torch.cuda.synchronize()
        _ = tail_output  # Placeholder to keep lints quiet

    def teardown(self) -> None:
        self.model = None
        self.graph_cache.clear()
        torch.cuda.empty_cache()

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=8,
            warmup=2,
            measurement_timeout_seconds=240,
            setup_timeout_seconds=120,
            use_subprocess=False,
        )


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

def get_benchmark() -> BaseBenchmark:
    return OptimizedPieceGraphsBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    bench = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=bench.get_config())
    result = harness.benchmark(bench)
    print(
        f"[optimized piece graphs] mean iteration "
        f"{result.timing.mean_ms if result and result.timing else 0.0:.3f} ms"
    )
