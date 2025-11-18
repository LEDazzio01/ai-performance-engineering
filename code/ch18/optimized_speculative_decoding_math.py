"""optimized_speculative_decoding_math.py - Flash-off variant for GB10."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


def _disable_flash_sdp() -> None:
    """Force math SDPA paths to avoid sm80-only flash kernels."""
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


class OptimizedSpeculativeDecodingMathBenchmark(BaseBenchmark):
    """Speculative decoding with flash disabled (math path for GB10)."""

    def __init__(self):
        super().__init__()
        self.target_model: Optional[nn.Module] = None
        self.draft_model: Optional[nn.Module] = None
        self.embedding: Optional[nn.Module] = None
        self.input_ids: Optional[torch.Tensor] = None
        self.memory: Optional[torch.Tensor] = None
        self.max_length = 20
        self.speculative_length = 4
        batch_size = 4
        seq_len = 10
        tokens = batch_size * (seq_len + self.max_length)
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        _disable_flash_sdp()
        torch.manual_seed(42)
        hidden_dim = 256
        vocab_size = 1000

        self.embedding = nn.Embedding(vocab_size, hidden_dim).to(self.device).eval()
        self.target_model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=4,
        ).to(self.device).eval()
        self.draft_model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=2,
        ).to(self.device).eval()

        batch_size = 4
        seq_len = 10
        self.input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
        self.memory = torch.randn(batch_size, seq_len, hidden_dim, device=self.device)
        self._synchronize()
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )

    def benchmark_fn(self) -> None:
        assert self.embedding is not None
        assert self.target_model is not None
        assert self.draft_model is not None
        assert self.input_ids is not None
        assert self.memory is not None

        with self._nvtx_range("optimized_speculative_decoding_math"):
            with torch.no_grad():
                current_ids = self.input_ids.clone()
                while current_ids.size(1) < self.input_ids.size(1) + self.max_length:
                    tgt_embedded = self.embedding(current_ids)
                    draft_output = self.draft_model(tgt_embedded, self.memory)
                    draft_tokens = draft_output[:, -self.speculative_length:, :].argmax(dim=-1)
                    current_ids = torch.cat([current_ids, draft_tokens], dim=1)
                    if current_ids.size(1) >= self.input_ids.size(1) + self.max_length:
                        break
        self._synchronize()

    def teardown(self) -> None:
        self.target_model = None
        self.draft_model = None
        self.embedding = None
        self.input_ids = None
        self.memory = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=2)

    def validate_result(self) -> Optional[str]:
        if self.target_model is None or self.draft_model is None:
            return "Models not initialized"
        if self.input_ids is None:
            return "Input IDs not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedSpeculativeDecodingMathBenchmark()


if __name__ == "__main__":
    bench = get_benchmark()
    result = bench.run(BenchmarkConfig(iterations=5, warmup=1))
    print(result)
