"""Monolithic inference path where prefill and decode block each other."""

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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch18")
    return torch.device("cuda")


class BaselineDisaggregatedBenchmark(Benchmark):
    """Baseline that executes prefill + decode serially inside one service."""

    def __init__(self):
        self.device = resolve_device()
        self.workload = WORKLOAD
        self.smoke_test = is_smoke_test()

        self.hidden_dim = self.workload.attention_hidden_dim
        self.num_heads = self.workload.attention_num_heads
        self.batch_size = self.workload.attention_batch_size
        self.prefill_seq = self.workload.seq_len(self.smoke_test)
        self.decode_seq = self.workload.decode_len(self.smoke_test)
        self.prefill_segments = self.workload.micro_batches_for_mode(self.smoke_test)
        self.decode_steps = self.workload.micro_batches_for_mode(self.smoke_test) * 4

        self.decoder: Optional[nn.TransformerDecoder] = None
        self.prefill_batches: list[torch.Tensor] = []
        self.decode_token: Optional[torch.Tensor] = None
        self.memory: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.manual_seed(42)
        layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            batch_first=True,
            dim_feedforward=self.hidden_dim * 4,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=4).to(self.device).half().eval()

        segment_len = max(32, self.prefill_seq // self.prefill_segments)
        self.prefill_batches = [
            torch.randn(self.batch_size, segment_len, self.hidden_dim, dtype=torch.float16, device=self.device)
            for _ in range(self.prefill_segments)
        ]
        self.decode_token = torch.randn(self.batch_size, 1, self.hidden_dim, dtype=torch.float16, device=self.device)
        self.memory = torch.randn(
            self.batch_size,
            segment_len,
            self.hidden_dim,
            dtype=torch.float16,
            device=self.device,
        )
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        assert self.decoder is not None
        assert self.decode_token is not None
        assert self.memory is not None

        with nvtx_range("baseline_disaggregated", enable=enable_nvtx):
            for step in range(self.decode_steps):
                prefill_batch = self.prefill_batches[step % len(self.prefill_batches)]
                _ = self.decoder(prefill_batch, self.memory)
                torch.cuda.synchronize()
                _ = self.decoder(self.decode_token, self.memory)
                torch.cuda.synchronize()

    def teardown(self) -> None:
        self.decoder = None
        self.prefill_batches = []
        self.decode_token = None
        self.memory = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=3,
            warmup=1,
            enable_memory_tracking=False,
            measurement_timeout_seconds=90,
        )

    def validate_result(self) -> Optional[str]:
        if self.decoder is None:
            return "Decoder not initialized"
        if not self.prefill_batches:
            return "Prefill inputs missing"
        if self.decode_token is None:
            return "Decode token missing"
        return None


def get_benchmark() -> Benchmark:
    return BaselineDisaggregatedBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=BenchmarkConfig(iterations=3, warmup=1))
    result = harness.benchmark(BaselineDisaggregatedBenchmark())
    print(f"Baseline disaggregated mean: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
