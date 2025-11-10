"""Disaggregated inference where prefill and decode run on separate streams."""

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


class OptimizedDisaggregatedBenchmark(Benchmark):
    """Prefill executes in parallel with decode thanks to stream-level separation."""

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

        self.prefill_model: Optional[nn.TransformerDecoder] = None
        self.decode_model: Optional[nn.GRU] = None
        self.prefill_batches: list[torch.Tensor] = []
        self.decode_tokens: Optional[torch.Tensor] = None
        self.memory: Optional[torch.Tensor] = None
        self.prefill_stream: Optional[torch.cuda.Stream] = None
        self.decode_stream: Optional[torch.cuda.Stream] = None

    def setup(self) -> None:
        torch.manual_seed(42)
        layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            batch_first=True,
            dim_feedforward=self.hidden_dim * 4,
        )
        self.prefill_model = nn.TransformerDecoder(layer, num_layers=4).to(self.device).half().eval()
        self.decode_model = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
        ).to(self.device).half().eval()

        segment_len = max(32, self.prefill_seq // self.prefill_segments)
        self.prefill_batches = [
            torch.randn(self.batch_size, segment_len, self.hidden_dim, dtype=torch.float16, device=self.device)
            for _ in range(self.prefill_segments)
        ]
        self.decode_tokens = torch.randn(
            self.batch_size,
            self.decode_seq,
            self.hidden_dim,
            dtype=torch.float16,
            device=self.device,
        )
        self.memory = torch.randn(
            self.batch_size,
            segment_len,
            self.hidden_dim,
            dtype=torch.float16,
            device=self.device,
        )
        self.prefill_stream = torch.cuda.Stream()
        self.decode_stream = torch.cuda.Stream()
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        assert self.prefill_model is not None
        assert self.decode_model is not None
        assert self.decode_tokens is not None
        assert self.memory is not None
        assert self.prefill_stream is not None
        assert self.decode_stream is not None

        events: list[torch.cuda.Event] = []
        with nvtx_range("optimized_disaggregated", enable=enable_nvtx):
            with torch.cuda.stream(self.prefill_stream):
                for batch in self.prefill_batches:
                    _ = self.prefill_model(batch, self.memory)
                    event = torch.cuda.Event()
                    event.record(self.prefill_stream)
                    events.append(event)

            with torch.cuda.stream(self.decode_stream):
                if events:
                    events[0].wait(self.decode_stream)
                hx: Optional[torch.Tensor] = None
                for step in range(self.decode_steps):
                    idx = step % self.decode_tokens.size(1)
                    token = self.decode_tokens[:, idx : idx + 1, :]
                    _, hx = self.decode_model(token, hx)

            self.prefill_stream.synchronize()
            self.decode_stream.synchronize()

    def teardown(self) -> None:
        self.prefill_model = None
        self.decode_model = None
        self.prefill_batches = []
        self.decode_tokens = None
        self.memory = None
        self.prefill_stream = None
        self.decode_stream = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=3,
            warmup=1,
            enable_memory_tracking=False,
            measurement_timeout_seconds=90,
        )

    def validate_result(self) -> Optional[str]:
        if self.prefill_model is None or self.decode_model is None:
            return "Models not initialized"
        if not self.prefill_batches:
            return "Prefill inputs missing"
        if self.decode_tokens is None:
            return "Decode tokens missing"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedDisaggregatedBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=BenchmarkConfig(iterations=3, warmup=1))
    result = harness.benchmark(OptimizedDisaggregatedBenchmark())
    print(f"Optimized disaggregated mean: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
