"""Baseline guided decoding - math-only SDP for compatibility."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

try:
    import ch1.arch_config  # noqa: F401
except ImportError:
    pass

from typing import Optional

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from common.python.compile_utils import enable_tf32


class BaselineGuidedDecodingMathBenchmark(BaseBenchmark):
    """Baseline: guided decoding with schema, forcing SDP to math only."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.input_ids = None
        self.schema = None
        self.max_length = 20
        self.batch_size = 4
        self.seq_len = 10
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
                torch.backends.cuda.enable_cudnn_sdp(False)
            except Exception:
                pass
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()

        torch.manual_seed(42)
        vocab_size = 1000
        hidden_dim = 256
        self.model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=1,
        ).to(self.device).eval()

        self.schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "count": {"type": "number"},
            },
            "required": ["summary"],
        }

        self.input_ids = torch.randint(0, vocab_size, (self.batch_size, self.seq_len), device=self.device)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        with self._nvtx_range("baseline_guided_decoding_math"):
            with torch.no_grad():
                embedded_input = torch.randn(self.input_ids.size(0), self.input_ids.size(1), 256, device=self.device)
                memory = torch.randn(self.input_ids.size(0), self.input_ids.size(1), 256, device=self.device)
                output = self.model(embedded_input, memory)
                _ = output.sum()
            self._synchronize()

    def teardown(self) -> None:
        self.model = None
        self.input_ids = None
        self.schema = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineGuidedDecodingMathBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Guided Decoding (math SDP): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
