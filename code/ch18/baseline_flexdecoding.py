"""Baseline harness for FlexDecoding with SDPA fallback."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from ch18 import flexdecoding as flexdemo  # noqa: E402


class FlexDecodingHarness(BaseBenchmark):
    """Shared benchmarking harness for baseline/optimized FlexDecoding runs."""

    def __init__(self, *, use_flex_attention: bool, require_flex: bool, decode_tokens: int = 128):
        super().__init__()
        self.use_flex_attention = use_flex_attention
        self.require_flex = require_flex
        self.decode_tokens = decode_tokens
        self.config = flexdemo.FlexDecodingConfig()
        self.model: Optional[flexdemo.FlexDecodingModule] = None
        self.prefill_tokens: Optional[torch.Tensor] = None
        self.decode_token: Optional[torch.Tensor] = None
        self._history: Dict[str, List[float]] = {"prefill_ms": [], "decode_ms": []}
        total_tokens = self.config.max_seq_len + self.decode_tokens
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(total_tokens),
        )

    # --------------------------------------------------------------------- setup
    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("FlexDecoding benchmarks require CUDA")
        if self.require_flex and not getattr(flexdemo, "HAS_FLEX", False):
            raise RuntimeError("FlexAttention is not available; optimized path cannot run.")

        previous_flag = flexdemo.HAS_FLEX
        flexdemo.HAS_FLEX = self.use_flex_attention and previous_flag
        self.model = flexdemo.FlexDecodingModule(self.config).to(self.device).eval()
        self.model.ensure_compiled()
        flexdemo.HAS_FLEX = previous_flag

        torch.manual_seed(0)
        self.prefill_tokens = torch.randn(1, self.config.window * 2, self.config.dim, device=self.device)
        self.decode_token = torch.randn(1, 1, self.config.dim, device=self.device)
        torch.cuda.synchronize(self.device)

    # --------------------------------------------------------------- benchmark_fn
    def benchmark_fn(self) -> Dict[str, List[float]]:
        if self.model is None or self.prefill_tokens is None or self.decode_token is None:
            raise RuntimeError("Model/tokens not initialized")

        prefill_times: List[float] = []
        decode_times: List[float] = []
        base_position = self.prefill_tokens.size(1)

        with torch.no_grad():
            with self._nvtx_range("flex_prefill"):
                start = time.perf_counter()
                _ = self.model.prefill(self.prefill_tokens)
                torch.cuda.synchronize(self.device)
                prefill_times.append((time.perf_counter() - start) * 1000.0)

            with self._nvtx_range("flex_decode"):
                for pos in range(self.decode_tokens):
                    start = time.perf_counter()
                    _ = self.model.decode(self.decode_token, base_position + pos)
                    torch.cuda.synchronize(self.device)
                    decode_times.append((time.perf_counter() - start) * 1000.0)

        self._history["prefill_ms"].extend(prefill_times)
        self._history["decode_ms"].extend(decode_times)
        return {"prefill_ms": prefill_times, "decode_ms": decode_times}

    # ---------------------------------------------------------------- lifecycle
    def teardown(self) -> None:
        self.model = None
        self.prefill_tokens = None
        self.decode_token = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------- configs
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=4, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        if not self._history["prefill_ms"]:
            return None
        return {
            "flex_prefill.mean_ms": float(statistics.mean(self._history["prefill_ms"])),
            "flex_decode.mean_ms": float(statistics.mean(self._history["decode_ms"])),
            "flex_decode.tokens_per_iter": float(self.decode_tokens),
            "flexdecode.uses_flex_attention": float(self.use_flex_attention),
        }

    def validate_result(self) -> Optional[str]:
        if not self._history["prefill_ms"]:
            return "No prefill samples collected"
        if not self._history["decode_ms"]:
            return "No decode samples collected"
        return None


class BaselineFlexDecodingBenchmark(FlexDecodingHarness):
    def __init__(self):
        super().__init__(use_flex_attention=False, require_flex=False, decode_tokens=128)


def get_benchmark() -> BaseBenchmark:
    return BaselineFlexDecodingBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    print(result)
