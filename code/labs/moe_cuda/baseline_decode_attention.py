"""labs.moe_cuda/baseline_decode_attention.py - Naive decode attention."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range


class BaselineDecodeAttentionBenchmark(BaseBenchmark):
    """Naive decode attention with many small matmuls."""

    def __init__(self) -> None:
        super().__init__()
        # Realistic decode workload where BF16 optimization shows benefit
        self.batch = 32
        self.num_heads = 12
        self.kv_seq = 512
        self.head_dim = 64
        self.module: Optional[torch.nn.Module] = None
        self.q: Optional[torch.Tensor] = None
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None
        tokens = self.batch * self.kv_seq
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )
        self._history: Dict[str, List[float]] = {"latency_ms": []}

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("labs.moe_cuda decode attention requires CUDA")

        torch.manual_seed(0)
        self.module = torch.nn.MultiheadAttention(
            embed_dim=self.num_heads * self.head_dim,
            num_heads=self.num_heads,
            batch_first=True,
            device=self.device,
            dtype=torch.float32,
        )
        self.q = torch.randn(
            self.batch,
            1,
            self.num_heads * self.head_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self.k = torch.randn(
            self.batch,
            self.kv_seq,
            self.num_heads * self.head_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self.v = torch.randn_like(self.k)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> Dict[str, List[float]]:
        if any(t is None for t in (self.module, self.q, self.k, self.v)):
            raise RuntimeError("Decode tensors missing")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        with nvtx_range("moe_cuda_decode_naive", enable=enable_nvtx):
            with torch.inference_mode():
                start = self._record_start()
                _ = self.module(self.q, self.k, self.v)
                torch.cuda.synchronize(self.device)
                self._history["latency_ms"].append(self._record_stop(start))
        return {"decode_ms": self._history["latency_ms"]}

    def teardown(self) -> None:
        torch.cuda.empty_cache()
        self.module = None
        self.q = None
        self.k = None
        self.v = None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=8, warmup=3)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        if not self._history["latency_ms"]:
            return None
        return {
            "decode.mean_ms": float(sum(self._history["latency_ms"]) / len(self._history["latency_ms"]))
        }

    def validate_result(self) -> Optional[str]:
        if any(t is None for t in (self.module, self.q, self.k, self.v)):
            return "Decode tensors missing"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineDecodeAttentionBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    mean = result.timing.mean_ms if result.timing else 0.0
    print(f"Baseline decode attention: {mean:.3f} ms")
