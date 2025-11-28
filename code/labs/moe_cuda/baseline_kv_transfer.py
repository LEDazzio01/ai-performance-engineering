"""labs.moe_cuda/baseline_kv_transfer.py - Sequential KV cache transfers."""

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


class BaselineKVTransferBenchmark(BaseBenchmark):
    """Sequential KV transfers (no overlap between compute and NVLink copies)."""

    def __init__(self) -> None:
        super().__init__()
        self.batch_size = 8
        self.hidden_size = 1024
        self.seq_len = 512
        self.kv_cache: Optional[torch.Tensor] = None
        self.decode_inputs: Optional[List[torch.Tensor]] = None
        tokens = self.batch_size * (self.hidden_size + self.seq_len)
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self._history: Dict[str, List[float]] = {"latency_ms": []}

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("labs.moe_cuda KV transfer requires CUDA")

        torch.manual_seed(0)
        self.kv_cache = torch.randn(
            self.batch_size,
            self.seq_len,
            self.hidden_size,
            device=self.device,
            dtype=torch.float16,
        )
        self.decode_inputs = [
            torch.randn(
                self.batch_size,
                1,
                self.hidden_size,
                device=self.device,
                dtype=torch.float16,
            )
            for _ in range(self.seq_len)
        ]
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> Dict[str, List[float]]:
        if self.kv_cache is None or self.decode_inputs is None:
            raise RuntimeError("KV cache or decode inputs missing")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        with nvtx_range("moe_cuda_kv_baseline", enable=enable_nvtx):
            latencies: List[float] = []
            for decode_tensor in self.decode_inputs:
                start = self._record_start()
                # Simulate copy then compute
                copied = self.kv_cache[:, -1:, :].clone()
                _ = decode_tensor + copied
                torch.cuda.synchronize(self.device)
                latencies.append(self._record_stop(start))
            self._history["latency_ms"].extend(latencies)
            return {"kv_transfer_ms": latencies}

    def teardown(self) -> None:
        self.kv_cache = None
        self.decode_inputs = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)  # Min warmup for CUDA

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        if not self._history["latency_ms"]:
            return None
        return {"kv_transfer.mean_ms": float(sum(self._history["latency_ms"]) / len(self._history["latency_ms"]))}

    def validate_result(self) -> Optional[str]:
        if self.kv_cache is None or self.decode_inputs is None:
            return "KV cache or decode inputs missing"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineKVTransferBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    mean = result.timing.mean_ms if result.timing else 0.0
    print(f"Baseline KV transfer: {mean:.3f} ms")
