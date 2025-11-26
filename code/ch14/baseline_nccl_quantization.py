"""Baseline NCCL quantization – quantize on CPU and serialize transfers."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
from typing import Optional

from common.python.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


class BaselineNCCLQuantizationBenchmark(BaseBenchmark):
    """Baseline: Simulate per-rank CPU-side quantization with serialized copies."""

    def __init__(self):
        super().__init__()
        self.tensor = None
        self.num_chunks = 16
        self.chunk_len = 1 << 14
        self._last = 0.0
        tokens = self.num_chunks * self.chunk_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.num_chunks),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        """Setup: initialize synthetic gradients."""
        torch.manual_seed(42)
        self.tensor = torch.randn(self.num_chunks, self.chunk_len, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        """Benchmark: CPU quantization + host/device transfers."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_nccl_quantization", enable=enable_nvtx):
            if self.tensor is None:
                raise RuntimeError("Tensor not initialized")
            total = 0.0
            for idx in range(self.num_chunks):
                chunk = self.tensor[idx].detach().cpu()
                max_abs = chunk.abs().max().clamp(min=1e-6)
                scale = 127.0 / max_abs
                q = torch.round(chunk * scale).to(torch.int8)
                dq = q.float() / scale
                total += float(dq.sum())
                self.tensor[idx].copy_(dq.to(self.device))
            self._last = total
        self._synchronize()


    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.tensor = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_triton_metrics
        return compute_triton_metrics(
            num_elements=getattr(self, 'N', getattr(self, 'num_elements', 1024)),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            block_size=getattr(self, 'BLOCK_SIZE', 1024),
            num_warps=getattr(self, 'num_warps', 4),
        )
}

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.tensor is None:
            return "Tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineNCCLQuantizationBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline NCCL Quantization (Single GPU): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(" Note: Single-GPU operation, no distributed computing")
