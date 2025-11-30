"""baseline_ddp_nvlink_naive.py

Simplified DDP-style training loop that blocks on gradient exchange and does
not bucket or overlap communication. Uses two microbatches to show the cost
of sequential reduce + compute. Falls back to single-GPU execution when
additional GPUs are unavailable.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from core.benchmark.gpu_requirements import skip_if_insufficient_gpus
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselineDdpNvlinkNaiveBenchmark(BaseBenchmark):
    """No overlap, naive gradient sync."""

    def __init__(self):
        super().__init__()
        self.models: List[nn.Linear] = []
        self.microbatches = 2
        self.batch_size = 8
        self.hidden = 512
        tokens = self.batch_size * self.hidden * self.microbatches
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size * self.microbatches),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(0)
        num = torch.cuda.device_count()
        skip_if_insufficient_gpus(2)
        for rank in range(num):
            device = f"cuda:{rank}"
            self.models.append(nn.Linear(self.hidden, self.hidden).to(device))
        self._synchronize()

    def _simulate_allreduce(self, grads: List[torch.Tensor]) -> None:
        """Simple blocking allreduce (sum + scatter) across model gradients."""
        if len(grads) == 1:
            return
        root = grads[0].device
        buf = torch.zeros_like(grads[0], device=root)
        for g in grads:
            buf.add_(g.to(root))
        buf.mul_(1.0 / len(grads))
        for g in grads:
            g.copy_(buf.to(g.device))

    def benchmark_fn(self) -> None:
        assert self.models
        with self._nvtx_range("baseline_ddp_nvlink_naive"):
            for _ in range(self.microbatches):
                grads = []
                for model in self.models:
                    x = torch.randn(self.batch_size, self.hidden, device=model.weight.device)
                    y = model(x)
                    loss = y.pow(2).mean()
                    loss.backward()
                    grads.append(model.weight.grad)
                # Blocking gradient sync
                self._simulate_allreduce(grads)
                for model in self.models:
                    with torch.no_grad():
                        model.weight.add_(-1e-3, model.weight.grad)
                        model.weight.grad.zero_()
                        model.bias.grad.zero_()
            self._synchronize()

    def teardown(self) -> None:
        self.models.clear()
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=5, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            transfer_type="hbm",
        )

    def validate_result(self) -> Optional[str]:
        if not self.models:
            return "Models not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineDdpNvlinkNaiveBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=2, warmup=5),
    )
    bench = BaselineDdpNvlinkNaiveBenchmark()
    result = harness.benchmark(bench)
    print(result)
