"""baseline_optimizer_replicated.py

Baseline training step that replicates optimizer state on every GPU.
Each device keeps its own momentum buffers and updates locally (no sharing),
which inflates memory and increases communication later.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselineOptimizerReplicatedBenchmark(BaseBenchmark):
    """Replicated optimizer state per GPU."""

    def __init__(self):
        super().__init__()
        self.models: List[nn.Linear] = []
        self.momentum: List[torch.Tensor] = []
        self.batch_size = 8
        self.hidden = 512
        tokens = self.batch_size * self.hidden
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(123)
        num_gpus = max(1, torch.cuda.device_count())
        for rank in range(num_gpus):
            device = f"cuda:{rank}"
            model = nn.Linear(self.hidden, self.hidden).to(device)
            buf = torch.zeros_like(model.weight, dtype=torch.float32, device=device)
            self.models.append(model)
            self.momentum.append(buf)
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert len(self.models) == len(self.momentum)
        with self._nvtx_range("baseline_optimizer_replicated"):
            for model, mom in zip(self.models, self.momentum):
                x = torch.randn(self.batch_size, self.hidden, device=model.weight.device)
                y = model(x)
                loss = y.pow(2).mean()
                loss.backward()
                # Local momentum update (per-GPU state)
                with torch.no_grad():
                    mom.mul_(0.9).add_(model.weight.grad)
                    model.weight.add_(-1e-3, mom)
                    model.weight.grad.zero_()
                    model.bias.grad.zero_()
            self._synchronize()

    def teardown(self) -> None:
        self.models.clear()
        self.momentum.clear()
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)

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
        if not self.models or not self.momentum:
            return "Models or optimizer state not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineOptimizerReplicatedBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=4, warmup=5),
    )
    bench = BaselineOptimizerReplicatedBenchmark()
    result = harness.benchmark(bench)
    print(result)
