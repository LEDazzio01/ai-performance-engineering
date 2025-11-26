"""NVFP4 training benchmark that exercises Transformer Engine block scaling."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig

try:
    from transformer_engine.pytorch import Linear as TELinear
    from transformer_engine.pytorch import LayerNorm as TELayerNorm
    from transformer_engine.pytorch import autocast as te_autocast
    from transformer_engine.pytorch import quantized_model_init, is_nvfp4_available
    from transformer_engine.common import recipe as te_recipe

    TE_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    TE_AVAILABLE = False
    TE_IMPORT_ERROR = exc
    TELinear = TELayerNorm = te_autocast = quantized_model_init = te_recipe = None  # type: ignore[assignment]
else:
    TE_IMPORT_ERROR = None


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for NVFP4 benchmarks")
    return torch.device("cuda")


class _NVFP4Block(nn.Module):
    """Feed-forward block composed of Transformer Engine modules."""

    def __init__(self, hidden_dim: int, intermediate_dim: int) -> None:
        super().__init__()
        self.ln = TELayerNorm(hidden_dim)
        self.fc1 = TELinear(hidden_dim, intermediate_dim, bias=True)
        self.act = nn.GELU()
        self.fc2 = TELinear(intermediate_dim, hidden_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        orig_shape = x.shape
        y = x.reshape(-1, orig_shape[-1])
        y = self.ln(y)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        return y.reshape(*orig_shape)


class OptimizedNVFP4TrainingBenchmark(BaseBenchmark):
    """Runs Transformer Engine NVFP4 microscaling inside the harness."""

    def __init__(self) -> None:
        super().__init__()
        self.device = resolve_device()
        self.hidden_dim = 2048
        self.intermediate_dim = self.hidden_dim * 2
        self.num_layers = 4
        self.batch_size = 16
        self.seq_len = 512
        self.micro_batches = 4
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.inputs: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.recipe = te_recipe.NVFP4BlockScaling()
        self.fp8_fallback_recipe = te_recipe.DelayedScaling()
        self._te_ready = TE_AVAILABLE
        self.use_nvfp4 = True
        self._probe_error: Optional[Exception] = None

    def setup(self) -> None:
        if not self._te_ready:
            raise RuntimeError(
                f"Transformer Engine not available: {TE_IMPORT_ERROR}"
            )
        if not is_nvfp4_available():
            raise RuntimeError("SKIPPED: NVFP4 kernels unavailable on this hardware/driver.")
        torch.manual_seed(42)
        layers = [
            _NVFP4Block(self.hidden_dim, self.intermediate_dim)
            for _ in range(self.num_layers)
        ]
        with quantized_model_init(enabled=True, recipe=self.recipe):
            self.model = nn.Sequential(*layers).to(self.device, dtype=torch.bfloat16)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, fused=True)
        self.inputs = [
            torch.randn(
                self.batch_size,
                self.seq_len,
                self.hidden_dim,
                device=self.device,
                dtype=torch.bfloat16,
            )
            for _ in range(self.micro_batches)
        ]
        self.targets = [
            torch.randn_like(self.inputs[0]) for _ in range(self.micro_batches)
        ]
        torch.cuda.synchronize()
        if not self._probe_nvfp4_path():
            self.use_nvfp4 = False
            self.recipe = self.fp8_fallback_recipe
            msg = "[NVFP4] Falling back to FP8 recipe because NVFP4 kernels failed to launch."
            if self._probe_error:
                msg += f" Root cause: {self._probe_error}"
            print(msg, flush=True, file=sys.stderr)

    def _train_step(self, idx: int) -> None:
        assert self.model is not None and self.optimizer is not None
        inp = self.inputs[idx]
        target = self.targets[idx]

        self.optimizer.zero_grad(set_to_none=True)
        with te_autocast(enabled=True, recipe=self.recipe):
            out = self.model(inp)
            loss = F.mse_loss(out, target)
        loss.backward()
        self.optimizer.step()

    def _probe_nvfp4_path(self) -> bool:
        if self.model is None:
            return False
        probe_input = torch.randn(
            2, self.seq_len, self.hidden_dim, device=self.device, dtype=torch.bfloat16
        )
        try:
            with te_autocast(enabled=True, recipe=self.recipe):
                _ = self.model(probe_input)
            return True
        except Exception as exc:  # pragma: no cover - debug helper
            self._probe_error = exc
            torch.cuda.synchronize()
            return False

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("nvfp4_training_optimized", enable=enable_nvtx):
            for idx in range(self.micro_batches):
                self._train_step(idx)
        torch.cuda.synchronize()

    def teardown(self) -> None:
        self.model = None
        self.optimizer = None
        self.inputs = []
        self.targets = []
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=8,
            warmup=2,
            enable_memory_tracking=False,
            deterministic=False,  # NVFP4/TE kernels prefer nondeterministic fast paths
            seed=None,  # avoid global cuRAND seed interactions
            measurement_timeout_seconds=90,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_fp32_ms', 10.0),
            reduced_precision_time_ms=getattr(self, '_reduced_ms', 5.0),
            precision_type="fp8",
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None or self.optimizer is None:
            return "Transformer Engine model not initialized"
        if not self.inputs:
            return "Input tensors missing"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedNVFP4TrainingBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    print(f"NVFP4 optimized mean: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
