"""NVFP4 training benchmark that exercises Transformer Engine block scaling.

Chapter 19 demonstrates NVFP4 (4-bit floating point) quantization for training,
which provides memory savings and potential speedups through reduced memory bandwidth.
"""

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
    is_nvfp4_available = lambda: False  # type: ignore[assignment]
else:
    TE_IMPORT_ERROR = None


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for NVFP4 benchmarks")
    return torch.device("cuda")


class _NVFP4Block(nn.Module):
    """Feed-forward block composed of Transformer Engine modules for NVFP4 quantization."""

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
    """NVFP4 quantized training using Transformer Engine.
    
    This demonstrates the memory and compute benefits of NVFP4 (4-bit) quantization
    compared to the BF16 baseline. NVFP4 provides:
    - 4x memory compression for activations
    - Reduced memory bandwidth requirements
    - Potential speedup from smaller data transfers
    """

    def __init__(self) -> None:
        super().__init__()
        self.device = resolve_device()
        # Larger workload to amortize TE overhead and show NVFP4 benefits
        self.hidden_dim = 4096
        self.intermediate_dim = self.hidden_dim * 4
        self.num_layers = 8
        self.batch_size = 32
        self.seq_len = 1024
        self.micro_batches = 4
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.inputs: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        
        # NVFP4 recipe with calibration
        self.nvfp4_recipe = (
            te_recipe.NVFP4BlockScaling(calibration_steps=20, amax_history_len=16, fp4_tensor_block=16)
            if TE_AVAILABLE
            else None
        )
        # Fallback to FP8 if NVFP4 unavailable
        self.fp8_recipe = (
            te_recipe.DelayedScaling(amax_history_len=16, amax_compute_algo="max")
            if TE_AVAILABLE
            else None
        )
        self.active_recipe = None
        self.use_nvfp4 = False
        self._probe_error: Optional[Exception] = None

    def setup(self) -> None:
        if not TE_AVAILABLE:
            raise RuntimeError(f"Transformer Engine not available: {TE_IMPORT_ERROR}")
        
        torch.manual_seed(42)
        
        # Determine which recipe to use
        if is_nvfp4_available() and self.nvfp4_recipe is not None:
            self.active_recipe = self.nvfp4_recipe
            self.use_nvfp4 = True
        else:
            self.active_recipe = self.fp8_recipe
            self.use_nvfp4 = False
            print("[NVFP4] Falling back to FP8 recipe (NVFP4 not available)", file=sys.stderr, flush=True)
        
        # Build model with TE modules
        layers = [
            _NVFP4Block(self.hidden_dim, self.intermediate_dim)
            for _ in range(self.num_layers)
        ]
        
        with quantized_model_init(enabled=True, recipe=self.active_recipe):
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
        
        # Calibration warmup (important for quantization)
        self._calibration_warmup()
        torch.cuda.synchronize()

    def _calibration_warmup(self) -> None:
        """Run calibration steps to collect scaling factors."""
        if self.model is None or self.active_recipe is None:
            return
        
        # Run several forward passes to calibrate quantization scales
        for _ in range(5):
            for idx in range(self.micro_batches):
                self._train_step(idx)
        torch.cuda.synchronize()

    def _train_step(self, idx: int) -> None:
        assert self.model is not None and self.optimizer is not None
        inp = self.inputs[idx]
        target = self.targets[idx]

        self.optimizer.zero_grad(set_to_none=True)
        with te_autocast(enabled=True, recipe=self.active_recipe):
            out = self.model(inp)
            loss = F.mse_loss(out, target)
        loss.backward()
        self.optimizer.step()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        label = "nvfp4_training" if self.use_nvfp4 else "fp8_training"
        with nvtx_range(label, enable=enable_nvtx):
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
            warmup=10,  # Extra warmup for quantization stability
            enable_memory_tracking=False,
            deterministic=False,
            seed=None,
            measurement_timeout_seconds=120,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return NVFP4-specific metrics."""
        return {
            "nvfp4.active": 1.0 if self.use_nvfp4 else 0.0,
            "nvfp4.compression_ratio": 4.0 if self.use_nvfp4 else 2.0,
            "nvfp4.micro_batches": float(self.micro_batches),
        }

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
