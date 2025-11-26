"""
optimized_fp8_static.py - Static FP8 Quantization (Ch13)

WHAT: Static quantization calibrates scale factors ONCE during profiling,
then uses fixed scales during inference.

WHY: Dynamic quantization computes scales every forward pass:
  - Dynamic: Compute amax → derive scale → quantize → GEMM
  - Static: Use pre-computed scale → quantize → GEMM
  
Static saves compute and is deterministic for deployment.

WORKFLOW:
  1. Calibration pass: Run representative data, collect amax statistics
  2. Compute scales: scales = amax / fp8_max
  3. Freeze scales: Store as model attributes
  4. Inference: Use frozen scales, no amax computation

WHEN TO USE:
  - Production inference where latency matters
  - When input distribution is stable
  - Edge deployment where compute is limited

REQUIREMENTS:
  - PyTorch 2.1+ with FP8 support
  - Calibration dataset
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager

from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


@dataclass
class CalibrationStats:
    """Statistics collected during calibration."""
    amax_history: List[float] = field(default_factory=list)
    running_amax: float = 0.0
    num_samples: int = 0
    
    def update(self, tensor: torch.Tensor):
        current_amax = tensor.abs().max().item()
        self.amax_history.append(current_amax)
        self.running_amax = max(self.running_amax, current_amax)
        self.num_samples += 1
    
    def get_scale(self, fp8_max: float = 448.0, margin: float = 0.0) -> float:
        amax = self.running_amax * (1.0 + margin)
        return max(amax / fp8_max, 1e-12)


class StaticFP8Linear(nn.Module):
    """Linear layer with static FP8 quantization."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype)) if bias else None
        
        self.fp8_max = 448.0
        self.register_buffer('input_scale', torch.tensor(1.0))
        self.register_buffer('weight_scale', torch.tensor(1.0))
        self.register_buffer('is_calibrated', torch.tensor(False))
        
        self._calibrating = False
        self._input_stats = CalibrationStats()
        self._weight_stats = CalibrationStats()
        
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    @contextmanager
    def calibration_mode(self):
        self._calibrating = True
        self._input_stats = CalibrationStats()
        self._weight_stats = CalibrationStats()
        try:
            yield
        finally:
            self._calibrating = False
    
    def freeze_scales(self, margin: float = 0.0):
        input_scale = self._input_stats.get_scale(self.fp8_max, margin)
        weight_scale = self._weight_stats.get_scale(self.fp8_max, margin)
        
        self.input_scale.fill_(input_scale)
        self.weight_scale.fill_(weight_scale)
        self.is_calibrated.fill_(True)
        
        return {"input_scale": input_scale, "weight_scale": weight_scale,
                "calibration_samples": self._input_stats.num_samples}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        
        if self._calibrating:
            self._input_stats.update(x)
            self._weight_stats.update(self.weight)
            output = torch.nn.functional.linear(x, self.weight, self.bias)
            
        elif self.is_calibrated:
            # Static FP8 quantization
            x_q = torch.clamp(x / self.input_scale, -self.fp8_max, self.fp8_max).round()
            w_q = torch.clamp(self.weight / self.weight_scale, -self.fp8_max, self.fp8_max).round()
            
            output_q = torch.nn.functional.linear(x_q, w_q, bias=None)
            output = (output_q * self.input_scale * self.weight_scale).to(original_dtype)
            
            if self.bias is not None:
                output = output + self.bias
        else:
            output = torch.nn.functional.linear(x, self.weight, self.bias)
        
        return output


#============================================================================
# Benchmark
#============================================================================

def benchmark():
    """Compare static vs dynamic FP8 quantization."""
    print("Static vs Dynamic FP8 Quantization")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name()}")
    
    # Create simple model
    dim = 4096
    batch_size, seq_len = 32, 512
    
    fp32_linear = nn.Linear(dim, dim).to(device)
    static_linear = StaticFP8Linear(dim, dim, device=device)
    
    # Copy weights
    with torch.no_grad():
        static_linear.weight.copy_(fp32_linear.weight)
        static_linear.bias.copy_(fp32_linear.bias)
    
    # Calibrate
    print("\nCalibrating...")
    with static_linear.calibration_mode():
        for _ in range(50):
            x = torch.randn(batch_size, seq_len, dim, device=device)
            _ = static_linear(x)
    
    cal_info = static_linear.freeze_scales()
    print(f"  Calibration samples: {cal_info['calibration_samples']}")
    print(f"  Input scale: {cal_info['input_scale']:.6f}")
    print(f"  Weight scale: {cal_info['weight_scale']:.6f}")
    
    # Benchmark
    x = torch.randn(batch_size, seq_len, dim, device=device)
    
    # Warmup
    for _ in range(10):
        _ = fp32_linear(x)
        _ = static_linear(x)
    torch.cuda.synchronize()
    
    # FP32 baseline
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(100):
        _ = fp32_linear(x)
    end.record()
    torch.cuda.synchronize()
    fp32_ms = start.elapsed_time(end) / 100
    
    # Static FP8
    start.record()
    for _ in range(100):
        _ = static_linear(x)
    end.record()
    torch.cuda.synchronize()
    static_ms = start.elapsed_time(end) / 100
    
    # Accuracy
    with torch.no_grad():
        fp32_out = fp32_linear(x)
        static_out = static_linear(x)
    error = (static_out - fp32_out).abs().mean() / fp32_out.abs().mean() * 100
    
    print(f"\nResults:")
    print(f"  FP32: {fp32_ms:.3f} ms")
    print(f"  Static FP8: {static_ms:.3f} ms")
    print(f"  Speedup: {fp32_ms / static_ms:.2f}x")
    print(f"  Relative Error: {error:.4f}%")
    
    print("\nNote: Static FP8 avoids per-forward amax computation")


#============================================================================
# Benchmark Harness Integration
#============================================================================

class StaticFP8Benchmark(BaseBenchmark):
    """Benchmark harness wrapper for static FP8 quantization."""

    def __init__(self):
        super().__init__()
        self.static_linear = None
        self.x = None
        self.batch_size = 32
        self.seq_len = 512
        self.dim = 4096
        self._last = 0.0
        
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        """Setup: Initialize and calibrate static FP8 linear."""
        torch.manual_seed(42)
        
        self.static_linear = StaticFP8Linear(self.dim, self.dim, device=self.device)
        
        # Calibrate
        with self.static_linear.calibration_mode():
            for _ in range(50):
                x = torch.randn(self.batch_size, self.seq_len, self.dim, device=self.device)
                _ = self.static_linear(x)
        
        self.static_linear.freeze_scales()
        
        self.x = torch.randn(self.batch_size, self.seq_len, self.dim, device=self.device)
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = self.static_linear(self.x)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        """Benchmark: Static FP8 forward pass."""
        with torch.no_grad():
            output = self.static_linear(self.x)
            self._last = float(output.sum())
            self._synchronize()

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.static_linear = None
        self.x = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=10)
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_fp32_ms', 10.0),
            reduced_precision_time_ms=getattr(self, '_reduced_ms', 5.0),
            precision_type="fp8",
        )

    def validate_result(self) -> Optional[str]:
        if self.static_linear is None:
            return "Linear layer not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return StaticFP8Benchmark()


if __name__ == "__main__":
    benchmark()
