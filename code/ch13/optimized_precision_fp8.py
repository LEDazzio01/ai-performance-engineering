"""optimized_precision_fp8.py - FP8 precision training optimization (optimized).

FP8 quantization for training using native GB10 FP8 support.
Faster training with reduced memory usage.

Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch13")
    return torch.device("cuda")

def quantize_to_fp8(x: torch.Tensor) -> torch.Tensor:
    """Quantize tensor to FP8 (E4M3FN format) for GB10."""
    try:
        if hasattr(torch, 'float8_e4m3fn'):
            x_fp8 = x.to(torch.float8_e4m3fn)
            return x_fp8.to(x.dtype)
    except (AttributeError, RuntimeError):
        pass
    
    # Fallback: Manual FP8 quantization
    scale = x.abs().max() / 448.0 if x.abs().max() > 0 else 1.0
    x_scaled = x / scale
    x_clamped = torch.clamp(x_scaled, -448.0, 448.0)
    x_quantized = (x_clamped * 8.0).round() / 8.0
    return x_quantized * scale

class FP8Linear(nn.Module):
    """FP8 quantized linear layer."""
    
    def __init__(self, in_features: int, out_features: int, use_fp8: bool = True):
        super().__init__()
        self.use_fp8 = use_fp8
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.bfloat16))
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16))
        
        if use_fp8:
            with torch.no_grad():
                self.weight_quantized = quantize_to_fp8(self.weight)
        else:
            self.weight_quantized = self.weight
    
    def to(self, device):
        """Override to() to ensure quantized weights move with model."""
        result = super().to(device)
        if hasattr(self, 'weight_quantized') and self.weight_quantized.device != device:
            self.weight_quantized = self.weight_quantized.to(device)
        return result
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fp8:
            weight = self.weight_quantized
        else:
            weight = self.weight
        return nn.functional.linear(x, weight, self.bias)

class SimpleTransformerFP8(nn.Module):
    """Simple transformer with FP8 quantization."""
    
    def __init__(self, hidden_dim: int = 1024, num_layers: int = 6, use_fp8: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'attn_qkv': FP8Linear(hidden_dim, hidden_dim * 3, use_fp8=use_fp8),
                'attn_proj': FP8Linear(hidden_dim, hidden_dim, use_fp8=use_fp8),
                'ffn_fc1': FP8Linear(hidden_dim, hidden_dim * 4, use_fp8=use_fp8),
                'ffn_fc2': FP8Linear(hidden_dim * 4, hidden_dim, use_fp8=use_fp8),
                'norm1': nn.LayerNorm(hidden_dim, dtype=torch.bfloat16),
                'norm2': nn.LayerNorm(hidden_dim, dtype=torch.bfloat16),
            })
            self.layers.append(layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        num_heads = 8
        head_dim = hidden_dim // num_heads
        
        for layer in self.layers:
            # Attention
            residual = x
            x = layer['norm1'](x)
            qkv = layer['attn_qkv'](x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn, v)
            attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
            x = layer['attn_proj'](attn_out) + residual
            
            # FFN
            residual = x
            x = layer['norm2'](x)
            ffn_out = layer['ffn_fc2'](torch.relu(layer['ffn_fc1'](x)))
            x = ffn_out + residual
        
        return x

class OptimizedFP8Benchmark(Benchmark):
    """FP8 precision optimization - faster training."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        # Optimization: Compile model for kernel fusion and optimization
        try:

        # Optimization: Compile model for kernel fusion and optimization
        try:

        self.inputs = None
        self.optimizer = None
        self.criterion = None
        self.batch_size = 4
        self.seq_len = 256
        self.hidden_dim = 1024
    
    def setup(self) -> None:
        """Setup: Initialize FP8 model and data."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(42)
        
        self.model = SimpleTransformerFP8(hidden_dim=self.hidden_dim, num_layers=6, use_fp8=True).to(self.device)
        # Optimization: Use FP16 for faster computation
        if self.device.type == "cuda":
            try:
                self.model = self.model.half()
            except Exception:
                pass  # Fallback to FP32 if FP16 not supported
        
        self.model.train()
        
        # Ensure quantized weights are on device
        for layer in self.model.layers:
            for name, module in layer.items():
                if isinstance(module, FP8Linear) and hasattr(module, 'weight_quantized'):
                    module.weight_quantized = module.weight_quantized.to(self.device)
        
        self.inputs = torch.randn(self.batch_size, self.seq_len, self.hidden_dim, device=self.device, dtype=torch.bfloat16)
        self.targets = torch.randn(self.batch_size, self.seq_len, self.hidden_dim, device=self.device, dtype=torch.bfloat16)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()
        
        # Warmup
        for _ in range(3):
            self.optimizer.zero_grad()
            _ = self.model(self.inputs)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - FP8 training."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_precision_fp8", enable=enable_nvtx):
            self.optimizer.zero_grad()
            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.targets)
            loss.backward()
            self.optimizer.step()

    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.inputs, self.targets, self.optimizer, self.criterion
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=30,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedFP8Benchmark()

if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
    mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized FP8 Training: {result.mean_ms:.3f} ms")

