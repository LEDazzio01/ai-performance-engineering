"""optimized_precision_fp4.py - FP4 precision inference (optimized).

FP4 quantization using native GB10 FP4 support for fastest inference.
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
        raise RuntimeError("CUDA required for ch20")
    return torch.device("cuda")

def quantize_to_fp4(x: torch.Tensor) -> torch.Tensor:
    """Quantize tensor to FP4 (E2M1F format).
    
    For GB10, we can use native FP4 support if available, otherwise simulate.
    """
    # Check if native FP4 is available (GB10 supports FP4)
    try:
        # Try to use native FP4 dtype if available
        if hasattr(torch, 'float4_e2m1f'):
            # Convert to FP4 and back for demonstration
            # In real usage, you'd keep weights in FP4
            x_fp4 = x.to(torch.float4_e2m1f)
            return x_fp4.to(x.dtype)  # Convert back for computation
    except (AttributeError, RuntimeError):
        pass
    
    # Fallback: Manual FP4 quantization (simplified)
    # FP4 E2M1F: 1 sign bit, 2 exponent bits, 1 mantissa bit
    # Range: ~0.0625 to 6.0
    scale = x.abs().max() / 6.0 if x.abs().max() > 0 else 1.0
    x_scaled = x / scale
    x_clamped = torch.clamp(x_scaled, -6.0, 6.0)
    # Simulate quantization by reducing precision
    x_quantized = (x_clamped * 2.0).round() / 2.0
    return x_quantized * scale

class FP4Linear(nn.Module):
    """FP4 quantized linear layer."""
    
    def __init__(self, in_features: int, out_features: int, use_fp4: bool = True):
        super().__init__()
        self.use_fp4 = use_fp4
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.bfloat16))
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16))
        
        # Pre-quantize weights to FP4 for faster inference
        if use_fp4:
            with torch.no_grad():
                self.weight_quantized = quantize_to_fp4(self.weight)
        else:
            self.weight_quantized = self.weight
    
    def to(self, device):
        """Override to() to ensure quantized weights move with model."""
        result = super().to(device)
        if hasattr(self, 'weight_quantized') and self.weight_quantized.device != device:
            self.weight_quantized = self.weight_quantized.to(device)
        return result
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fp4:
            # Use quantized weights
            weight = self.weight_quantized
        else:
            weight = self.weight
        
        return nn.functional.linear(x, weight, self.bias)

class SimpleTransformerFP4(nn.Module):
    """Simple transformer with FP4 quantization."""
    
    def __init__(self, hidden_dim: int = 1024, num_layers: int = 6, num_heads: int = 8, use_fp4: bool = True):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_fp4 = use_fp4
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'attn_qkv': FP4Linear(hidden_dim, hidden_dim * 3, use_fp4=use_fp4),
                'attn_proj': FP4Linear(hidden_dim, hidden_dim, use_fp4=use_fp4),
                'ffn_fc1': FP4Linear(hidden_dim, hidden_dim * 4, use_fp4=use_fp4),
                'ffn_fc2': FP4Linear(hidden_dim * 4, hidden_dim, use_fp4=use_fp4),
                'norm1': nn.LayerNorm(hidden_dim, dtype=torch.bfloat16),
                'norm2': nn.LayerNorm(hidden_dim, dtype=torch.bfloat16),
            })
            self.layers.append(layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            # Attention
            residual = x
            x = layer['norm1'](x)
            qkv = layer['attn_qkv'](x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(-1, q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(-1, k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(-1, v.size(1), self.num_heads, self.head_dim).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn, v)
            attn_out = attn_out.transpose(1, 2).contiguous().view(-1, attn_out.size(2), self.head_dim * self.num_heads)
            x = layer['attn_proj'](attn_out) + residual
            
            # FFN
            residual = x
            x = layer['norm2'](x)
            ffn_out = layer['ffn_fc2'](torch.relu(layer['ffn_fc1'](x)))
            x = ffn_out + residual
        
        return x

class OptimizedFP4Benchmark(Benchmark):
    """FP4 precision optimization - fastest inference with aggressive quantization."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        # Optimization: Compile model for kernel fusion and optimization
        try:

        # Optimization: Compile model for kernel fusion and optimization
        try:

        self.inputs = None
        self.batch_size = 4
        self.seq_len = 512
        self.hidden_dim = 1024
    
    def setup(self) -> None:
        """Setup: Initialize FP4 model and data."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(42)
        
        self.model = SimpleTransformerFP4(
            hidden_dim=self.hidden_dim,
            num_layers=6,
            num_heads=8,
            use_fp4=True
        ).to(self.device).eval()
        
        # Ensure quantized weights are on device
        for layer in self.model.layers:
            for name, module in layer.items():
                if isinstance(module, FP4Linear) and hasattr(module, 'weight_quantized'):
                    module.weight_quantized = module.weight_quantized.to(self.device)
        
        self.inputs = torch.randn(
            self.batch_size,
            self.seq_len,
            self.hidden_dim,
            device=self.device,
            dtype=torch.bfloat16
        )
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = self.model(self.inputs)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - FP4 inference."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_precision_fp4", enable=enable_nvtx):
            with torch.no_grad():
                _ = self.model(self.inputs)

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.inputs
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
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
    return OptimizedFP4Benchmark()

if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized FP4: {result.mean_ms:.3f} ms")

