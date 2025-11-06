"""baseline_precision_bf16.py - BF16 precision training baseline (baseline).

Standard BF16 training without FP8 quantization.
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


class SimpleTransformer(nn.Module):
    """Simple transformer for precision comparison."""
    
    def __init__(self, hidden_dim: int = 1024, num_layers: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'attn_qkv': nn.Linear(hidden_dim, hidden_dim * 3, dtype=torch.bfloat16),
                'attn_proj': nn.Linear(hidden_dim, hidden_dim, dtype=torch.bfloat16),
                'ffn_fc1': nn.Linear(hidden_dim, hidden_dim * 4, dtype=torch.bfloat16),
                'ffn_fc2': nn.Linear(hidden_dim * 4, hidden_dim, dtype=torch.bfloat16),
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


class BaselineBF16Benchmark(Benchmark):
    """BF16 precision baseline - standard training."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.inputs = None
        self.optimizer = None
        self.criterion = None
        self.batch_size = 4
        self.seq_len = 256
        self.hidden_dim = 1024
    
    def setup(self) -> None:
        """Setup: Initialize model and data."""
        torch.manual_seed(42)
        
        self.model = SimpleTransformer(hidden_dim=self.hidden_dim, num_layers=6).to(self.device).train()
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
        """Function to benchmark - BF16 training."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_precision_bf16", enable=enable_nvtx):
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
    return BaselineBF16Benchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline BF16 Training: {result.mean_ms:.3f} ms")

