"""baseline_attention.py - Baseline scaled dot-product attention (unoptimized).

Demonstrates naive attention implementation without optimizations.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

try:
    import ch1.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass
import torch.nn as nn
import torch.nn.functional as F

from common.python.compile_utils import compile_model

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return a usable device, falling back to CPU if CUDA is unavailable or unsupported."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.zeros(1, device="cuda")
        return torch.device("cuda")
    except Exception as exc:
        print(f"WARNING: CUDA unavailable or unsupported ({exc}); falling back to CPU.")
        return torch.device("cpu")


class BaselineAttention(nn.Module):
    """Baseline multi-head attention implementation (unoptimized)."""
    
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Projections for Q, K, V
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: naive scaled dot-product attention."""
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq, hidden)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, seq, head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores (naive implementation)
        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch, heads, seq, seq)
        
        # Causal mask (lower triangular)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)  # (batch, heads, seq, seq)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (batch, heads, seq, head_dim)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(attn_output)
        
        return output


class BaselineAttentionBenchmark(Benchmark):
    """Benchmark implementation following Benchmark protocol."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.data = None
    
    def setup(self) -> None:
        """Setup: initialize attention model and data."""
        self.model = BaselineAttention(hidden_dim=256, num_heads=8)
        
        if self.device.type == "cuda":
            try:
                self.model = self.model.to(self.device)
            except Exception as exc:
                print(f"WARNING: GPU initialization failed: {exc}. Falling back to CPU.")
                self.device = torch.device("cpu")
                self.model = self.model.cpu()
        else:
            self.model = self.model.to(self.device)
        
        self.model.eval()
        # Match optimized sequence length for fair comparison
        seq_len = 256  # Match optimized sequence length
        self.data = torch.randn(4, seq_len, 256, device=self.device)  # (batch, seq_len, hidden_dim)
    
    def benchmark_fn(self) -> None:
        """Function to benchmark."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_attention", enable=enable_nvtx):
            with torch.no_grad():
                _ = self.model(self.data)

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.data is None:
            return "Data not initialized"
        # Check that model can produce valid output
        try:
            with torch.no_grad():
                test_output = self.model(self.data)
                if test_output.shape != self.data.shape:
                    return f"Output shape mismatch: expected {self.data.shape}, got {test_output.shape}"
        except Exception as e:
            return f"Model forward pass failed: {e}"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineAttentionBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=20, warmup=5)
    )
    benchmark = BaselineAttentionBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print("Baseline: Scaled Dot-Product Attention")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()

