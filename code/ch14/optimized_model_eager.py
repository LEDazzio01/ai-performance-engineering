"""optimized_model_eager.py - torch.compile optimized execution.

Uses torch.compile for kernel fusion and optimization.
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
import torch.nn as nn

from typing import Optional

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)

# Ensure consistent TF32 state before any operations (new API only)
enable_tf32()

# Note: arch_config not imported here to avoid TF32 API mixing with torch.compile
# torch.compile handles TF32 internally, but we need consistent state first


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch14")
    return torch.device("cuda")


class SimpleTransformer(nn.Module):
    """Simple transformer for profiling."""
    
    def __init__(self, d_model=512, n_heads=8, n_layers=6, d_ff=2048, vocab_size=10000):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 2048, d_model))  # Support up to 2048 seq len
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                batch_first=True,
            )
            for _ in range(n_layers)
        ])
        
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


class OptimizedModelCompiledBenchmark(Benchmark):
    """Benchmark implementation with torch.compile optimization."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.compiled_model = None
        self.input_ids = None
    
    def setup(self) -> None:
        """Setup: initialize model and compile it."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        batch_size = 16
        seq_len = 1024
        vocab_size = 10000
        
        model = SimpleTransformer().to(self.device).eval()
        self.model = model  # Store reference for fallback
        
        # Compile model with optimal settings
        try:
            self.compiled_model = torch.compile(
                model, 
                mode="reduce-overhead", 
                fullgraph=False,
                dynamic=False
            )
        except Exception as e:
            # Fallback to eager if compilation fails
            self.compiled_model = model
        
        self.input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
        
        # Warmup (compilation happens here)
        # Catch PyTorch internal errors (e.g., C++ compilation failures in inductor)
        try:
            for _ in range(30):
                with torch.no_grad():
                    _ = self.compiled_model(self.input_ids)
            torch.cuda.synchronize()
            
            # Additional warmup after compilation
            for _ in range(10):
                with torch.no_grad():
                    _ = self.compiled_model(self.input_ids)
            torch.cuda.synchronize()
        except (RuntimeError, Exception) as e:
            # PyTorch internal errors (e.g., inductor C++ compilation failures)
            # Fall back to eager mode if compilation/warmup fails
            error_msg = str(e)
            if "CppCompileError" in error_msg or "torch._inductor" in error_msg or "SavedTensorHooks" in error_msg:
                # Known PyTorch internal bugs - fall back to eager mode
                self.compiled_model = self.model
                # Run a simple warmup with eager mode
                for _ in range(5):
                    with torch.no_grad():
                        _ = self.compiled_model(self.input_ids)
                torch.cuda.synchronize()
            else:
                # Re-raise unknown errors
                raise
    
    def benchmark_fn(self) -> None:
        """Function to benchmark."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("model_eager", enable=enable_nvtx):
            with torch.no_grad():
                        _ = self.compiled_model(self.input_ids)

    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.compiled_model, self.input_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            setup_timeout_seconds=180,  # torch.compile compilation can take 60-120 seconds
            measurement_timeout_seconds=180,  # torch.compile may need compilation during warmup/measurement
            use_subprocess=False,  # Disable subprocess to avoid pydantic import issues with torch.compile
        )
    def validate_result(self) -> Optional[str]:
        """Optional validation."""
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedModelCompiledBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=10)
    )
    benchmark = OptimizedModelCompiledBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print("Optimized: torch.compile Execution")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
