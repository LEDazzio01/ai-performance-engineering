"""optimized_model_eager.py - torch.compile optimized execution.

Uses torch.compile for kernel fusion and optimization.
Implements BaseBenchmark for harness integration.
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

from core.utils.compile_utils import enable_tf32, compile_model
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
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


class OptimizedModelCompiledBenchmark(BaseBenchmark):
    """Benchmark implementation with torch.compile optimization.
    
    Chapter 14 Optimization: torch.compile with max-autotune mode for optimal
    kernel selection and fusion. Demonstrates:
    1. Kernel fusion (multiple ops -> single kernel)
    2. Memory layout optimization
    3. Inductor-based code generation
    """
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.compiled_model = None
        self.input_ids = None
        # Increase work so torch.compile overhead is amortized and speedups are clearer.
        self.batch_size = 24
        self.seq_len = 1536
        self.vocab_size = 10000
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: initialize model and compile it."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()  # Enable TF32 for speedup
        
        # Use BF16 for tensor core acceleration (key optimization over baseline FP32)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = SimpleTransformer().to(self.device, dtype=dtype).eval()
        self.model = model
        
        # Use max-autotune for best performance (searches through kernel configs)
        self.compiled_model = compile_model(
            model,
            mode="max-autotune",  # Better than reduce-overhead for sustained workloads
            fullgraph=False,
            dynamic=False,
        )
        
        self.input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len), device=self.device)
        
        # Extensive warmup for compilation and autotuning
        for _ in range(50):
            with torch.no_grad():
                _ = self.compiled_model(self.input_ids)
        torch.cuda.synchronize(self.device)
        
        # Additional warmup after compilation
        for _ in range(20):
            with torch.no_grad():
                _ = self.compiled_model(self.input_ids)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark."""
        # Use conditional NVTX ranges - only enabled when profiling

        from core.profiling.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("model_eager", enable=enable_nvtx):
            with torch.no_grad():
                        _ = self.compiled_model(self.input_ids)
        self._synchronize()

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
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_triton_metrics
        return compute_triton_metrics(
            num_elements=getattr(self, 'N', getattr(self, 'num_elements', 1024)),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            block_size=getattr(self, 'BLOCK_SIZE', 1024),
            num_warps=getattr(self, 'num_warps', 4),
        )

    def validate_result(self) -> Optional[str]:
        """Optional validation."""
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return OptimizedModelCompiledBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
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
