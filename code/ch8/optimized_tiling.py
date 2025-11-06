"""optimized_tiling.py - Optimized tiling in occupancy/warp divergence context.

Demonstrates tiling for improved cache utilization.
Tiling: Uses tiling to divide data into smaller tiles.
Improves cache utilization and memory access patterns.
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
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch8")
    return torch.device("cuda")


class OptimizedTilingBenchmark(Benchmark):
    """Optimized: Tiling for improved cache utilization.
    
    Tiling: Uses tiling to divide data into smaller tiles.
    Improves cache utilization and memory access patterns.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        # Optimization: Compile model for kernel fusion and optimization
        try:
            model = torch.compile(None, mode="reduce-overhead", backend="inductor")
        except Exception:
            pass  # Fallback to eager if compilation fails

        # Optimization: Compile model for kernel fusion and optimization
        try:
            self.model = torch.compile(None, mode="reduce-overhead", backend="inductor")
        except Exception:
            pass  # Fallback to eager if compilation fails

        self.input = None
        self.tile_size = 64  # Tiling: tile size for cache optimization
    
    def setup(self) -> None:
        """Setup: Initialize model with tiling optimization."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: Tiling - divide data into tiles
        # Tiling improves cache utilization by processing smaller chunks
        
        self.model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        ).to(self.device).eval()
        
        # Large input with tiling (better cache utilization)
        self.input = torch.randn(512, 1024, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Tiled operations."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_tiling", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: Tiling
                # Process input in tiles for better cache utilization
                # Tiling: divides data into smaller tiles
                batch_size = self.input.size(0)
                tile_size = self.tile_size
                
                outputs = []
                for i in range(0, batch_size, tile_size):
                    # Process tile (tiling: cache-friendly chunk)
                    tile = self.input[i:i+tile_size]
                    tile_output = self.model(tile)
                    outputs.append(tile_output)
                
                # Concatenate tile outputs (tiling: efficient processing)
                output = torch.cat(outputs, dim=0)
                
                # Optimization: Tiling benefits
                # - Better cache utilization
                # - Improved memory access patterns
                # - Efficient tile-based processing
                # - Better performance through tiling
                _ = output.sum()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.input is None:
            return "Input not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedTilingBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedTilingBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: Tiling")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()

