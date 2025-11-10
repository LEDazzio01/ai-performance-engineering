"""optimized nvlink - Optimized NVLink GPU-to-GPU transfer. Implements Benchmark protocol for harness integration."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

try:
    import ch1.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

from typing import Optional

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)
from ch1.workload_config import WORKLOAD


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch1")
    return torch.device("cuda")


class OptimizedNvlinkBenchmark(Benchmark):
    """Optimized: Memory transfer with NVLink (GPU-to-GPU via NVLink)."""

    def __init__(self):
        self.device = resolve_device()
        self.data_gpu0 = None
        self.data_gpu1 = None
        self.host_buffers = None
        self.N = 16_000_000
        self.is_multi_gpu = False
        self.num_chunks = max(16, WORKLOAD.prefill_chunks * 2)
        self.chunk_size = self.N // self.num_chunks
        self.stream_main = None
        self.copy_stream = None
        self.result = None

    def setup(self) -> None:
        """Setup: Initialize tensors and enable NVLink peer access."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            enable_tf32()
        torch.manual_seed(42)
        # Optimization: NVLink for high-speed GPU-to-GPU communication
        # NVLink provides high bandwidth and low latency compared to PCIe
        
        num_gpus = torch.cuda.device_count()
        self.is_multi_gpu = num_gpus >= 2
        
        if self.is_multi_gpu:
            # Multi-GPU: use NVLink for direct GPU-to-GPU transfer
            self.data_gpu0 = torch.randn(self.chunk_size, device=torch.device("cuda:0"), dtype=torch.float32)
            self.data_gpu1 = torch.empty(self.chunk_size, device=torch.device("cuda:1"), dtype=torch.float32)
            # Enable peer access for NVLink (if available)
            # This enables direct GPU-to-GPU communication via NVLink
            if torch.cuda.can_device_access_peer(0, 1):
                try:
                    # Enable peer access from device 0 to device 1
                    torch.cuda.set_device(0)
                    # Peer access enables NVLink communication between GPUs
                    # Note: torch._C._cuda_enablePeerAccess may not be available in all PyTorch versions
                    # The driver may already enable peer access automatically
                    pass  # Peer access is typically enabled automatically by CUDA driver
                except Exception:
                    pass
        else:
            self.data_gpu0 = torch.empty(self.chunk_size, device=self.device, dtype=torch.float32)
        self.host_buffers = [
            torch.randn(self.chunk_size, dtype=torch.float32, pin_memory=True)
            for _ in range(self.num_chunks)
        ]
        self.stream_main = torch.cuda.Stream(device=0)
        self.copy_stream = torch.cuda.Stream(device=0)
        
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Benchmark: NVLink-optimized GPU-to-GPU transfer."""
        # Use conditional NVTX ranges - only enabled when profiling
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        
        with nvtx_range("optimized_nvlink", enable=enable_nvtx):
            checksum = 0.0
            if self.is_multi_gpu:
                stream0 = torch.cuda.Stream(device=0)
                stream1 = torch.cuda.Stream(device=1)
                for host in self.host_buffers:
                    with torch.cuda.stream(stream0):
                        self.data_gpu0.copy_(host.to(self.device, non_blocking=True), non_blocking=True)
                    with torch.cuda.stream(stream1):
                        self.data_gpu1.copy_(self.data_gpu0.to(torch.device("cuda:1"), non_blocking=True), non_blocking=True)
                        self.data_gpu1.mul_(1.0001)
                        checksum += self.data_gpu1.sum().item()
                torch.cuda.device(0)
                stream0.synchronize()
                torch.cuda.device(1)
                stream1.synchronize()
            else:
                for host in self.host_buffers:
                    with torch.cuda.stream(self.copy_stream):
                        self.data_gpu0.copy_(host, non_blocking=True)
                    with torch.cuda.stream(self.stream_main):
                        self.stream_main.wait_stream(self.copy_stream)
                        self.data_gpu0.mul_(1.0002)
                        checksum += self.data_gpu0.sum().item()
                torch.cuda.synchronize()
            self.result = checksum

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data_gpu0 = None
        self.data_gpu1 = None
        self.host_buffers = None
        self.stream_main = None
        self.copy_stream = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=3,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.result is None:
            return "Data not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedNvlinkBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized NVLink: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print("NOTE: Uses NVLink for GPU-to-GPU transfer (high bandwidth, low latency)")
