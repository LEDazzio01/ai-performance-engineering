"""optimized_memory.py - Optimized GPU memory management."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata

BATCH_SIZE = 512
INPUT_DIM = 2048
HIDDEN_DIM = 2048
REPETITIONS = 8


class OptimizedMemoryBenchmark(BaseBenchmark):
    """Optimized: GPU memory management with capture and buffer reuse."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.batch_size = BATCH_SIZE
        self.input_dim = INPUT_DIM
        self.device_buffer: Optional[torch.Tensor] = None
        self.transform_buffer: Optional[torch.Tensor] = None
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.graph_output: Optional[torch.Tensor] = None
        self.repetitions = REPETITIONS
        tokens = self.batch_size * self.input_dim * self.repetitions
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.repetitions),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        
        torch.manual_seed(42)
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, self.input_dim),
        ).to(self.device, dtype=torch.float32).eval()
        
        self.device_buffer = torch.empty(
            self.batch_size,
            self.input_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self.transform_buffer = torch.empty_like(self.device_buffer)
        self.graph_output = torch.empty_like(self.device_buffer)
        self._synchronize()

        with torch.no_grad():
            _ = self.model(self.device_buffer)
        self._synchronize()

        self.graph = torch.cuda.CUDAGraph()
        self.device_buffer.uniform_(0.0, 255.0)
        self._synchronize()
        with torch.cuda.graph(self.graph):
            self.transform_buffer.copy_(self.device_buffer)
            self.transform_buffer.mul_(1.0 / 255.0)
            self.transform_buffer.add_(-0.5)
            self.transform_buffer.mul_(2.0)
            self.transform_buffer.tanh_()
            self.graph_output.copy_(self.model(self.transform_buffer))
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        if (
            self.model is None
            or self.device_buffer is None
            or self.graph is None
            or self.graph_output is None
        ):
            raise RuntimeError("Optimized memory benchmark not initialized")

        with self._nvtx_range("optimized_memory"):
            with torch.no_grad():
                for _ in range(self.repetitions):
                    self.device_buffer.uniform_(0.0, 255.0)
                    self.graph.replay()
        self._synchronize()
    
    def teardown(self) -> None:
        self.model = None
        self.device_buffer = None
        self.transform_buffer = None
        self.graph_output = None
        self.graph = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=200,
            warmup=10,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_inference_metrics
        return compute_inference_metrics(
            ttft_ms=getattr(self, '_ttft_ms', 50.0),
            tpot_ms=getattr(self, '_tpot_ms', 10.0),
            total_tokens=getattr(self, 'total_tokens', 256),
            total_requests=getattr(self, 'total_requests', 1),
            batch_size=getattr(self, 'batch_size', 1),
            max_batch_size=getattr(self, 'max_batch_size', 32),
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> OptimizedMemoryBenchmark:
    return OptimizedMemoryBenchmark()
