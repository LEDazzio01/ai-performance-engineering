"""optimized_expert_parallelism.py - Optimized expert parallelism for MoE.

Demonstrates expert parallelism by distributing experts across multiple GPUs.
Expert parallelism: Distributes experts across GPUs for parallel processing.
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

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch19")
    return torch.device("cuda")


class ExpertLayer(nn.Module):
    """Single expert in MoE model."""
    
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.expert = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expert(x)


class OptimizedExpertParallelismBenchmark(Benchmark):
    """Optimized: Expert parallelism with experts distributed across GPUs."""
    
    def __init__(self):
        self.device = resolve_device()
        self.experts = None
        self.router = None
        self.input_data = None
        self.compute_streams: list[torch.cuda.Stream] | None = None
        self.last_output = None
        self.num_experts = 32
        self.top_k = 4  # Top-k experts per token
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    def setup(self) -> None:
        """Setup: Initialize MoE model with experts distributed across GPUs."""
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        
        torch.manual_seed(42)
        # Optimization: Expert parallelism distributes experts across multiple GPUs
        # Each GPU hosts a subset of experts for parallel processing
        # This enables parallel processing of different experts simultaneously
        self.experts = nn.ModuleList()
        for expert_id in range(self.num_experts):
            gpu_id = expert_id % self.num_gpus
            expert = ExpertLayer(256).to(torch.device(f"cuda:{gpu_id}")).half().eval()
            self.experts.append(expert)
        
        # Router on first GPU
        self.router = nn.Linear(256, self.num_experts).to(self.device)
        self.compute_streams = [torch.cuda.Stream() for _ in range(min(self.num_experts, 8))]
        
        self.input_data = torch.randn(512, 256, device=self.device, dtype=torch.float16)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Expert parallelism processing across multiple GPUs."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled
        
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        
        # Optimization: Process experts in parallel across GPUs
        # Expert parallelism enables parallel processing of different experts
        with nvtx_range("optimized_expert_parallelism", enable=enable_nvtx):
            with torch.no_grad():
                router_logits = self.router(self.input_data.float())
                top_k_weights, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
                top_k_weights = torch.softmax(top_k_weights, dim=-1)
                
                output = torch.zeros(self.input_data.shape[0], self.input_data.shape[1], device=self.device, dtype=torch.float32)
                work_items = []
                for expert_id in range(self.num_experts):
                    expert_mask = (top_k_indices == expert_id).any(dim=-1)
                    if expert_mask.any():
                        token_indices = torch.nonzero(expert_mask, as_tuple=False).squeeze(-1)
                        work_items.append((expert_id, token_indices))

                pending = []
                if work_items and self.compute_streams:
                    for idx, (expert_id, token_indices) in enumerate(work_items):
                        stream = self.compute_streams[idx % len(self.compute_streams)]
                        expert = self.experts[expert_id]
                        expert_device = torch.device(f"cuda:{expert_id % self.num_gpus}")
                        with torch.cuda.stream(stream):
                            expert_input = torch.index_select(self.input_data, 0, token_indices).to(
                                expert_device, dtype=torch.float16
                            )
                            expert_output = expert(expert_input)
                            event = torch.cuda.Event(blocking=False)
                            event.record(stream)
                            pending.append((token_indices, expert_output, event, expert_device))

                for token_indices, expert_output, event, expert_device in pending:
                    event.synchronize()
                    scaled = expert_output.to(self.device, dtype=torch.float32)
                    output.index_add_(0, token_indices, scaled)
                self.last_output = output
        
        torch.cuda.synchronize()
    
    def teardown(self) -> None:
        """Cleanup: Clear CUDA cache."""
        if torch.cuda.is_available():
            for gpu_id in range(self.num_gpus):
                torch.cuda.empty_cache()
        self.compute_streams = None
        self.last_output = None
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=3,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.experts is None or len(self.experts) == 0:
            return "Experts not initialized"
        if self.router is None:
            return "Router not initialized"
        if self.input_data is None:
            return "Input data not initialized"
        if self.last_output is None:
            return "No expert output computed"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedExpertParallelismBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)
