"""optimized_pipeline_parallelism.py - Optimized pipeline parallelism across GPUs."""

from __future__ import annotations

import argparse
from typing import Optional, List

import torch
import torch.nn as nn

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


class OptimizedPipelineParallelismBenchmark(BaseBenchmark):
    """Optimized: Pipeline parallelism with layers split across GPUs.
    
    When only 1 GPU is available, uses optimized single-GPU path with:
    - torch.compile for kernel fusion
    - Autocast for mixed precision
    - Efficient layer execution without pipeline overhead
    """

    def __init__(self, micro_batches: Optional[int] = None):
        super().__init__()
        self.pipeline_stages: List[nn.Module] = []
        self.hidden_size = 1024
        self.batch_size = 256
        self.micro_batches = 4
        if micro_batches is not None:
            self.micro_batches = max(1, int(micro_batches))
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.stage_streams: List[torch.cuda.Stream] = []
        self.stage_events: List[List[torch.cuda.Event]] = []
        self.microbatch_inputs: Optional[List[torch.Tensor]] = None
        self._last_stage_durations_ms: List[float] = []
        self._bubble_fraction: float = 0.0
        self._single_gpu_mode: bool = False
        self._compiled_model: Optional[nn.Module] = None
        self._input_data: Optional[torch.Tensor] = None
        tokens = self.batch_size * self.hidden_size
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.micro_batches),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()

        torch.manual_seed(42)
        
        # Use single-GPU optimized path when only 1 GPU available
        self._single_gpu_mode = (self.num_gpus == 1)
        
        if self._single_gpu_mode:
            # Single GPU: use compiled sequential model (faster than pipeline overhead)
            model = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 4),
                nn.GELU(),
                nn.Linear(self.hidden_size * 4, self.hidden_size * 4),
                nn.GELU(),
                nn.Linear(self.hidden_size * 4, self.hidden_size * 2),
                nn.GELU(),
                nn.Linear(self.hidden_size * 2, self.hidden_size),
            ).to(self.device, dtype=torch.bfloat16).eval()
            
            # Compile for kernel fusion and optimization
            try:
                self._compiled_model = torch.compile(model, mode="reduce-overhead")
            except Exception:
                self._compiled_model = model
            
            self._input_data = torch.randn(
                self.batch_size, self.hidden_size, 
                device=self.device, dtype=torch.bfloat16
            )
        else:
            # Multi-GPU: use pipeline parallelism
            layers_per_stage = [
                [nn.Linear(self.hidden_size, self.hidden_size * 4), nn.GELU()],
                [nn.Linear(self.hidden_size * 4, self.hidden_size * 4), nn.GELU()],
                [nn.Linear(self.hidden_size * 4, self.hidden_size * 2), nn.GELU()],
                [nn.Linear(self.hidden_size * 2, self.hidden_size)],
            ]

            self.pipeline_stages = []
            for stage_id, layer_stack in enumerate(layers_per_stage):
                gpu_id = stage_id % self.num_gpus
                stage = nn.Sequential(*layer_stack).to(torch.device(f"cuda:{gpu_id}"), dtype=torch.bfloat16).eval()
                self.pipeline_stages.append(stage)

            self.microbatch_inputs = torch.randn(
                self.batch_size, self.hidden_size, device=torch.device("cuda:0"), dtype=torch.bfloat16
            ).chunk(self.micro_batches, dim=0)

            self.stage_streams = [torch.cuda.Stream(priority=-1) for _ in self.pipeline_stages]
            self.stage_events = [
                [torch.cuda.Event(enable_timing=False) for _ in range(self.micro_batches)]
                for _ in self.pipeline_stages
            ]
        
        # Refresh workload metadata
        tokens = self.batch_size * self.hidden_size
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.micro_batches),
            tokens_per_iteration=float(tokens),
        )
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self._single_gpu_mode:
            # Optimized single-GPU path: compiled model, no pipeline overhead
            if self._compiled_model is None or self._input_data is None:
                raise RuntimeError("Single-GPU model not initialized")
            
            with self._nvtx_range("optimized_single_gpu"):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    _ = self._compiled_model(self._input_data)
            self._synchronize()
            self._bubble_fraction = 0.0  # No pipeline bubble
            self._last_stage_durations_ms = [0.0]
            return
        
        # Multi-GPU pipeline path
        if not self.pipeline_stages or self.microbatch_inputs is None:
            raise RuntimeError("Pipeline not initialized")

        num_stages = len(self.pipeline_stages)
        self._last_stage_durations_ms = [0.0 for _ in range(num_stages)]
        stage_buffers: List[List[Optional[torch.Tensor]]] = [
            [None for _ in range(self.micro_batches)] for _ in range(num_stages + 1)
        ]
        stage_buffers[0] = list(self.microbatch_inputs)

        stage_devices = [next(stage.parameters()).device for stage in self.pipeline_stages]

        with self._nvtx_range("optimized_pipeline_parallelism"):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for micro_idx in range(self.micro_batches + num_stages - 1):
                    for stage_idx, stage in enumerate(self.pipeline_stages):
                        chunk_idx = micro_idx - stage_idx
                        if chunk_idx < 0 or chunk_idx >= self.micro_batches:
                            continue
                        stream = self.stage_streams[stage_idx]
                        with torch.cuda.stream(stream):
                            if stage_idx > 0:
                                stream.wait_event(self.stage_events[stage_idx - 1][chunk_idx])
                            x = stage_buffers[stage_idx][chunk_idx]
                            if x is None:
                                continue
                            stage_start = self._record_start()
                            with self._nvtx_range(f"stage{stage_idx}_mb{chunk_idx}"):
                                out = stage(x.to(stage_devices[stage_idx]))
                            self._last_stage_durations_ms[stage_idx] += self._record_stop(stage_start)
                            next_stage_idx = stage_idx + 1
                            if next_stage_idx < len(stage_devices):
                                next_device = stage_devices[next_stage_idx]
                                if next_device != stage_devices[stage_idx]:
                                    out = out.to(next_device)
                            stage_buffers[next_stage_idx][chunk_idx] = out
                            self.stage_events[stage_idx][chunk_idx].record(stream)

        for stream in self.stage_streams:
            stream.synchronize()
        self._synchronize()
        # Bubble fraction approximates fill/drain overhead: (S-1)/M
        self._bubble_fraction = (num_stages - 1) / float(self.micro_batches)

    def teardown(self) -> None:
        self.pipeline_stages = []
        self.microbatch_inputs = None
        self.stage_streams = []
        self.stage_events = []
        self._compiled_model = None
        self._input_data = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=12,
            warmup=5,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Expose bubble math and per-stage timing to spot imbalance."""
        if self._single_gpu_mode:
            return {
                "mode": "single_gpu_compiled",
                "pipeline_stages": 1,
                "microbatches": 1,
                "bubble_fraction": 0.0,
                "torch_compile": 1.0,
            }
        
        if not self._last_stage_durations_ms:
            return None
        max_stage = max(self._last_stage_durations_ms)
        min_stage = min(self._last_stage_durations_ms)
        imbalance = (max_stage / min_stage) if min_stage > 0 else float("inf")
        return {
            "mode": "multi_gpu_pipeline",
            "pipeline_stages": len(self._last_stage_durations_ms),
            "microbatches": self.micro_batches,
            "bubble_fraction": self._bubble_fraction,
            "stage_time_max_ms": max_stage,
            "stage_time_min_ms": min_stage,
            "stage_imbalance_ratio": imbalance,
        }

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if not self.pipeline_stages:
            return "Pipeline stages not initialized"
        return None


def get_benchmark() -> OptimizedPipelineParallelismBenchmark:
    """Factory function for harness discovery."""
    return OptimizedPipelineParallelismBenchmark()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimized pipeline parallel benchmark")
    parser.add_argument(
        "--microbatches",
        type=int,
        default=None,
        help="Number of microbatches to pipeline (default: 4)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    benchmark = OptimizedPipelineParallelismBenchmark(micro_batches=args.microbatches)
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(result)
