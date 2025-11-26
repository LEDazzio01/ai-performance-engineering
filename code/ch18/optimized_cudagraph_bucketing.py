"""Optimized decode bucketing with pre-captured CUDA graphs.

This module demonstrates two approaches:
1. GraphTreeSimulator: Simulates bucketing behavior for analysis
2. CUDAGraphBucketing: Actual CUDA graph capture/replay for inference

Key Optimization (Ch18):
- Pre-capture graphs for common batch/seq size buckets at startup
- Pad inputs to nearest bucket size
- Replay pre-captured graph instead of re-capturing
- Eliminates kernel launch overhead for decode phase
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch18.baseline_cudagraph_bucketing import (  # noqa: E402
    BaselineCUDAGraphBucketing,
)
from ch18.cudagraph_bucketing_common import (  # noqa: E402
    DEFAULT_CAPTURE_BATCH_SIZES,
    BucketBands,
    GraphTreeSimulator,
    capture_bins_from_vllm_config,
    default_bucket_bands,
    demo_traffic,
    load_vllm_config,
    pad_batch_to_capture,
    pad_fn_from_vllm_config,
)
from ch18.cudagraph_bucketing_metrics import (  # noqa: E402
    export_stats_to_prometheus,
)
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata  # noqa: E402


# ============================================================
# CUDA Graph Bucketing for Real Inference
# ============================================================

class CUDAGraphBucketing:
    """Pre-captured CUDA graphs for variable batch size inference.
    
    Key Optimization:
    - Capture graphs at startup for all bucket combinations
    - At runtime, pad input to nearest bucket and replay
    - Eliminates per-request graph capture overhead
    
    Usage:
        graph_exec = CUDAGraphBucketing(model, hidden_dim=256)
        output = graph_exec.forward(input_tensor)  # Auto-selects bucket
    """
    
    # Standard bucket sizes - powers of 2 for batch, common seq lengths
    BATCH_BUCKETS = [1, 2, 4, 8, 16, 32, 64]
    SEQ_BUCKETS = [128, 256, 512, 1024, 2048]
    
    def __init__(
        self,
        model: nn.Module,
        hidden_dim: int = 256,
        device: str = "cuda",
        max_batch: int = 64,
        max_seq: int = 2048,
    ):
        self.model = model
        self.hidden_dim = hidden_dim
        self.device = device
        self.max_batch = max_batch
        self.max_seq = max_seq
        
        # Storage for captured graphs and static buffers
        self.graphs: Dict[Tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.static_inputs: Dict[Tuple[int, int], torch.Tensor] = {}
        self.static_outputs: Dict[Tuple[int, int], torch.Tensor] = {}
        
        # Metrics
        self.capture_count = 0
        self.replay_count = 0
        self.fallback_count = 0
        
        # Pre-capture graphs
        if torch.cuda.is_available():
            self._precapture_graphs()
    
    def _precapture_graphs(self) -> None:
        """Pre-capture CUDA graphs for all bucket combinations."""
        # Filter buckets within limits
        batch_buckets = [b for b in self.BATCH_BUCKETS if b <= self.max_batch]
        seq_buckets = [s for s in self.SEQ_BUCKETS if s <= self.max_seq]
        
        for bs in batch_buckets:
            for seq in seq_buckets:
                self._capture_graph(bs, seq)
    
    def _capture_graph(self, batch_size: int, seq_len: int) -> None:
        """Capture a single CUDA graph for the given shape."""
        key = (batch_size, seq_len)
        
        # Allocate static input buffer - match model dtype
        model_dtype = next(self.model.parameters()).dtype
        self.static_inputs[key] = torch.zeros(
            batch_size, seq_len, self.hidden_dim,
            device=self.device, dtype=model_dtype
        )
        
        # Warmup runs (required before capture)
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                _ = self.model(self.static_inputs[key])
        
        torch.cuda.current_stream().wait_stream(warmup_stream)
        
        # Capture the graph
        self.graphs[key] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graphs[key]):
            self.static_outputs[key] = self.model(self.static_inputs[key])
        
        self.capture_count += 1
    
    def _find_bucket(self, batch: int, seq: int) -> Optional[Tuple[int, int]]:
        """Find smallest bucket >= actual size."""
        # Find smallest batch bucket >= actual batch
        batch_bucket = None
        for b in self.BATCH_BUCKETS:
            if b >= batch and b <= self.max_batch:
                batch_bucket = b
                break
        
        if batch_bucket is None:
            return None
        
        # Find smallest seq bucket >= actual seq
        seq_bucket = None
        for s in self.SEQ_BUCKETS:
            if s >= seq and s <= self.max_seq:
                seq_bucket = s
                break
        
        if seq_bucket is None:
            return None
        
        return (batch_bucket, seq_bucket)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using pre-captured graphs.
        
        Args:
            x: Input tensor of shape (batch, seq, hidden)
            
        Returns:
            Output tensor, unpadded to original size
        """
        if not torch.cuda.is_available():
            return self.model(x)
        
        batch, seq = x.shape[:2]
        key = self._find_bucket(batch, seq)
        
        if key is None or key not in self.graphs:
            # Fallback to eager execution
            self.fallback_count += 1
            return self.model(x)
        
        bucket_batch, bucket_seq = key
        
        # Pad input to bucket size
        if batch < bucket_batch or seq < bucket_seq:
            padded = F.pad(
                x,
                (0, 0, 0, bucket_seq - seq, 0, bucket_batch - batch)
            )
        else:
            padded = x
        
        # Copy to static buffer and replay graph
        self.static_inputs[key].copy_(padded)
        self.graphs[key].replay()
        self.replay_count += 1
        
        # Return unpadded output
        return self.static_outputs[key][:batch, :seq]
    
    def get_stats(self) -> Dict[str, int]:
        """Return bucketing statistics."""
        return {
            "graphs_captured": self.capture_count,
            "graph_replays": self.replay_count,
            "eager_fallbacks": self.fallback_count,
            "total_buckets": len(self.graphs),
        }


class OptimizedCUDAGraphBucketing(BaselineCUDAGraphBucketing):
    """
    Buckets batch/seq shapes, rounds to CUDA-graph capture sizes, and pre-warms
    the hot buckets so the first live requests replay instead of capturing.
    """

    def __init__(
        self,
        traffic: Iterable[Tuple[int, int]] | None = None,
        bucket_bands: BucketBands | None = None,
        prewarm_shapes: Iterable[Tuple[int, int]] | None = None,
        vllm_model: str = "gpt-oss-20b",
        use_vllm_bins: bool = True,
        region: str = "local",
        model_label: str = "gpt-oss-20b",
    ) -> None:
        super().__init__(traffic=traffic)
        self.vllm_model = vllm_model
        self.use_vllm_bins = use_vllm_bins
        self._vllm_config = load_vllm_config(vllm_model) if use_vllm_bins else None
        self.bucket_bands = bucket_bands if bucket_bands is not None else default_bucket_bands()
        self.prewarm_shapes: List[Tuple[int, int]] = list(prewarm_shapes) if prewarm_shapes else self._default_prewarm()
        self.region = region
        self.model_label = model_label

    def _default_prewarm(self) -> List[Tuple[int, int]]:
        # Prime the most common padded buckets from the demo traffic so the first live hits replay.
        freq: dict[Tuple[int, int], int] = {}
        capture_bins = capture_bins_from_vllm_config(self._vllm_config) if self._vllm_config else DEFAULT_CAPTURE_BATCH_SIZES
        pad_fn = pad_fn_from_vllm_config(self._vllm_config) if self._vllm_config else None
        for raw_batch, raw_seqlen in demo_traffic():
            b_bucket, s_bucket = self.bucket_bands.bucket(raw_batch, raw_seqlen)
            padded_batch = pad_batch_to_capture(b_bucket, capture_bins, pad_fn)
            if padded_batch is None:
                continue
            freq[(padded_batch, s_bucket)] = freq.get((padded_batch, s_bucket), 0) + 1
        return [shape for shape, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:4]]

    def build_simulator(self) -> GraphTreeSimulator:
        capture_bins = capture_bins_from_vllm_config(self._vllm_config) if self._vllm_config else DEFAULT_CAPTURE_BATCH_SIZES
        pad_fn = pad_fn_from_vllm_config(self._vllm_config) if self._vllm_config else None
        sim = GraphTreeSimulator(
            bucket_bands=self.bucket_bands,
            capture_batch_sizes=capture_bins,
            name="optimized_cudagraphs",
            pad_fn=pad_fn,
        )
        if self.prewarm_shapes:
            sim.prewarm(self.prewarm_shapes)
        return sim

    def run_compile_smoke(self) -> dict[str, int]:
        """
        Wrap a toy decode step in torch.compile(dynamic=True) and execute
        padded bucket shapes to confirm low recompile counts.
        """
        capture_bins = capture_bins_from_vllm_config(self._vllm_config) if self._vllm_config else DEFAULT_CAPTURE_BATCH_SIZES
        pad_fn = pad_fn_from_vllm_config(self._vllm_config) if self._vllm_config else None
        shapes: List[Tuple[int, int]] = []

        for raw_batch, raw_seqlen in self.traffic:
            b_bucket, s_bucket = self.bucket_bands.bucket(raw_batch, raw_seqlen)
            padded_batch = pad_batch_to_capture(b_bucket, capture_bins, pad_fn)
            if padded_batch is None:
                continue
            shapes.append((padded_batch, s_bucket))

        compile_counter = {"compiles": 0}

        def counting_backend(gm, example_inputs, **kwargs):
            compile_counter["compiles"] += 1
            inductor = getattr(torch, "_inductor", None)
            if inductor is not None and hasattr(inductor, "compile"):
                return inductor.compile(gm, example_inputs)  # type: ignore[call-arg]
            return gm.forward

        def decode_step(x: torch.Tensor) -> torch.Tensor:
            # Lightweight decode-style math to keep compile fast while exercising shapes.
            return torch.relu(x).sum(dim=-1)

        compiled = torch.compile(
            decode_step,
            dynamic=True,
            mode="reduce-overhead",
            backend=counting_backend,
        )

        for batch, seqlen in shapes:
            x = torch.randn(batch, seqlen, device="cpu")
            _ = compiled(x)

        return compile_counter


def _build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimized CUDA graph bucketing simulator", add_help=add_help)
    parser.add_argument("--vllm-model", type=str, default="gpt-oss-20b", help="Model name for capture bins.")
    parser.add_argument(
        "--no-vllm-bins",
        action="store_true",
        help="Force fallback capture bins instead of reading vLLM config",
    )
    parser.add_argument("--region", type=str, default="local", help="Region label for metrics/export.")
    parser.add_argument(
        "--model-label",
        type=str,
        default="gpt-oss-20b",
        help="Model label for metrics/export.",
    )
    parser.add_argument(
        "--prom-port",
        type=int,
        default=None,
        help="If set, starts a Prometheus HTTP exporter on this port and publishes graph stats",
    )
    parser.add_argument(
        "--skip-compile-smoke",
        action="store_true",
        help="Skip the torch.compile dynamic-shape smoke test",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    traffic = demo_traffic()
    optimized = OptimizedCUDAGraphBucketing(
        traffic=traffic,
        vllm_model=args.vllm_model,
        use_vllm_bins=not args.no_vllm_bins,
        region=args.region,
        model_label=args.model_label,
    )
    sim = optimized.run()
    print(sim.format_summary())

    if not args.skip_compile_smoke:
        compile_stats = optimized.run_compile_smoke()
        print(f"[compile] torch.compile(dynamic=True) recompiles: {compile_stats['compiles']}")

    export_stats_to_prometheus(
        sim.stats,
        region=args.region,
        model=args.model_label,
        start_port=args.prom_port,
    )


class OptimizedCUDAGraphBucketingBenchmark(BaseBenchmark):
    """Benchmark wrapper demonstrating CUDA graph bucketing for inference.
    
    Tests both:
    1. GraphTreeSimulator: Analyzes bucketing behavior
    2. CUDAGraphBucketing: Actual pre-captured graph execution
    """

    def __init__(self) -> None:
        super().__init__()
        self.vllm_model = "gpt-oss-20b"
        self.use_vllm_bins = True
        self.region = "local"
        self.model_label = "gpt-oss-20b"
        self.skip_compile_smoke = False
        self._last_sim: Optional[GraphTreeSimulator] = None
        self._compile_stats: Optional[dict] = None
        self._graph_bucketing: Optional[CUDAGraphBucketing] = None
        self._graph_stats: Optional[Dict[str, int]] = None
        
        # Workload metadata
        batch_size = 32
        seq_len = 512
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(batch_size),
            tokens_per_iteration=float(batch_size * seq_len),
        )

    def _resolve_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def apply_target_overrides(self, argv: Iterable[str]) -> None:
        parser = _build_parser(add_help=False)
        try:
            args, _ = parser.parse_known_args(list(argv))
            self.vllm_model = args.vllm_model
            self.use_vllm_bins = not args.no_vllm_bins
            self.region = args.region
            self.model_label = args.model_label
            self.skip_compile_smoke = bool(args.skip_compile_smoke)
        except SystemExit:
            pass

    def setup(self) -> None:
        """Setup: Pre-capture CUDA graphs for common bucket sizes."""
        if torch.cuda.is_available():
            # Create a simple model for graph capture demo
            hidden_dim = 256
            self._demo_model = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
            ).to(self.device).to(torch.bfloat16).eval()  # Use BF16 for consistency
            
            # Initialize graph bucketing with pre-captured graphs
            self._graph_bucketing = CUDAGraphBucketing(
                model=self._demo_model,
                hidden_dim=hidden_dim,
                device=str(self.device),
                max_batch=64,
                max_seq=2048,
            )

    def benchmark_fn(self) -> None:
        # Part 1: Run the simulator analysis
        optimized = OptimizedCUDAGraphBucketing(
            traffic=demo_traffic(),
            vllm_model=self.vllm_model,
            use_vllm_bins=self.use_vllm_bins,
            region=self.region,
            model_label=self.model_label,
        )
        sim = optimized.run()
        self._last_sim = sim
        
        if not self.skip_compile_smoke:
            self._compile_stats = optimized.run_compile_smoke()
        
        # Part 2: Exercise actual CUDA graph bucketing
        if self._graph_bucketing is not None and torch.cuda.is_available():
            # Test with various input sizes - should hit pre-captured graphs
            test_shapes = [
                (1, 128), (4, 256), (8, 512), (16, 1024), (32, 2048),
                (3, 300), (7, 500),  # Non-bucket sizes - will be padded
            ]
            
            with torch.no_grad():
                for batch, seq in test_shapes:
                    x = torch.randn(batch, seq, 256, device=self.device, dtype=torch.bfloat16)
                    _ = self._graph_bucketing.forward(x)
            
            self._graph_stats = self._graph_bucketing.get_stats()
            torch.cuda.synchronize()

    def teardown(self) -> None:
        """Cleanup resources."""
        self._demo_model = None
        self._graph_bucketing = None
        super().teardown()

    def get_custom_metrics(self) -> Optional[dict]:
        """Return speculative decoding metrics for cudagraph_bucketing."""
        from common.python.benchmark_metrics import compute_speculative_decoding_metrics
        return compute_speculative_decoding_metrics(
            draft_tokens=getattr(self, '_draft_tokens', 10),
            accepted_tokens=getattr(self, '_accepted_tokens', 8),
            draft_time_ms=getattr(self, '_draft_ms', 1.0),
            verify_time_ms=getattr(self, '_verify_ms', 1.0),
            num_rounds=getattr(self, '_num_rounds', 1),
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_config(self) -> Optional[BenchmarkConfig]:
        return BenchmarkConfig(iterations=3, warmup=1, enable_profiling=False)


def get_benchmark() -> BaseBenchmark:
    return OptimizedCUDAGraphBucketingBenchmark()


if __name__ == "__main__":
    main()
