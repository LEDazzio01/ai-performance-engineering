"""Decode loop microbenchmark with baseline and optimized paths.

Shared helpers and configs for the decode_optimization lab variants.
Tests serving optimizations (pinned memory, streams, CUDA graphs, FP8/FP4, torch.compile)
on a simplified MLP-based decode loop.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
from contextlib import nullcontext

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from arch_config import prefer_sdpa_backends  # type: ignore
    from core.utils.compile_utils import enable_tf32  # type: ignore
except Exception:  # pragma: no cover - defensive import
    prefer_sdpa_backends = None  # type: ignore
    enable_tf32 = None  # type: ignore

from core.harness.hardware_capabilities import detect_capabilities  # noqa: E402
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig  # noqa: E402

try:  # Optional but strongly recommended for fast variants
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch import Linear as TELinear
    from transformer_engine.common.recipe import DelayedScaling
    import transformer_engine.pytorch.constants as te_constants

    TE_AVAILABLE = True
except Exception:  # pragma: no cover - safe fallback
    te = None  # type: ignore
    TELinear = None  # type: ignore
    DelayedScaling = None  # type: ignore
    te_constants = None  # type: ignore
    TE_AVAILABLE = False


def attach_benchmark_metadata(bench: BaseBenchmark, module_file: str):
    """Annotate a benchmark so subprocess runner can re-import via get_benchmark."""
    bench._module_file_override = module_file
    bench._factory_name_override = "get_benchmark"
    return bench


def _te_version_at_least(major: int, minor: int = 0) -> bool:
    if not TE_AVAILABLE or not hasattr(te, "__version__"):
        return False
    try:
        parts = te.__version__.split(".")
        return int(parts[0]) > major or (int(parts[0]) == major and int(parts[1]) >= minor)
    except Exception:
        return False


def _is_blackwell_family() -> bool:
    cap = detect_capabilities()
    if cap is not None:
        # Treat Blackwell (SM100/SM103) and Grace-Blackwell (SM12x) as Blackwell-class for defaults.
        return cap.architecture in {"blackwell", "blackwell_ultra", "grace_blackwell"}
    if not torch.cuda.is_available():
        raise RuntimeError("Cannot determine architecture: capability probe unavailable and CUDA not available.")
    cc_major, _ = torch.cuda.get_device_capability()
    return cc_major >= 10


@dataclass
class DecodeConfig:
    batch_size: int = 4
    prompt_tokens: int = 256
    decode_tokens: int = 64
    hidden_size: int = 1024
    vocab_size: int = 8192
    use_fp8: bool = False
    use_fp4: bool = False
    use_pinned_host: bool = False
    use_copy_stream: bool = False
    use_compute_stream: bool = False
    use_cuda_graphs: bool = False
    graph_full_iteration: bool = False
    use_torch_compile: bool = False
    iterations: int = 8
    warmup: int = 10
    label: str = "decode_optimization"


class DecodeBenchmark(BaseBenchmark):
    """Lightweight decode loop benchmark for testing serving optimizations."""

    def __init__(self, cfg: DecodeConfig):
        super().__init__()
        self.cfg = cfg
        self.dtype = torch.bfloat16
        self.copy_stream: Optional[torch.cuda.Stream] = None
        self.compute_stream: Optional[torch.cuda.Stream] = None
        self.graph_stream: Optional[torch.cuda.Stream] = None
        self.decode_graph: Optional[torch.cuda.CUDAGraph] = None
        self.graph_includes_prefill: bool = False
        self.graph_logits: Optional[torch.Tensor] = None
        self.graph_next_token: Optional[torch.Tensor] = None
        self._custom_metrics: Dict[str, float] = {}
        self._fp8_enabled: bool = False
        self._fp4_enabled: bool = False
        self._graph_error: Optional[str] = None
        self._compile_error: Optional[str] = None
        self.sdpa_ctx_factory = prefer_sdpa_backends if prefer_sdpa_backends is not None else nullcontext
        self.fp8_recipe = None
        if TE_AVAILABLE:
            if self.cfg.use_fp4 and getattr(te_constants, "NVFP4_BLOCK_SCALING_SIZE", None) is not None:
                try:
                    from transformer_engine.common.recipe import NVFP4BlockScaling

                    self.fp8_recipe = DelayedScaling(float8_block_scaling=NVFP4BlockScaling())
                    self._fp4_enabled = True
                except Exception:
                    self.fp8_recipe = None
                    self._fp4_enabled = False
            elif self.cfg.use_fp8:
                try:
                    self.fp8_recipe = DelayedScaling()
                    self._fp8_enabled = True
                except Exception:
                    self.fp8_recipe = None

    def setup(self) -> None:
        import gc
        
        # CRITICAL: Clean up CUDA state from previous benchmarks
        # This prevents "Offset increment outside graph capture" errors
        # and index out of bounds from corrupted CUDA state
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        try:
            if hasattr(torch.cuda, 'graph_pool_trim'):
                torch.cuda.graph_pool_trim()
        except Exception:
            pass
        
        # Reset CUDA RNG state
        try:
            device_idx = torch.cuda.current_device()
            gen = torch.cuda.default_generators[device_idx]
            gen.set_offset(0)
            gen.manual_seed(42)
        except Exception:
            pass
        
        try:
            torch._dynamo.reset()
        except Exception:
            pass
        
        try:
            torch._inductor.cudagraph_trees.reset_cudagraph_trees()
        except Exception:
            pass
        
        # Ensure deterministic behavior for verification
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        # Enable deterministic algorithms where possible
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if enable_tf32 is not None:
            enable_tf32(set_global_precision=True)
        else:
                torch.set_float32_matmul_precision("high")
        if self.cfg.use_copy_stream:
            self.copy_stream = torch.cuda.Stream()
        if self.cfg.use_compute_stream:
            self.compute_stream = torch.cuda.Stream()
        self._init_model()
        self._init_buffers()
        self._cache_te_weight_workspaces()
        # Default to eager helpers; swap in compiled variants below when enabled.
        self.prefill_fn = self._prefill
        self.decode_fn = self._decode_step
        if self.cfg.use_torch_compile:
            self._maybe_compile()
        if self.cfg.use_cuda_graphs:
            self._capture_decode_graph()
        total_tokens = self.cfg.batch_size * (self.cfg.prompt_tokens + self.cfg.decode_tokens)
        self.register_workload_metadata(
            requests_per_iteration=float(self.cfg.batch_size),
            tokens_per_iteration=float(total_tokens),
        )

    # Model + buffer init
    def _init_model(self) -> None:
        hs = self.cfg.hidden_size
        vs = self.cfg.vocab_size
        # Create embedding on CPU first to avoid CUDA RNG graph capture issues
        # then move to device. This ensures parameter init uses CPU RNG.
        self.embedding = nn.Embedding(vs, hs, dtype=self.dtype).to(self.device)

        def _linear(in_features: int, out_features: int, *, bias: bool = True) -> nn.Module:
            if (self.cfg.use_fp8 or self.cfg.use_fp4) and TE_AVAILABLE:
                # TE Linear must be created on device
                return TELinear(
                    in_features,
                    out_features,
                    bias=bias,
                    params_dtype=self.dtype,
                    device=self.device,
                )
            # Create on CPU first, then move to device to avoid CUDA RNG issues
            return nn.Linear(in_features, out_features, bias=bias, dtype=self.dtype).to(self.device)

        # Create modules on CPU first, then move to device
        self.prefill_mlp = nn.Sequential(
            nn.LayerNorm(hs, dtype=self.dtype).to(self.device),
            _linear(hs, hs * 2),
            nn.GELU(),
            _linear(hs * 2, hs),
        )
        self.decode_mlp = nn.Sequential(
            nn.LayerNorm(hs, dtype=self.dtype).to(self.device),
            _linear(hs, hs),
            nn.GELU(),
            _linear(hs, hs),
        )
        self.lm_head = _linear(hs, vs, bias=False)
        if self.cfg.use_fp8 and TE_AVAILABLE and not self._fp4_enabled:
            self._fp8_enabled = True

    def _cache_te_weight_workspaces(self) -> None:
        """Pre-quantize TE weights by running a warmup forward pass.
        
        The correct way to initialize FP8 workspaces is via forward passes under
        fp8_autocast, not by calling get_weight_workspace() manually.
        """
        if (
            not TE_AVAILABLE
            or not (self._fp8_enabled or self._fp4_enabled)
            or os.getenv("DECODE_SKIP_TE_CACHE") == "1"
        ):
            return
        
        # Warmup FP8 caches by running forward passes - this is the proper API
        # Use CPU randn + to(device) to avoid CUDA RNG graph capture issues
        bsz = self.cfg.batch_size
        hs = self.cfg.hidden_size
        dummy_hidden = torch.randn(bsz, hs, dtype=self.dtype).to(self.device)
        dummy_seq = torch.randn(bsz, self.cfg.prompt_tokens, hs, dtype=self.dtype).to(self.device)
        
        with torch.no_grad(), te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
            # Warmup prefill MLP
            _ = self.prefill_mlp(dummy_seq)
            # Warmup decode MLP  
            _ = self.decode_mlp(dummy_hidden)
            # Warmup lm_head
            _ = self.lm_head(dummy_hidden)
        
        torch.cuda.synchronize()

    def _init_buffers(self) -> None:
        bsz, prompt = self.cfg.batch_size, self.cfg.prompt_tokens
        self.host_prompt = torch.randint(
            0, self.cfg.vocab_size, (bsz, prompt), dtype=torch.long, device="cpu"
        )
        if self.cfg.use_pinned_host:
            self.host_prompt = self.host_prompt.pin_memory()

        self.gpu_prompt = torch.empty_like(self.host_prompt, device=self.device)
        self.state_buffer = torch.zeros((bsz, self.cfg.hidden_size), device=self.device, dtype=self.dtype)
        self.current_tokens = torch.empty((bsz,), device=self.device, dtype=torch.long)
        self.next_token_out = torch.empty_like(self.current_tokens)

    # Compiled / graphed helpers
    def _maybe_compile(self) -> None:
        # NO FALLBACK - torch.compile must work
        # When using explicit CUDA graphs, don't use reduce-overhead mode (which uses internal graphs)
        # as this causes "Cannot prepare for replay during capturing stage" errors
        compile_mode = "default" if self.cfg.use_cuda_graphs else "reduce-overhead"
        self.prefill_fn = torch.compile(self._prefill, mode=compile_mode, fullgraph=False)
        self.decode_fn = torch.compile(self._decode_step, mode=compile_mode, fullgraph=True)
        self._compile_error = None

    def _capture_decode_graph(self) -> None:
        # Allocate static outputs once
        bsz = self.cfg.batch_size
        self.graph_stream = torch.cuda.Stream()
        self.graph_logits = torch.empty((bsz, self.cfg.vocab_size), device=self.device, dtype=self.dtype)
        self.graph_next_token = torch.empty((bsz,), device=self.device, dtype=torch.long)
        # Ensure prompt buffer is initialized with valid tokens before capture
        self.gpu_prompt.copy_(self.host_prompt, non_blocking=False)
        
        # NO FALLBACK - CUDA graph capture must succeed
        # Warm up to populate kernels/caches prior to capture
        with torch.cuda.stream(self.graph_stream):
            for _ in range(2):
                if self.cfg.graph_full_iteration:
                    prefill_state = self.prefill_fn(self.gpu_prompt)
                    self.state_buffer.copy_(prefill_state)
                    self.current_tokens.copy_(self.gpu_prompt[:, -1])
                logits, next_state, next_token = self.decode_fn(self.current_tokens, self.state_buffer)
                self.state_buffer.copy_(next_state)
                self.graph_next_token.copy_(next_token)
                self.current_tokens.copy_(next_token)
        torch.cuda.synchronize()
        self.decode_graph = torch.cuda.CUDAGraph()
        # Reset state before capture for determinism
        self.state_buffer.zero_()
        self.current_tokens.zero_()
        torch.cuda.synchronize()
        with torch.cuda.graph(self.decode_graph, stream=self.graph_stream):
            if self.cfg.graph_full_iteration:
                prefill_state = self.prefill_fn(self.gpu_prompt)
                self.state_buffer.copy_(prefill_state)
                self.current_tokens.copy_(self.gpu_prompt[:, -1])
            for _ in range(self.cfg.decode_tokens):
                logits, next_state, next_token = self.decode_fn(self.current_tokens, self.state_buffer)
                self.state_buffer.copy_(next_state)
                self.graph_logits.copy_(logits)
                self.graph_next_token.copy_(next_token)
                if self.cfg.graph_full_iteration:
                    self.current_tokens.copy_(next_token)
        torch.cuda.synchronize()
        self.graph_includes_prefill = bool(self.cfg.graph_full_iteration)
        self._graph_error = None

    # Core math - NOTE: fp8_autocast should be managed at benchmark_fn level
    # to avoid per-call overhead and memory leaks.
    # All operations use torch.no_grad() since this is inference (no backward pass).
    def _prefill(self, tokens: torch.Tensor) -> torch.Tensor:
        """Prefill phase - fp8_autocast managed externally."""
        with torch.no_grad(), self.sdpa_ctx_factory():
            embeds = self.embedding(tokens)
            hidden = self.prefill_mlp(embeds)
        return hidden[:, -1, :]

    def _decode_step(
        self, tokens: torch.Tensor, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single decode step - fp8_autocast managed externally."""
        with torch.no_grad(), self.sdpa_ctx_factory():
            token_hidden = self.embedding(tokens)
            combined = token_hidden + state
            hidden = self.decode_mlp(combined)
            logits = self.lm_head(hidden)
        next_token = torch.argmax(logits, dim=-1)
        return logits, hidden, next_token
    
    def _get_fp8_context(self):
        """Return fp8_autocast context if FP8 is enabled, else nullcontext."""
        if (self._fp8_enabled or self._fp4_enabled) and te is not None and self.fp8_recipe is not None:
            return te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe)
        return nullcontext()

    # Execution helpers
    def _copy_prompts_to_device(self) -> None:
        non_blocking = bool(self.cfg.use_pinned_host)
        if self.copy_stream is not None:
            with torch.cuda.stream(self.copy_stream):
                self.gpu_prompt.copy_(self.host_prompt, non_blocking=non_blocking)
            torch.cuda.current_stream().wait_stream(self.copy_stream)
        else:
            self.gpu_prompt.copy_(self.host_prompt, non_blocking=non_blocking)
            # CRITICAL: If non_blocking copy with no stream, must synchronize
            # Otherwise gpu_prompt may contain garbage when immediately accessed
            if non_blocking:
                torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        # Timers via CUDA events
        prefill_start = torch.cuda.Event(enable_timing=True)
        prefill_end = torch.cuda.Event(enable_timing=True)
        decode_start = torch.cuda.Event(enable_timing=True)
        decode_end = torch.cuda.Event(enable_timing=True)

        # Choose streams for work/timing
        prefill_stream = self.compute_stream or torch.cuda.current_stream()
        decode_stream = self.graph_stream if self.decode_graph is not None else prefill_stream
        timing_stream = decode_stream

        # NVTX ranges for profiling clarity
        try:
            import torch.cuda.nvtx as nvtx  # type: ignore
        except Exception:
            nvtx = None  # type: ignore

        prefill_start.record(prefill_stream)
        self._copy_prompts_to_device()
        if nvtx:
            nvtx.range_push("prefill")

        # Single FP8 context for entire forward pass to avoid workspace churn
        with self._get_fp8_context():
            # Prefill (or reset when full graph already contains it)
            if self.decode_graph is not None and self.graph_includes_prefill:
                self.state_buffer.zero_()
                self.current_tokens.zero_()
            else:
                prefill_state = self.prefill_fn(self.gpu_prompt)
                self.state_buffer.copy_(prefill_state)
                self.current_tokens.copy_(self.gpu_prompt[:, -1])
            prefill_end.record(prefill_stream)

            # Ensure decode stream waits for prefill when streams differ
            if decode_stream is not prefill_stream:
                decode_stream.wait_event(prefill_end)

            if nvtx:
                nvtx.range_pop()  # prefill
                nvtx.range_push("decode")

            # Decode
            decode_start.record(timing_stream)
            if self.decode_graph is not None:
                # Replay once; graph already captures the decode loop
                if self.graph_stream is not None:
                    with torch.cuda.stream(self.graph_stream):
                        self.decode_graph.replay()
                    torch.cuda.current_stream().wait_stream(self.graph_stream)
                else:
                    self.decode_graph.replay()
            else:
                with torch.cuda.stream(self.compute_stream or torch.cuda.current_stream()):
                    for _ in range(self.cfg.decode_tokens):
                        logits, next_state, next_token = self.decode_fn(self.current_tokens, self.state_buffer)
                        self.state_buffer.copy_(next_state)
                        self.current_tokens.copy_(next_token)
                if self.compute_stream is not None:
                    torch.cuda.current_stream().wait_stream(self.compute_stream)
            decode_end.record(timing_stream)

        if nvtx:
            nvtx.range_pop()

        torch.cuda.synchronize()

        ttft_ms = prefill_end.elapsed_time(prefill_start) if prefill_end.query() else 0.0
        decode_ms = decode_end.elapsed_time(decode_start) if decode_end.query() else 0.0
        total_ms = decode_end.elapsed_time(prefill_start) if decode_end.query() else ttft_ms + decode_ms

        # Defensive clamp to avoid negative/zero timing artifacts
        eps_ms = 1e-6
        ttft_ms = max(ttft_ms, eps_ms)
        decode_ms = max(decode_ms, eps_ms)
        total_ms = max(total_ms, ttft_ms + decode_ms)

        tpot_ms = decode_ms / max(self.cfg.decode_tokens, 1)
        tokens_per_iter = float(self.cfg.batch_size * (self.cfg.prompt_tokens + self.cfg.decode_tokens))
        tokens_per_s = tokens_per_iter / max(total_ms / 1000.0, 1e-6)

        self._custom_metrics = {
            "tokens_per_iteration": tokens_per_iter,
            "prompt_tokens": float(self.cfg.prompt_tokens),
            "decode_tokens": float(self.cfg.decode_tokens),
            "hidden_size": float(self.cfg.hidden_size),
            "use_pinned_host": float(self.cfg.use_pinned_host),
            "use_copy_stream": float(self.cfg.use_copy_stream),
            "use_compute_stream": float(self.cfg.use_compute_stream),
            "use_cuda_graphs": float(self.decode_graph is not None),
            "graph_full_iteration": float(self.graph_includes_prefill),
            "use_torch_compile": float(self.cfg.use_torch_compile and not self._compile_error),
            "use_fp8": float(self._fp8_enabled),
            "fp8_fallback": float(1.0 if (self.cfg.use_fp8 and not self._fp8_enabled) else 0.0),
            "use_fp4": float(self._fp4_enabled),
            "ttft_ms": float(ttft_ms),
            "decode_time_ms": float(decode_ms),
            "tpot_mean_ms": float(tpot_ms),
            "tokens_per_s": float(tokens_per_s),
            "total_time_ms": float(total_ms),
        }
        if self._compile_error:
            self._custom_metrics["compile_error"] = 1.0
        if self._graph_error:
            self._custom_metrics["graph_capture_failed"] = 1.0

    def validate_result(self) -> Optional[str]:
        if torch.isnan(self.state_buffer).any():
            return "NaN detected in decode state"
        return None

    def teardown(self) -> None:
        # Release model buffers between variants to keep allocator usage low.
        for attr in ("embedding", "prefill_mlp", "decode_mlp", "lm_head", "host_prompt", "gpu_prompt", "state_buffer", "current_tokens", "next_token_out", "graph_logits", "graph_next_token"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=self.cfg.iterations,
            warmup=self.cfg.warmup,
            percentiles=[50, 90, 99],
        )


    def get_custom_metrics(self) -> Dict[str, float]:
        return self._custom_metrics
