#!/usr/bin/env python3
"""Baseline: vLLM v1 integration without optimization.

Demonstrates basic vLLM v1 usage without advanced features like:
- Bucketed CUDA graphs
- Optimized KV cache management
- Prefix caching
"""

import torch
import time
from typing import Dict, Any, List, Optional
import random
import sys
from pathlib import Path

# Ensure the hack/numba stub is importable before vLLM touches numba.
repo_root = Path(__file__).resolve().parents[1]
hack_path = repo_root / "hack"
if str(hack_path) not in sys.path:
    sys.path.insert(0, str(hack_path))
# Import numba (will resolve to hack/numba) so vLLM sees a compatible module.
import numba  # noqa: F401

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.utils.logger import get_logger

logger = get_logger(__name__)

# Check for vLLM
try:
    from vllm import LLM, SamplingParams
    from vllm.inputs.data import TokensPrompt
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not available, using simulation mode")


class BaselineVLLMV1Integration:
    """Baseline vLLM v1 without optimizations."""
    
    def __init__(
        self,
        model_name: str = "facebook/opt-125m",
        max_tokens: int = 128,
        batch_size: int = 8,
        use_vllm: bool = True,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.use_vllm = use_vllm and VLLM_AVAILABLE
        
        if not self.use_vllm:
            logger.info("Running in simulation mode (vLLM not available)")
    
    def setup(self):
        """Initialize vLLM model."""
        random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        if self.use_vllm:
            import gc
            
            # Force cleanup before vLLM initialization to avoid resource conflicts
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            try:
                # Baseline: No CUDA graphs, no prefix caching
                self.llm = LLM(
                    model=self.model_name,
                    enforce_eager=True,  # Disable CUDA graphs (baseline)
                    enable_prefix_caching=False,  # Disable prefix caching
                    enable_chunked_prefill=True,
                    gpu_memory_utilization=0.7,  # Lower to avoid OOM during engine init
                    dtype="bfloat16",
                    tensor_parallel_size=1,  # Single GPU to avoid coordination issues
                    max_model_len=512,  # Limit context length to reduce memory
                )
                
                logger.info(f"Loaded model: {self.model_name}")
                logger.info("Baseline config: eager execution, no prefix caching")
            except RuntimeError as e:
                error_msg = str(e)
                if "Engine core initialization failed" in error_msg:
                    raise RuntimeError(
                        f"SKIPPED: vLLM engine initialization failed - likely due to GPU resource "
                        f"contention from previous benchmark. Original error: {error_msg}"
                    ) from e
                raise

        if not self.use_vllm:
            raise RuntimeError("vLLM required for this benchmark. Install with: pip install vllm")

        # Build pre-tokenized prompts so CPU tokenization doesn't dominate, and so
        # prefix caching (in the optimized variant) can skip meaningful prefill.
        tokenizer = self.llm.get_tokenizer()
        base_ids = tokenizer.encode("Once upon a time in a land far away, ", add_special_tokens=False)
        if not base_ids:
            raise RuntimeError("Tokenizer returned empty token IDs for baseline prefix")

        max_prompt_len = 512 - self.max_tokens
        suffix_ids: List[List[int]] = []
        for i in range(self.batch_size):
            ids = tokenizer.encode(f"there was a {i}.", add_special_tokens=False)
            if not ids:
                raise RuntimeError(f"Tokenizer returned empty token IDs for suffix {i}")
            suffix_ids.append(ids)
        max_suffix = max(len(ids) for ids in suffix_ids)
        if max_suffix >= max_prompt_len:
            raise RuntimeError("Suffix length exceeds max prompt length; reduce max_tokens or suffix text.")

        target_prefix_len = max_prompt_len - max_suffix
        repeats = (target_prefix_len + len(base_ids) - 1) // len(base_ids)
        prefix_ids = (base_ids * repeats)[:target_prefix_len]

        self.prompts = [
            TokensPrompt(prompt_token_ids=prefix_ids + suffix_ids[i])
            for i in range(self.batch_size)
        ]
        
        # Sampling parameters
        self.sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=42,
        )
    
    def run(self) -> Dict[str, float]:
        """Execute baseline vLLM inference."""
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # Generate
        outputs = self.llm.generate(self.prompts, self.sampling_params, use_tqdm=False)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        elapsed = end - start
        
        # Calculate metrics
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        first_ids = outputs[0].outputs[0].token_ids if outputs else []
        token_ids = list(first_ids[:16])
        throughput = total_tokens / elapsed
        mean_latency_ms = (elapsed / len(self.prompts)) * 1000
        
        logger.info(f"Throughput: {throughput:.2f} tokens/sec")
        logger.info(f"Mean latency: {mean_latency_ms:.2f} ms")
        
        return {
            "mean_latency_ms": mean_latency_ms,
            "throughput_tokens_per_sec": throughput,
            "total_tokens": total_tokens,
            "token_ids": token_ids,
        }
    
    def cleanup(self):
        """Clean up resources."""
        if self.use_vllm and hasattr(self, 'llm'):
            del self.llm
        torch.cuda.empty_cache()


class BaselineVLLMV1IntegrationBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Harness wrapper for the baseline vLLM integration."""

    def __init__(self):
        super().__init__()
        self.runner = BaselineVLLMV1Integration()
        self._metrics: Dict[str, Any] = {}
        self.output: Optional[torch.Tensor] = None
        self._last_token_ids: Optional[torch.Tensor] = None
        self._verification_payload = None
        self.register_workload_metadata(requests_per_iteration=8.0)

    def setup(self) -> None:
        self.runner.setup()
        self._metrics = {}
        self.output = None
        self._last_token_ids = None

    def benchmark_fn(self) -> None:
        self._metrics = self.runner.run()
        token_ids = self._metrics.get("token_ids")
        if token_ids is None:
            raise RuntimeError("Runner did not return token_ids for verification")
        self._last_token_ids = torch.as_tensor(token_ids, dtype=torch.int32)
        self.output = self._last_token_ids
        self._synchronize()

    def capture_verification_payload(self) -> None:
        if self._last_token_ids is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={
                "batch_size": torch.tensor(self.runner.batch_size),
                "max_tokens": torch.tensor(self.runner.max_tokens),
            },
            output=self.output,
            batch_size=self.runner.batch_size,
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": True, "fp8": False, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(0.0, 0.0),
        )

    def teardown(self) -> None:
        self.runner.cleanup()
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=3,
            warmup=1,
            use_subprocess=True,
            setup_timeout_seconds=600,
            measurement_timeout_seconds=600,
            timing_method="wall_clock",
        )

    def get_custom_metrics(self) -> Dict[str, Any]:
        return self._metrics


def get_benchmark() -> BaseBenchmark:
    return BaselineVLLMV1IntegrationBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
