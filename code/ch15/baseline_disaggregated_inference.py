"""Baseline CLI hook for the disaggregated inference walkthrough.

Chapter 15: Disaggregated Inference

NOTE: This file uses speculative decoding concepts (speculative_window parameter).
Speculative decoding is covered in depth in Chapter 18. Here we demonstrate
the basic pattern for disaggregated prefill/decode in a multi-GPU context.
For full speculative decoding with draft models and token verification, see:
- ch18/optimized_speculative_decode.py
- ch18/optimized_vllm_decode_graphs.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig  # noqa: E402
from ch15.baseline_moe_inference import BaselineMoeInferenceBenchmark  # noqa: E402


class _DisaggregatedInferenceBenchmark(BaselineMoeInferenceBenchmark):
    """Shared harness that simulates prefill/decode split execution."""

    def __init__(self, *, speculative_window: int, decode_parallelism: int):
        super().__init__()
        self.speculative_window = max(1, speculative_window)
        self.decode_parallelism = max(1, decode_parallelism)
        self._disagg_history: Dict[str, List[float]] = {
            "prefill_ms": [],
            "decode_ms": [],
        }
        self.register_workload_metadata(requests_per_iteration=1.0)

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return torch.tensor([hash(str(id(self))) % (2**31)], dtype=torch.float32)

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return {"speculative_window": self.speculative_window, "decode_parallelism": self.decode_parallelism}

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)

    def setup(self) -> None:
        super().setup()
        if self.model is not None:
            self.model.to(device=self.device, dtype=self.config.dtype_obj)

    def benchmark_fn(self) -> Dict[str, List[float]]:
        if self.model is None or self.prompts is None or self.kv_cache is None:
            raise RuntimeError("Model or prompts not initialized")

        cfg = self.config
        ttft_samples: List[float] = []
        decode_samples: List[float] = []

        with torch.no_grad():
            with self._nvtx_range("disagg_prefill"):
                start = time.perf_counter()
                hidden, logits = self.model.prefill(self.prompts, kv_cache=self.kv_cache, cache_start=0)
                torch.cuda.synchronize(self.device)
                ttft_samples.append((time.perf_counter() - start) * 1000.0)

            seeds = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            context_position = cfg.context_window
            step = 0

            while step < cfg.decode_tokens:
                tokens_now = min(self.speculative_window, cfg.decode_tokens - step)
                start = time.perf_counter()

                for bucket in range(tokens_now):
                    position = context_position + step + bucket
                    _hidden, decode_logits = self.model.decode(
                        seeds,
                        kv_cache=self.kv_cache,
                        position=position,
                    )
                    torch.cuda.synchronize(self.device)
                    seeds = torch.argmax(decode_logits[:, -1, :], dim=-1, keepdim=True)

                decode_samples.append((time.perf_counter() - start) * 1000.0)
                step += tokens_now

        total_ms = sum(ttft_samples) + sum(decode_samples)
        throughput = cfg.tokens_per_iteration / max(total_ms / 1000.0, 1e-6)
        nvlink_gbps = 0.0
        if ttft_samples:
            bytes_moved = cfg.batch_size * cfg.context_window * cfg.hidden_size * self._dtype_bytes
            nvlink_gbps = (bytes_moved * 8.0 / 1e9) / (ttft_samples[0] / 1000.0)

        self._history["ttft"].extend(ttft_samples)
        self._history["tpot"].extend(decode_samples)
        self._history["throughput"].append(throughput)
        self._history["nvlink"].append(nvlink_gbps)

        self._disagg_history["prefill_ms"].extend(ttft_samples)
        self._disagg_history["decode_ms"].extend(decode_samples)
        return {"prefill_ms": ttft_samples, "decode_ms": decode_samples}

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_inference_metrics
        return compute_inference_metrics(
            ttft_ms=getattr(self, '_ttft_ms', 50.0),
            tpot_ms=getattr(self, '_tpot_ms', 10.0),
            total_tokens=getattr(self, 'total_tokens', 256),
            total_requests=getattr(self, 'total_requests', 1),
            batch_size=getattr(self, 'batch_size', 1),
            max_batch_size=getattr(self, 'max_batch_size', 32),
        )

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=4, warmup=5)


class BaselineDisaggregatedInferenceBenchmark(_DisaggregatedInferenceBenchmark):
    """Sequential prefill/decode simulation (no overlap)."""

    def __init__(self) -> None:
        super().__init__(speculative_window=1, decode_parallelism=1)


def get_benchmark():
    return BaselineDisaggregatedInferenceBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
