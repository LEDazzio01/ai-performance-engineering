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

    def __init__(
        self,
        *,
        speculative_window: int,
        decode_parallelism: int,
        overlap_kv_transfer: bool,
        transfer_stride: int,
    ):
        super().__init__()
        self.speculative_window = max(1, speculative_window)
        self.decode_parallelism = max(1, decode_parallelism)
        self.overlap_kv_transfer = bool(overlap_kv_transfer)
        self.transfer_stride = max(1, transfer_stride)
        # Favor transfer-bound behavior to make batching gains visible.
        self.config.num_layers = 2
        self.config.num_moe_layers = 1
        self.config.ffn_size = 1024
        self.config.num_experts = 8
        self.config.context_window = 256
        self.config.decode_tokens = 256
        self.output = None
        self._disagg_history: Dict[str, List[float]] = {
            "prefill_ms": [],
            "decode_ms": [],
        }
        self._kv_transfer_buffer: Optional[torch.Tensor] = None
        self._kv_transfer_stream: Optional[torch.cuda.Stream] = None
        self._kv_transfer_tokens: int = 0
        self.register_workload_metadata(requests_per_iteration=1.0)

    def setup(self) -> None:
        super().setup()
        if self.model is not None:
            self.model.to(device=self.device, dtype=self.config.dtype_obj)
        self._kv_transfer_tokens = max(1, min(self.transfer_stride, self.config.decode_tokens))
        self._kv_transfer_buffer = torch.empty(
            (self.config.batch_size, self._kv_transfer_tokens, self.config.hidden_size),
            device="cpu",
            dtype=self.config.dtype_obj,
            pin_memory=True,
        )
        if self.device.type == "cuda" and self.overlap_kv_transfer:
            self._kv_transfer_stream = torch.cuda.Stream()

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
            window = max(1, int(self.speculative_window) * int(self.decode_parallelism))

            if window > 1:
                start_decode = torch.cuda.Event(enable_timing=True)
                end_decode = torch.cuda.Event(enable_timing=True)
                start_decode.record()
                while step < cfg.decode_tokens:
                    tokens_now = min(window, cfg.decode_tokens - step)
                    for bucket in range(tokens_now):
                        position = context_position + step + bucket
                        _hidden, decode_logits = self.model.decode(
                            seeds,
                            kv_cache=self.kv_cache,
                            position=position,
                        )
                        seeds = torch.argmax(decode_logits[:, -1, :], dim=-1, keepdim=True)
                    if self._kv_transfer_buffer is not None and self.kv_cache is not None:
                        transfer_start = context_position + step
                        transfer_end = transfer_start + tokens_now
                        kv_slice = self.kv_cache[:, transfer_start:transfer_end, :]
                        if self._kv_transfer_stream is not None:
                            self._kv_transfer_stream.wait_stream(torch.cuda.current_stream(self.device))
                            with torch.cuda.stream(self._kv_transfer_stream):
                                self._kv_transfer_buffer[:, :tokens_now].copy_(kv_slice, non_blocking=True)
                        else:
                            self._kv_transfer_buffer[:, :tokens_now].copy_(kv_slice, non_blocking=False)
                    step += tokens_now
                if self._kv_transfer_stream is not None:
                    torch.cuda.current_stream(self.device).wait_stream(self._kv_transfer_stream)
                end_decode.record()
                end_decode.synchronize()
                total_decode_ms = float(start_decode.elapsed_time(end_decode))
                avg_tpot_ms = total_decode_ms / max(float(cfg.decode_tokens), 1.0)
                decode_samples.extend([avg_tpot_ms] * cfg.decode_tokens)
            else:
                while step < cfg.decode_tokens:
                    tokens_now = min(window, cfg.decode_tokens - step)
                    start = time.perf_counter()

                    for bucket in range(tokens_now):
                        position = context_position + step + bucket
                        _hidden, decode_logits = self.model.decode(
                            seeds,
                            kv_cache=self.kv_cache,
                            position=position,
                        )
                        seeds = torch.argmax(decode_logits[:, -1, :], dim=-1, keepdim=True)

                    if self._kv_transfer_buffer is not None and self.kv_cache is not None:
                        transfer_start = context_position + step
                        transfer_end = transfer_start + tokens_now
                        kv_slice = self.kv_cache[:, transfer_start:transfer_end, :]
                        self._kv_transfer_buffer[:, :tokens_now].copy_(kv_slice, non_blocking=False)

                    torch.cuda.synchronize(self.device)
                    decode_samples.append((time.perf_counter() - start) * 1000.0)
                    step += tokens_now
        
        # Capture output for verification (final token predictions)
        self.output = seeds.detach()
        self._synchronize()

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
        if not self._disagg_history["prefill_ms"] or not self._disagg_history["decode_ms"]:
            raise RuntimeError("Disaggregated inference metrics require timing samples")
        return compute_inference_metrics(
            ttft_ms=float(statistics.mean(self._disagg_history["prefill_ms"])),
            tpot_ms=float(statistics.mean(self._disagg_history["decode_ms"])),
            total_tokens=int(self.config.tokens_per_iteration),
            total_requests=int(self.config.batch_size),
            batch_size=int(self.config.batch_size),
            max_batch_size=int(self.config.batch_size),
        )

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=4, warmup=5)


class BaselineDisaggregatedInferenceBenchmark(_DisaggregatedInferenceBenchmark):
    """Sequential prefill/decode simulation (no overlap)."""

    def __init__(self) -> None:
        super().__init__(
            speculative_window=1,
            decode_parallelism=1,
            overlap_kv_transfer=False,
            transfer_stride=1,
        )


def get_benchmark():
    return BaselineDisaggregatedInferenceBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
