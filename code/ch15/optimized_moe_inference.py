"""Optimized MoE inference benchmark (single-GPU prefill + decode without per-token sync)."""

from __future__ import annotations

from typing import Dict, List

import torch

from ch15.baseline_moe_inference import BaselineMoeInferenceBenchmark
from core.optimization.moe_inference import MoEFeedForward, MoEFeedForwardSortedDispatch
from core.profiling.gpu_telemetry import query_gpu_telemetry


class OptimizedMoeInferenceBenchmark(BaselineMoeInferenceBenchmark):
    """Optimized: remove per-token host sync + wall-clock timing inside the decode loop.

    This keeps the workload equivalent (same model/prompt/cache + greedy decode),
    but avoids synchronizing the GPU every token, which is a common throughput killer.
    """

    def setup(self) -> None:
        """Build the same model/prompt/cache as baseline for a fair comparison."""
        super().setup()
        if self.model is None:
            raise RuntimeError("Model not initialized")

        # Swap MoE FFNs to a faster dispatch strategy (same math, less Python/mask overhead).
        for block in getattr(self.model, "layers", []):
            ff = getattr(block, "ff", None)
            if ff is None or not isinstance(ff, MoEFeedForward):
                continue
            if isinstance(ff, MoEFeedForwardSortedDispatch):
                continue
            replacement = MoEFeedForwardSortedDispatch(
                self.config.hidden_size,
                self.config.ffn_size,
                num_experts=self.config.num_experts,
                top_k=self.config.top_k,
                router_noise=self.config.router_noise,
                capacity_factor=self.config.capacity_factor,
                device=self.device,
                dtype=self.config.dtype_obj,
            )
            replacement.load_state_dict(ff.state_dict(), strict=True)
            block.ff = replacement

    def benchmark_fn(self) -> Dict[str, List[float]]:
        if self.model is None or self.prompts is None or self.kv_cache is None:
            raise RuntimeError("Model, prompts, or KV cache not initialized")

        cfg = self.config
        ttft_times: List[float] = []
        tpot_times: List[float] = []

        if torch.cuda.is_available() and hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats(self.device)
        logical_index = self.device.index if self.device.index is not None else None
        telemetry_before = query_gpu_telemetry(logical_index)

        with torch.no_grad():
            # Time TTFT with CUDA events (one device sync via event).
            start_prefill = torch.cuda.Event(enable_timing=True)
            end_prefill = torch.cuda.Event(enable_timing=True)
            start_prefill.record()
            _hidden, logits = self.model.prefill(self.prompts, kv_cache=self.kv_cache, cache_start=0)
            end_prefill.record()
            end_prefill.synchronize()
            ttft_ms = float(start_prefill.elapsed_time(end_prefill))
            ttft_times.append(ttft_ms)

            seed_tokens = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

            # Time the entire decode loop once (no per-step host sync).
            start_decode = torch.cuda.Event(enable_timing=True)
            end_decode = torch.cuda.Event(enable_timing=True)
            start_decode.record()
            for step in range(cfg.decode_tokens):
                _hidden, decode_logits = self.model.decode(
                    seed_tokens,
                    kv_cache=self.kv_cache,
                    position=cfg.context_window + step,
                )
                seed_tokens = torch.argmax(decode_logits[:, -1, :], dim=-1, keepdim=True)
            end_decode.record()
            end_decode.synchronize()

            total_decode_ms = float(start_decode.elapsed_time(end_decode))
            avg_tpot_ms = total_decode_ms / max(float(cfg.decode_tokens), 1.0)
            tpot_times.extend([avg_tpot_ms] * cfg.decode_tokens)

            self.output = seed_tokens.detach()

        telemetry_after = query_gpu_telemetry(logical_index)

        total_time_s = (sum(ttft_times) + sum(tpot_times)) / 1000.0
        throughput = cfg.tokens_per_iteration / max(total_time_s, 1e-6)
        nvlink_gbps = telemetry_after.get("nvlink_tx_gbps") or 0.0
        measured_nvlink = self._compute_nvlink_delta(telemetry_before, telemetry_after, total_time_s)
        self._nvlink_status = telemetry_after.get("nvlink_status", "unknown")

        self._history["ttft"].extend(ttft_times)
        self._history["tpot"].extend(tpot_times)
        self._history["throughput"].append(throughput)
        self._history["nvlink"].append(nvlink_gbps)
        if measured_nvlink is not None:
            self._history["nvlink_measured"].append(measured_nvlink)
        else:
            if not self._nvlink_warned:
                self._nvlink_warned = True
        if torch.cuda.is_available():
            peak_bytes = torch.cuda.max_memory_allocated(self.device)  # type: ignore[arg-type]
            if peak_bytes:
                self._history["memory_gb"].append(peak_bytes / (1024 ** 3))

        return {
            "ttft_times_ms": ttft_times,
            "tpot_times_ms": tpot_times,
        }


def get_benchmark():
    return OptimizedMoeInferenceBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
