"""Baseline MoE inference benchmark (sequential prefill + decode on one GPU)."""

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

from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.profiling.gpu_memory_logger import (  # noqa: E402
    GpuMemoryLogger,
    resolve_gpu_log_interval,
    resolve_gpu_log_path,
)
from core.profiling.gpu_telemetry import query_gpu_telemetry  # noqa: E402
from core.optimization.moe_inference import (  # noqa: E402
    MoeInferenceConfig,
    SimpleMoEGPT,
    allocate_kv_cache,
    dtype_bytes,
    env_override_float,
    env_override_int,
)
from ch15.verification_payload_mixin import VerificationPayloadMixin  # noqa: E402


class BaselineMoeInferenceBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Implements the HuggingFace-style baseline described in the MoE showcase doc."""

    def __init__(self) -> None:
        super().__init__()
        self.config = self._build_config()
        self.model: Optional[SimpleMoEGPT] = None
        self.prompts: Optional[torch.Tensor] = None
        self.kv_cache: Optional[torch.Tensor] = None
        self._dtype_bytes = dtype_bytes(self.config.dtype_obj)
        self._history: Dict[str, List[float]] = {
            "ttft": [],
            "tpot": [],
            "throughput": [],
            "nvlink": [],
            "nvlink_measured": [],
            "memory_gb": [],
        }
        self._workload_metadata = WorkloadMetadata(
            requests_per_iteration=float(self.config.batch_size),
            tokens_per_iteration=float(self.config.tokens_per_iteration),
        )
        self.output = None
        self.register_workload_metadata(
            requests_per_iteration=float(self.config.batch_size),
            tokens_per_iteration=float(self.config.tokens_per_iteration),
        )
        self._mem_logger: Optional[GpuMemoryLogger] = None
        self._mem_log_path: Optional[Path] = None
        self._nvlink_warned: bool = False
        self._nvlink_status: str = "unknown"

    def _build_config(self) -> MoeInferenceConfig:
        """Allow environment overrides to keep the workload tractable on smaller GPUs."""
        return MoeInferenceConfig(
            vocab_size=env_override_int("BASELINE_MOE_VOCAB", 16384),
            hidden_size=env_override_int("BASELINE_MOE_HIDDEN", 1024),
            ffn_size=env_override_int("BASELINE_MOE_FFN", 4096),
            num_layers=env_override_int("BASELINE_MOE_LAYERS", 8),
            num_moe_layers=env_override_int("BASELINE_MOE_MOE_LAYERS", 4),
            num_experts=env_override_int("BASELINE_MOE_EXPERTS", 32),
            top_k=2,
            moe_layer_frequency=max(1, env_override_int("BASELINE_MOE_MOE_FREQ", 2)),
            batch_size=env_override_int("BASELINE_MOE_BATCH", 1),
            context_window=env_override_int("BASELINE_MOE_CONTEXT", 512),
            decode_tokens=env_override_int("BASELINE_MOE_DECODE", 64),
            router_noise=env_override_float("BASELINE_MOE_ROUTER_NOISE", 0.0),
            dtype=torch.bfloat16,
        )

    # --------------------------------------------------------------------- setup
    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        cfg = self.config
        self.model = SimpleMoEGPT(cfg, device=self.device).eval()
        self.prompts = torch.randint(
            0,
            cfg.vocab_size,
            (cfg.batch_size, cfg.context_window),
            device=self.device,
        )
        total_tokens = cfg.context_window + cfg.decode_tokens
        self.kv_cache = allocate_kv_cache(
            cfg.batch_size,
            total_tokens,
            cfg.hidden_size,
            cfg.dtype_obj,
            self.device,
        )
        torch.cuda.synchronize(self.device)
        if torch.cuda.is_available() and hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats(self.device)
            log_path = resolve_gpu_log_path(None)
            logger = GpuMemoryLogger(
                device=self.device,
                interval=resolve_gpu_log_interval(1.0),
                log_path=log_path,
            )
            if logger.start():
                self._mem_logger = logger
                self._mem_log_path = log_path

    # --------------------------------------------------------------- benchmark_fn
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
            with self._nvtx_range("baseline_prefill"):
                request_start = time.perf_counter()
                hidden, logits = self.model.prefill(self.prompts, kv_cache=self.kv_cache, cache_start=0)
                torch.cuda.synchronize(self.device)
                ttft_ms = (time.perf_counter() - request_start) * 1000.0
                ttft_times.append(ttft_ms)

            seed_tokens = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

            with self._nvtx_range("baseline_decode"):
                for step in range(cfg.decode_tokens):
                    decode_start = time.perf_counter()
                    _hidden, decode_logits = self.model.decode(
                        seed_tokens,
                        kv_cache=self.kv_cache,
                        position=cfg.context_window + step,
                    )
                    torch.cuda.synchronize(self.device)
                    step_ms = (time.perf_counter() - decode_start) * 1000.0
                    tpot_times.append(step_ms)
                    seed_tokens = torch.argmax(decode_logits[:, -1, :], dim=-1, keepdim=True)
            # Capture the final token ids for verification (payload is set post-timing)
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

    def _compute_nvlink_delta(
        self,
        telemetry_before: Dict[str, Optional[float]],
        telemetry_after: Dict[str, Optional[float]],
        elapsed_s: float,
    ) -> Optional[float]:
        if elapsed_s <= 0:
            return None
        tx_before = telemetry_before.get("nvlink_tx_bytes_total") if telemetry_before else None
        tx_after = telemetry_after.get("nvlink_tx_bytes_total") if telemetry_after else None
        rx_before = telemetry_before.get("nvlink_rx_bytes_total") if telemetry_before else None
        rx_after = telemetry_after.get("nvlink_rx_bytes_total") if telemetry_after else None
        if None in (tx_before, tx_after, rx_before, rx_after):
            return None
        delta_tx = max(0.0, tx_after - tx_before)
        delta_rx = max(0.0, rx_after - rx_before)
        total_delta = delta_tx + delta_rx
        if total_delta <= 0.0:
            return None
        return (total_delta * 8.0) / (elapsed_s * 1e9)

    # ------------------------------------------------------------------ lifecycle
    def teardown(self) -> None:
        self.model = None
        self.prompts = None
        self.kv_cache = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self._mem_logger is not None:
            self._mem_logger.stop()
            self._mem_logger = None

    # ------------------------------------------------------------------- configs
    def get_config(self) -> BenchmarkConfig:
        # This benchmark optimizes end-to-end inference latency (host sync + decode loop),
        # so we intentionally time with a CPU wall-clock timer. Both baseline and
        # optimized variants synchronize GPU work before returning.
        return BenchmarkConfig(
            iterations=8,
            warmup=5,
            device=torch.device("cpu"),
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload_metadata

    def get_custom_metrics(self) -> Optional[dict]:
        """Return inference metrics for moe_inference."""
        from core.benchmark.metrics import compute_inference_metrics
        return compute_inference_metrics(
            ttft_ms=getattr(self, '_ttft_ms', 10.0),
            tpot_ms=getattr(self, '_tpot_ms', 1.0),
            total_tokens=getattr(self, '_total_tokens', 100),
            total_requests=getattr(self, '_total_requests', 1),
            batch_size=getattr(self, 'batch_size', 1),
            max_batch_size=getattr(self, 'max_batch_size', 32),
        )

    def validate_result(self) -> Optional[str]:
        if not self._history["ttft"]:
            return "No TTFT samples recorded"
        if not self._history["tpot"]:
            return "No TPOT samples recorded"
        return None

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return super().get_input_signature()

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return super().get_output_tolerance()

    def capture_verification_payload(self) -> None:
        if self.output is None or self.prompts is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        param_count = sum(p.numel() for p in self.model.parameters()) if self.model is not None else 0
        self._set_verification_payload(
            inputs={"prompt": self.prompts},
            output=self.output.to(dtype=torch.float32),
            batch_size=int(self.prompts.shape[0]),
            parameter_count=param_count,
            precision_flags={
                "fp16": self.config.dtype_obj == torch.float16,
                "bf16": self.config.dtype_obj == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-2, 1e-2),
        )


def get_benchmark() -> BaseBenchmark:
    return BaselineMoeInferenceBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
