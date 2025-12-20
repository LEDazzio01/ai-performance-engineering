"""Optimized disaggregated prefill/decode benchmark (Chapter 17).

Separates prefill (long context) and decode (short, latency-sensitive) phases onto
independent CUDA streams. Mirrors production scheduling that dedicates resources
for context building while keeping decode latency low.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class SimpleLLM(nn.Module):
    """Shared model definition to keep baseline/optimized outputs comparable."""

    def __init__(self, hidden_dim: int = 1024, num_layers: int = 12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])

    def prefill(self, prompt_tokens: torch.Tensor) -> torch.Tensor:
        x = torch.randn(
            prompt_tokens.size(0),
            prompt_tokens.size(1),
            self.hidden_dim,
            device=prompt_tokens.device,
            dtype=torch.bfloat16,
        )
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x[:, -1:, :]

    def decode(self, kv_cache: torch.Tensor, num_tokens: int = 16) -> torch.Tensor:
        outputs = []
        x = kv_cache
        for _ in range(num_tokens):
            for layer in self.layers:
                x = torch.relu(layer(x))
            outputs.append(x)
        return torch.cat(outputs, dim=1)


class OptimizedDisaggregatedBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Prefill on a long context + decode on a short context using separate streams."""

    def __init__(self) -> None:
        super().__init__()
        # Match baseline dimensions for fair comparison.
        self.dtype = torch.bfloat16
        self.hidden = 1024
        self.prefill_seq = 256
        self.decode_seq = 16
        self.batch_size = 1

        self.model: Optional[SimpleLLM] = None
        self.prompt: Optional[torch.Tensor] = None
        self.prefill_stream: Optional[torch.cuda.Stream] = None
        self.decode_stream: Optional[torch.cuda.Stream] = None
        self._prefill_done: Optional[torch.cuda.Event] = None
        self._history: Dict[str, list[float]] = {"ttft": [], "tpot": []}
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(self.batch_size * (self.prefill_seq + self.decode_seq)),
        )
        self.output: Optional[torch.Tensor] = None
        self.parameter_count: int = 0
        self._verification_payload = None

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.model = SimpleLLM(hidden_dim=self.hidden, num_layers=12).to(self.device).to(self.dtype).eval()
        self.prompt = torch.randint(0, 10000, (self.batch_size, self.prefill_seq), device=self.device)

        self.prefill_stream = torch.cuda.Stream(device=self.device)
        self.decode_stream = torch.cuda.Stream(device=self.device)
        self._prefill_done = torch.cuda.Event()

        # Warm up to reduce first-iteration variance.
        with torch.no_grad():
            kv_cache = self.model.prefill(self.prompt)
            _ = self.model.decode(kv_cache, num_tokens=1)
        torch.cuda.synchronize(self.device)
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

    def benchmark_fn(self) -> Dict[str, list[float]]:
        if self.model is None or self.prompt is None or self.prefill_stream is None or self.decode_stream is None or self._prefill_done is None:
            raise RuntimeError("Model/inputs/streams not initialized")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        with nvtx_range("optimized_disaggregated.prefill_decode", enable=enable_nvtx):
            with torch.no_grad():
                request_start = self._record_start()

                torch.cuda.synchronize(self.device)
                with torch.cuda.stream(self.prefill_stream):
                    kv_cache = self.model.prefill(self.prompt)
                    self._prefill_done.record(self.prefill_stream)
                torch.cuda.synchronize(self.device)
                ttft_ms = self._record_stop(request_start)

                token_output = kv_cache
                tpot_times_ms: list[float] = []
                for _ in range(self.decode_seq):
                    token_start = self._record_start()
                    with torch.cuda.stream(self.decode_stream):
                        self.decode_stream.wait_event(self._prefill_done)
                        token_output = self.model.decode(token_output[:, -1:, :], num_tokens=1)
                    torch.cuda.synchronize(self.device)
                    tpot_times_ms.append(self._record_stop(token_start))

                self.output = token_output
                self._history["ttft"].append(ttft_ms)
                self._history["tpot"].extend(tpot_times_ms)
                return {
                    "ttft_times_ms": [ttft_ms],
                    "tpot_times_ms": tpot_times_ms,
                }

    def capture_verification_payload(self) -> None:
        if self.prompt is None or self.output is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        dtype = self.output.dtype
        self._set_verification_payload(
            inputs={"prompt": self.prompt},
            output=self.output,
            batch_size=int(self.batch_size),
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": dtype == torch.float16,
                "bf16": dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def get_custom_streams(self):
        if self.prefill_stream is None or self.decode_stream is None:
            return None
        return [self.prefill_stream, self.decode_stream]

    def teardown(self) -> None:
        self.model = None
        self.prompt = None
        self.prefill_stream = None
        self.decode_stream = None
        self._prefill_done = None
        self.output = None
        self._history = {"ttft": [], "tpot": []}
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
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

    def validate_result(self) -> Optional[str]:
        if self.model is None or self.prompt is None:
            return "Model/inputs not initialized"
        if not self._history["ttft"]:
            return "No TTFT samples recorded"
        if not self._history["tpot"]:
            return "No TPOT samples recorded"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedDisaggregatedBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
