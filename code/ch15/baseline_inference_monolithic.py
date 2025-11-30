"""baseline_inference_monolithic.py - Monolithic inference (baseline).

Single service handles both prefill and decode - blocks each other.
Implements BaseBenchmark for harness integration.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class SimpleLLM(nn.Module):
    """Simplified LLM for inference simulation."""
    
    def __init__(self, hidden_dim=1024, num_layers=12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
    
    def prefill(self, prompt_tokens):
        """Prefill: Process full prompt (compute-bound)."""
        x = torch.randn(prompt_tokens.size(0), prompt_tokens.size(1), self.hidden_dim,
                       device=prompt_tokens.device, dtype=torch.bfloat16)
        for layer in self.layers:
            x = layer(x)
            x = torch.relu(x)
        return x[:, -1:, :]
    
    def decode(self, kv_cache, num_tokens=16):
        """Decode: Generate tokens (memory-bound)."""
        outputs = []
        x = kv_cache
        for _ in range(num_tokens):
            for layer in self.layers:
                x = layer(x)
                x = torch.relu(x)
            outputs.append(x)
        return torch.cat(outputs, dim=1)


class BaselineInferenceMonolithicBenchmark(BaseBenchmark):
    """Monolithic inference baseline using the shared harness conventions."""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[SimpleLLM] = None
        self.prompt: Optional[torch.Tensor] = None
        self.kv_cache: Optional[torch.Tensor] = None
        self._history: Dict[str, List[float]] = {"ttft": [], "tpot": []}
        # Workload dimensions for signature matching
        self.batch_size = 1
        self.prefill_seq = 256
        self.num_tokens = 16
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=self.prefill_seq + self.num_tokens,
        )
    
    def setup(self) -> None:
        """Setup: initialize model and data."""
        self.model = SimpleLLM(hidden_dim=1024, num_layers=12).to(self.device).to(torch.bfloat16).eval()
        self.prompt = torch.randint(0, 10000, (1, 256), device=self.device)
        
        with torch.no_grad():
            self.kv_cache = self.model.prefill(self.prompt)
        torch.cuda.synchronize(self.device)
    
    def benchmark_fn(self) -> Optional[dict]:
        if self.model is None or self.prompt is None:
            raise RuntimeError("Model or prompt not initialized")

        enable_nvtx = get_nvtx_enabled(self.get_config())

        with nvtx_range("inference_monolithic", enable=enable_nvtx):
            with torch.no_grad():
                request_start = self._record_start()
                
                torch.cuda.synchronize(self.device)
                prefill_start = self._record_start()
                kv_cache = self.model.prefill(self.prompt)
                torch.cuda.synchronize(self.device)
                ttft_ms = self._record_stop(request_start)
                
                num_tokens = 16
                tpot_times_ms = []
                
                for i in range(num_tokens):
                    token_start = self._record_start()
                    if i == 0:
                        token_output = self.model.decode(kv_cache, num_tokens=1)
                    else:
                        token_output = self.model.decode(token_output[:, -1:, :], num_tokens=1)
                    torch.cuda.synchronize(self.device)
                    tpot_times_ms.append(self._record_stop(token_start))
                
                self._history["ttft"].append(ttft_ms)
                self._history["tpot"].extend(tpot_times_ms)
                return {
                    "ttft_times_ms": [ttft_ms],
                    "tpot_times_ms": tpot_times_ms,
                }

    def teardown(self) -> None:
        self.model = None
        self.prompt = None
        self.kv_cache = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        if not self._history["ttft"]:
            return None
        return {
            "monolithic.ttft_ms": float(sum(self._history["ttft"]) / len(self._history["ttft"])),
            "monolithic.tpot_mean_ms": float(sum(self._history["tpot"]) / len(self._history["tpot"])),
        }

    def validate_result(self) -> Optional[str]:
        if not self._history["ttft"]:
            return "No TTFT samples recorded"
        if not self._history["tpot"]:
            return "No TPOT samples recorded"
        return None

    def get_input_signature(self) -> dict:
        """Return workload signature for input verification."""
        return {
            "batch_size": self.batch_size,
            "num_tokens": self.num_tokens,
        }


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselineInferenceMonolithicBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=20, warmup=5)
    )
    benchmark = BaselineInferenceMonolithicBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print("Baseline: Monolithic Inference")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
