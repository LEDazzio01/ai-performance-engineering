"""Baseline MoE validation: measure loss/throughput without routing guardrails."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from common.python.moe_inference import MoeInferenceConfig, SimpleMoEGPT, allocate_kv_cache  # noqa: E402


class BaselineMoeValidationBenchmark(BaseBenchmark):
    """Runs the MoE stack with minimal telemetry (throughput + average loss)."""

    def __init__(self, config: Optional[MoeInferenceConfig] = None) -> None:
        super().__init__()
        self._config_override = config
        self.config = self._build_config()
        self.model: Optional[SimpleMoEGPT] = None
        self.prompts: Optional[torch.Tensor] = None
        self.labels: Optional[torch.Tensor] = None
        self.kv_cache: Optional[torch.Tensor] = None
        self._history: Dict[str, list] = {"loss": [], "throughput": []}
        self._workload_metadata = WorkloadMetadata(
            requests_per_iteration=float(self.config.batch_size),
            tokens_per_iteration=float(self.config.tokens_per_iteration),
        )

    def apply_target_overrides(self, argv: list[str]) -> None:
        """Allow benchmark_cli --target-extra-arg to override config."""
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--vocab-size", type=int)
        parser.add_argument("--hidden-size", type=int)
        parser.add_argument("--ffn-size", type=int)
        parser.add_argument("--layers", type=int)
        parser.add_argument("--moe-layers", type=int)
        parser.add_argument("--experts", type=int)
        parser.add_argument("--top-k", type=int)
        parser.add_argument("--moe-frequency", type=int)
        parser.add_argument("--batch-size", type=int)
        parser.add_argument("--context-window", type=int)
        parser.add_argument("--decode-tokens", type=int)
        parser.add_argument("--router-noise", type=float)
        args, _ = parser.parse_known_args(argv)
        cfg = self.config
        updated = MoeInferenceConfig(
            vocab_size=args.vocab_size or cfg.vocab_size,
            hidden_size=args.hidden_size or cfg.hidden_size,
            ffn_size=args.ffn_size or cfg.ffn_size,
            num_layers=args.layers or cfg.num_layers,
            num_moe_layers=args.moe_layers or cfg.num_moe_layers,
            num_experts=args.experts or cfg.num_experts,
            top_k=args.top_k or cfg.top_k,
            moe_layer_frequency=args.moe_frequency or cfg.moe_layer_frequency,
            batch_size=args.batch_size or cfg.batch_size,
            context_window=args.context_window or cfg.context_window,
            decode_tokens=args.decode_tokens or cfg.decode_tokens,
            router_noise=args.router_noise if args.router_noise is not None else cfg.router_noise,
            capacity_factor=None,
            dtype=cfg.dtype_obj,
        )
        self.config = updated
        self._workload_metadata = WorkloadMetadata(
            requests_per_iteration=float(self.config.batch_size),
            tokens_per_iteration=float(self.config.tokens_per_iteration),
        )

    def _resolve_device(self) -> torch.device:  # type: ignore[override]
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _build_config(self) -> MoeInferenceConfig:
        """Smaller fixed config so CPU-only runs stay tractable (no env overrides)."""
        if self._config_override is not None:
            return self._config_override
        return MoeInferenceConfig(
            vocab_size=32768,
            hidden_size=1024,
            ffn_size=4096,
            num_layers=6,
            num_moe_layers=3,
            num_experts=16,
            top_k=1,
            moe_layer_frequency=2,
            batch_size=2,
            context_window=512,
            decode_tokens=16,
            router_noise=0.0,
            capacity_factor=None,
            dtype=torch.bfloat16,
        )

    # --------------------------------------------------------------------- setup
    def setup(self) -> None:
        torch.manual_seed(7)
        cfg = self.config
        self.model = SimpleMoEGPT(cfg, device=self.device).eval()
        self.prompts = torch.randint(
            0,
            cfg.vocab_size,
            (cfg.batch_size, cfg.context_window),
            device=self.device,
        )
        total_tokens = cfg.context_window + cfg.decode_tokens
        self.labels = torch.randint(
            0,
            cfg.vocab_size,
            (cfg.batch_size, total_tokens),
            device=self.device,
        )
        self.kv_cache = allocate_kv_cache(
            cfg.batch_size,
            total_tokens,
            cfg.hidden_size,
            cfg.dtype_obj,
            self.device,
        )
        if torch.cuda.is_available() and hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats(self.device)

    # --------------------------------------------------------------- benchmark_fn
    def benchmark_fn(self) -> Dict[str, float]:
        if self.model is None or self.prompts is None or self.labels is None or self.kv_cache is None:
            raise RuntimeError("Model, prompts, labels, or KV cache not initialized")

        cfg = self.config
        with torch.no_grad():
            start = time.perf_counter()
            hidden, logits = self.model.prefill(self.prompts, kv_cache=self.kv_cache, cache_start=0)
            token_loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size),
                self.labels[:, : cfg.context_window].reshape(-1),
            )

            seed_tokens = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            decode_losses = []
            for step in range(cfg.decode_tokens):
                _, decode_logits = self.model.decode(
                    seed_tokens,
                    kv_cache=self.kv_cache,
                    position=cfg.context_window + step,
                )
                step_loss = F.cross_entropy(
                    decode_logits.reshape(-1, cfg.vocab_size),
                    self.labels[:, cfg.context_window + step].reshape(-1),
                )
                decode_losses.append(step_loss)
                seed_tokens = torch.argmax(decode_logits[:, -1, :], dim=-1, keepdim=True)

            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            elapsed_s = max(time.perf_counter() - start, 1e-6)

        avg_loss = float(
            token_loss.item()
            + (sum(dl.item() for dl in decode_losses) / max(len(decode_losses), 1))
        )
        throughput = cfg.tokens_per_iteration / elapsed_s
        self._history["loss"].append(avg_loss)
        self._history["throughput"].append(throughput)

        return {"loss": avg_loss, "elapsed_s": elapsed_s}

    # ------------------------------------------------------------------- configs
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=6, warmup=1)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload_metadata

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        if not self._history["loss"]:
            return None
        return {
            "baseline_moe_val.loss": float(statistics.mean(self._history["loss"])),
            "baseline_moe_val.throughput_tok_s": float(statistics.mean(self._history["throughput"])),
        }

    def validate_result(self) -> Optional[str]:
        if not self._history["throughput"]:
            return "No throughput samples recorded"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineMoeValidationBenchmark()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline MoE validation benchmark.")
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--ffn-size", type=int, default=4096)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--moe-layers", type=int, default=3)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--moe-frequency", type=int, default=2, help="Every N layers is MoE.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--context-window", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--router-noise", type=float, default=0.0)
    args = parser.parse_args()

    cfg = MoeInferenceConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        ffn_size=args.ffn_size,
        num_layers=args.layers,
        num_moe_layers=args.moe_layers,
        num_experts=args.experts,
        top_k=args.top_k,
        moe_layer_frequency=args.moe_frequency,
        batch_size=args.batch_size,
        context_window=args.context_window,
        decode_tokens=args.decode_tokens,
        router_noise=args.router_noise,
        capacity_factor=None,
        dtype=torch.bfloat16,
    )
    bench = BaselineMoeValidationBenchmark(config=cfg)
    bench.setup()
    bench.benchmark_fn()
    print(bench.get_custom_metrics())
