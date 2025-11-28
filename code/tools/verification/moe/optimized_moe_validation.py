"""Optimized MoE validation: log routing health + throughput across sweeps."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

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


def compute_gini(counts: torch.Tensor) -> float:
    counts = counts.to(torch.float32)
    if counts.numel() == 0:
        return 0.0
    total = counts.sum()
    if total <= 0:
        return 0.0
    sorted_counts, _ = torch.sort(counts)
    n = counts.numel()
    index = torch.arange(1, n + 1, dtype=torch.float32, device=counts.device)
    gini = 1.0 + 1.0 / n - 2.0 * torch.sum((n + 1 - index) * sorted_counts) / (n * total)
    return float(gini)


class MoEStatsLogger:
    def __init__(self, num_experts: int) -> None:
        self.num_experts = num_experts
        self.reset()

    def reset(self) -> None:
        self.expert_counts = torch.zeros(self.num_experts, dtype=torch.long)
        self.overflow_tokens = 0
        self.total_tokens = 0
        self.entropy: List[float] = []

    def update(self, stats: Dict[str, torch.Tensor]) -> None:
        if not stats:
            return
        expert_indices = stats.get("expert_indices")
        if expert_indices is not None:
            flat = expert_indices.reshape(-1)
            valid = (flat >= 0) & (flat < self.num_experts)
            if valid.any():
                self.expert_counts += torch.bincount(
                    flat[valid],
                    minlength=self.num_experts,
                ).cpu()
            self.total_tokens += int(expert_indices.shape[0])

        overflow_mask = stats.get("overflow_mask")
        if overflow_mask is not None:
            self.overflow_tokens += int(overflow_mask.sum().item())

        entropy_val = stats.get("router_entropy")
        if entropy_val is not None:
            self.entropy.append(float(entropy_val))

    def summarize(self) -> Dict[str, float]:
        overflow_rate = (
            self.overflow_tokens / self.total_tokens if self.total_tokens > 0 else 0.0
        )
        gini = compute_gini(self.expert_counts)
        entropy = statistics.mean(self.entropy) if self.entropy else 0.0
        return {
            "overflow_rate": float(overflow_rate),
            "gini": float(gini),
            "router_entropy": float(entropy),
        }


def _set_router_config(model: SimpleMoEGPT, top_k: int, capacity_factor: float) -> None:
    for block in model.layers:
        ff = getattr(block, "ff", None)
        if hasattr(ff, "top_k"):
            ff.top_k = top_k  # type: ignore[attr-defined]
        if hasattr(ff, "capacity_factor"):
            ff.capacity_factor = capacity_factor  # type: ignore[attr-defined]


class OptimizedMoeValidationBenchmark(BaseBenchmark):
    """Runs routing sweeps and reports overflow/Gini/entropy along with throughput."""

    def __init__(
        self,
        config: Optional[MoeInferenceConfig] = None,
        k_values: Optional[List[int]] = None,
        capacity_factors: Optional[List[float]] = None,
        eval_seeds: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self._config_override = config
        self.config = self._build_config()
        self.k_values = k_values or [1, 2]
        self.capacity_factors = capacity_factors or [1.0, 1.25, 1.5]
        self.eval_seeds = eval_seeds or [3, 13]
        self.model: Optional[SimpleMoEGPT] = None
        self._records: List[Dict[str, float]] = []
        self._workload_metadata = WorkloadMetadata(
            requests_per_iteration=float(self.config.batch_size),
            tokens_per_iteration=float(self.config.tokens_per_iteration),
        )

    def apply_target_overrides(self, argv: list[str]) -> None:
        """Allow aisp bench --target-extra-arg to tweak sweep knobs."""
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
        parser.add_argument("--capacity-factor", type=float)
        parser.add_argument("--k-values", type=str)
        parser.add_argument("--capacity-factors", type=str)
        parser.add_argument("--seeds", type=str)
        args, _ = parser.parse_known_args(argv)

        def _parse_csv(raw: Optional[str], cast) -> Optional[List]:
            if not raw:
                return None
            return [cast(item.strip()) for item in raw.split(",") if item.strip()]

        cfg = self.config
        updated_cfg = MoeInferenceConfig(
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
            capacity_factor=args.capacity_factor if args.capacity_factor is not None else cfg.capacity_factor,
            dtype=cfg.dtype_obj,
        )
        self.config = updated_cfg
        parsed_k = _parse_csv(args.k_values, int)
        parsed_cf = _parse_csv(args.capacity_factors, float)
        parsed_seeds = _parse_csv(args.seeds, int)
        if parsed_k:
            self.k_values = parsed_k
        if parsed_cf:
            self.capacity_factors = parsed_cf
        if parsed_seeds:
            self.eval_seeds = parsed_seeds
        self._workload_metadata = WorkloadMetadata(
            requests_per_iteration=float(self.config.batch_size),
            tokens_per_iteration=float(self.config.tokens_per_iteration),
        )

    def _resolve_device(self) -> torch.device:  # type: ignore[override]
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _build_config(self) -> MoeInferenceConfig:
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
        torch.manual_seed(21)
        self.model = SimpleMoEGPT(self.config, device=self.device).eval()
        if torch.cuda.is_available() and hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats(self.device)

    def _make_batch(self, seed: int) -> Dict[str, torch.Tensor]:
        torch.manual_seed(seed)
        cfg = self.config
        prompts = torch.randint(
            0,
            cfg.vocab_size,
            (cfg.batch_size, cfg.context_window),
            device=self.device,
        )
        total_tokens = cfg.context_window + cfg.decode_tokens
        labels = torch.randint(
            0,
            cfg.vocab_size,
            (cfg.batch_size, total_tokens),
            device=self.device,
        )
        return {"prompts": prompts, "labels": labels}

    def _run_once(
        self,
        prompts: torch.Tensor,
        labels: torch.Tensor,
        top_k: int,
        capacity_factor: float,
    ) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("Model not initialized")
        _set_router_config(self.model, top_k=top_k, capacity_factor=capacity_factor)
        moe_logger = MoEStatsLogger(num_experts=self.config.num_experts)
        cfg = self.config
        total_tokens = cfg.context_window + cfg.decode_tokens
        kv_cache = allocate_kv_cache(
            cfg.batch_size,
            total_tokens,
            cfg.hidden_size,
            cfg.dtype_obj,
            self.device,
        )

        with torch.no_grad():
            start = time.perf_counter()
            hidden, logits, router_stats = self.model.prefill(
                prompts,
                kv_cache=kv_cache,
                cache_start=0,
                output_router_stats=True,
            )
            token_loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size),
                labels[:, : cfg.context_window].reshape(-1),
            )
            for stats in router_stats:
                moe_logger.update(stats)
            seed_tokens = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            decode_losses: List[torch.Tensor] = []
            for step in range(cfg.decode_tokens):
                _, decode_logits, decode_stats = self.model.decode(
                    seed_tokens,
                    kv_cache=kv_cache,
                    position=cfg.context_window + step,
                    output_router_stats=True,
                )
                step_loss = F.cross_entropy(
                    decode_logits.reshape(-1, cfg.vocab_size),
                    labels[:, cfg.context_window + step].reshape(-1),
                )
                decode_losses.append(step_loss)
                for stats in decode_stats:
                    moe_logger.update(stats)
                seed_tokens = torch.argmax(decode_logits[:, -1, :], dim=-1, keepdim=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            elapsed_s = max(time.perf_counter() - start, 1e-6)

        summary = moe_logger.summarize()
        avg_decode_loss = (
            sum(loss.item() for loss in decode_losses) / max(len(decode_losses), 1)
            if decode_losses
            else 0.0
        )
        avg_loss = float(token_loss.item() + avg_decode_loss)
        record = {
            "top_k": float(top_k),
            "capacity_factor": float(capacity_factor),
            "loss": avg_loss,
            "tokens_per_sec": float(cfg.tokens_per_iteration) / elapsed_s,
            "overflow_rate": summary["overflow_rate"],
            "gini": summary["gini"],
            "router_entropy": summary["router_entropy"],
        }
        return record

    # --------------------------------------------------------------- benchmark_fn
    def benchmark_fn(self) -> List[Dict[str, float]]:
        self._records = []
        for seed in self.eval_seeds:
            batch = self._make_batch(seed)
            for k in self.k_values:
                for cf in self.capacity_factors:
                    record = self._run_once(batch["prompts"], batch["labels"], k, cf)
                    record["seed"] = float(seed)
                    self._records.append(record)
        return self._records

    # ------------------------------------------------------------------- configs
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=1, warmup=0, measurement_timeout_seconds=120)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload_metadata

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        if not self._records:
            return None
        best = max(self._records, key=lambda r: r["tokens_per_sec"])
        overflow_mean = statistics.mean(r["overflow_rate"] for r in self._records)
        gini_mean = statistics.mean(r["gini"] for r in self._records)
        loss_std = statistics.pstdev(r["loss"] for r in self._records) if len(self._records) > 1 else 0.0
        return {
            "optimized_moe_val.best_tok_s": float(best["tokens_per_sec"]),
            "optimized_moe_val.best_loss": float(best["loss"]),
            "optimized_moe_val.avg_overflow": float(overflow_mean),
            "optimized_moe_val.avg_gini": float(gini_mean),
            "optimized_moe_val.loss_seed_std": float(loss_std),
        }

    def validate_result(self) -> Optional[str]:
        if not self._records:
            return "No sweep data captured"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedMoeValidationBenchmark()


if __name__ == "__main__":
    def _parse_csv(raw: str, cast) -> List:
        return [cast(item.strip()) for item in raw.split(",") if item.strip()]

    parser = argparse.ArgumentParser(description="Optimized MoE validation sweeps.")
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
    parser.add_argument("--capacity-factor", type=float, default=0.0)
    parser.add_argument("--k-values", type=str, default="1,2", help="Comma-separated list.")
    parser.add_argument("--capacity-factors", type=str, default="1.0,1.25,1.5", help="Comma-separated list.")
    parser.add_argument("--seeds", type=str, default="3,13", help="Comma-separated list.")
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
        capacity_factor=None if args.capacity_factor == 0.0 else args.capacity_factor,
        dtype=torch.bfloat16,
    )
    k_vals = _parse_csv(args.k_values, int)
    cf_vals = _parse_csv(args.capacity_factors, float)
    seed_vals = _parse_csv(args.seeds, int)

    bench = OptimizedMoeValidationBenchmark(
        config=cfg,
        k_values=k_vals,
        capacity_factors=cf_vals,
        eval_seeds=seed_vals,
    )
    bench.setup()
    results = bench.benchmark_fn()
    print(results)
