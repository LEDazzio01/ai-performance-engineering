"""
Load testing harness for the 8x B200 inference server.

Run with torchrun:
    torchrun --nproc_per_node=8 ch16/inference_server_load_test.py \
        --duration 60 --target-qps 400 --output-json results.json

The harness feeds synthetic requests (random prompt lengths, configurable QPS)
into the `InferenceServer8GPU` implementation and collects latency/throughput
statistics. All ranks receive the same request stream via broadcast so the
continuous batching state remains consistent across tensor-parallel workers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist

from ch16.inference_serving_8xb200 import (
    DemoCausalLM,
    InferenceRequest,
    InferenceServer8GPU,
    RequestState,
)


@dataclass
class CompletionRecord:
    request_id: str
    prompt_tokens: int
    generated_tokens: int
    latency_ms: float


def _init_distributed() -> None:
    if not dist.is_initialized():
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            backend = "nccl"
            dist.init_process_group(backend=backend, device_id=local_rank)
        else:
            backend = "gloo"
            dist.init_process_group(backend=backend)


def _broadcast_requests(requests: Optional[List[Dict]]) -> List[Dict]:
    obj_list = [requests]
    dist.broadcast_object_list(obj_list, src=0)
    result = obj_list[0]
    return result if result is not None else []


def run_load_test(
    server: InferenceServer8GPU,
    *,
    duration: float,
    target_qps: float,
    interval: float,
    prompt_range: tuple[int, int],
    max_new_tokens: int,
    temperature: float,
    rng: np.random.Generator,
) -> Dict:
    """Main load execution loop (identical across all ranks)."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    start_time = time.time()
    next_tick = start_time
    generated_counter = 0

    completions: List[CompletionRecord] = []

    max_prompt_len = max(1, min(prompt_range[1], server.max_seq_len - 1))
    min_prompt_len = max(1, min(prompt_range[0], max_prompt_len))

    def _on_complete(state: RequestState, completed_at: float) -> None:
        latency_ms = (completed_at - state.request.arrived_at) * 1000.0
        completions.append(
            CompletionRecord(
                request_id=state.request.request_id,
                prompt_tokens=len(state.request.prompt_tokens),
                generated_tokens=len(state.generated_tokens),
                latency_ms=latency_ms,
            )
        )

    server._completion_callback = _on_complete  # type: ignore[attr-defined]

    # Precompute world size to distribute IDs
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    while time.time() - start_time < duration:
        tick_start = time.time()

        # Generate load on rank 0 and broadcast to others
        if rank == 0:
            expected = max(target_qps * interval, 0.0)
            num_requests = rng.poisson(expected)
            new_requests = []
            for _ in range(num_requests):
                prompt_len = int(rng.integers(min_prompt_len, max_prompt_len + 1))
                prompt_tokens = list(range(prompt_len))
                request_id = f"req_{generated_counter}"
                generated_counter += 1
                new_requests.append(
                    {
                        "request_id": request_id,
                        "prompt_tokens": prompt_tokens,
                        "max_new_tokens": max_new_tokens,
                        "temperature": float(temperature),
                        "top_k": 50,
                        "priority": generated_counter % 4,
                        "arrived_at": time.time(),
                    }
                )
        else:
            new_requests = None

        for payload in _broadcast_requests(new_requests):
            request = InferenceRequest(**payload)
            server.scheduler.add_request(request)

        # Continuous batching iteration (replicates serve_loop body)
        batch = server.scheduler.get_next_batch(server.kv_cache)
        if batch:
            server.generate_batch(batch, num_tokens=1)
            server.scheduler.update_completions(
                batch,
                server.kv_cache,
                server._on_request_complete,  # type: ignore[attr-defined]
            )

        # Maintain target tick rate
        tick_elapsed = time.time() - tick_start
        sleep_time = interval - tick_elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    dist.barrier()

    stats = server.scheduler.get_stats()
    elapsed = time.time() - start_time
    return {
        "rank": rank,
        "elapsed": elapsed,
        "stats": stats,
        "completions": [asdict(record) for record in completions],
        "world_size": world_size,
    }


def aggregate_results(local_result: Dict) -> Dict:
    gathered: List[Dict] = [None] * dist.get_world_size()  # type: ignore
    dist.all_gather_object(gathered, local_result)

    total_requests = sum(item["stats"]["total_requests"] for item in gathered)
    completed_requests = sum(item["stats"]["completed_requests"] for item in gathered)
    rejected_requests = sum(item["stats"]["rejected_requests"] for item in gathered)
    tokens_generated = sum(item["stats"]["total_tokens_generated"] for item in gathered)
    elapsed = max(item["elapsed"] for item in gathered)
    latencies = [rec["latency_ms"] for item in gathered for rec in item["completions"]]

    throughput = tokens_generated / elapsed if elapsed > 0 else 0.0
    p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
    p90 = float(np.percentile(latencies, 90)) if latencies else 0.0
    p99 = float(np.percentile(latencies, 99)) if latencies else 0.0

    return {
        "elapsed": elapsed,
        "total_requests": int(total_requests),
        "completed_requests": int(completed_requests),
        "rejected_requests": int(rejected_requests),
        "tokens_generated": int(tokens_generated),
        "throughput_tok_per_s": throughput,
        "latency_p50_ms": p50,
        "latency_p90_ms": p90,
        "latency_p99_ms": p99,
        "samples_collected": len(latencies),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Load test for 8x B200 inference server")
    parser.add_argument("--duration", type=float, default=30.0, help="Total test duration in seconds.")
    parser.add_argument("--target-qps", type=float, default=200.0, help="Synthetic request rate.")
    parser.add_argument("--tick-interval", type=float, default=0.02, help="Scheduler tick interval (seconds).")
    parser.add_argument("--prompt-len-min", type=int, default=64, help="Minimum prompt length.")
    parser.add_argument("--prompt-len-max", type=int, default=2048, help="Maximum prompt length.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="New tokens generated per request.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-layers", type=int, default=32, help="Number of transformer layers in the demo model.")
    parser.add_argument("--d-model", type=int, default=4096, help="Hidden dimension of the demo model.")
    parser.add_argument("--num-heads", type=int, default=64, help="Number of attention heads in the demo model.")
    parser.add_argument("--max-batch-size", type=int, default=256, help="Maximum batch size for the server/model.")
    parser.add_argument("--max-seq-len", type=int, default=16384, help="Maximum sequence length for KV cache and requests.")
    parser.add_argument(
        "--allow-smaller-world",
        action="store_true",
        help="Allow running with world size != 8 (useful for dry runs on limited hardware).",
    )
    parser.add_argument("--output-json", type=Path, help="Optional path to write aggregated results.")
    parser.add_argument("--disable-compile", action="store_true", help="Disable torch.compile on server submodules.")
    parser.add_argument("--disable-prefill-graph", action="store_true", help="Disable CUDA graph capture for prefill phase.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _init_distributed()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size != 8 and not args.allow_smaller_world and rank == 0:
        raise SystemExit(
            f"This harness expects 8 ranks (found {world_size}). "
            "Use --allow-smaller-world to bypass for dry runs."
        )

    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)
    rng = np.random.default_rng(args.seed + rank)

    # Instantiate demo model
    model = DemoCausalLM(
        vocab_size=50000,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_gpus=world_size,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
    )

    server = InferenceServer8GPU(
        model=model,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        enable_compile=not args.disable_compile,
    )

    if args.disable_prefill_graph:
        server._prefill_graph = None
        server._prefill_graph_available = False

    local_result = run_load_test(
        server,
        duration=args.duration,
        target_qps=args.target_qps,
        interval=args.tick_interval,
        prompt_range=(args.prompt_len_min, args.prompt_len_max),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        rng=rng,
    )

    aggregate = aggregate_results(local_result)
    if rank == 0:
        print("\n=== Load Test Summary ===")
        print(f"Duration: {aggregate['elapsed']:.1f}s")
        print(f"Throughput: {aggregate['throughput_tok_per_s']:.0f} tokens/s")
        print(
            f"Requests: total={aggregate['total_requests']} "
            f"completed={aggregate['completed_requests']} "
            f"rejected={aggregate['rejected_requests']}"
        )
        print(
            "Latency (ms): "
            f"P50={aggregate['latency_p50_ms']:.2f} "
            f"P90={aggregate['latency_p90_ms']:.2f} "
            f"P99={aggregate['latency_p99_ms']:.2f}"
        )
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(aggregate, indent=2))
            print(f"Saved aggregate metrics to {args.output_json}")


if __name__ == "__main__":
    main()
