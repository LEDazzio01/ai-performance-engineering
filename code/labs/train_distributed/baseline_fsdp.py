"""Baseline FSDP2 example using native torchrun."""

from __future__ import annotations

import argparse
import os
import random
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler

from labs.train_distributed.training_utils.torchrun_harness import TorchrunScriptBenchmark
from labs.train_distributed.utils import (
    ThroughputTracker,
    create_collate_fn,
    get_model_flops_per_token,
    gpu_memory_usage,
    load_tinystories,
    setup_tokenizer,
)

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline FSDP2 BF16 training")
    parser.add_argument("--steps", type=int, default=100, help="Number of optimizer steps to run")
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--micro-batch-size", type=int, default=1, help="Per-rank microbatch size")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    return parser.parse_args()


def _init_distributed() -> tuple[int, int, int]:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_dataloader(seq_len: int, micro_batch: int, rank: int, world_size: int):
    tokenizer = setup_tokenizer(MODEL_ID)
    dataset = load_tinystories(tokenizer, seq_len, is_main_process=rank == 0)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=create_collate_fn(),
    )
    return dataloader, sampler


def _wrap_fsdp(model: torch.nn.Module) -> FSDP:
    try:
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    except ImportError as exc:
        raise RuntimeError("_wrap_fsdp() requires the `transformers` package") from exc
    auto_wrap = partial(transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer})
    mp_policy = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        use_orig_params=True,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
    )


def main():
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("baseline_fsdp requires the `transformers` package") from exc

    args = parse_args()
    rank, world_size, local_rank = _init_distributed()
    _set_seed(1337 + rank)


    dataloader, sampler = _build_dataloader(args.sequence_length, args.micro_batch_size, rank, world_size)

    config = AutoConfig.from_pretrained(MODEL_ID, use_cache=False, attn_implementation="eager")
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16, attn_implementation="eager")
    fsdp_model = _wrap_fsdp(model)
    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)

    flop_per_token = get_model_flops_per_token(fsdp_model.module.config, args.sequence_length)
    tracker = ThroughputTracker(warmup_steps=5)

    total_updates = args.steps
    is_main = rank == 0
    optimizer_step = 0
    micro_step = 0
    epoch = 0

    while optimizer_step < total_updates:
        sampler.set_epoch(epoch)
        for batch in dataloader:
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = fsdp_model(**batch)
                loss = outputs.loss / args.grad_accum

            loss.backward()
            micro_step += 1
            loss_value = loss.item() * args.grad_accum

            should_step = micro_step % args.grad_accum == 0
            if should_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            metrics = tracker.step(batch["input_ids"].numel(), flop_per_token)
            if (
                metrics
                and should_step
                and is_main
                and (optimizer_step % 5 == 0 or optimizer_step == total_updates)
            ):
                metrics.update(gpu_memory_usage(local_rank))
                msg = (
                    f"[baseline_fsdp] step {optimizer_step}/{total_updates} "
                    f"loss={loss_value:.4f}" + ThroughputTracker.format(metrics, include_memory=True)
                )
                print(msg, flush=True)

            if optimizer_step >= total_updates:
                break

        epoch += 1

    dist.barrier()
    if is_main:
        print("[baseline_fsdp] training completed", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()


def get_benchmark():
    """Expose torchrun-wrapped benchmark for the harness."""
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parent / "train_fsdp.py",
        base_args=["--mode", "baseline"],
        config_arg_map={"iterations": "--steps"},
        target_label="labs/train_distributed:fsdp",
        default_nproc_per_node=None,
        name="baseline_fsdp",
    )
