"""Optimized DDP training loop showcasing perf levers from the book/labs."""

from __future__ import annotations

import argparse
import math
from time import perf_counter
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import torch.distributed as dist
from accelerate import PartialState
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path

try:
    from arch_config import prefer_sdpa_backends  # type: ignore
    from common.python.compile_utils import enable_tf32  # type: ignore
except Exception:  # pragma: no cover - defensive
    prefer_sdpa_backends = None  # type: ignore
    enable_tf32 = None  # type: ignore

from labs.train_distributed.training_utils.utils import (
    build_dataloader,
    build_text_model,
    build_tokenizer,
    get_dataset,
    set_seed,
)
from labs.train_distributed.training_utils.torchrun_harness import TorchrunScriptBenchmark


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200, help="Number of optimization steps.")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-rank microbatch size.")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="AdamW learning rate.")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile on the model.")
    return parser.parse_args()


def _maybe_fused_adamw(params, lr):
    try:
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            fused=True,
        )
    except TypeError:
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )


def main():
    args = parse_args()
    state = PartialState()
    device = state.device

    set_seed(2025 + state.process_index)
    if enable_tf32 is not None:
        enable_tf32(set_global_precision=True)
    else:
        try:
            torch.set_float32_matmul_precision("high")
            matmul_backend = getattr(torch.backends.cuda, "matmul", None)
            if matmul_backend is not None and hasattr(matmul_backend, "fp32_precision"):
                matmul_backend.fp32_precision = "high"  # type: ignore[attr-defined]
            cudnn_conv = getattr(torch.backends.cudnn, "conv", None)
            if cudnn_conv is not None and hasattr(cudnn_conv, "fp32_precision"):
                cudnn_conv.fp32_precision = "tf32"  # type: ignore[attr-defined]
        except Exception:
            pass
    torch.backends.cudnn.benchmark = True

    tokenizer = build_tokenizer()
    dataset = get_dataset()["train"]

    dataloader = build_dataloader(
        dataset,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        distributed=True,
        num_workers=4,
        prefetch_factor=4,
        pin_memory=True,
    )

    model = build_text_model()
    model.to(device)
    model.train()

    if not torch.distributed.is_initialized():
        raise RuntimeError("DDP optimized run requires torch.distributed process group to be initialized.")

    ddp_model = DDP(
        model,
        device_ids=[device],
        static_graph=True,
        bucket_cap_mb=50,
        gradient_as_bucket_view=True,
    )

    if args.compile:
        ddp_model = torch.compile(ddp_model, mode="max-autotune", fullgraph=True, dynamic=False)

    optimizer = _maybe_fused_adamw(ddp_model.parameters(), args.learning_rate)
    grad_clip = 1.0
    scaler = torch.cuda.amp.GradScaler(enabled=False)  # bf16 path does not need scaling

    num_steps = min(args.steps, len(dataloader))
    total_tokens = 0
    start_time = perf_counter()

    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break

        micro_step = step % args.grad_accum
        sdpa_ctx = prefer_sdpa_backends() if prefer_sdpa_backends is not None else nullcontext()
        with sdpa_ctx, torch.cuda.amp.autocast(dtype=torch.bfloat16):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            batch["labels"] = batch["input_ids"].clone()
            outputs = ddp_model(**batch)
            loss = outputs.loss / args.grad_accum

        scaler.scale(loss).backward()

        if micro_step == args.grad_accum - 1:
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_tokens += batch["input_ids"].numel()

        if step % 10 == 0 and state.is_main_process:
            print(
                f"[optimized-ddp] step {step}/{num_steps} "
                f"loss={loss.item():.4f} "
                f"tokens/step={batch['input_ids'].numel():,}"
            )

    torch.cuda.synchronize(device)
    total_time = perf_counter() - start_time
    if state.is_main_process:
        toks_sec = total_tokens / total_time if total_time > 0 else 0.0
        effective_bs = args.batch_size * args.grad_accum * state.num_processes
        print(
            f"[optimized-ddp] {num_steps} steps | total tokens {total_tokens:,} | "
            f"global batch {effective_bs} | {toks_sec:,.0f} toks/s per rank"
        )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()


def get_benchmark():
    """Expose torchrun-wrapped benchmark for the harness."""
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parent / "ddp.py",
        base_args=["--mode", "optimized"],
        config_arg_map={"iterations": "--steps"},
        target_label="labs/train_distributed:ddp",
        default_nproc_per_node=None,
        name="optimized_ddp",
    )
