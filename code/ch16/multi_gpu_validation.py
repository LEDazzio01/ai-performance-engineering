"""
Utility script to sanity-check tensor-parallel execution of the GPT benchmark model.

Example:
    python ch16/multi_gpu_validation.py --tensor-parallel-gpus 4 --dtype bfloat16
"""

from __future__ import annotations

import argparse

import torch

from ch16.test_gpt_large_optimized import (
    GPTConfig,
    GPTModel,
    validate_multi_gpu_equivalence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Validate GPT tensor-parallel execution consistency")
    parser.add_argument("--tensor-parallel-gpus", type=int, default=2, help="Number of GPUs to test.")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--attention-backend", choices=["auto", "sdpa", "flex"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for multi-GPU validation.")

    available = torch.cuda.device_count()
    if args.tensor_parallel_gpus > available:
        raise SystemExit(f"Requested {args.tensor_parallel_gpus} GPUs but only {available} available.")

    devices = [torch.device(f"cuda:{idx}") for idx in range(args.tensor_parallel_gpus)]
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    config = GPTConfig(
        vocab_size=32000,
        n_layers=6,
        n_heads=16,
        d_model=1024,
        d_ff=4096,
        max_seq_len=2048,
        attention_backend=args.attention_backend,
        attention_window=None,
    )

    diff = validate_multi_gpu_equivalence(config, devices, dtype)
    if diff is None:
        print("Validation skipped: requires at least 2 GPUs.")
        return

    print("=== Tensor Parallel Consistency Check ===")
    print(f"GPUs tested: {len(devices)}")
    print(f"Dtype: {args.dtype}")
    print(f"Attention backend: {args.attention_backend}")
    print(f"Max absolute deviation: {diff:.3e}")
    if diff < 1e-3:
        print("Result: PASS (outputs aligned across tensor partitions)")
    else:
        print("Result: FAIL (consider investigating numerical drift)")


if __name__ == "__main__":
    main()
