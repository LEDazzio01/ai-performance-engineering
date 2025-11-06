from __future__ import annotations

"""Profiling helper for DeepSeek Coder training (Chapter 13).

Demonstrates DeepSeek architecture training with:
- DeepSeek Coder 6.7B model (real DeepSeek architecture)
- Warmup loop outside the profiler
- AMP/fused optimizer for B200 performance
- CUDA Graph capture compatible data
- Proper profiling workflow for large models

Note: Uses DeepSeek Coder 6.7B (manageable size for single GPU)
For full DeepSeek-V3, see multi-GPU examples in extras/ch13/fsdp_example.py
"""

import pathlib
import sys

_EXTRAS_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_EXTRAS_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTRAS_REPO_ROOT))

from pathlib import Path
import os



import json
from contextlib import nullcontext

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, AutoTokenizer
from common.device_utils import get_preferred_device


def _select_model() -> tuple[str, int, int]:
    """Choose an appropriate model/batch based on environment."""
    override = os.environ.get("DEEPSEEK_CODER_MODEL")
    quick_mode = (
        os.environ.get("QUICK_PROFILE") == "1"
        or os.environ.get("RUN_ALL_CHAPTERS") == "1"
        or os.environ.get("BENCHMARK_QUICK") == "1"
        or os.environ.get("SKIP_HEAVY_MODELS") == "1"
    )
    if override:
        return override, BATCH, PROFILE_STEPS
    if quick_mode:
        # Tiny GPT-2 stands in for heavy model during CI/automation runs.
        return "sshleifer/tiny-gpt2", 1, 1
    return MODEL_NAME, BATCH, PROFILE_STEPS

# Using real DeepSeek Coder model (6.7B parameters)
# This is a real DeepSeek architecture, not GPT-2!
MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-base"
BATCH = 2
WARMUP = 2
PROFILE_STEPS = 3


def main() -> None:
    device, cuda_err = get_preferred_device()
    if cuda_err:
        print(f"WARNING: CUDA unavailable ({cuda_err}); using CPU.")
    model_name, batch_size, profile_steps = _select_model()
    if model_name != MODEL_NAME:
        print(
            f"Using lightweight model '{model_name}' "
            f"(batch_size={batch_size}, profile_steps={profile_steps}) for quick profiling."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and model_name != "sshleifer/tiny-gpt2":
            # Fallback to tiny model when the requested model exhausts memory.
            print("Falling back to tiny GPT-2 due to OOM during model load.")
            model_name = "sshleifer/tiny-gpt2"
            batch_size = 1
            profile_steps = 1
            device = torch.device("cpu")
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        else:
            raise

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=device.type == "cuda")

    texts = ["DeepSeek Coder is optimized for code generation." for _ in range(batch_size)]
    batch = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    labels = batch["input_ids"].clone()

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    autocast_ctx = torch.autocast(device_type="cuda") if device.type == "cuda" else nullcontext()

    model.train()
    warmup_steps = min(WARMUP, profile_steps + 1)
    for _ in range(warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            out = model(**batch, labels=labels)
            loss = out.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        for _ in range(profile_steps):
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx:
                out = model(**batch, labels=labels)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    prof.export_chrome_trace("deepseek_coder_trace.json")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

    hta_dir = "hta_traces"
    os.makedirs(hta_dir, exist_ok=True)
    with open(os.path.join(hta_dir, "rank_0.json"), "w") as f:
        json.dump(json.loads(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)), f)


if __name__ == "__main__":
    main()
