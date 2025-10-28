"""Profiling helper for DeepSeek Coder training (Chapter 13).

Demonstrates DeepSeek architecture training with:
- DeepSeek Coder 6.7B model (real DeepSeek architecture)
- Warmup loop outside the profiler
- AMP/fused optimizer for B200 performance
- CUDA Graph capture compatible data
- Proper profiling workflow for large models

Note: Uses DeepSeek Coder 6.7B (manageable size for single GPU)
For full DeepSeek-V3, see multi-GPU examples in ch13/fsdp_example.py
"""

from __future__ import annotations
import arch_config  # noqa: F401 - Configure Blackwell optimizations

import json
import os
from contextlib import nullcontext

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, AutoTokenizer

# Using real DeepSeek Coder model (6.7B parameters)
# This is a real DeepSeek architecture, not GPT-2!
MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-base"
BATCH = 2
WARMUP = 2
PROFILE_STEPS = 3


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=device.type == "cuda")

    texts = ["DeepSeek Coder is optimized for code generation." for _ in range(BATCH)]
    batch = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    labels = batch["input_ids"].clone()

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    autocast_ctx = torch.autocast(device_type="cuda") if device.type == "cuda" else nullcontext()

    model.train()
    for _ in range(WARMUP):
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
        for _ in range(PROFILE_STEPS):
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
