"""Shared workload configuration for Chapter 13 training/KV-cache benchmarks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chapter13Workload:
    hidden_dim: int = 512
    num_heads: int = 8
    head_dim: int = 64
    tokens_per_step: int = 4
    decode_steps: int = 512
    num_requests: int = 8
    batch_size: int = 4

    # Training standard config - same model for fair checkpointing comparison
    # Larger model to demonstrate checkpointing memory savings
    training_hidden_dim: int = 4096
    training_layers_baseline: int = 48  # Deeper model for activation memory pressure
    training_layers_optimized: int = 48  # Same model with checkpointing
    global_batch_size: int = 256  # Larger batch for more activation memory
    micro_batch_size: int = 32


WORKLOAD = Chapter13Workload()
