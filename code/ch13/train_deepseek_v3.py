#!/usr/bin/env python3
"""Synthetic DeepSeek workload referenced in Chapter 13 profiling walkthrough."""

import math
import torch
from torch import nn


class ExpertMLP(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden * 4)
        self.fc2 = nn.Linear(hidden * 4, hidden)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DeepSeekToy(nn.Module):
    def __init__(self, hidden: int, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(ExpertMLP(hidden) for _ in range(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return x


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    hidden = 3072
    layers = 12
    model = DeepSeekToy(hidden, layers).to(device)
    model = torch.compile(model, mode="reduce-overhead")

    batch = 8
    seq = 1024
    data = torch.randn(batch, seq, hidden, device=device)

    with torch.inference_mode():
        out = model(data)
        checksum = out.square().mean().sqrt()

    tokens = batch * seq
    print(f"DeepSeek v3 synthetic forward pass complete on {tokens} tokens.")
    print(f"Output RMS: {checksum.item():.4f}")


if __name__ == "__main__":
    main()

