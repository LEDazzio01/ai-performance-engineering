#!/usr/bin/env python3
"""Minimal training script used for NUMA pinning examples in Chapter 3."""

import argparse
import torch
from torch import nn, optim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(1024, 2048), nn.GELU(), nn.Linear(2048, 1024)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    data = torch.randn(512, 1024, device=device)
    target = torch.randn(512, 1024, device=device)

    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        out = model(data)
        loss = nn.functional.mse_loss(out, target)
        loss.backward()
        optimizer.step()

    print(f"train.py finished on device {device} with loss {loss.item():.4f}")


if __name__ == "__main__":
    main()

