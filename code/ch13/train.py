#!/usr/bin/env python3
"""Minimal training loop used in Chapter 13 NUMA examples."""

import torch
from torch import nn, optim


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    model = nn.Sequential(
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    data = torch.randn(512, 1024, device=device)
    target = torch.randn(512, 1, device=device)

    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        out = model(data)
        loss = nn.functional.mse_loss(out, target)
        loss.backward()
        optimizer.step()

    print("Training loop completed; final loss {:.4f}".format(loss.item()))


if __name__ == "__main__":
    main()

