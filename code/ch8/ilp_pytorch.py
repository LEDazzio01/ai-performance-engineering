"""Instruction-level parallelism (ILP) illustrations in PyTorch.

The goal is to show how grouping independent work, fusing kernels, and using
mixed precision improves throughput on recent CUDA GPUs.
"""

from __future__ import annotations
import arch_config  # noqa: F401 - Configure Blackwell optimizations

import time
import torch


def _ensure_cuda() -> bool:
    if not torch.cuda.is_available():
        print("CUDA not available; skipping demos.")
        return False
    return True


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _benchmark(label: str, fn, iters: int = 10) -> float:
    _sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    elapsed_ms = (time.perf_counter() - start) * 1_000 / iters
    print(f"{label:<30}: {elapsed_ms:7.3f} ms")
    return elapsed_ms


def basic_ilp_demo() -> None:
    """Compare dependent vs. independent elementwise ops."""
    if not _ensure_cuda():
        return

    device = torch.device("cuda")
    n = 1_000_000
    a, b, c, d = [torch.randn(n, device=device) for _ in range(4)]

    def dependent():
        x = a * a
        y = x + b
        z = y * c
        return z + d

    def independent():
        acc = (a * a) + (b * b)
        acc += (c * c) + (d * d)
        return acc

    print("\n=== Elementwise ILP ===")
    dep = _benchmark("Dependent chain", dependent)
    indep = _benchmark("Independent accumulators", independent)
    print(f"Speedup (dependent / independent): {dep / indep:5.2f}x")


def fusion_with_torch_compile() -> None:
    """Show how torch.compile can fuse chained elementwise work."""
    if not _ensure_cuda():
        return
    if not hasattr(torch, "compile"):
        print("torch.compile unavailable; skipping fusion demo.")
        return

    device = torch.device("cuda")
    x = torch.randn(1_000_000, device=device)

    def unfused(inp):
        y1 = torch.sin(inp)
        y2 = torch.cos(inp)
        y3 = torch.exp(inp * 0.1)
        y4 = torch.log(torch.abs(inp) + 1)
        return y1 + y2 + y3 + y4

    # Do not force fullgraph so dynamic shapes remain supported; enable manually for stable workloads.
    compiled = torch.compile(unfused, mode="reduce-overhead")
    compiled(x)  # warm-up

    print("\n=== Kernel fusion with torch.compile ===")
    unfused_ms = _benchmark("Eager", lambda: unfused(x))
    fused_ms = _benchmark("torch.compile", lambda: compiled(x))
    speedup = unfused_ms / fused_ms if fused_ms else float("inf")
    print(f"Fusion speedup: {speedup:5.2f}x")


def mixed_precision_ilp() -> None:
    """Demonstrate FP32 vs FP16 throughput advantages."""
    if not _ensure_cuda():
        return

    device = torch.device("cuda")
    m = 2048
    a_fp32 = torch.randn(m, m, device=device, dtype=torch.float32)
    b_fp32 = torch.randn(m, m, device=device, dtype=torch.float32)
    a_fp16 = a_fp32.to(torch.float16)
    b_fp16 = b_fp32.to(torch.float16)

    print("\n=== Mixed precision matmul throughput ===")
    fp32_ms = _benchmark("torch.mm FP32", lambda: torch.mm(a_fp32, b_fp32), iters=5)
    fp16_ms = _benchmark("torch.mm FP16", lambda: torch.mm(a_fp16, b_fp16), iters=5)

    def autocast_mm():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return torch.mm(a_fp32, b_fp32)

    autocast_ms = _benchmark("Autocast FP16", autocast_mm, iters=5)
    print(f"FP16 speedup:   {fp32_ms / fp16_ms:5.2f}x")
    print(f"Autocast speedup: {fp32_ms / autocast_ms:5.2f}x")


def reduction_ilp() -> None:
    """Highlight chunked reductions (multiple accumulators)."""
    if not _ensure_cuda():
        return

    device = torch.device("cuda")
    data = torch.randn(1_000_000, device=device)

    print("\n=== Reduction ILP ===")

    def standard_sum():
        return data.sum()

    def chunked_sum(chunk: int = 4):
        length = data.numel()
        pad = (-length) % chunk
        if pad:
            padded = torch.nn.functional.pad(data, (0, pad))
        else:
            padded = data
        reshaped = padded.view(-1, chunk)
        partial = reshaped.sum(dim=1)
        return partial.sum()

    standard_ms = _benchmark("torch.sum", standard_sum)
    chunked_ms = _benchmark("Chunked sum", chunked_sum)
    print(f"Chunking speedup: {standard_ms / chunked_ms:5.2f}x")


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available; ILP demos require a GPU")
        return

    print("=== PyTorch ILP demonstrations ===")
    print(f"Device: {torch.cuda.get_device_name()}\n")

    basic_ilp_demo()
    fusion_with_torch_compile()
    mixed_precision_ilp()
    reduction_ilp()

    print("\nTakeaways:")
    print("- Expose independent math to keep the GPU issue pipeline full.")
    print("- torch.compile can fuse long chains of pointwise ops.")
    print("- FP16/Autocast increases math throughput on Tensor Core GPUs.")
    print("- Chunked reductions mimic manual ILP (multiple accumulators).")


if __name__ == "__main__":
    main()
