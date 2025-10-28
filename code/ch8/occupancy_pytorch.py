"""GPU occupancy heuristics with PyTorch (CUDA 13.0 / PyTorch 2.9).

The focus is on sizing work to keep SMs busy, batching small ops, fusing chains,
and choosing tensor layouts that map to efficient kernels.
"""

from __future__ import annotations
import arch_config  # noqa: F401 - Configure Blackwell optimizations

import time
import torch


def _need_cuda() -> bool:
    if not torch.cuda.is_available():
        print("CUDA not available; occupancy demos require a GPU.")
        return False
    return True


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _benchmark(label: str, fn, iters: int) -> float:
    _sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    elapsed_ms = (time.perf_counter() - start) * 1_000 / iters
    print(f"{label:<32}: {elapsed_ms:7.3f} ms")
    return elapsed_ms


def matmul_size_demo() -> None:
    if not _need_cuda():
        return

    device = torch.device("cuda")
    small = (128, 128)
    large = (4096, 4096)
    a_small = torch.randn(*small, device=device)
    b_small = torch.randn(*small, device=device)
    a_large = torch.randn(*large, device=device)
    b_large = torch.randn(*large, device=device)

    print("\n=== Matrix size and occupancy ===")
    _benchmark("mm 128x128", lambda: torch.mm(a_small, b_small), iters=200)
    _benchmark("mm 4096x4096", lambda: torch.mm(a_large, b_large), iters=5)


def batching_demo() -> None:
    if not _need_cuda():
        return

    device = torch.device("cuda")
    tensors = [torch.randn(256, 256, device=device) for _ in range(32)]
    batched = torch.stack(tensors, dim=0)

    print("\n=== Batching small pointwise ops ===")

    def individual():
        out = []
        for t in tensors:
            y = torch.nn.functional.relu(t)
            y = torch.nn.functional.silu(y)
            out.append(y)
        return out

    def batched_op():
        y = torch.nn.functional.relu(batched)
        return torch.nn.functional.silu(y)

    indiv_ms = _benchmark("Individual 32x", individual, iters=5)
    batch_ms = _benchmark("Batched (N=32)", batched_op, iters=5)
    print(f"Speedup (individual / batched): {indiv_ms / batch_ms:5.2f}x")


def compile_fusion_demo() -> None:
    if not _need_cuda():
        return
    if not hasattr(torch, "compile"):
        print("torch.compile not available; skipping fusion demo.")
        return

    device = torch.device("cuda")
    x = torch.randn(2048, 2048, device=device)

    def eager(inp):
        out = torch.nn.functional.relu(inp)
        out = out * 2.0
        out = torch.nn.functional.gelu(out)
        return out + 1.0

    # Avoid forcing fullgraph so TorchDynamo can handle dynamic shapes; toggle manually if shapes are static.
    compiled = torch.compile(eager, mode="reduce-overhead")
    compiled(x)  # warm-up

    print("\n=== torch.compile for occupancy (kernel fusion) ===")
    eager_ms = _benchmark("Eager", lambda: eager(x), iters=10)
    compiled_ms = _benchmark("Compiled", lambda: compiled(x), iters=10)
    print(f"Speedup (eager / compiled): {eager_ms / compiled_ms:5.2f}x")


def layout_demo() -> None:
    if not _need_cuda():
        return

    device = torch.device("cuda")
    conv = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1).to(device)

    nchw = torch.randn(32, 128, 64, 64, device=device)
    nhwc = nchw.to(memory_format=torch.channels_last)

    print("\n=== Memory layout impact (channels-last) ===")
    print(f"NCHW contiguous: {nchw.is_contiguous()}")
    print(f"NHWC contiguous (channels_last): {nhwc.is_contiguous(memory_format=torch.channels_last)}")

    _benchmark("Conv2d NCHW", lambda: conv(nchw), iters=20)
    _benchmark("Conv2d NHWC", lambda: conv(nhwc), iters=20)


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available; nothing to demonstrate")
        return

    props = torch.cuda.get_device_properties(0)
    print("=== PyTorch occupancy heuristics ===")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"SM count: {props.multi_processor_count}, total memory: {props.total_memory / 1e9:.1f} GB")

    matmul_size_demo()
    batching_demo()
    compile_fusion_demo()
    layout_demo()

    print("\nSuggested profiling commands:")
    print("- ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active python occupancy_pytorch.py")
    print("- torch.profiler.profile(..., activities=[ProfilerActivity.CUDA])")


if __name__ == "__main__":
    main()
