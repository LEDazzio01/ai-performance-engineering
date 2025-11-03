#!/usr/bin/env python3
"""Dynamic KV cache quantization demo from Chapter 19."""

import torch


def quantize_to_fp8(tensor: torch.Tensor) -> torch.Tensor:
    scale = torch.max(torch.abs(tensor)).clamp(min=1e-6)
    q = torch.clamp((tensor / scale) * 127.0, -127.0, 127.0).round()
    return q.to(torch.int8), scale


def dequantize_from_fp8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (q.to(torch.float32) / 127.0) * scale


def main() -> None:
    torch.manual_seed(0)
    kv_cache = torch.randn(8, 16, 128, device="cuda", dtype=torch.float32)
    quantized, scale = quantize_to_fp8(kv_cache)
    restored = dequantize_from_fp8(quantized, scale)
    error = (kv_cache - restored).abs().max().item()
    print(f"Quantized cache restored with max error {error:.4f}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required for dynamic_quantized_cache demo.")
    main()

