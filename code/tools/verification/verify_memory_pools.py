#!/usr/bin/env python3
"""Verify cudaMallocAsync / memory pool support."""

from __future__ import annotations

import sys

from verify_utils import CudaDriver, format_driver_version

CUDADEV_ATTR_MEMORY_POOLS_SUPPORTED = 115


def main() -> int:
    try:
        cuda = CudaDriver()
    except RuntimeError as exc:
        print(f"[verify_memory_pools] CUDA driver unavailable: {exc}", file=sys.stderr)
        return 1

    print(f"[verify_memory_pools] CUDA driver version: {format_driver_version(cuda.driver_version())}")
    count = cuda.device_count()
    if count == 0:
        print("[verify_memory_pools] No CUDA devices detected.")
        return 1

    failures = []
    for ordinal in range(count):
        device = cuda.device_handle(ordinal)
        name = cuda.device_name(device)
        supported = cuda.attribute(device, CUDADEV_ATTR_MEMORY_POOLS_SUPPORTED)
        print(f"Device {ordinal}: {name}")
        if supported:
            print("  ✓ cudaMallocAsync / memory pools supported.")
        else:
            print("  ✗ Memory pool APIs not supported on this device.")
            failures.append(ordinal)
        print()

    if failures:
        print("[verify_memory_pools] Some devices do not support memory pools.")
        return 1

    print("[verify_memory_pools] Memory pools supported on all devices.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
