#!/usr/bin/env python3
"""Verify managed/unified memory capabilities."""

from __future__ import annotations

import sys

from verify_utils import CudaDriver, format_driver_version

CUDADEV_ATTR_MANAGED_MEMORY = 83
CUDADEV_ATTR_PAGEABLE_ACCESS = 88
CUDADEV_ATTR_CONCURRENT_MANAGED = 89
CUDADEV_ATTR_DIRECT_MANAGED_FROM_HOST = 101


def main() -> int:
    try:
        cuda = CudaDriver()
    except RuntimeError as exc:
        print(f"[verify_managed_memory] CUDA driver unavailable: {exc}", file=sys.stderr)
        return 1

    print(f"[verify_managed_memory] CUDA driver version: {format_driver_version(cuda.driver_version())}")
    count = cuda.device_count()
    if count == 0:
        print("[verify_managed_memory] No CUDA devices detected.")
        return 1

    failures = []
    for ordinal in range(count):
        device = cuda.device_handle(ordinal)
        name = cuda.device_name(device)
        managed = cuda.attribute(device, CUDADEV_ATTR_MANAGED_MEMORY)
        pageable = cuda.attribute(device, CUDADEV_ATTR_PAGEABLE_ACCESS)
        concurrent = cuda.attribute(device, CUDADEV_ATTR_CONCURRENT_MANAGED)
        direct_host = cuda.attribute(device, CUDADEV_ATTR_DIRECT_MANAGED_FROM_HOST)

        print(f"Device {ordinal}: {name}")
        if managed:
            print("  ✓ Managed memory supported.")
        else:
            print("  ✗ Managed memory allocation not supported.")
            failures.append(ordinal)

        print(f"  {'✓' if pageable else '✗'} Coherent access to pageable memory")
        print(f"  {'✓' if concurrent else '✗'} Concurrent managed access (GPU/CPU overlap)")
        print(f"  {'✓' if direct_host else '✗'} Direct host access to managed memory")
        print()

    if failures:
        print("[verify_managed_memory] Some devices lack core managed-memory support.")
        return 1

    print("[verify_managed_memory] Managed memory fully supported on all devices.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
