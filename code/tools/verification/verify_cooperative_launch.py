#!/usr/bin/env python3
"""Verify cooperative-kernel launch capability."""

from __future__ import annotations

import sys

from verify_utils import CudaDriver, format_driver_version

CUDADEV_ATTR_COMPUTE_CAPABILITY_MAJOR = 75
CUDADEV_ATTR_COMPUTE_CAPABILITY_MINOR = 76
CUDADEV_ATTR_COOPERATIVE_LAUNCH = 95


def main() -> int:
    try:
        cuda = CudaDriver()
    except RuntimeError as exc:
        print(f"[verify_cooperative_launch] CUDA driver unavailable: {exc}", file=sys.stderr)
        return 1

    print(f"[verify_cooperative_launch] CUDA driver version: {format_driver_version(cuda.driver_version())}")
    count = cuda.device_count()
    if count == 0:
        print("[verify_cooperative_launch] No CUDA devices detected.")
        return 1

    failures = []
    for ordinal in range(count):
        device = cuda.device_handle(ordinal)
        name = cuda.device_name(device)
        major = cuda.attribute(device, CUDADEV_ATTR_COMPUTE_CAPABILITY_MAJOR)
        minor = cuda.attribute(device, CUDADEV_ATTR_COMPUTE_CAPABILITY_MINOR)
        coop = cuda.attribute(device, CUDADEV_ATTR_COOPERATIVE_LAUNCH)

        print(f"Device {ordinal}: {name} (CC {major}.{minor})")
        if coop:
            print("  ✓ Cooperative launch supported.")
        else:
            print("  ✗ Cooperative launch not supported on this device.")
            failures.append(ordinal)
        print()

    if failures:
        print("[verify_cooperative_launch] One or more devices lack cooperative launch support.")
        return 1

    print("[verify_cooperative_launch] All devices support cooperative kernel launch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
