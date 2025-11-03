#!/usr/bin/env python3
"""Verify readiness for device-initiated CUDA Graph launches."""

from __future__ import annotations

import sys

from verify_utils import CudaDriver, format_driver_version

CUDADEV_ATTR_COMPUTE_CAPABILITY_MAJOR = 75
CUDADEV_ATTR_COMPUTE_CAPABILITY_MINOR = 76

REQUIRED_CC = (8, 0)  # device-side graph launch available Hopper+


def supports_device_graph_launch(major: int, minor: int) -> bool:
    return (major, minor) >= REQUIRED_CC


def main() -> int:
    try:
        cuda = CudaDriver()
    except RuntimeError as exc:
        print(f"[verify_graph_launch] CUDA driver unavailable: {exc}", file=sys.stderr)
        return 1

    print(f"[verify_graph_launch] CUDA driver version: {format_driver_version(cuda.driver_version())}")
    count = cuda.device_count()
    if count == 0:
        print("[verify_graph_launch] No CUDA devices detected.")
        return 1

    failures = []
    for ordinal in range(count):
        device = cuda.device_handle(ordinal)
        name = cuda.device_name(device)
        major = cuda.attribute(device, CUDADEV_ATTR_COMPUTE_CAPABILITY_MAJOR)
        minor = cuda.attribute(device, CUDADEV_ATTR_COMPUTE_CAPABILITY_MINOR)

        print(f"Device {ordinal}: {name} (CC {major}.{minor})")
        if supports_device_graph_launch(major, minor):
            print("  ✓ Device-initiated CUDA Graph launches supported (CC ≥ 8.0).")
        else:
            print("  ✗ Requires compute capability 8.0 or newer for device-side graph launch.")
            failures.append(ordinal)
        print()

    if failures:
        print("[verify_graph_launch] Some devices cannot issue device-side CUDA Graph launches.")
        return 1

    print("[verify_graph_launch] All devices meet requirements for device-side CUDA Graph launches.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
