#!/usr/bin/env python3
"""Report SM/resources limits for each CUDA device."""

from __future__ import annotations

import sys

from verify_utils import CudaDriver, format_driver_version

# Attribute constants
ATTR = {
    "multi_processor_count": (16, "SM count"),
    "max_threads_per_sm": (39, "Max threads per SM"),
    "max_threads_per_block": (1, "Max threads per block"),
    "max_blocks_per_sm": (106, "Max blocks per SM"),
    "warp_size": (10, "Warp size"),
    "max_shared_mem_block": (8, "Shared memory per block (bytes)"),
    "max_shared_mem_block_optin": (97, "Shared memory per block optin (bytes)"),
    "max_shared_mem_sm": (81, "Shared memory per SM (bytes)"),
    "max_registers_per_block": (12, "Registers per block"),
    "max_registers_per_sm": (82, "Registers per SM"),
    "max_block_dim_x": (2, "Max block dim X"),
    "max_block_dim_y": (3, "Max block dim Y"),
    "max_block_dim_z": (4, "Max block dim Z"),
    "max_grid_dim_x": (5, "Max grid dim X"),
    "max_grid_dim_y": (6, "Max grid dim Y"),
    "max_grid_dim_z": (7, "Max grid dim Z"),
    "clock_rate": (13, "SM clock (kHz)"),
    "l2_cache_size": (38, "L2 cache size (bytes)"),
}


def main() -> int:
    try:
        cuda = CudaDriver()
    except RuntimeError as exc:
        print(f"[verify_sm_resources] CUDA driver unavailable: {exc}", file=sys.stderr)
        return 1

    print(f"[verify_sm_resources] CUDA driver version: {format_driver_version(cuda.driver_version())}")
    count = cuda.device_count()
    if count == 0:
        print("[verify_sm_resources] No CUDA devices detected.")
        return 1

    for ordinal in range(count):
        device = cuda.device_handle(ordinal)
        name = cuda.device_name(device)
        total_mem = cuda.total_mem(device)

        print(f"\nDevice {ordinal}: {name}")
        print(f"  Global memory: {total_mem / (1024 ** 3):.2f} GiB")

        for key, (attr_id, label) in ATTR.items():
            value = cuda.attribute(device, attr_id)
            if key == "clock_rate":
                print(f"  {label}: {value / 1000:.2f} MHz")
            else:
                print(f"  {label}: {value}")

    print()
    print("[verify_sm_resources] Resource limits listed above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
