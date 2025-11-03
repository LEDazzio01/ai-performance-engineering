#!/usr/bin/env python3
"""
verify_hbm.py
=============
Report high-bandwidth-memory (HBM) characteristics for each CUDA device and
compute the theoretical peak bandwidth based on reported bus width and memory
clock.

The script exits with status 0 but highlights devices where the driver fails to
report the required properties.
"""

from __future__ import annotations

import ctypes
import math
import sys


CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36          # in kHz
CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37    # in bits
CU_DEVICE_ATTRIBUTE_MEMORY_POOL_SUPPORTED = 105     # optional sanity check

CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76


class CudaDriver:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcuda.so")
        self._bind()
        self._init()

    def _bind(self) -> None:
        lib = self.lib
        lib.cuInit.restype = ctypes.c_int
        lib.cuInit.argtypes = [ctypes.c_uint]

        lib.cuDriverGetVersion.restype = ctypes.c_int
        lib.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]

        lib.cuDeviceGetCount.restype = ctypes.c_int
        lib.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]

        lib.cuDeviceGet.restype = ctypes.c_int
        lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]

        lib.cuDeviceGetName.restype = ctypes.c_int
        lib.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]

        lib.cuDeviceTotalMem_v2.restype = ctypes.c_int
        lib.cuDeviceTotalMem_v2.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]

        lib.cuDeviceGetAttribute.restype = ctypes.c_int
        lib.cuDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]

        lib.cuGetErrorName.restype = ctypes.c_int
        lib.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]

        lib.cuGetErrorString.restype = ctypes.c_int
        lib.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]

    def _init(self) -> None:
        self._check(self.lib.cuInit(0), "cuInit")

    def _check(self, status: int, func: str) -> None:
        if status == CUDA_SUCCESS:
            return
        err_name = ctypes.c_char_p()
        err_str = ctypes.c_char_p()
        if self.lib.cuGetErrorName(status, ctypes.byref(err_name)) != CUDA_SUCCESS:
            err_name.value = b"CUDA_ERROR_UNKNOWN"
        if self.lib.cuGetErrorString(status, ctypes.byref(err_str)) != CUDA_SUCCESS:
            err_str.value = b"Unknown error"
        raise RuntimeError(f"{func} failed: {err_name.value.decode()} - {err_str.value.decode()}")

    def driver_version(self) -> int:
        version = ctypes.c_int()
        self._check(self.lib.cuDriverGetVersion(ctypes.byref(version)), "cuDriverGetVersion")
        return version.value

    def device_count(self) -> int:
        count = ctypes.c_int()
        self._check(self.lib.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
        return count.value

    def device_handle(self, ordinal: int) -> int:
        dev = ctypes.c_int()
        self._check(self.lib.cuDeviceGet(ctypes.byref(dev), ordinal), "cuDeviceGet")
        return dev.value

    def device_name(self, device: int) -> str:
        buffer = ctypes.create_string_buffer(100)
        self._check(self.lib.cuDeviceGetName(buffer, len(buffer), device), "cuDeviceGetName")
        return buffer.value.decode()

    def total_mem(self, device: int) -> int:
        value = ctypes.c_size_t()
        self._check(self.lib.cuDeviceTotalMem_v2(ctypes.byref(value), device), "cuDeviceTotalMem")
        return value.value

    def attribute(self, device: int, attr: int) -> int:
        value = ctypes.c_int()
        self._check(self.lib.cuDeviceGetAttribute(ctypes.byref(value), attr, device),
                    f"cuDeviceGetAttribute(attr={attr})")
        return value.value


def format_driver_version(version: int) -> str:
    major = version // 1000
    minor = (version % 1000) // 10
    return f"{major}.{minor}"


def bytes_to_gib(bytes_value: int) -> float:
    return bytes_value / (1024 ** 3)


def bandwidth_bytes_per_sec(clock_khz: int, bus_width_bits: int) -> float:
    # DDR: two transfers per cycle. clock_khz -> cycles per second * 1000
    return 2.0 * clock_khz * 1_000.0 * (bus_width_bits / 8.0)


def main() -> int:
    try:
        cuda = CudaDriver()
    except RuntimeError as exc:
        print(f"[verify_hbm] CUDA driver unavailable: {exc}", file=sys.stderr)
        return 1

    print(f"[verify_hbm] CUDA driver version: {format_driver_version(cuda.driver_version())}")
    count = cuda.device_count()
    if count == 0:
        print("[verify_hbm] No CUDA devices detected.")
        return 1

    for ordinal in range(count):
        device = cuda.device_handle(ordinal)
        name = cuda.device_name(device)
        total_mem_bytes = cuda.total_mem(device)
        major = cuda.attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        minor = cuda.attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)

        mem_clock_khz = cuda.attribute(device, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)
        bus_width_bits = cuda.attribute(device, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)

        print(f"\nDevice {ordinal}: {name} (CC {major}.{minor})")
        print(f"  Total global memory: {bytes_to_gib(total_mem_bytes):.2f} GiB")

        if mem_clock_khz <= 0 or bus_width_bits <= 0:
            print("  ⚠️  Driver did not report memory clock / bus width; unable to compute theoretical bandwidth.")
            continue

        peak_bytes = bandwidth_bytes_per_sec(mem_clock_khz, bus_width_bits)
        peak_tbps = peak_bytes / 1e12
        peak_gbps = peak_bytes / 1e9

        print(f"  Memory clock: {mem_clock_khz / 1_000:.2f} MHz")
        print(f"  Bus width:    {bus_width_bits} bits")
        print(f"  Theoretical peak bandwidth: {peak_tbps:.2f} TB/s ({peak_gbps:.1f} GB/s)")

    print()
    print("[verify_hbm] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
