#!/usr/bin/env python3
"""
verify_thread_block_clusters.py
==============================
Verify that the active CUDA devices support thread-block cluster launch,
which underpins features such as distributed shared memory and the
warp-specialized pipelines used by Blackwell-era kernels.

The script exits with status 0 when every device advertises cluster launch
support, otherwise 1.
"""

from __future__ import annotations

import ctypes
import sys

CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120


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

        lib.cuDeviceGetAttribute.restype = ctypes.c_int
        lib.cuDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]

        lib.cuGetErrorName.restype = ctypes.c_int
        lib.cuGetErrorName.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]

        lib.cuGetErrorString.restype = ctypes.c_int
        lib.cuGetErrorString.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]

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

    def attribute(self, device: int, attr: int) -> int:
        value = ctypes.c_int()
        self._check(self.lib.cuDeviceGetAttribute(ctypes.byref(value), attr, device),
                    f"cuDeviceGetAttribute(attr={attr})")
        return value.value


def format_driver_version(version: int) -> str:
    major = version // 1000
    minor = (version % 1000) // 10
    return f"{major}.{minor}"


def main() -> int:
    try:
        cuda = CudaDriver()
    except RuntimeError as exc:
        print(f"[verify_thread_block_clusters] CUDA driver unavailable: {exc}", file=sys.stderr)
        return 1

    print(f"[verify_thread_block_clusters] CUDA driver version: {format_driver_version(cuda.driver_version())}")
    count = cuda.device_count()
    if count == 0:
        print("[verify_thread_block_clusters] No CUDA devices detected.")
        return 1

    failures = []
    for ordinal in range(count):
        device = cuda.device_handle(ordinal)
        name = cuda.device_name(device)
        major = cuda.attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        minor = cuda.attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        cluster = cuda.attribute(device, CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH)

        print(f"Device {ordinal}: {name} (CC {major}.{minor})")
        if cluster == 0:
            print("  ✗ Thread-block cluster launch not supported.")
            failures.append(ordinal)
        else:
            print("  ✓ Thread-block cluster launch supported.")
        print()

    if failures:
        print("[verify_thread_block_clusters] One or more devices are missing cluster launch support.")
        return 1

    print("[verify_thread_block_clusters] All devices support thread-block clusters.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
