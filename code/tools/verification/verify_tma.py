#!/usr/bin/env python3
"""
verify_tma.py
==============
Check whether each visible CUDA device supports the Tensor Memory Accelerator (TMA)
introduced with Hopper/Blackwell GPUs.

The script exits with status code 0 when all devices support TMA, otherwise 1.
"""

from __future__ import annotations

import ctypes
import sys
from typing import Callable


CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = 127


class CudaDriver:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcuda.so")
        self._bind_functions()
        self._init_driver()

    def _bind_functions(self) -> None:
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

    def _init_driver(self) -> None:
        status = self.lib.cuInit(0)
        self._check(status, "cuInit")

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

    def device_attribute(self, device: int, attr: int) -> int:
        value = ctypes.c_int()
        status = self.lib.cuDeviceGetAttribute(ctypes.byref(value), attr, device)
        self._check(status, f"cuDeviceGetAttribute(attr={attr})")
        return value.value


def format_driver_version(version: int) -> str:
    major = version // 1000
    minor = (version % 1000) // 10
    return f"{major}.{minor}"


def main() -> int:
    try:
        cuda = CudaDriver()
    except RuntimeError as exc:
        print(f"[verify_tma] CUDA driver unavailable: {exc}", file=sys.stderr)
        return 1

    driver_version = format_driver_version(cuda.driver_version())
    count = cuda.device_count()
    if count == 0:
        print("[verify_tma] No CUDA devices detected.")
        return 1

    print(f"[verify_tma] CUDA driver version: {driver_version}")
    print(f"[verify_tma] Detected {count} CUDA device(s)\n")

    failures = []
    for ordinal in range(count):
        device = cuda.device_handle(ordinal)
        name = cuda.device_name(device)
        major = cuda.device_attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        minor = cuda.device_attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        tma_supported = cuda.device_attribute(
            device, CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED
        )

        print(f"Device {ordinal}: {name} (CC {major}.{minor})")
        if major < 9:
            print("  ✗ Tensor Memory Accelerator introduced in SM 90 (Hopper) – unsupported.")
            failures.append(ordinal)
        elif tma_supported == 0:
            print("  ✗ Driver reports no Tensor Map access support.")
            failures.append(ordinal)
        else:
            print("  ✓ Tensor Memory Accelerator supported.")
        print()

    if failures:
        print("[verify_tma] One or more devices lack TMA support.")
        return 1

    print("[verify_tma] All devices support Tensor Memory Accelerator.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
