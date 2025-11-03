#!/usr/bin/env python3
"""Verify peer-to-peer (P2P) access between CUDA devices."""

from __future__ import annotations

import ctypes
import sys

from verify_utils import format_driver_version

CUDA_SUCCESS = 0


class CudaRuntime:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL('libcudart.so')
        self._bind()

    def _bind(self) -> None:
        lib = self.lib
        lib.cudaGetDeviceCount.restype = ctypes.c_int
        lib.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]

        lib.cudaSetDevice.restype = ctypes.c_int
        lib.cudaSetDevice.argtypes = [ctypes.c_int]

        lib.cudaDeviceCanAccessPeer.restype = ctypes.c_int
        lib.cudaDeviceCanAccessPeer.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]

        lib.cudaDeviceGetPCIBusId.restype = ctypes.c_int
        lib.cudaDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]

        lib.cudaDriverGetVersion.restype = ctypes.c_int
        lib.cudaDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]

        lib.cudaGetDeviceProperties.restype = ctypes.c_int
        # not used currently

    def check(self, status: int, func: str) -> None:
        if status != CUDA_SUCCESS:
            raise RuntimeError(f"{func} failed with error code {status}")

    def device_count(self) -> int:
        count = ctypes.c_int()
        self.check(self.lib.cudaGetDeviceCount(ctypes.byref(count)), 'cudaGetDeviceCount')
        return count.value

    def driver_version(self) -> int:
        version = ctypes.c_int()
        self.check(self.lib.cudaDriverGetVersion(ctypes.byref(version)), 'cudaDriverGetVersion')
        return version.value

    def device_bus_id(self, device: int) -> str:
        buffer = ctypes.create_string_buffer(20)
        status = self.lib.cudaDeviceGetPCIBusId(buffer, len(buffer), device)
        if status != CUDA_SUCCESS:
            return 'unknown'
        return buffer.value.decode()

    def can_access_peer(self, actor: int, target: int) -> bool:
        capable = ctypes.c_int()
        self.check(self.lib.cudaDeviceCanAccessPeer(ctypes.byref(capable), actor, target), 'cudaDeviceCanAccessPeer')
        return bool(capable.value)


def main() -> int:
    try:
        runtime = CudaRuntime()
    except RuntimeError as exc:
        print(f"[verify_peer_access] Unable to load libcudart: {exc}", file=sys.stderr)
        return 1

    driver_ver = format_driver_version(runtime.driver_version())
    count = runtime.device_count()
    print(f"[verify_peer_access] CUDA driver version: {driver_ver}")
    if count <= 1:
        print("[verify_peer_access] <2 devices; nothing to verify.")
        return 0

    failures = []
    for i in range(count):
        for j in range(count):
            if i == j:
                continue
            ok = runtime.can_access_peer(i, j)
            bus_i = runtime.device_bus_id(i)
            bus_j = runtime.device_bus_id(j)
            if ok:
                print(f"Device {i} (bus {bus_i}) → Device {j} (bus {bus_j}): ✓ peer access enabled")
            else:
                print(f"Device {i} (bus {bus_i}) → Device {j} (bus {bus_j}): ✗ peer access disabled")
                failures.append((i, j))

    if failures:
        print("[verify_peer_access] One or more device pairs lack peer-to-peer access.")
        return 1

    print("[verify_peer_access] Peer access enabled for all device pairs.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
