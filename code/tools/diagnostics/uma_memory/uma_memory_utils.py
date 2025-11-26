"""Helpers for UMA memory reporting on GPU-first systems."""

from __future__ import annotations

import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Optional

import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass
class MeminfoSnapshot:
    """Parsed fields from /proc/meminfo with huge-page awareness."""

    mem_available_kb: int
    swap_free_kb: int
    huge_total_pages: int
    huge_free_pages: int
    huge_page_size_kb: int

    def effective_available_kb(self) -> int:
        """Honor hugetlbfs behavior: huge pages are not swappable."""
        if self.huge_total_pages not in (0, -1):
            return self.huge_free_pages * self.huge_page_size_kb
        return self.mem_available_kb

    def allocatable_bytes(self, reclaim_fraction: float = 1.0) -> int:
        """Estimate allocatable bytes assuming some DRAM can be reclaimed to SWAP."""
        reclaim_fraction = max(0.0, min(reclaim_fraction, 1.0))
        available_kb = max(self.effective_available_kb(), 0)
        if self.huge_total_pages not in (0, -1):
            swap_free_kb = 0
        else:
            swap_free_kb = max(self.swap_free_kb, 0)
        allocatable_kb = available_kb + reclaim_fraction * swap_free_kb
        return int(allocatable_kb * 1024)


def read_meminfo(path: str = "/proc/meminfo") -> Optional[MeminfoSnapshot]:
    """Parse /proc/meminfo analog to the C snippet from the NVIDIA note."""
    mem_available_kb = swap_free_kb = huge_total_pages = huge_free_pages = huge_page_size_kb = -1
    try:
        with open(path, "r", encoding="utf-8") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    mem_available_kb = int(line.split()[1])
                elif line.startswith("SwapFree:"):
                    swap_free_kb = int(line.split()[1])
                elif line.startswith("HugePages_Total:"):
                    huge_total_pages = int(line.split()[1])
                elif line.startswith("HugePages_Free:"):
                    huge_free_pages = int(line.split()[1])
                elif line.startswith("Hugepagesize:"):
                    huge_page_size_kb = int(line.split()[1])
    except OSError:
        return None

    if mem_available_kb == -1 and huge_total_pages in (-1, 0):
        return None

    return MeminfoSnapshot(
        mem_available_kb=mem_available_kb if mem_available_kb != -1 else 0,
        swap_free_kb=swap_free_kb if swap_free_kb != -1 else 0,
        huge_total_pages=huge_total_pages,
        huge_free_pages=huge_free_pages if huge_free_pages != -1 else 0,
        huge_page_size_kb=huge_page_size_kb if huge_page_size_kb != -1 else 0,
    )


def is_integrated_gpu(device_index: int = 0) -> bool:
    """Best-effort detection of iGPU/UMA devices."""
    try:
        props = torch.cuda.get_device_properties(device_index)
    except Exception:
        return False

    for attr in ("is_integrated", "integrated"):
        flag = getattr(props, attr, None)
        if flag is not None:
            return bool(flag)
    return False


def format_bytes(value: int) -> str:
    """Human-readable byte formatter."""
    suffixes = ["B", "KB", "MB", "GB", "TB"]
    float_val = float(value)
    for suffix in suffixes:
        if float_val < 1024.0:
            return f"{float_val:.2f} {suffix}"
        float_val /= 1024.0
    return f"{float_val:.2f} PB"
