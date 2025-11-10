"""Shared helpers for tcgen05-capable benchmarks."""

from __future__ import annotations

from typing import Callable, Optional

import torch

try:  # Ensure TORCH_CUDA_ARCH_LIST gets clamped to sm_120 on GB10.
    import arch_config  # noqa: F401
except ImportError:  # pragma: no cover - optional during import bootstrap
    arch_config = None  # type: ignore[assignment]

from common.python.blackwell_requirements import ensure_blackwell_tma_supported


def ensure_tcgen05_supported(
    loader: Optional[Callable[[], object]] = None,
    *,
    module_name: str = "tcgen05 kernel",
) -> None:
    """Raise a SKIPPED error if tcgen05 kernels cannot run."""
    ensure_blackwell_tma_supported(module_name)
    major, minor = torch.cuda.get_device_capability()
    if major < 10:
        raise RuntimeError(
            "SKIPPED: tcgen05 kernels require SM100-class Tensor Cores."
        )
    if major > 10:
        raise RuntimeError(
            (
                "SKIPPED: tcgen05 kernels currently target SM100 (B100/B200) "
                f"but detected sm_{major}{minor:02d} hardware."
            )
        )
    if loader is None:
        return
    try:
        loader()
    except RuntimeError as exc:
        raise RuntimeError(f"SKIPPED: {module_name} unavailable ({exc})") from exc


def check_tcgen05_support(
    loader: Optional[Callable[[], object]] = None,
    *,
    module_name: str = "tcgen05 kernel",
) -> tuple[bool, Optional[str]]:
    """Return (is_supported, reason) without raising on SKIPPED errors."""
    try:
        ensure_tcgen05_supported(loader=loader, module_name=module_name)
        return True, None
    except RuntimeError as exc:
        message = str(exc)
        if "SKIPPED" not in message:
            raise
        return False, message
