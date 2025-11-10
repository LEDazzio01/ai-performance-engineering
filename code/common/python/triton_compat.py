"""Utilities that keep Triton compatible with Grace-Blackwell GPUs.

Triton 3.5+ may emit architecture names such as ``sm_121a`` that ``ptxas``
doesn't understand even though the underlying GPU (e.g. GB10) supports the
feature set.  Importing this module patches Triton's CUDA backend so the
generated PTX/CUBIN requests ``sm_121`` instead.  The patch is idempotent and
can be disabled by setting ``ENABLE_TRITON_PATCH=0`` in the environment.
"""

from __future__ import annotations

import os
from typing import Any

import torch

ENABLE_TRITON_PATCH = os.environ.get("ENABLE_TRITON_PATCH", "1") == "1"
_VERBOSE = os.environ.get("VERBOSE_EXPERIMENTAL_FEATURES", "0") == "1"
_TRITON_ARCH_PATCHED = False


def _clamp_triton_codegen_arch() -> None:
    """Force Triton to target sm_120 on GB10 until CUDA 13.1 adds sm_121."""
    if "TRITON_CODEGEN_ARCH" in os.environ:
        return
    if not torch.cuda.is_available():
        return
    major, _ = torch.cuda.get_device_capability()
    if major >= 12:
        os.environ["TRITON_CODEGEN_ARCH"] = "sm_120"
        _log("INFO: Triton codegen clamped to sm_120 for GB10 compatibility")


def _log(message: str) -> None:
    if _VERBOSE:
        print(message)


def ensure_triton_compat() -> None:
    """Apply the Triton SM-architecture patch if needed."""
    global _TRITON_ARCH_PATCHED

    if _TRITON_ARCH_PATCHED:
        return
    if not ENABLE_TRITON_PATCH:
        _log("INFO:  Triton SM arch patch disabled via ENABLE_TRITON_PATCH=0")
        return

    try:
        import triton  # type: ignore
        import triton.backends.nvidia.compiler as triton_compiler  # type: ignore
    except (ImportError, ModuleNotFoundError):
        _log("WARNING: Triton SM arch patch: Triton not available")
        return

    try:
        version_tokens = getattr(triton, "__version__", "0.0.0").split(".")
        triton_version = tuple(int(token) for token in version_tokens[:3])
        if triton_version < (3, 5, 0):
            _log(
                f"INFO:  Triton SM arch patch: Skipping "
                f"(Triton {getattr(triton, '__version__', 'unknown')} < 3.5.0)"
            )
            return
    except Exception as exc:  # pragma: no cover - defensive
        _log(f"WARNING: Triton SM arch patch: Could not check version ({exc}), applying patch anyway")

    if getattr(triton_compiler, "_sm_arch_patch_applied", False):
        _TRITON_ARCH_PATCHED = True
        _log("PASSED: Triton SM arch patch: Already applied")
        return

    original_sm_arch = triton_compiler.sm_arch_from_capability

    def _safe_sm_arch_from_capability(capability: int, _orig=original_sm_arch) -> str:
        arch = _orig(capability)
        if arch.endswith("a"):
            arch = arch[:-1]
        if arch == "sm_121":
            # CUDA 13.0's ptxas still lacks native sm_121 support. Clamp to sm_120
            # so Triton kernels build on GB10 until NVIDIA ships an updated assembler.
            arch = "sm_120"
        return arch

    try:
        triton_compiler.sm_arch_from_capability = _safe_sm_arch_from_capability  # type: ignore[assignment]
        triton_compiler._sm_arch_patch_applied = True  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - defensive
        _log(f"WARNING: Triton SM arch patch: Failed to patch sm_arch_from_capability ({exc})")
        return

    implementation_cls: Any = getattr(triton_compiler, "CUDABackend", None)
    if implementation_cls is not None and not hasattr(implementation_cls, "_capability_clamp_patch"):
        original_make_ptx = implementation_cls.make_ptx
        original_make_cubin = implementation_cls.make_cubin

        def _clamp_capability(capability: int) -> int:
            # Triton passes major * 10 + minor here (e.g., 121). Clamp GB10 to 120
            # so both ptxas flags and PTX headers agree on a supported target.
            return 120 if capability == 121 else capability

        def _make_ptx_with_clamp(self, src, metadata, opt, capability, _orig=original_make_ptx):
            return _orig(self, src, metadata, opt, _clamp_capability(capability))

        def _make_cubin_with_clamp(self, src, metadata, opt, capability, _orig=original_make_cubin):
            return _orig(self, src, metadata, opt, _clamp_capability(capability))

        try:
            implementation_cls.make_ptx = _make_ptx_with_clamp  # type: ignore[assignment]
            implementation_cls.make_cubin = _make_cubin_with_clamp  # type: ignore[assignment]
            implementation_cls._capability_clamp_patch = True  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - defensive
            _log(f"WARNING: Triton SM arch patch: Failed to patch CUDABackend ({exc})")

    _TRITON_ARCH_PATCHED = True
    _log("PASSED: Triton SM arch patch: Applied successfully")


# Apply the patch immediately so importing this module is enough.
_clamp_triton_codegen_arch()
ensure_triton_compat()

__all__ = ["ENABLE_TRITON_PATCH", "ensure_triton_compat"]
