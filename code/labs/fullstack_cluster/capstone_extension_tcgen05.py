"""Torch extension loader for the inline tcgen05 kernels."""

from __future__ import annotations

from pathlib import Path

from core.utils.extension_loader_template import load_cuda_extension_v2

try:  # Ensure arch_config clamps TORCH_CUDA_ARCH_LIST before building.
    import arch_config  # noqa: F401
except ImportError:  # pragma: no cover - optional when module unavailable
    arch_config = None  # type: ignore[assignment]

_MODULE = None
_EXT_NAME = "blackwell_capstone_tcgen05_inline_ext"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def load_tcgen05_module():
    """Compile (if needed) and return the inline tcgen05 extension.
    
    Note: We don't cache build errors because they may be transient
    (e.g., CUDA state issues that get resolved). The module cache
    handles successful loads.
    """
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    te_cutlass = _REPO_ROOT / "third_party" / "TransformerEngine" / "3rdparty" / "cutlass" / "include"
    upstream_cutlass = _REPO_ROOT / "third_party" / "cutlass" / "include"
    clang_host = _REPO_ROOT / "third_party" / "llvm" / "bin" / "clang++"

    include_flags = []
    if te_cutlass.exists():
        include_flags.append(f"-I{te_cutlass}")
    include_flags.append(f"-I{upstream_cutlass}")

    extra_cuda_cflags = [
        "-std=c++20",
        "-gencode=arch=compute_100,code=sm_100",
        "-lineinfo",
        "-DCUTE_PREFETCH_COPY_ATOM_DISABLED",
    ] + include_flags
    if clang_host.exists():
        extra_cuda_cflags.append(f"-ccbin={clang_host}")

    try:
        _MODULE = load_cuda_extension_v2(
            name=_EXT_NAME,
            sources=[_REPO_ROOT / "labs" / "fullstack_cluster" / "capstone_kernels_tcgen05.cu"],
            extra_cuda_cflags=extra_cuda_cflags,
            extra_cflags=["-std=c++20"],
            extra_ldflags=["-lcuda"],
        )
    except Exception as exc:  # pragma: no cover - depends on toolchain
        raise RuntimeError(
            "failed to build inline tcgen05 extension. "
            "Requires SM100 hardware and CUDA toolchain with tcgen05 support. "
            f"Original error: {exc}"
        ) from exc

    return _MODULE


__all__ = ["load_tcgen05_module"]
